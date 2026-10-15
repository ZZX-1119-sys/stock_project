"""CPU-friendly MambaTS-style reference forecaster.

This module follows the patch embedding, variable-time token ordering,
selective state-space blocks, and per-variable prediction head used by
MambaTS. It uses a PyTorch reference scan so it runs on Windows/CPU without
the official CUDA extensions. It is intended for functional experiments,
not kernel-level speed comparisons with optimized Mamba.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.base import ForecastModel
from src.models.mamba_scan_reference import selective_scan_reference


class _SelectiveStateSpaceBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=4):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(a.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, hidden):
        residual = hidden
        hidden = self.norm(hidden)
        x, gate = self.in_proj(hidden).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., :hidden.shape[1]]
        x = F.silu(x).transpose(1, 2)

        params = self.x_proj(x)
        dt_low, b_value, c_value = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        scanned = selective_scan_reference(
            u=x.transpose(1, 2),
            delta=self.dt_proj(dt_low).transpose(1, 2),
            A=-torch.exp(self.A_log.float()),
            B=b_value.transpose(1, 2),
            C=c_value.transpose(1, 2),
            D=self.D.float(),
            z=gate.transpose(1, 2),
            delta_softplus=True,
        )
        return residual + self.out_proj(scanned.transpose(1, 2))


class MambaTS(ForecastModel):
    """MambaTS-style forecaster with [B,L,C] input/output API."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        target_dim,
        d_model=64,
        d_state=16,
        e_layers=2,
        dropout=0.1,
        patch_len=16,
        stride=16,
        expand=2,
        target_idx=3,
        **kwargs,
    ):
        super().__init__(seq_len, pred_len, enc_in, target_dim)
        if target_dim != 1:
            raise ValueError("MambaTS wrapper currently supports one target feature")
        if not 0 <= target_idx < enc_in:
            raise ValueError("target_idx must be within the input feature range")
        if patch_len < 1 or patch_len > seq_len:
            raise ValueError("patch_len must be between 1 and seq_len")
        if stride < 1:
            raise ValueError("stride must be positive")
        if e_layers < 1:
            raise ValueError("e_layers must be positive")

        self.target_idx = target_idx
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.encoder = nn.ModuleList(
            [_SelectiveStateSpaceBlock(d_model, d_state, expand) for _ in range(e_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.prediction_head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Input must have shape [Batch, Seq_Len, Features]")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.shape[1]}")
        if x.shape[2] != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x.shape[2]}")

        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev

        patches = normalized.transpose(1, 2).unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        batch_size = x.shape[0]
        tokens = self.patch_embedding(patches)
        tokens = tokens.reshape(batch_size, self.enc_in * self.num_patches, -1)
        for block in self.encoder:
            tokens = block(tokens)
        tokens = self.dropout(tokens)

        per_variate = tokens.reshape(
            batch_size, self.enc_in, self.num_patches, -1
        ).reshape(batch_size * self.enc_in, -1)
        forecast = self.prediction_head(per_variate)
        forecast = forecast.reshape(batch_size, self.enc_in, self.pred_len).transpose(1, 2)
        forecast = forecast * stdev + means
        return forecast[:, :, self.target_idx:self.target_idx + 1]
