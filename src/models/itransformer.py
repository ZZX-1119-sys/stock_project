
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.base import ForecastModel


REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "_reference" / "iTransformer"


def _load_official_model():
    root = str(REFERENCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from model.iTransformer import Model

    return Model


class ITransformer(ForecastModel):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        target_dim,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
        activation="gelu",
        use_norm=True,
        **kwargs,
    ):
        super().__init__(seq_len, pred_len, enc_in, target_dim)

        target_idx = kwargs.pop("target_idx", 3 if target_dim == 1 else 0)
        if target_idx < 0 or target_idx + target_dim > enc_in:
            raise ValueError("target_idx and target_dim must select input channels")
        self.target_idx = target_idx

        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            output_attention=False,
            use_norm=use_norm,
            d_model=d_model,
            embed="timeF",
            freq="h",
            dropout=dropout,
            class_strategy="projection",
            factor=1,
            n_heads=n_heads,
            d_ff=d_ff,
            activation=activation,
            e_layers=e_layers,
        )

        official_model = _load_official_model()
        self.model = official_model(configs)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Input must have shape [Batch, Seq_Len, Features]")

        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.shape[1]}"
            )

        if x.shape[2] != self.enc_in:
            raise ValueError(
                f"Expected {self.enc_in} input features, got {x.shape[2]}"
            )

        batch_size = x.shape[0]
        x_mark_enc = torch.zeros(
            batch_size, self.seq_len, 4, device=x.device, dtype=x.dtype
        )
        x_dec = torch.zeros(
            batch_size,
            self.pred_len,
            self.enc_in,
            device=x.device,
            dtype=x.dtype,
        )
        x_mark_dec = torch.zeros(
            batch_size, self.pred_len, 4, device=x.device, dtype=x.dtype
        )

        output = self.model(x, x_mark_enc, x_dec, x_mark_dec)

        if isinstance(output, tuple):
            output = output[0]

        if self.target_idx == 0 and self.target_dim == self.enc_in:
            return output

        return output[:, :, self.target_idx:self.target_idx + self.target_dim]
