from abc import ABC, abstractmethod

import torch
from torch import nn


class ForecastModel(nn.Module, ABC):
    def __init__(self, seq_len, pred_len, enc_in, target_dim):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.target_dim = target_dim

    @abstractmethod
    def forward(self, x):
        pass


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def estimate_flops(model, input_shape):
    """Estimate FLOPs for profiler-supported operators on one forward pass.

    PyTorch's profiler reports FLOPs for supported operations (not every
    elementwise or custom operation), so this is a documented partial
    estimate rather than a complete symbolic operation count.
    """
    from torch.profiler import ProfilerActivity, profile

    if len(input_shape) != 3:
        raise ValueError("input_shape must be (batch, seq_len, features)")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    sample = torch.zeros(tuple(input_shape), device=device, dtype=dtype)
    was_training = model.training
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    try:
        model.eval()
        with torch.inference_mode(), profile(
            activities=activities,
            with_flops=True,
        ) as prof:
            model(sample)
    finally:
        model.train(was_training)

    operator_events = [
        event for event in prof.events() if event.name.startswith("aten::")
    ]
    counted_events = [
        event for event in operator_events
        if isinstance(event.flops, (int, float)) and event.flops > 0
    ]
    return {
        "profiler_flops": int(sum(event.flops for event in counted_events)),
        "flop_counted_operator_events": len(counted_events),
        "flop_uncounted_operator_events": len(operator_events) - len(counted_events),
        "flops_method": "torch.profiler with_flops=True; supported operators only",
    }


def get_model_size(model, input_shape=None):
    info = count_parameters(model)
    info["parameter_size_mb"] = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)
    if input_shape is not None:
        info.update(estimate_flops(model, input_shape))
    return info
