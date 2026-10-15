
import torch
import torch.nn.functional as F


def selective_scan_reference(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
):
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must have shape [batch, channels, length]")

    batch, channels, length = u.shape

    if A.ndim != 2 or A.shape[0] != channels:
        raise ValueError("A must have shape [channels, state_dim]")

    state_dim = A.shape[1]

    if B.shape != (batch, state_dim, length):
        raise ValueError("B must have shape [batch, state_dim, length]")

    if C.shape != (batch, state_dim, length):
        raise ValueError("C must have shape [batch, state_dim, length]")

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)

    if delta_softplus:
        delta = F.softplus(delta)

    state = u.new_zeros(batch, channels, state_dim)
    outputs = []

    for t in range(length):
        dt = delta[:, :, t]
        ut = u[:, :, t]
        bt = B[:, :, t]
        ct = C[:, :, t]

        transition = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        input_term = dt.unsqueeze(-1) * ut.unsqueeze(-1) * bt.unsqueeze(1)

        state = transition * state + input_term
        yt = (state * ct.unsqueeze(1)).sum(dim=-1)
        outputs.append(yt)

    y = torch.stack(outputs, dim=-1)

    if D is not None:
        y = y + D.view(1, -1, 1) * u

    if z is not None:
        y = y * F.silu(z)

    return y
