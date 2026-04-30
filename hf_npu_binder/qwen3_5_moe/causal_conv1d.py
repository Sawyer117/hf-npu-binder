"""qwen3_5_moe ``causal_conv1d`` operator (decode path).

Contract (matches the HF / alloy ``Qwen3_5MoeGatedDeltaNet`` decode-path call site)::

    fn(hidden_states, conv_state, weight, bias, activation) -> Tensor

Used to update the rolling depthwise-conv state on QKV during decode.

Backend status
--------------
Real ``triton`` and ``flash`` ports of this op (wrapping the ``causal_conv1d``
pip package's ``causal_conv1d_update`` or an ascendc equivalent) are not yet
written. Until they are, both backends delegate to a pure-torch reference
(:func:`_torch_causal_conv1d_update`) which is correct under the same
contract. The decode path is single-token, small-dimension, so the torch
fallback is acceptable in the interim — it just won't get the speedup a
fused kernel would provide.

Drop-in upgrade path: replace the body of :func:`triton` (or :func:`flash`)
with a real kernel call. Callers and the alloy bridge see the same
function signatures, no other code needs to change.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Pure-torch reference for the rolling depthwise-conv state update.

    Equivalent to ``causal_conv1d_update`` from the causal_conv1d pip
    package: roll ``conv_state`` left, append the new token(s), depthwise
    conv across the kernel window, optionally apply silu, return the conv
    output for the current token positions.

    ``conv_state`` is mutated in place to advance the rolling buffer —
    same convention as the C++/triton backends so callers can swap any of
    them without changing surrounding code.
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias,
        padding=0, groups=hidden_size,
    )
    if activation == "silu":
        out = F.silu(out[:, :, -seq_len:])
    else:
        out = out[:, :, -seq_len:]
    return out.to(hidden_states.dtype)


def triton(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """``triton`` backend — currently a torch fallback.

    Triton kernel for ``causal_conv1d_update`` not yet ported. Replace the
    body of this function with the real triton call when one exists; the
    contract above stays the same.
    """
    return _torch_causal_conv1d_update(
        hidden_states, conv_state, weight, bias, activation,
    )


def flash(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """``flash`` backend — currently a torch fallback.

    ascendc / causal_conv1d-pip-package wrap not yet implemented. Replace
    the body of this function with the real call when one exists.
    """
    return _torch_causal_conv1d_update(
        hidden_states, conv_state, weight, bias, activation,
    )
