"""Stubs for the qwen3_5_moe ``causal_conv1d`` operator.

Contract (matches the HF / alloy ``Qwen3_5MoeGatedDeltaNet`` decode-path call site)::

    fn(hidden_states, conv_state, weight, bias, activation) -> Tensor

Used to update the rolling depthwise-conv state on QKV during decode.
Real backends wrap the ``causal_conv1d`` pip package (``causal_conv1d_update``)
or its ascendc equivalent.
"""
from __future__ import annotations

import torch


def triton(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    raise NotImplementedError(
        "hf_npu_binder.qwen3_5_moe.causal_conv1d.triton: stub."
    )


def flash(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    raise NotImplementedError(
        "hf_npu_binder.qwen3_5_moe.causal_conv1d.flash: stub."
    )
