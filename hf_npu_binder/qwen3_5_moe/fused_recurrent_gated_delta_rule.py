"""Stubs for the qwen3_5_moe ``fused_recurrent_gated_delta_rule`` operator (decode path).

Contract::

    fn(query, key, value, *,
       g, beta,
       initial_state, output_final_state, use_qk_l2norm_in_kernel)
        -> tuple[Tensor, Tensor | None]   # (core_attn_out, last_recurrent_state)

Real implementations wrap fla / mindspeed-lite recurrent kernels.
"""
from __future__ import annotations

import torch


def triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    raise NotImplementedError(
        "hf_npu_binder.qwen3_5_moe.fused_recurrent_gated_delta_rule.triton: stub."
    )


def flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    raise NotImplementedError(
        "hf_npu_binder.qwen3_5_moe.fused_recurrent_gated_delta_rule.flash: stub."
    )
