"""Stubs for the qwen3_5_moe ``chunk_gated_delta_rule`` operator (prefill path).

Both backends share the contract used by HF ``Qwen3_5MoeGatedDeltaNet`` /
alloy's ``Qwen35GatedDeltaNet`` chunked-prefill call site::

    fn(query, key, value, *,
       g, beta,
       initial_state, output_final_state, use_qk_l2norm_in_kernel)
        -> tuple[Tensor, Tensor | None]   # (core_attn_out, last_recurrent_state)

Real implementations will wrap:
  - ``triton``: pure triton kernels from ``mindspeed.lite.ops.triton.*``
    (cf. ``mindspeed_mm/fsdp/models/qwen3_5/chunk_gated_delta_rule.py``).
  - ``flash``: ascendc + triton hybrid using ``torch_npu`` fused ops
    where available (cf. ``flash_gated_delta_rule.py``).

Heavy deps (``triton``, ``torch_npu``) are imported lazily inside each
function body so this module loads on a CPU box.
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
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    raise NotImplementedError(
        "hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule.triton: stub. "
        "Real triton kernel will land in qwen3_5_moe/kernels/."
    )


def flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    raise NotImplementedError(
        "hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule.flash: stub. "
        "Real ascendc + triton hybrid kernel coming next."
    )
