"""Stubs for the qwen3_5_moe experts forward (GMM-based fused MoE path).

This is the **whole** experts forward — not just grouped matmul. The real
NPU fast path chains five fused ops:

    npu_moe_token_permute      gather tokens by expert assignment
    npu_grouped_matmul         gate_up projection (this is the "GMM")
    npu_swiglu                 fused gate * silu(up) activation
    npu_grouped_matmul         down projection (second GMM)
    npu_moe_token_unpermute    scatter tokens back, applying routing weights

Reference: ``mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py``
``Qwen3_5MoeExperts.forward`` and ``mindspeed_mm/models/common/gmm.py``.

Contract — matches HuggingFace ``ALL_EXPERTS_FUNCTIONS`` entries:

    fn(self, hidden_states, top_k_index, top_k_weights) -> Tensor

``self`` is the experts ``nn.Module``; the function reads
``self.gate_up_proj`` / ``self.down_proj`` (both 3D :
``[num_experts, *, *]``) directly.

Note on weight layout: alloy's ``_Experts`` stores HF-canonical
``gate_up_proj`` as ``[E, 2*I, H]`` (output-first), whereas mindspeed_mm's
reference uses ``[E, H, 2*I]``. The real implementation will need to handle
this — likely by transposing in-place or selecting a torch_npu call form
that matches alloy's layout.

Heavy deps (``torch_npu``) are imported lazily inside the function body so
this module loads on a CPU box.
"""
from __future__ import annotations

import torch


def flash(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "hf_npu_binder.qwen3_5_moe.experts.flash: stub. "
        "Real ascendc + grouped_matmul fused MoE path coming next."
    )
