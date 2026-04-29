"""Stubs for the qwen3_5_moe experts forward (GMM-based fused MoE path).

This is the **whole** experts forward — not just grouped matmul. The real
NPU fast path chains five fused ops:

    npu_moe_token_permute      gather tokens by expert assignment
    shared.gmm.flash           gate_up projection (grouped matmul primitive)
    npu_swiglu                 fused gate * silu(up) activation
    shared.gmm.flash           down projection (second grouped matmul)
    npu_moe_token_unpermute    scatter tokens back, applying routing weights

The grouped-matmul primitive lives in ``hf_npu_binder.shared.gmm`` so other
MoE families (mistral_moe, dbrx, ...) can reuse the exact same wrapper +
custom backward without duplication. ``experts.flash`` here is purely the
qwen3_5-specific composition.

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

# Re-exported for convenience — call sites inside experts.flash will use
# ``gmm_flash(...)`` rather than reaching across packages each time.
from ..shared.gmm import flash as gmm_flash  # noqa: F401  -- staged for the real impl


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
