"""qwen3_5_moe experts — re-export of the cross-family GMM-based ``flash``.

The actual implementation lives in :mod:`hf_npu_binder.shared.moe_experts`
because HuggingFace's ``ALL_EXPERTS_FUNCTIONS`` is a single global table:
one ``"flash"`` entry must serve every MoE family that wires through
``@use_experts_implementation``. Branching on ``self.limit`` inside the
shared implementation handles the per-architecture differences
(Qwen3.5: no clamp; DSV4: clamped SwiGLU).

This module exists for two reasons:
  1. Backward-compatible import path (``hf_npu_binder.qwen3_5_moe.experts``)
     for the alloy bridge and downstream tests written before the
     refactor.
  2. Anchor for the ``qwen3_5_moe.experts`` DEFAULTS key — that
     intent → impl translation is still keyed by the HF model family,
     even though the underlying callable is shared.
"""
from __future__ import annotations

from ..shared.moe_experts import _expert_weight_in_out, flash

__all__ = ["flash", "_expert_weight_in_out"]
