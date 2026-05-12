"""Cross-family NPU primitives.

Modules under ``shared/`` are primitives that any HuggingFace model family
inside this package may import — grouped matmul, fused norms, etc. Anything
that is genuinely reusable across MoE / hybrid-arch models lives here, not
under a specific ``<hf_family>/`` directory.

Family-specific operators (``qwen3_5_moe.chunk_gated_delta_rule``) compose
primitives from this directory. Some operators that are formally
"family-specific" in the registry (the ``qwen3_5_moe.experts``
DEFAULTS key, for instance) have implementations that are actually
family-agnostic — those live here and the per-family module just
re-exports.
"""
from __future__ import annotations

from . import gmm
from . import moe_experts

__all__ = ["gmm", "moe_experts"]
