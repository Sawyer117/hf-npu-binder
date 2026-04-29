"""Cross-family NPU primitives.

Modules under ``shared/`` are primitives that any HuggingFace model family
inside this package may import — grouped matmul, fused norms, etc. Anything
that is genuinely reusable across MoE / hybrid-arch models lives here, not
under a specific ``<hf_family>/`` directory.

Family-specific operators (``qwen3_5_moe.chunk_gated_delta_rule``,
``qwen3_5_moe.experts``) compose primitives from this directory.
"""
from __future__ import annotations

from . import gmm

__all__ = ["gmm"]
