"""Fast-path callables for ``transformers.models.qwen3_5_moe``.

Mirrors HF's source layout. Each operator gets its own file; backend
implementations are top-level functions named after the backend
(``triton``, ``flash``, ...).

Real triton kernels land in ``kernels/`` once ported.
"""
from __future__ import annotations

from . import causal_conv1d
from . import chunk_gated_delta_rule
from . import experts
from . import fused_recurrent_gated_delta_rule

__all__ = [
    "causal_conv1d",
    "chunk_gated_delta_rule",
    "experts",
    "fused_recurrent_gated_delta_rule",
]
