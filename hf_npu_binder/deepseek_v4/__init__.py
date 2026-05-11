"""Fast-path callables for ``transformers.models.deepseek_v4``.

Mirrors HF's source layout. Each operator gets its own file; backend
implementations are top-level functions named after the backend
(``triton``, ``flash``, ...). Real kernels land in ``kernels/`` once
ported.

Current state — scaffold only:

  * ``sparse_flash_attention.triton`` defined but raises
    ``NotImplementedError``; the kernel port from
    ``MindSpeed-LLM/.../g2_attention_kernel.py`` is pending. The alloy
    bridge therefore does **not** register this entry, so callers
    requesting ``"triton"`` quietly fall back to alloy's own torch impl
    via the ``IMPL_REGISTRY`` ``fallback="torch"`` chain.

When the triton kernel is ported, the bridge picks it up automatically
(see ``hf_npu_binder.DEFAULTS`` for the recommended-impl table).
"""
from __future__ import annotations

from . import sparse_flash_attention

__all__ = ["sparse_flash_attention"]
