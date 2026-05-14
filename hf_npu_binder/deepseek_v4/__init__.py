"""Fast-path callables for ``transformers.models.deepseek_v4``.

Mirrors HF's source layout. Each operator gets its own file; backend
implementations are top-level functions named after the backend
(``triton``, ``ascendc``, ``flash``, ...). Vendored triton kernels live
under ``kernels/``.

Modules:

  * ``sparse_flash_attention`` — CSA attention (Lightning-Indexer
    sliding + per-query topk over compressed KV). Two backends:
    ``triton`` (vendored MindSpeed SFA + BHSD adapter + combined-topk
    construction) and ``ascendc`` (CANN ``aclnnSparseAttnSharedkv``
    fused op; needs CANN 9.0.0 RC+).

  * ``compressed_attention`` — HCA + sliding-only layers. Shares the
    same SFA kernel as CSA; differs only in topk construction (no
    Lightning Indexer; HCA attends to compressed range in full,
    sliding has no compressed range at all). ``triton`` backend only
    so far — ascendc port (HCA via ``aclnnSparseAttnSharedkv`` without
    indexer picks) is a future extension.

  * ``hyper_connection`` — MHC (Mixture of Head Clusters /
    HyperConnection) for DSV4's residual stream mixing. ``triton``
    backend composes 3 vendored MindSpeed kernels:
    ``mhc/rmsnorm_without_weight``, ``mhc/sinkhorn``,
    ``mhc/pre_bmm``. **Hardcoded hc_mult=4** (DSV4 paper config);
    raises clear ValueError otherwise.

See ``hf_npu_binder.DEFAULTS`` for the recommended-impl table per
intent (``auto`` / ``flash`` / ``triton`` / ``ascendc`` / ``torch``).
"""
from __future__ import annotations

from . import compressed_attention
from . import hyper_connection
from . import sparse_flash_attention

__all__ = ["sparse_flash_attention", "compressed_attention", "hyper_connection"]
