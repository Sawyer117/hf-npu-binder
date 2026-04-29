"""Vendored triton kernels for qwen3_5_moe gated delta rule.

Ported verbatim from ``mindspeed_mm/fsdp/models/qwen3_5/triton/``. The
internal relative imports (``from .utils import ...``) work unchanged
because all 7 files moved as a group.

This package is intentionally NOT auto-imported by
``hf_npu_binder.qwen3_5_moe`` — every kernel module pulls ``triton`` at
top level, which would break ``import hf_npu_binder`` on a CPU dev box.
The public-API file ``chunk_gated_delta_rule.py`` lazy-imports these
kernels inside its function bodies.
"""
