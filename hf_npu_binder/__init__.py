"""Optional NPU / triton fast-path implementations for HuggingFace-native models.

This package exposes pure callables organised by HuggingFace model family
(``hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule.triton`` and friends).
It does **not** register itself anywhere, patch any model class, or import
any consumer library. Callers — vanilla HF code, alloy, mindspeed_mm,
custom HF model packages — wire these callables into their own dispatch
machinery on their own terms.

Heavy deps (``torch_npu``, ``triton``) are imported lazily inside each
implementation's body, so this package loads on a CPU box.
"""
from __future__ import annotations

from . import qwen3_5_moe  # noqa: F401  -- re-export for convenience

__all__ = ["qwen3_5_moe"]
__version__ = "0.0.1"
