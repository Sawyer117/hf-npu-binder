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

from . import deepseek_v4   # noqa: F401  -- re-export for convenience
from . import qwen3_5_moe   # noqa: F401  -- re-export for convenience
from . import shared        # noqa: F401  -- cross-family primitives

# ---------------------------------------------------------------------------
# Recommended-impl table consulted by ``alloy.integrations.hf_npu_binder``
# when broadcasting a single user intent (``"auto"`` / ``"flash"`` /
# ``"triton"`` / ...) across every dispatch surface this package wires up.
#
# Schema::
#
#   DEFAULTS: dict[binder_operator_key, dict[user_intent, actual_impl_name]]
#
# Each binder operator declares (a) what it considers the current best
# default under ``"auto"`` for the typical NPU target, and (b) explicit
# per-intent translations so that a single user-facing knob covers operators
# that have different available backends. When a user calls
# ``activate(model, "flash")`` and an operator has no flash kernel, this
# table is where the per-operator fallback is declared.
#
# Source of truth: this file. alloy consults it but never overrides — alloy
# is backend-agnostic; backend-specific knowledge ("on Ascend 910B,
# triton beats flash for sparse_flash_attention") lives here, in the
# package that actually ships those kernels.
#
# Entries that are not registered into alloy's IMPL_REGISTRY (because the
# kernel port is pending) point ``"auto"`` at ``"torch"`` so that
# ``activate(model, "auto")`` still resolves to a working callable.
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, dict[str, str]] = {
    "qwen3_5_moe.chunk_gated_delta_rule": {
        "auto":   "triton",
        "flash":  "flash",
        "triton": "triton",
    },
    "qwen3_5_moe.fused_recurrent_gated_delta_rule": {
        "auto":   "triton",
        "flash":  "flash",
        "triton": "triton",
    },
    "qwen3_5_moe.causal_conv1d": {
        "auto":   "triton",
        "flash":  "flash",
        "triton": "triton",
    },
    "qwen3_5_moe.experts": {
        "auto":   "flash",
        "flash":  "flash",
        "triton": "torch",   # no triton experts; HF dispatch falls back to torch
    },
    # ``deepseek_v4.sparse_flash_attention``: scaffold only — kernel port
    # from MindSpeed-LLM pending. "auto" maps to "torch" so an
    # ``activate(model, "auto")`` still gets a working callable through
    # alloy's own torch impl registered under ``"dsv4_csa.attention"``.
    "deepseek_v4.sparse_flash_attention": {
        "auto":   "torch",
        "flash":  "torch",
        "triton": "torch",
    },
}


__all__ = ["DEFAULTS", "deepseek_v4", "qwen3_5_moe", "shared"]
__version__ = "0.0.2"
