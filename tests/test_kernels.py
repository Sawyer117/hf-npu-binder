"""Smoke-test that hf-npu-binder's stub callables are present, callable, and
loud about being unimplemented.

This package does not depend on alloy. The test deliberately does not import
alloy — that wiring is tested on the alloy side (in
``alloy/tests/test_hf_npu_binder_integration.py``).

Pure CPU torch.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hf_npu_binder.qwen3_5_moe import (
    causal_conv1d,
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)


_OPERATORS = [
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
    causal_conv1d,
]
_BACKENDS = ["triton", "flash"]


def test_every_operator_exposes_every_backend() -> None:
    """Each operator file ships exactly the same set of backend names."""
    for module in _OPERATORS:
        for backend in _BACKENDS:
            fn = getattr(module, backend, None)
            assert callable(fn), (
                f"{module.__name__}.{backend} missing or not callable"
            )


def test_chunk_and_recurrent_share_signature() -> None:
    """Prefill and decode kernels MUST take the same args (the GDN block calls
    one or the other based on a flag, so the contract has to match).
    """
    expected = ["query", "key", "value", "g", "beta",
                "initial_state", "output_final_state", "use_qk_l2norm_in_kernel"]
    for module in (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule):
        for backend in _BACKENDS:
            sig = inspect.signature(getattr(module, backend))
            params = list(sig.parameters)
            assert params == expected, (
                f"{module.__name__}.{backend} signature drift: "
                f"{params} != {expected}"
            )


def test_causal_conv1d_signature() -> None:
    expected = ["hidden_states", "conv_state", "weight", "bias", "activation"]
    for backend in _BACKENDS:
        sig = inspect.signature(getattr(causal_conv1d, backend))
        params = list(sig.parameters)
        assert params == expected, (
            f"causal_conv1d.{backend} signature drift: {params} != {expected}"
        )


def test_stubs_raise_not_implemented() -> None:
    """Stubs must be loud, not silent."""
    z = torch.zeros(1)
    cases = [
        # (callable, args, kwargs)
        (chunk_gated_delta_rule.triton, (z, z, z),
         dict(g=z, beta=z, initial_state=None, output_final_state=False)),
        (chunk_gated_delta_rule.flash, (z, z, z),
         dict(g=z, beta=z, initial_state=None, output_final_state=False)),
        (fused_recurrent_gated_delta_rule.triton, (z, z, z),
         dict(g=z, beta=z, initial_state=None, output_final_state=False)),
        (fused_recurrent_gated_delta_rule.flash, (z, z, z),
         dict(g=z, beta=z, initial_state=None, output_final_state=False)),
        (causal_conv1d.triton, (z, z, z), dict(bias=None, activation=None)),
        (causal_conv1d.flash,  (z, z, z), dict(bias=None, activation=None)),
    ]
    for fn, args, kwargs in cases:
        try:
            fn(*args, **kwargs)
        except NotImplementedError as e:
            assert "stub" in str(e), e
        else:
            raise AssertionError(f"{fn.__module__}.{fn.__name__} did not raise NotImplementedError")


def test_package_does_not_import_alloy() -> None:
    """Binder must remain consumer-agnostic. Importing it should not have
    pulled alloy into sys.modules.
    """
    # If alloy was already imported (e.g. by a parent test runner) we can't
    # blame binder. So we only assert binder doesn't put alloy modules in
    # sys.modules under names like 'alloy.modules.registry' that *we*
    # would be the only plausible importer of.
    suspect = [m for m in sys.modules if m == "alloy" or m.startswith("alloy.")]
    if suspect:
        # alloy may already be installed and previously imported. We only
        # care that binder didn't pull alloy.modules.* — registry is the
        # specific surface a stale wiring would touch.
        assert "alloy.modules.registry" not in sys.modules, (
            "binder appears to have imported alloy.modules.registry — "
            "the package should be alloy-unaware"
        )


_TESTS = [
    test_every_operator_exposes_every_backend,
    test_chunk_and_recurrent_share_signature,
    test_causal_conv1d_signature,
    test_stubs_raise_not_implemented,
    test_package_does_not_import_alloy,
]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"  OK  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(_TESTS)} test(s) failed.")
        return 1
    print(f"\nAll {len(_TESTS)} kernel tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
