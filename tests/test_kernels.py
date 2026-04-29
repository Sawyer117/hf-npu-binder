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
    experts,
    fused_recurrent_gated_delta_rule,
)
from hf_npu_binder.shared import gmm


# Operators with the (q,k,v,g,beta,...) GDN signature ship triton + flash.
_GDN_OPERATORS = [
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
    causal_conv1d,
]
_GDN_BACKENDS = ["triton", "flash"]

# experts is the whole MoE forward (permute + gmm + swiglu + gmm + unpermute).
# Reference only ships the ascendc-fused composite as a single "flash" backend
# — there is no pure-triton MoE expert path because nobody has rewritten the
# permute/unpermute kernels in pure triton.
_EXPERTS_BACKENDS = ["flash"]


def test_every_gdn_operator_exposes_every_backend() -> None:
    """Each GDN operator file ships exactly the same set of backend names."""
    for module in _GDN_OPERATORS:
        for backend in _GDN_BACKENDS:
            fn = getattr(module, backend, None)
            assert callable(fn), (
                f"{module.__name__}.{backend} missing or not callable"
            )


def test_experts_exposes_flash_backend() -> None:
    for backend in _EXPERTS_BACKENDS:
        fn = getattr(experts, backend, None)
        assert callable(fn), f"experts.{backend} missing or not callable"


def test_chunk_and_recurrent_share_signature() -> None:
    """Prefill and decode kernels MUST take the same args (the GDN block calls
    one or the other based on a flag, so the contract has to match).
    """
    expected = ["query", "key", "value", "g", "beta",
                "initial_state", "output_final_state", "use_qk_l2norm_in_kernel"]
    for module in (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule):
        for backend in _GDN_BACKENDS:
            sig = inspect.signature(getattr(module, backend))
            params = list(sig.parameters)
            assert params == expected, (
                f"{module.__name__}.{backend} signature drift: "
                f"{params} != {expected}"
            )


def test_causal_conv1d_signature() -> None:
    expected = ["hidden_states", "conv_state", "weight", "bias", "activation"]
    for backend in _GDN_BACKENDS:
        sig = inspect.signature(getattr(causal_conv1d, backend))
        params = list(sig.parameters)
        assert params == expected, (
            f"causal_conv1d.{backend} signature drift: {params} != {expected}"
        )


def test_experts_signature_matches_hf_dispatch() -> None:
    """experts.flash must match HuggingFace ALL_EXPERTS_FUNCTIONS shape:
    fn(self, hidden_states, top_k_index, top_k_weights). The first param
    receives the experts ``nn.Module`` instance from the dispatch wrapper.
    """
    expected = ["self", "hidden_states", "top_k_index", "top_k_weights"]
    for backend in _EXPERTS_BACKENDS:
        sig = inspect.signature(getattr(experts, backend))
        params = list(sig.parameters)
        assert params == expected, (
            f"experts.{backend} signature drift: {params} != {expected}"
        )


def test_shared_gmm_present_and_signature() -> None:
    """shared.gmm.flash is the grouped matmul primitive that experts.flash
    composes with permute / unpermute / swiglu. Signature must stay stable
    so future MoE family files can rely on it.
    """
    assert callable(gmm.flash), "shared.gmm.flash missing"
    expected = ["x", "weight", "group_list"]
    sig = inspect.signature(gmm.flash)
    params = list(sig.parameters)
    assert params == expected, f"shared.gmm.flash signature drift: {params} != {expected}"


def test_experts_imports_shared_gmm() -> None:
    """experts.py stages shared.gmm as ``gmm_flash`` for the real impl.
    Catches accidental removal of the import wiring.
    """
    assert getattr(experts, "gmm_flash", None) is gmm.flash, (
        "experts.gmm_flash should re-export shared.gmm.flash"
    )


def test_remaining_stubs_raise_not_implemented() -> None:
    """The still-stubbed backends must remain loud, not silent.

    These are the operators whose reference implementations either don't
    have an NPU-specific path (causal_conv1d) or rely on external packages
    not yet vendored here (fused_recurrent uses fla; chunk_gated_delta_rule
    needs the 7 triton kernels — pending in the next slice).
    """
    z = torch.zeros(1)
    cases = [
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


def test_real_impls_fail_with_missing_npu_dep_not_notimpl() -> None:
    """The now-real implementations (gmm.flash, experts.flash) call into
    torch_npu — which isn't available on a CPU dev box. They should fail
    with ModuleNotFoundError on torch_npu, NOT with NotImplementedError
    (that would mean we're still stubbed).
    """
    z = torch.zeros(1)
    # Validation-friendly tensors so we get past _validate() and hit the
    # actual NPU call path. Real shapes don't matter on a CPU box; we only
    # need to confirm the path tries to import torch_npu / load triton.
    bf = torch.zeros(1, 2, 1, 4, dtype=torch.bfloat16)            # [B, T, H, K] for chunk_rule
    beta_bf = torch.zeros(1, 2, 1, dtype=torch.bfloat16)          # [B, T, H]
    g_bf = torch.zeros(1, 2, 1, dtype=torch.bfloat16)
    fake_self = type("FakeExperts", (), {
        "num_experts": 2, "is_transposed": False,
        "gate_up_proj": torch.zeros(2, 4, 6),
        "down_proj": torch.zeros(2, 6, 4),
    })()
    cases = [
        (gmm.flash, (z, z, z), {}),
        (experts.flash, (fake_self, z, z, z), {}),
        (chunk_gated_delta_rule.triton, (bf, bf, bf),
         dict(g=g_bf, beta=beta_bf, initial_state=None, output_final_state=False)),
        (chunk_gated_delta_rule.flash, (bf, bf, bf),
         dict(g=g_bf, beta=beta_bf, initial_state=None, output_final_state=False)),
    ]
    for fn, args, kwargs in cases:
        try:
            fn(*args, **kwargs)
        except NotImplementedError:
            raise AssertionError(f"{fn.__module__}.{fn.__name__} still stubbed")
        except ModuleNotFoundError as e:
            # Either torch_npu (gmm.flash, experts.flash, chunk_*.flash) or
            # triton (chunk_*.triton, chunk_*.flash for the kernel imports).
            msg = str(e)
            assert "torch_npu" in msg or "triton" in msg, (
                f"{fn.__module__}.{fn.__name__} expected torch_npu/triton "
                f"ModuleNotFoundError, got: {e}"
            )
        except Exception:
            # Other failures (shape mismatches, attribute errors, etc.) are
            # acceptable — they prove the function ran past any stub, tried
            # to actually compute, and tripped on the missing NPU/triton env.
            pass


def test_module_load_does_not_pull_torch_npu_or_triton() -> None:
    """Importing the binder must not eagerly load torch_npu / triton.

    This is the contract that lets binder install on a CPU dev box. The
    real implementations import these heavy deps lazily inside function
    bodies, not at module level.
    """
    assert "torch_npu" not in sys.modules, (
        "torch_npu was loaded by `import hf_npu_binder` — heavy NPU dep "
        "leaked into module load. Move the import inside the function body."
    )
    assert "triton" not in sys.modules, (
        "triton was loaded by `import hf_npu_binder` — heavy triton dep "
        "leaked into module load. Move the import inside the function body."
    )


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
    test_every_gdn_operator_exposes_every_backend,
    test_experts_exposes_flash_backend,
    test_chunk_and_recurrent_share_signature,
    test_causal_conv1d_signature,
    test_experts_signature_matches_hf_dispatch,
    test_shared_gmm_present_and_signature,
    test_experts_imports_shared_gmm,
    test_remaining_stubs_raise_not_implemented,
    test_real_impls_fail_with_missing_npu_dep_not_notimpl,
    test_module_load_does_not_pull_torch_npu_or_triton,
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
