"""Smoke + sanity test for the vendored AscendC sparse-attn-shared-kv op.

Three layers of validation, in order of increasing strictness:

  1. Builder.load() compiles the vendored .cpp against the user's
     CANN / torch_npu (~10s first run, cached afterwards). If this
     fails, the vendored sources / headers / build flags are wrong
     and nothing else can run.

  2. SBND-direct call: invoke ``ascendc_sparse_attn_shared_kv.npu_sparse_attn_shared_kv``
     directly with random tensors in Megatron-native SBND/SBD/BSK layouts.
     Check shape + finiteness + dtype. No reference comparison.

  3. BHSD adapter: invoke ``sparse_flash_attention.ascendc(...)`` (the
     alloy-facing entry) with BHSD tensors. Confirm the layout permute +
     KV split work and the result shape matches what alloy expects.

  4. Backward: run a backward over the ascendc adapter's output. Confirm
     gradients are produced for query / sliding_kv / compressed_kv / sinks
     and are finite.

Run on NPU (CANN env sourced)::

    cd hf-npu-binder
    PYTHONPATH=. python debug/test_ascendc_sparse_attn.py

This does NOT compare numerical correctness against a torch reference —
that comes later via alloy's tests/npu/compare_binder_vs_torch.py
(extended for DSV4 CSA in Phase 2.5). The point of this script is to
prove the JIT compile + the API plumbing work end-to-end on real
hardware before we wire the ascendc backend into alloy's bridge.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _try_import_npu():
    try:
        import torch_npu  # noqa: F401
        return torch_npu
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Stage 1: JIT compile via the op-builder
# ---------------------------------------------------------------------------
def stage_1_compile(verbose: bool = True) -> bool:
    print("\n[1/4] AscendcOpBuilder.load() — JIT compile the vendored .cpp")
    t0 = time.perf_counter()
    try:
        from hf_npu_binder.deepseek_v4.ascendc_sparse_attn_shared_kv import _op_builder
        module = _op_builder.load(verbose=verbose)
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}")
        return False
    elapsed = time.perf_counter() - t0
    pybinds = [n for n in dir(module) if not n.startswith("_")]
    print(f"   OK   compile + load took {elapsed:.2f}s   pybinds: {pybinds}")
    return True


# ---------------------------------------------------------------------------
# Stage 2: direct SBND call
# ---------------------------------------------------------------------------
def stage_2_sbnd_direct(device: torch.device) -> bool:
    """Build random SBND/SBD/BSK inputs and call the op directly."""
    print("\n[2/4] SBND-direct call: npu_sparse_attn_shared_kv(q, ori_kv, cmp_kv, ...)")

    from hf_npu_binder.deepseek_v4.ascendc_sparse_attn_shared_kv import (
        npu_sparse_attn_shared_kv,
    )

    B, S, H, D = 1, 128, 8, 64
    S_ori = 128
    S_cmp = 16
    K = 32
    sliding_window = 128

    g = torch.Generator(device="cpu").manual_seed(0)
    q_sbnd = torch.randn(S, B, H, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    ori_kv_sbd = torch.randn(S_ori, B, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    cmp_kv_sbd = torch.randn(S_cmp, B, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    sinks = torch.randn(H, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.float32
    ).requires_grad_(True)
    # Build a synthetic topk: each query picks K random valid compressed slots.
    topk = torch.randint(0, S_cmp, (B, S, K), generator=g, dtype=torch.int32).to(device)
    # Inject one -1 sentinel to exercise the mask path.
    topk[0, 0, -1] = -1
    softmax_scale = D ** -0.5

    try:
        out = npu_sparse_attn_shared_kv(
            q_sbnd, ori_kv_sbd, cmp_kv_sbd, topk, sinks,
            softmax_scale, cmp_ratio=4,
            ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=sliding_window - 1, ori_win_right=0,
        )
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}")
        return False

    expected_shape = (S, B, H, D)
    shape_ok = tuple(out.shape) == expected_shape
    finite = torch.isfinite(out.float()).all().item()
    print(f"   out shape: {tuple(out.shape)}   expected {expected_shape}   shape_ok={shape_ok}")
    print(f"   finite (no NaN / Inf):   {finite}")
    print(f"   dtype: {out.dtype}")
    if not (shape_ok and finite):
        print("   FAIL")
        return False
    # Stash for backward stage.
    stage_2_sbnd_direct._out = out
    stage_2_sbnd_direct._tensors = (q_sbnd, ori_kv_sbd, cmp_kv_sbd, sinks)
    print("   OK")
    return True


# ---------------------------------------------------------------------------
# Stage 3: BHSD adapter via sparse_flash_attention.ascendc
# ---------------------------------------------------------------------------
def stage_3_bhsd_adapter(device: torch.device) -> bool:
    """Run through the alloy-facing wrapper: BHSD inputs, KV concat'd."""
    print("\n[3/4] BHSD adapter: sparse_flash_attention.ascendc(...)")

    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as sfa

    B, H, S, D = 1, 8, 128, 64
    S_sliding = 128
    T = 16
    K = 32
    sliding_window = 128
    kv_len = S_sliding + T

    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, H, S, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    # Single KV head (DSV4 MQA-style), pre-concat'd as alloy passes.
    kv = torch.randn(B, 1, kv_len, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    sinks = torch.randn(H, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.float32
    ).requires_grad_(True)
    topk = torch.randint(0, T, (B, S, K), generator=g, dtype=torch.int32).to(device)
    topk[0, 0, -1] = -1  # exercise mask path

    # Make a fake module that carries .sinks (alloy passes via s_aux kwarg, so
    # the module doesn't need to actually carry it, but keep the shape).
    class _FakeAttn(torch.nn.Module):
        def __init__(self): super().__init__()

    try:
        attn_output, attn_weights = sfa.ascendc(
            _FakeAttn(), q, kv, kv,
            attention_mask=None,
            scaling=D ** -0.5,
            sliding_window=sliding_window,
            s_aux=sinks,
            csa_topk_idxs=topk,
            compressed_seq_len=T,
        )
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}")
        return False

    expected = (B, S, H, D)
    shape_ok = tuple(attn_output.shape) == expected
    finite = torch.isfinite(attn_output.float()).all().item()
    print(f"   attn_output shape: {tuple(attn_output.shape)}   expected {expected}   shape_ok={shape_ok}")
    print(f"   finite (no NaN / Inf):   {finite}")
    print(f"   attn_weights (expected None):   {attn_weights}")
    if not (shape_ok and finite and attn_weights is None):
        print("   FAIL")
        return False
    stage_3_bhsd_adapter._out = attn_output
    stage_3_bhsd_adapter._tensors = (q, kv, sinks)
    print("   OK")
    return True


# ---------------------------------------------------------------------------
# Stage 4: backward sanity (gradients exist + finite)
# ---------------------------------------------------------------------------
def stage_4_backward(device: torch.device) -> bool:
    print("\n[4/4] Backward pass over the BHSD adapter output")
    if not hasattr(stage_3_bhsd_adapter, "_out"):
        print("   SKIP (stage 3 did not produce an output)")
        return False
    out = stage_3_bhsd_adapter._out
    q, kv, sinks = stage_3_bhsd_adapter._tensors
    grad_out = torch.randn_like(out)
    try:
        out.backward(grad_out)
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}")
        return False
    checks = [
        ("q.grad",     q.grad),
        ("kv.grad",    kv.grad),
        ("sinks.grad", sinks.grad),
    ]
    all_ok = True
    for name, g in checks:
        if g is None:
            print(f"   FAIL: {name} is None (no gradient flowed)")
            all_ok = False
            continue
        finite = torch.isfinite(g.float()).all().item()
        print(f"   {name}: shape={tuple(g.shape)}   finite={finite}")
        if not finite:
            all_ok = False
    if not all_ok:
        print("   FAIL")
        return False
    print("   OK")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="npu / cuda / cpu (default: npu if available)")
    p.add_argument("--skip-compile", action="store_true", help="skip stage 1 (use cached .so)")
    args = p.parse_args()

    npu_mod = _try_import_npu()
    if args.device is None:
        if npu_mod is not None and torch.npu.is_available():
            device = torch.device("npu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"torch={torch.__version__}   device={device}")
    if device.type == "npu":
        print(f"  NPU: {torch.npu.get_device_name(device)}")

    stages = []
    if not args.skip_compile:
        stages.append(("compile", stage_1_compile))
    stages.append(("sbnd",     lambda: stage_2_sbnd_direct(device)))
    stages.append(("bhsd",     lambda: stage_3_bhsd_adapter(device)))
    stages.append(("backward", lambda: stage_4_backward(device)))

    results = []
    for name, fn in stages:
        try:
            ok = fn()
        except Exception as e:
            print(f"\n[{name}] EXCEPTION: {type(e).__name__}: {e}")
            ok = False
        results.append((name, ok))
        if not ok:
            break  # fail-fast — later stages depend on earlier ones

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}   stage_{name}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} stage(s) failed: {failed}")
        return 1
    print("\nAll stages passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
