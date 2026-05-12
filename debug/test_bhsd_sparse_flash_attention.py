"""Smoke test for the alloy-facing BHSD adapter on the triton SFA kernel.

This test exercises the wrapper that takes alloy's ``dsv4_csa.attention``
calling convention (BHSD inputs, concatenated sliding + compressed KV,
compressor topk into the compressed range) and adapts it to the
vendored triton kernel (SBHD inputs, single combined topk tensor over
the cat'd KV buffer).

Three stages:

  1. Forward call: random BHSD inputs through
     ``sparse_flash_attention.triton(...)``; check output shape +
     finiteness. No reference comparison (the low-level kernel vs
     ``pytorch_reference`` comparison lives in
     ``test_sparse_flash_attention.py``).

  2. Backward: gradient flow through the adapter to query / kv / sinks.

  3. Width-constraint guard: pass a topk width that ``CONFIG_MAP``
     doesn't support; assert a clear ValueError fires at the adapter
     level (before getting deep into the kernel).

Run on NPU (CANN + torch_npu + triton-ascend installed)::

    cd hf-npu-binder
    PYTHONPATH=. python debug/test_bhsd_sparse_flash_attention.py

This is the binder-side counterpart to the alloy bridge test
``alloy/tests/test_hf_npu_binder_deepseek_v4.py`` — that one verifies
registration; this one verifies the registered callable actually runs
on real NPU silicon.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _try_import_npu():
    try:
        import torch_npu  # noqa: F401
        return torch_npu
    except ImportError:
        return None


def _build_inputs(device: torch.device):
    """Standard BHSD-shape inputs matching alloy's dsv4_csa.attention call.

    Shape chosen so ``sliding_window + index_topk = 128 + 32 = 160``
    which is one of the triton kernel's supported CONFIG_MAP widths.
    """
    B, H, S, D = 1, 8, 128, 64
    S_sliding = 128
    T = 16
    K = 32
    sliding_window = 128

    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, H, S, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    kv = torch.randn(B, 1, S_sliding + T, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    sinks = torch.randn(H, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.float32
    ).requires_grad_(True)
    topk = torch.randint(0, T, (B, S, K), generator=g, dtype=torch.int32).to(device)
    topk[0, 0, -1] = -1  # exercise the -1 sentinel path

    return {
        "B": B, "H": H, "S": S, "D": D,
        "S_sliding": S_sliding, "T": T, "K": K,
        "sliding_window": sliding_window,
        "q": q, "kv": kv, "sinks": sinks, "topk": topk,
    }


def stage_forward(device: torch.device) -> bool:
    print("\n[1/3] BHSD triton adapter forward call")
    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as sfa

    cfg = _build_inputs(device)
    B, H, S, D, T = cfg["B"], cfg["H"], cfg["S"], cfg["D"], cfg["T"]

    class _FakeAttn(torch.nn.Module):
        def __init__(self): super().__init__()

    try:
        attn_output, attn_weights = sfa.triton(
            _FakeAttn(), cfg["q"], cfg["kv"], cfg["kv"],
            attention_mask=None,
            scaling=D ** -0.5,
            sliding_window=cfg["sliding_window"],
            s_aux=cfg["sinks"],
            csa_topk_idxs=cfg["topk"],
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
    stage_forward._out = attn_output
    stage_forward._cfg = cfg
    print("   OK")
    return True


def stage_backward(device: torch.device) -> bool:
    print("\n[2/3] Backward pass through the BHSD triton adapter")
    if not hasattr(stage_forward, "_out"):
        print("   SKIP (forward did not produce an output)")
        return False
    out = stage_forward._out
    cfg = stage_forward._cfg
    grad_out = torch.randn_like(out)
    try:
        out.backward(grad_out)
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}")
        return False
    checks = [
        ("q.grad",     cfg["q"].grad),
        ("kv.grad",    cfg["kv"].grad),
        ("sinks.grad", cfg["sinks"].grad),
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


def stage_width_guard(device: torch.device) -> bool:
    """Hand the adapter a configuration whose total topk width
    (sliding_window + K) is NOT in CONFIG_MAP, and assert a clear
    ValueError fires at the adapter (rather than letting it bubble out
    of the triton kernel as an opaque error)."""
    print("\n[3/3] Width-constraint guard: total topk not in CONFIG_MAP")
    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as sfa

    cfg = _build_inputs(device)
    # 128 sliding + 31 csa-topk = 159, NOT in {128, 160, 640}.
    bad_k = 31
    bad_topk = torch.randint(0, cfg["T"], (cfg["B"], cfg["S"], bad_k),
                             dtype=torch.int32).to(device)

    class _FakeAttn(torch.nn.Module):
        def __init__(self): super().__init__()

    try:
        sfa.triton(
            _FakeAttn(), cfg["q"], cfg["kv"], cfg["kv"],
            attention_mask=None,
            scaling=cfg["D"] ** -0.5,
            sliding_window=cfg["sliding_window"],
            s_aux=cfg["sinks"],
            csa_topk_idxs=bad_topk,
            compressed_seq_len=cfg["T"],
        )
    except ValueError as e:
        msg = str(e)
        ok = "CONFIG_MAP" in msg or "topk widths" in msg
        print(f"   raised ValueError as expected: {msg[:100]}...")
        if not ok:
            print("   FAIL: error message didn't mention CONFIG_MAP / topk widths")
            return False
        print("   OK")
        return True
    except Exception as e:
        print(f"   FAIL: wrong exception type {type(e).__name__}: {e}")
        return False
    print("   FAIL: no exception raised; expected ValueError on bad total topk")
    return False


def main() -> int:
    npu_mod = _try_import_npu()
    if npu_mod is not None and torch.npu.is_available():
        device = torch.device("npu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"torch={torch.__version__}   device={device}")
    if device.type == "npu":
        print(f"  NPU: {torch.npu.get_device_name(device)}")

    stages = [
        ("forward",     lambda: stage_forward(device)),
        ("backward",    lambda: stage_backward(device)),
        ("width_guard", lambda: stage_width_guard(device)),
    ]
    results = []
    for name, fn in stages:
        try:
            ok = fn()
        except Exception as e:
            print(f"\n[{name}] EXCEPTION: {type(e).__name__}: {e}")
            ok = False
        results.append((name, ok))
        if not ok and name != "width_guard":
            break

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
