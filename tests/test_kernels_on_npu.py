"""Run binder kernels on real NPU and confirm they finish + return sane output.

This is a lightweight per-kernel smoke. It does NOT verify numerical
correctness against a torch reference (that comparison happens in alloy's
``tests/npu/compare_binder_vs_torch.py`` end-to-end). Here we just check:

  - The kernel actually completes a forward
  - Output shape is what we expect
  - Output is finite (no NaN / Inf)
  - Print wall-clock time (first call, second call, n-call avg) so a quick
    look at the numbers tells you whether autograd-class building is being
    warmed up correctly between calls

Skip cleanly when ``torch_npu`` (or ``triton`` for the triton backend) is
absent — we want this script to be safe to invoke on any dev box, not
just NPU hardware.

Usage::

    python tests/test_kernels_on_npu.py [--dtype bf16] [--n-repeat 5]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _try_import_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        return True
    except ImportError:
        return False


def _try_import_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _device(prefer_npu: bool) -> torch.device:
    if prefer_npu and torch.npu.is_available():
        return torch.device("npu")
    raise RuntimeError("This script needs NPU. Run on Ascend hardware.")


def _bench(fn: Callable[[], torch.Tensor], n_warmup: int, n_repeat: int, device: torch.device):
    """Warm up, then time n_repeat calls with NPU sync. Returns (output, t_first, t_avg)."""
    # First call (includes the lazy autograd.Function build cost)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    t_first = time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.npu.synchronize()

    # Time
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        out = fn()
    torch.npu.synchronize()
    t_avg = (time.perf_counter() - t0) / n_repeat

    return out, t_first, t_avg


def _check(name: str, out: torch.Tensor, expected_shape: tuple) -> bool:
    ok = True
    if tuple(out.shape) != expected_shape:
        print(f"  FAIL [{name}] shape {tuple(out.shape)} != expected {expected_shape}")
        ok = False
    if not torch.isfinite(out).all():
        n_nan = torch.isnan(out).sum().item()
        n_inf = torch.isinf(out).sum().item()
        print(f"  FAIL [{name}] non-finite output: nan={n_nan} inf={n_inf}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Per-kernel smokes
# ---------------------------------------------------------------------------
def smoke_gmm(args, device, dtype) -> bool:
    from hf_npu_binder.shared.gmm import flash as gmm_flash

    E, total, in_dim, out_dim = 4, 256, 64, 128
    torch.manual_seed(args.seed)
    x = torch.randn(total, in_dim, device=device, dtype=dtype)
    weight = torch.randn(E, in_dim, out_dim, device=device, dtype=dtype)
    # group_list_type=1 means cumulative offsets — torch_npu.npu_grouped_matmul wants int64
    tokens_per_expert = torch.tensor([60, 70, 80, 46], device=device, dtype=torch.int64)
    group_list = torch.cumsum(tokens_per_expert, dim=0)

    print(f"\n[gmm.flash]  x={tuple(x.shape)}  w={tuple(weight.shape)}  group_list={group_list.tolist()}")
    out, t1, ta = _bench(
        lambda: gmm_flash(x, weight, group_list),
        args.n_warmup, args.n_repeat, device,
    )
    print(f"           first {t1*1000:.2f} ms  /  avg {ta*1000:.2f} ms over {args.n_repeat} runs")
    return _check("gmm.flash", out, (total, out_dim))


def smoke_chunk_triton(args, device, dtype) -> bool:
    from hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule import triton as chunk_triton

    B, T, H, K, V = 1, 128, 4, 64, 64
    torch.manual_seed(args.seed)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.randn(B, T, H, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))

    print(f"\n[chunk_gated_delta_rule.triton]  q/k/v={B}x{T}x{H}x{{K={K}, V={V}}}")
    out, t1, ta = _bench(
        lambda: chunk_triton(q, k, v, g=g, beta=beta,
                             initial_state=None, output_final_state=False,
                             use_qk_l2norm_in_kernel=True)[0],
        args.n_warmup, args.n_repeat, device,
    )
    print(f"          first {t1*1000:.2f} ms  /  avg {ta*1000:.2f} ms over {args.n_repeat} runs")
    return _check("chunk.triton", out, (B, T, H, V))


def smoke_chunk_flash(args, device, dtype) -> bool:
    from hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule import flash as chunk_flash

    B, T, H, K, V = 1, 128, 4, 64, 64
    torch.manual_seed(args.seed)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.randn(B, T, H, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))

    print(f"\n[chunk_gated_delta_rule.flash]  q/k/v={B}x{T}x{H}x{{K={K}, V={V}}}")
    out, t1, ta = _bench(
        lambda: chunk_flash(q, k, v, g=g, beta=beta,
                            initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=True)[0],
        args.n_warmup, args.n_repeat, device,
    )
    print(f"          first {t1*1000:.2f} ms  /  avg {ta*1000:.2f} ms over {args.n_repeat} runs")
    return _check("chunk.flash", out, (B, T, H, V))


def smoke_experts_flash(args, device, dtype) -> bool:
    from hf_npu_binder.qwen3_5_moe.experts import flash as experts_flash

    E, num_top_k, total, hidden, intermediate = 8, 2, 256, 128, 256
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(total, hidden, device=device, dtype=dtype)
    top_k_index = torch.randint(0, E, (total, num_top_k), device=device, dtype=torch.int64)
    top_k_weights = torch.softmax(
        torch.randn(total, num_top_k, device=device, dtype=dtype), dim=-1
    )
    # Build a fake experts module with weights matching alloy's [E, 2*I, H] / [E, H, I] layout.
    fake_self = type("FakeExperts", (), {
        "num_experts": E,
        "is_transposed": False,
        "gate_up_proj": torch.randn(E, 2 * intermediate, hidden, device=device, dtype=dtype),
        "down_proj":    torch.randn(E, hidden, intermediate, device=device, dtype=dtype),
    })()

    print(f"\n[experts.flash]  hidden={hidden}  intermediate={intermediate}  experts={E} (top_k={num_top_k})  total_tokens={total}")
    out, t1, ta = _bench(
        lambda: experts_flash(fake_self, hidden_states, top_k_index, top_k_weights),
        args.n_warmup, args.n_repeat, device,
    )
    print(f"           first {t1*1000:.2f} ms  /  avg {ta*1000:.2f} ms over {args.n_repeat} runs")
    return _check("experts.flash", out, (total, hidden))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip", default="", help="comma-separated: gmm,chunk_triton,chunk_flash,experts")
    args = parser.parse_args()

    if not _try_import_npu():
        print("SKIP — torch_npu not available; this script must run on NPU hardware.")
        return 0

    has_triton = _try_import_triton()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    device = _device(prefer_npu=True)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print(f"running on {device}, dtype={dtype}, n_repeat={args.n_repeat}")

    smokes = []
    smokes.append(("gmm",          smoke_gmm))
    if has_triton:
        smokes.append(("chunk_triton", smoke_chunk_triton))
        smokes.append(("chunk_flash",  smoke_chunk_flash))
    else:
        print("\nSKIP chunk_gated_delta_rule.{triton,flash} — triton not installed")
    smokes.append(("experts",      smoke_experts_flash))

    n_failed = 0
    for name, fn in smokes:
        if name in skip:
            print(f"\n[SKIP] {name} (per --skip)")
            continue
        try:
            ok = fn(args, device, dtype)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR  [{name}] {type(e).__name__}: {e}")
            ok = False
        if not ok:
            n_failed += 1

    if n_failed:
        print(f"\n{n_failed}/{len(smokes)} smoke(s) failed.")
        return 1
    print(f"\nAll {len(smokes)} kernel smokes OK on {device}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
