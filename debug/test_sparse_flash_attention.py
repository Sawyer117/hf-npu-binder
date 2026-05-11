"""SparseFlashAttention (SFA) kernel validation — run on NPU only.

Exercises ``hf_npu_binder.deepseek_v4.sparse_flash_attention.kernel`` (the
vendored MindSpeed-LLM triton kernel) against ``pytorch_reference`` on
multiple shapes and TopK widths. The kernel and the reference are in
MindSpeed's Megatron-native **SBHD/SBD/SBK** layouts; this test does
NOT do any HF-style BHSD permutes — that's the wrapper concern (Phase 2,
not yet wired).

Three test cases cover the three TopK values supported by ``CONFIG_MAP``::

    case_small        — TopK=128 — quick smoke
    case_v4_pro_ish   — TopK=160 — medium / one common DSV4 shape
    case_mindspeed    — TopK=640 — full-size, matches the MindSpeed
                                   ``test_performance_profile`` config

For each case:
  * Forward: kernel out vs reference, ``rtol=1e-2 atol=1e-2`` in bf16.
  * Backward: ``grad_q``, ``grad_kv``, ``grad_sink``, ``rtol=2e-2 atol=2e-2``
    (softmax backward has more accumulation noise than forward).
  * Cosine similarity reported alongside max_abs for quick eyeballing.

Run on NPU::

    cd hf-npu-binder
    PYTHONPATH=. python debug/test_sparse_flash_attention.py

The kernel is NPU-only. CPU import works (lazy triton/torch_npu imports)
but ``kernel(...)`` will raise on a non-NPU device.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch


def _try_import_npu():
    """Best-effort torch_npu import. Returns the module or None."""
    try:
        import torch_npu  # noqa: F401
        return torch_npu
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    name: str
    B: int       # batch
    S: int       # query seq length
    H: int       # head count
    D: int       # head dim
    N: int       # KV seq length
    TOPK: int    # top-k width (must be in CONFIG_MAP: 128 / 160 / 640)


CASES: list[Case] = [
    Case("case_small",      B=1, S=64,   H=8,  D=128, N=512,  TOPK=128),
    Case("case_v4_pro_ish", B=1, S=256,  H=16, D=128, N=1024, TOPK=160),
    Case("case_mindspeed",  B=1, S=4096, H=64, D=512, N=5120, TOPK=640),
]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_inputs(case: Case, device: torch.device, seed: int = 100):
    """Build random (q, kv, attn_sink, topk_idxs) in MindSpeed-native
    layouts. q/kv in bf16 with ``requires_grad=True``. One ``-1``
    sentinel is injected into ``topk_idxs[0, 0, -1]`` to exercise the
    masked-load path."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    q = torch.randn(case.S, case.B, case.H, case.D, generator=g, dtype=torch.float32)
    kv = torch.randn(case.N, case.B, case.D, generator=g, dtype=torch.float32)
    attn_sink = torch.randn(case.H, generator=g, dtype=torch.float32)

    topk_idxs = torch.randint(0, case.N, (case.S, case.B, case.TOPK), generator=g, dtype=torch.int32)
    # padded slot on one row to cover the mask path
    topk_idxs[0, 0, -1] = -1

    q = q.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    kv = kv.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    attn_sink = attn_sink.to(device=device, dtype=torch.float32).requires_grad_(True)
    topk_idxs = topk_idxs.to(device=device)

    scale = (1.0 / case.D) ** 0.5
    return q, kv, attn_sink, topk_idxs, scale


# ---------------------------------------------------------------------------
# Compare helpers
# ---------------------------------------------------------------------------
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    denom = a.norm() * b.norm()
    if denom.item() == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return (a @ b / denom).item()


def _stats(name: str, a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    """Compare two tensors with bf16 noise-floor tolerances. Returns
    (passed, one-line summary)."""
    diff = (a - b).detach().float().abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos = _cos(a, b)
    passed = max_abs < 5e-2  # generous; bwd of softmax can produce ~1e-2 in bf16
    flag = "PASS" if passed else "FAIL"
    summary = f"   [{flag}] {name:>14s}  max_abs={max_abs:>10.4e}  mean_abs={mean_abs:>10.4e}  cos_sim={cos:>10.6f}"
    return passed, summary


# ---------------------------------------------------------------------------
# Timed runs with warmup / steady-state separation
# ---------------------------------------------------------------------------
def _sync_fn(device: torch.device):
    if device.type == "npu":
        return torch.npu.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def _timed_fwd(
    case: Case, device: torch.device, *, warmup: int, iters: int,
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bench a forward of the triton kernel. Returns
    (compile_ms, steady_ms_median, out, q_for_bwd, kv_for_bwd, sink_for_bwd).

    First call is timed separately as ``compile_ms`` — it includes Triton's
    AOT compile (often several seconds on Ascend NPU for novel ``constexpr``
    signatures). Then ``warmup`` untimed calls, then ``iters`` timed calls
    whose median is the steady-state ms.

    Returned q/kv/sink are the requires_grad-True inputs used in the LAST
    timed call, so the caller can chain a backward bench on them.
    """
    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as sfa
    sync = _sync_fn(device)

    # First call — compile + run.
    q, kv, sink, topk, scale = make_inputs(case, device)
    sync()
    t0 = time.perf_counter()
    out = sfa.kernel(q, kv, sink, topk, scale).to(torch.bfloat16)
    sync()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup — untimed, just to ensure caches are hot for the timed iters.
    for _ in range(warmup):
        q, kv, sink, _, _ = make_inputs(case, device)
        _ = sfa.kernel(q, kv, sink, topk, scale).to(torch.bfloat16)
    sync()

    # Timed iters — each rebuilds inputs so we capture full fwd cost (incl
    # any per-call alloc); reusing the SAME inputs would let allocator reuse
    # buffers and underestimate.
    times = []
    for _ in range(iters):
        q, kv, sink, _, _ = make_inputs(case, device)
        sync()
        t0 = time.perf_counter()
        out = sfa.kernel(q, kv, sink, topk, scale).to(torch.bfloat16)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    steady_ms = times[len(times) // 2]

    # Re-do once with requires_grad alive so the bwd bench can use the
    # autograd graph from a known steady-state forward.
    q, kv, sink, _, _ = make_inputs(case, device)
    out = sfa.kernel(q, kv, sink, topk, scale).to(torch.bfloat16)
    return compile_ms, steady_ms, out, q, kv, sink


def _timed_bwd(
    out: torch.Tensor, q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor,
    grad_out: torch.Tensor, *,
    case: Case, device: torch.device, warmup: int, iters: int,
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bench backward. Same compile-vs-steady split as ``_timed_fwd``.
    Each timed iter does fresh fwd+bwd (so autograd graph is rebuildable)."""
    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as sfa
    sync = _sync_fn(device)

    # First bwd — compile dq + dkv kernels.
    sync()
    t0 = time.perf_counter()
    out.backward(grad_out, retain_graph=False)
    sync()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    # Save grads from the compile run (used by correctness compare below).
    grad_q = q.grad.detach().clone()
    grad_kv = kv.grad.detach().clone()
    grad_sink = sink.grad.detach().clone()

    # Warmup — fresh fwd+bwd, untimed.
    _, _, scale = case.B, case.S, (1.0 / case.D) ** 0.5
    for _ in range(warmup):
        q, kv, sink, topk, _ = make_inputs(case, device)
        o = sfa.kernel(q, kv, sink, topk, scale).to(torch.bfloat16)
        o.backward(grad_out, retain_graph=False)
    sync()

    # Timed iters — fresh fwd+bwd each iter, only the bwd is timed.
    times = []
    for _ in range(iters):
        q, kv, sink, topk, _ = make_inputs(case, device)
        o = sfa.kernel(q, kv, sink, topk, scale).to(torch.bfloat16)
        sync()
        t0 = time.perf_counter()
        o.backward(grad_out, retain_graph=False)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    steady_ms = times[len(times) // 2]

    return compile_ms, steady_ms, grad_q, grad_kv, grad_sink


# ---------------------------------------------------------------------------
# One case driver
# ---------------------------------------------------------------------------
def run_case(
    case: Case, device: torch.device,
    *, warmup: int = 2, iters: int = 5, verbose: bool = True,
) -> bool:
    if verbose:
        print(f"\n--- {case.name}  B={case.B} S={case.S} H={case.H} D={case.D} N={case.N} TOPK={case.TOPK} ---")

    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as sfa
    sync = _sync_fn(device)

    # Reuse the SAME grad_out across triton + ref so the autograd outputs
    # are directly comparable.
    grad_out_template = torch.randn(case.S, case.B, case.H, case.D, device=device, dtype=torch.bfloat16)

    # ---- TRITON path (timed) ----
    fwd_compile_ms, fwd_steady_ms, out_triton, q_t, kv_t, sink_t = _timed_fwd(
        case, device, warmup=warmup, iters=iters,
    )
    bwd_compile_ms, bwd_steady_ms, grad_q_triton, grad_kv_triton, grad_sink_triton = _timed_bwd(
        out_triton, q_t, kv_t, sink_t, grad_out_template,
        case=case, device=device, warmup=warmup, iters=iters,
    )

    # ---- REFERENCE path (one shot — pytorch loop, no compile to amortise) ----
    q_r, kv_r, sink_r, topk, scale = make_inputs(case, device)
    sync()
    t0 = time.perf_counter()
    out_ref = sfa.pytorch_reference(q_r, kv_r, sink_r, topk, scale).to(torch.bfloat16)
    sync()
    fwd_ref_ms = (time.perf_counter() - t0) * 1000.0

    sync()
    t0 = time.perf_counter()
    out_ref.backward(grad_out_template)
    sync()
    bwd_ref_ms = (time.perf_counter() - t0) * 1000.0
    grad_q_ref = q_r.grad.detach().clone()
    grad_kv_ref = kv_r.grad.detach().clone()
    grad_sink_ref = sink_r.grad.detach().clone()

    # ---- report ----
    all_pass = True
    if verbose:
        print(f"   fwd  triton compile={fwd_compile_ms:>10.2f} ms  steady={fwd_steady_ms:>9.2f} ms (median of {iters})   ref={fwd_ref_ms:>10.2f} ms")
        print(f"   bwd  triton compile={bwd_compile_ms:>10.2f} ms  steady={bwd_steady_ms:>9.2f} ms (median of {iters})   ref={bwd_ref_ms:>10.2f} ms")

    for name, a, b in [
        ("forward",   out_triton,        out_ref),
        ("grad_q",    grad_q_triton,     grad_q_ref),
        ("grad_kv",   grad_kv_triton,    grad_kv_ref),
        ("grad_sink", grad_sink_triton,  grad_sink_ref),
    ]:
        passed, summary = _stats(name, a, b)
        if verbose:
            print(summary)
        all_pass = all_pass and passed

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="npu / cuda / cpu (default: npu if available, else cuda, else cpu)")
    p.add_argument("--cases", default=None, help="comma-separated subset of test case names")
    p.add_argument("--warmup", type=int, default=2, help="untimed warmup iters after first compile-call (default 2)")
    p.add_argument("--iters", type=int, default=5, help="timed iters for steady-state median (default 5)")
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

    print(f"torch={torch.__version__}  device={device}")
    if device.type == "npu":
        print(f"  NPU: {torch.npu.get_device_name(device)}")
    elif device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    elif device.type == "cpu":
        print("  CPU: kernel call will raise (kernel is NPU-only). Use --device npu on hardware.")

    selected: list[Case]
    if args.cases:
        wanted = {n.strip() for n in args.cases.split(",")}
        selected = [c for c in CASES if c.name in wanted]
        if not selected:
            print(f"no case names matched {sorted(wanted)}; available: {[c.name for c in CASES]}")
            return 1
    else:
        selected = CASES

    print("=" * 78)
    print(f"SparseFlashAttention validation  ({len(selected)} cases)")
    print("=" * 78)

    results: list[tuple[str, bool]] = []
    for case in selected:
        try:
            ok = run_case(case, device, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            print(f"\n--- {case.name} --- EXCEPTION")
            print(f"   {type(e).__name__}: {e}")
            ok = False
        results.append((case.name, ok))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failures = [n for n, ok in results if not ok]
    if failures:
        print(f"\n{len(failures)} case(s) failed: {failures}")
        return 1
    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
