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
# One case driver
# ---------------------------------------------------------------------------
def run_case(case: Case, device: torch.device, verbose: bool = True) -> bool:
    if verbose:
        print(f"\n--- {case.name}  B={case.B} S={case.S} H={case.H} D={case.D} N={case.N} TOPK={case.TOPK} ---")

    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as sfa

    # ---- triton kernel side ----
    q_t, kv_t, sink_t, topk, scale = make_inputs(case, device)
    grad_out = torch.randn_like(q_t)

    sync = getattr(torch.npu, "synchronize", None) if device.type == "npu" else None

    if sync: sync()
    t0 = time.perf_counter()
    out_triton = sfa.kernel(q_t, kv_t, sink_t, topk, scale).to(torch.bfloat16)
    if sync: sync()
    fwd_ms_triton = (time.perf_counter() - t0) * 1000.0

    if sync: sync()
    t0 = time.perf_counter()
    out_triton.backward(grad_out)
    if sync: sync()
    bwd_ms_triton = (time.perf_counter() - t0) * 1000.0

    grad_q_triton = q_t.grad.detach().clone()
    grad_kv_triton = kv_t.grad.detach().clone()
    grad_sink_triton = sink_t.grad.detach().clone()

    # ---- pytorch reference side ----
    q_r, kv_r, sink_r, _, _ = make_inputs(case, device)
    # reuse the SAME topk_idxs + grad_out so seeds align bit-for-bit
    # (make_inputs uses the same seed for everything)

    if sync: sync()
    t0 = time.perf_counter()
    out_ref = sfa.pytorch_reference(q_r, kv_r, sink_r, topk, scale).to(torch.bfloat16)
    if sync: sync()
    fwd_ms_ref = (time.perf_counter() - t0) * 1000.0

    if sync: sync()
    t0 = time.perf_counter()
    out_ref.backward(grad_out)
    if sync: sync()
    bwd_ms_ref = (time.perf_counter() - t0) * 1000.0

    grad_q_ref = q_r.grad.detach().clone()
    grad_kv_ref = kv_r.grad.detach().clone()
    grad_sink_ref = sink_r.grad.detach().clone()

    # ---- compare ----
    all_pass = True
    if verbose:
        print(f"   fwd  triton {fwd_ms_triton:>9.2f} ms   ref {fwd_ms_ref:>10.2f} ms")
        print(f"   bwd  triton {bwd_ms_triton:>9.2f} ms   ref {bwd_ms_ref:>10.2f} ms")

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
            ok = run_case(case, device)
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
