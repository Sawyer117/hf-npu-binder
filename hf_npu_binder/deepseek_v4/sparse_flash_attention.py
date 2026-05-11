"""``deepseek_v4`` CSA attention operator (Sparse Flash Attention).

Two entry points:

  :func:`kernel`            — direct SFA kernel call with MindSpeed-native
                              Megatron **SBHD/SBD/SBK** layouts. Used by
                              the binder's own validation tests in
                              ``hf-npu-binder/debug/`` and by any consumer
                              that already speaks SBHD natively.

  :func:`triton`            — alloy-facing adapter (HF **BHSD** layout,
                              sliding+compressed concat'd KV). **Not yet
                              wired** — the layer-level sliding↔SFA
                              combining is Phase 2 of this work. The
                              alloy bridge therefore omits this entry
                              from ``IMPL_REGISTRY`` registration; CSA
                              dispatch falls through to alloy's torch
                              impl (``dsv4_csa.attention:torch``).

The actual kernel autograd Function lives in
``kernels/sparse_flash_attention_triton.py`` (vendored byte-for-byte
from MindSpeed-LLM with the ``megatron.get_args`` dependency removed).

Heavy deps (``triton``, ``torch_npu``) are imported lazily inside
:func:`_load_kernel` so this module imports clean on a CPU dev box.
"""
from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Lazy kernel loader
# ---------------------------------------------------------------------------
_SFA = None  # set on first kernel() call


def _load_kernel():
    """Resolve the kernel autograd Function. Triggers the lazy triton +
    torch_npu imports inside the kernels module."""
    global _SFA
    if _SFA is None:
        from .kernels.sparse_flash_attention_triton import SparseFlashAttentionTriton
        _SFA = SparseFlashAttentionTriton
    return _SFA


def _load_pytorch_reference():
    """The pure-pytorch reference impl, used by validation tests. Imports
    lazily for symmetry with :func:`_load_kernel`."""
    from .kernels.sparse_flash_attention_triton import sparse_attn_pytorch
    return sparse_attn_pytorch


# ---------------------------------------------------------------------------
# Low-level entry: direct kernel call with MindSpeed-native SBHD layouts
# ---------------------------------------------------------------------------
def kernel(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Direct SFA kernel call. Inputs are in MindSpeed-native Megatron
    layouts; the wrapper does NOT permute. For BHSD-native callers, use
    :func:`triton` (once Phase 2 is wired) or permute at the call site.

    Args:
        q:          [Seq, Batch, Head, Dim] — BSHD layout, query
        kv:         [Seq_kv, Batch, Dim] — shared single-head KV
        attn_sink:  [Head] — per-head learnable sink
        topk_idxs:  [Seq, Batch, TopK] — int32, ``-1`` for padded slots
        sm_scale:   float — softmax scale (typically ``1/sqrt(Dim)``)

    Returns:
        ``out [Seq, Batch, Head, Dim]`` in ``q.dtype``.

    Supported ``TopK`` values are 128 / 160 / 640 (see ``CONFIG_MAP`` in
    ``kernels/sparse_flash_attention_triton.py``). The op is
    differentiable — ``out.backward(grad_out)`` writes into
    ``q.grad``, ``kv.grad``, ``attn_sink.grad``.
    """
    SFA = _load_kernel()
    return SFA.apply(q, kv, attn_sink, topk_idxs, sm_scale)


def pytorch_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure-pytorch reference impl matching :func:`kernel` semantics.

    Numerically close (``rtol=1e-2, atol=1e-2`` in bf16) but not
    bit-exact — the kernel uses log2-exp and a fused-pipe schedule
    that produces small accumulation-order differences.
    """
    fn = _load_pytorch_reference()
    return fn(q, kv, attn_sink, topk_idxs, sm_scale)


# ---------------------------------------------------------------------------
# alloy-facing entry (Phase 2 — not wired)
# ---------------------------------------------------------------------------
def triton(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float | int = 0.0,
    sliding_window: Optional[int] = None,
    s_aux: Optional[torch.Tensor] = None,
    csa_topk_idxs: Optional[torch.Tensor] = None,
    compressed_seq_len: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """alloy-facing CSA attention. **Not yet wired** — sliding window +
    SFA combining (log-sum-exp merge of two attention outputs) is Phase
    2. Until that lands, the alloy bridge does not register this entry,
    so ``activate(model, "triton")`` for DSV4 CSA quietly resolves to
    alloy's own torch impl via the ``fallback="torch"`` chain in
    ``get_implementation``.

    See ``hf-npu-binder/debug/test_sparse_flash_attention.py`` for the
    low-level kernel validation; that exercises :func:`kernel` directly
    against :func:`pytorch_reference` on the COMPRESSED-only path
    (sliding KV is the separate concern blocked here).
    """
    raise NotImplementedError(
        "hf_npu_binder.deepseek_v4.sparse_flash_attention.triton (alloy adapter) "
        "is Phase 2 work — needs sliding-window attention + SFA combining via "
        "log-sum-exp merge. The standalone kernel (call via "
        "sparse_flash_attention.kernel(...)) is ready and validated by "
        "hf-npu-binder/debug/test_sparse_flash_attention.py. The alloy bridge "
        "does not register this adapter, so a config request of "
        "_dsv4_csa_implementation = 'triton' transparently falls back to "
        "alloy's torch impl."
    )
