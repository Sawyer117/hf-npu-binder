"""``deepseek_v4`` CSA attention operators.

Four entry points, two abstraction levels:

Low-level (Megatron-native SBND/SBD/BSK layouts; used by binder validation
tests and by callers that already speak SBND):

  :func:`kernel`            — direct triton SFA call (compressed-KV only;
                              callers must pre-combine sliding+compressed
                              indices into ``topk_idxs``). Vendored from
                              ``MindSpeed-LLM/.../g2_attention_kernel.py``.

  :func:`pytorch_reference` — pure-torch reference for the triton kernel.

alloy-facing (HF **BHSD** layout, sliding+compressed concat'd KV;
register-able under ``dsv4_csa.attention`` in alloy IMPL_REGISTRY):

  :func:`ascendc`           — wrap CANN's ``aclnnSparseAttnSharedkv`` via
                              the vendored op in
                              ``ascendc_sparse_attn_shared_kv.py``. This
                              is what MindSpeed's production
                              ``use_sparse_flash_attn=True`` path uses —
                              one call handles sliding + compressed +
                              sink in fused ascendc kernels. Recommended
                              default once compiled.

  :func:`triton`            — would wrap the vendored triton kernel with
                              MindSpeed-style "combined topk" trick
                              (concat sliding window indices with
                              compressed topk picks). Not yet implemented;
                              the alloy bridge skips registration so CSA
                              dispatch falls back to alloy's torch impl.

Heavy deps (``triton``, ``torch_npu``, CANN aclnn JIT compile) are
imported / triggered lazily inside the loaders so this module imports
clean on a CPU dev box.
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
# alloy-facing entry: AscendC (CANN aclnn) — production path
# ---------------------------------------------------------------------------
def _load_ascendc_op():
    """Lazy import of the JIT C++ wrapper. First call triggers ~10s compile
    against the user's torch_npu / CANN; subsequent calls are cached."""
    from .ascendc_sparse_attn_shared_kv import npu_sparse_attn_shared_kv
    return npu_sparse_attn_shared_kv


def ascendc(
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
) -> tuple[torch.Tensor, None]:
    """CSA attention via CANN's ``aclnnSparseAttnSharedkv`` op.

    Matches alloy's ``dsv4_csa.attention`` dispatch contract — same
    signature as :func:`_torch_csa_attention`. Receives BHSD inputs
    where ``key == value`` has shape ``[B, 1, S_sliding + T, D]`` (alloy
    concat'd sliding KV + compressed KV before the dispatch call). The
    adapter splits the two ranges, permutes to Megatron SBND/SBD, and
    invokes the vendored CANN op (which handles sliding + sparse +
    softmax + sink in a single fused kernel).

    Args:
        query:           [B, H, S, D] bfloat16 (BHSD)
        key, value:      [B, 1, kv_len, D] bfloat16, kv_len = S_sliding + T;
                         key == value in DSV4 (K=V).
        attention_mask:  unused — the CANN op constructs its own from the
                         sliding-window + topk metadata.
        scaling:         float, softmax scale (typically 1 / sqrt(D)).
        dropout:         unused; the CANN op does not currently surface
                         attention dropout.
        sliding_window:  int, sliding window width (e.g. 128). The CANN
                         op's band-mask uses ``ori_win_left = sliding_window - 1``
                         and ``ori_win_right = 0`` (causal).
        s_aux:           [H] float32, per-head learnable sinks
                         (gpt-oss-style).
        csa_topk_idxs:   [B, S, K] int32 from alloy's Lightning Indexer.
                         ``-1`` sentinels for early-query invalid picks.
        compressed_seq_len: int, length of the compressed-KV segment T
                            (in the cat'd ``key`` tensor).
        **kwargs:        ignored.

    Returns:
        ``(attn_output [B, S, H, D], None)`` — matches what alloy's
        attention forward expects. ``attn_weights`` is None because the
        fused op does not surface them.

    Compile happens on first call (~10s). Subsequent calls reuse the
    cached .so from ``~/.cache/torch_extensions/``.
    """
    if compressed_seq_len <= 0:
        raise ValueError(
            "ascendc adapter only supports CSA layers (compressed_seq_len > 0); "
            "received compressed_seq_len=0. Sliding-only or HCA layers should "
            "stay on alloy's torch fallback or use a different binder entry."
        )
    if s_aux is None:
        raise ValueError(
            "ascendc adapter requires sinks (s_aux); DSV4 CSA layers always "
            "carry a per-head learnable sink. Got None."
        )
    if csa_topk_idxs is None:
        raise ValueError(
            "ascendc adapter requires csa_topk_idxs from the Lightning Indexer; "
            "got None. (alloy threads this through DeepseekV4Attention.forward.)"
        )
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(
            f"ascendc adapter requires sliding_window > 0; got {sliding_window!r}."
        )

    op = _load_ascendc_op()

    B, H, S, D = query.shape
    kv_len = key.shape[2]
    S_sliding = kv_len - compressed_seq_len

    # BHSD -> SBND for the kernel's preferred layout.
    q_sbnd = query.permute(2, 0, 1, 3).contiguous()  # [S, B, H, D]

    # Split [B, 1, kv_len, D] into sliding ([B, 1, S_sliding, D]) and
    # compressed ([B, 1, T, D]) ranges, then squeeze the KV head dim and
    # permute to [S, B, D].
    ori_kv_bhsd = key[:, :, :S_sliding, :]
    cmp_kv_bhsd = key[:, :, S_sliding:, :]
    ori_kv_sbd = ori_kv_bhsd.squeeze(1).permute(1, 0, 2).contiguous()  # [S_sliding, B, D]
    cmp_kv_sbd = cmp_kv_bhsd.squeeze(1).permute(1, 0, 2).contiguous()  # [T, B, D]

    # The CANN op takes int32 topk indices; alloy hands us int64 typically.
    if csa_topk_idxs.dtype != torch.int32:
        csa_topk_idxs = csa_topk_idxs.to(torch.int32)
    csa_topk_idxs = csa_topk_idxs.contiguous()

    # Sinks: CANN op wants float32 (per-head). alloy stores in self.sinks
    # which is the parameter dtype — coerce defensively.
    sinks_f32 = s_aux.float()

    # ori_win_left = sliding_window - 1 (current token sees self + prev N-1).
    ori_win_left = sliding_window - 1

    # CANN op contract: returns [S, B, H, D] (SBND).
    out_sbnd = op(
        q_sbnd,
        ori_kv_sbd,
        cmp_kv_sbd,
        csa_topk_idxs,
        sinks_f32,
        float(scaling),
        4,                    # cmp_ratio: DSV4 CSA uses compress_rate=4
        4,                    # ori_mask_mode: sliding-window-with-left-N
        3,                    # cmp_mask_mode: per-query topk
        ori_win_left,
        0,                    # ori_win_right: causal (no future)
    )

    # SBND -> BSHD (which is what alloy expects from the CSA attention call;
    # the call site does ``.transpose(1, 2)`` immediately to BHSD for RoPE).
    attn_output = out_sbnd.permute(1, 0, 2, 3).contiguous()  # [B, S, H, D]
    return attn_output, None


# ---------------------------------------------------------------------------
# alloy-facing entry: triton (combined-topk variant) — not yet implemented
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
    """alloy-facing CSA attention via vendored Triton kernel. **Not yet
    implemented** — requires constructing the MindSpeed-style "combined
    topk" (sliding-window indices ⨁ compressed topk picks) so a single
    SFA call covers both attention ranges; the kernel's ``CONFIG_MAP``
    also constrains the legal total topk widths to {128, 160, 640}, which
    in turn constrains the compressed_topk values alloy can use.

    Prefer :func:`ascendc` for production — it uses CANN's fused op
    which has no such width constraints. This Triton path is a
    self-contained backup for environments where the CANN
    ``aclnnSparseAttnSharedkv`` API is unavailable (rare).

    The alloy bridge does not register this entry until implemented.
    """
    raise NotImplementedError(
        "hf_npu_binder.deepseek_v4.sparse_flash_attention.triton (alloy adapter "
        "for the vendored triton SFA kernel) is not yet implemented. The "
        "production path is sparse_flash_attention.ascendc — vendored from "
        "MindSpeed's npu_sparse_attn_shared_kv aclnn op, no CONFIG_MAP width "
        "constraint. The alloy bridge does not register this triton adapter, "
        "so a config request of _dsv4_csa_implementation = 'triton' "
        "transparently falls back to alloy's torch impl."
    )
