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

  :func:`triton`            — wrap the vendored triton kernel by
                              materialising the sliding-window range as
                              explicit topk indices concatenated with
                              the compressor's compressed picks (the
                              triton kernel doesn't know sliding-window
                              semantics natively). Total topk width is
                              constrained by ``CONFIG_MAP``
                              ({128, 160, 640}); pick alloy's
                              ``sliding_window`` + ``index_topk`` so the
                              sum lands on a supported value.

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
# alloy-facing entry: triton (combined-topk variant)
# ---------------------------------------------------------------------------
def _build_sliding_indices(
    seq_len: int, sliding_window: int, device: torch.device,
) -> torch.Tensor:
    """For each query position q, return the sliding-window KV indices it
    attends to. Causal: query q sees keys in ``[max(0, q-W+1), q]``.

    Layout matches MindSpeed-LLM's ``DeepSeek4SelfAttention.get_window_topk_idxs``
    (``mindspeed_llm/.../deepseek4/g2_attention.py``) — valid indices are
    packed at the FRONT of each row in ascending order, with ``-1``
    sentinels at the END for early queries (q < window_size - 1). This
    front-packed layout is what the vendored SFA kernel was developed
    against; the equivalent back-packed layout has the same softmax
    semantics but is untested against the kernel's vectorisation
    assumptions, so we mirror MindSpeed exactly.

    Returns: ``[seq_len, min(seq_len, sliding_window)]`` int32.
    """
    base = torch.arange(seq_len, device=device).unsqueeze(1)               # [S, 1]
    win = torch.arange(min(seq_len, sliding_window), device=device)        # [W']
    matrix = (base - sliding_window + 1).clamp(min=0) + win                # [S, W']
    return torch.where(matrix > base, torch.full_like(matrix, -1), matrix).to(torch.int32)


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
) -> tuple[torch.Tensor, None]:
    """CSA attention via the vendored MindSpeed triton SFA kernel.

    Matches alloy's ``dsv4_csa.attention`` dispatch contract — same
    signature as :func:`ascendc` and :func:`_torch_csa_attention`. The
    triton kernel doesn't know about sliding-window semantics natively;
    it just does "gather these specific KV positions and attend over
    them" per query. So this adapter materialises the sliding range as
    explicit topk indices, concatenates them with the compressor's
    compressed-range picks, and hands the kernel a single fused
    ``topk_idxs`` covering both attention ranges.

    The kernel's ``CONFIG_MAP`` only supports total topk widths in
    ``{128, 160, 640}`` (combined sliding + compressed). Typical DSV4
    setting is ``sliding_window=128`` + ``index_topk=32`` -> total 160.

    Args:
        query:           [B, H, S, D] bf16
        key, value:      [B, 1, kv_len, D] bf16, kv_len = S_sliding + T;
                         key == value in DSV4 (K=V).
        attention_mask:  unused — the per-query bias is encoded in
                         ``csa_topk_idxs`` directly.
        scaling:         softmax scale.
        sliding_window:  W, sliding-window width (e.g. 128).
        s_aux:           [H] float32, per-head learnable sinks.
        csa_topk_idxs:   [B, S, K] int32, indices into the compressed
                         range ``[0, T)``. ``-1`` sentinels OK.
        compressed_seq_len: int T, length of the compressed-KV tail in
                            the concatenated KV buffer.
        **kwargs:        ignored.

    Returns:
        ``(attn_output [B, S, H, D], None)`` — ``attn_weights`` is None
        because the fused kernel does not surface them. Same BSHD layout
        the alloy call site expects.

    Prefer :func:`ascendc` when CANN's ``aclnnSparseAttnSharedkv`` op is
    available — its production path is faster and has no CONFIG_MAP
    width constraint. This triton wrapper is the self-contained
    fallback for environments where the CANN op is missing (e.g. older
    CANN releases that pre-date the aclnn op).
    """
    if compressed_seq_len <= 0:
        raise ValueError(
            "triton adapter only supports CSA layers (compressed_seq_len > 0); "
            "received compressed_seq_len=0."
        )
    if s_aux is None:
        raise ValueError(
            "triton adapter requires sinks (s_aux); DSV4 CSA layers always "
            "carry a per-head learnable sink. Got None."
        )
    if csa_topk_idxs is None:
        raise ValueError(
            "triton adapter requires csa_topk_idxs from the Lightning Indexer; "
            "got None. (alloy threads this through DeepseekV4Attention.forward.)"
        )
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(
            f"triton adapter requires sliding_window > 0; got {sliding_window!r}."
        )

    SFA = _load_kernel()

    B, H, S, D = query.shape
    kv_len = key.shape[2]
    S_sliding = kv_len - compressed_seq_len
    K = csa_topk_idxs.shape[-1]

    # BHSD -> SBHD for the kernel's preferred layout.
    q_sbnd = query.permute(2, 0, 1, 3).contiguous()  # [S, B, H, D]

    # Concatenated KV: alloy hands [B, 1, S_sliding+T, D]. Drop the
    # single KV-head dim and permute to SBD. Matches MindSpeed's
    # ``torch.cat([ori_kv, cmp_kv], dim=0)`` (sliding first, compressed second).
    kv_sbd = key.squeeze(1).permute(1, 0, 2).contiguous()  # [S_sliding + T, B, D]

    # ----- combined topk construction --------------------------------------
    # Sliding part: per-query causal window indices into [0, S_sliding).
    # Layout (front-packed valid, -1 padding at the end for early queries)
    # mirrors MindSpeed-LLM's ``get_window_topk_idxs``. Width is
    # ``min(S, sliding_window)`` — falls back to S only for the rare
    # short-sequence case.
    sliding_sw = _build_sliding_indices(S, sliding_window, query.device)  # [S, W']
    W_eff = sliding_sw.shape[-1]
    sliding_bsk = sliding_sw.unsqueeze(0).expand(B, S, W_eff)             # [B, S, W']

    # Compressed part: csa_topk_idxs is into [0, T); offset to [S_sliding, kv_len)
    # in the concatenated buffer, preserving -1 sentinels. MindSpeed's
    # indexer applies this offset internally; alloy's indexer does not, so
    # we apply it here.
    if csa_topk_idxs.dtype != torch.int32:
        csa_topk_idxs = csa_topk_idxs.to(torch.int32)
    cmp_offset = csa_topk_idxs + S_sliding
    cmp_bsk = torch.where(csa_topk_idxs >= 0, cmp_offset, csa_topk_idxs)  # [B, S, K]

    # Stack and re-permute to the kernel's [Seq, Batch, TopK] layout.
    combined_bsk = torch.cat([sliding_bsk, cmp_bsk], dim=-1)              # [B, S, W'+K]
    topk_idxs = combined_bsk.permute(1, 0, 2).contiguous()                # [S, B, W'+K]

    # CONFIG_MAP check on the actually-constructed total width. The kernel
    # raises an identical error on mismatch; doing it here gives the user
    # a clearer callsite ValueError pointing at the alloy-level knob
    # (sliding_window + index_topk + seq_len when short).
    total_topk = W_eff + K
    from .kernels.sparse_flash_attention_triton import CONFIG_MAP
    if total_topk not in CONFIG_MAP:
        raise ValueError(
            f"triton SFA kernel only supports total topk widths in "
            f"{sorted(CONFIG_MAP.keys())}; got min(seq_len={S}, "
            f"sliding_window={sliding_window})={W_eff} + index_topk={K} = "
            f"{total_topk}. Adjust the DSV4 config so the sum lands on a "
            f"supported width, or add a TilingBlockConfig entry to CONFIG_MAP."
        )

    # Sinks: kernel signature documents float32.
    sinks_f32 = s_aux.float() if s_aux.dtype != torch.float32 else s_aux

    out_sbnd = SFA.apply(q_sbnd, kv_sbd, sinks_f32, topk_idxs, float(scaling))

    # SBHD -> BSHD (alloy's call site immediately transposes to BHSD for RoPE).
    attn_output = out_sbnd.permute(1, 0, 2, 3).contiguous()  # [B, S, H, D]
    return attn_output, None
