"""``deepseek_v4`` HCA / sliding attention operators.

Sister module to :mod:`sparse_flash_attention`. The two share the same
underlying triton SFA kernel (``SparseFlashAttentionTriton``); they
differ in how ``topk_idxs`` is constructed:

  * ``sparse_flash_attention.triton`` — CSA layers. Combines a sliding
    window with Lightning-Indexer picks (``csa_topk_idxs``).

  * ``compressed_attention.triton``   — HCA / sliding layers. *No*
    Lightning Indexer; for HCA the compressed-KV range is gated by a
    *per-query causal compress mask* (query at position ``p`` may see
    compressed entry ``w`` only if ``w < (p + 1) // compress_rate_hca``,
    matching alloy's HCA torch reference and HF main ``ac372e10f2``).

Layer-type routing happens at the alloy bridge level
(``alloy/integrations/hf_npu_binder.py``); this file only sees the
already-resolved ``compressed_seq_len`` value:

  * ``compressed_seq_len == 0`` → sliding-only layer; topk is just the
    sliding-window indices (width = ``min(S, sliding_window)``).
  * ``compressed_seq_len  > 0`` → HCA layer; topk is sliding-window
    indices ++ per-query compressed indices ``[0, threshold(p))`` with
    ``-1`` sentinel padding to width ``compressed_seq_len``. Adapter
    reads ``position_ids`` from kwargs and ``compress_rate`` from
    ``module.compressor`` to build the threshold.

The kernel's ``CONFIG_MAP`` only supports total topk widths in
``{128, 160, 640}``; the adapter checks the constructed width (= W +
compressed_seq_len for HCA, W for sliding) and gives a clear error if
it falls outside. Heavy deps (``triton``, ``torch_npu``) are loaded
lazily by :mod:`sparse_flash_attention` so this module is CPU-import-safe.
"""
from __future__ import annotations

from typing import Optional

import torch

# Reuse the kernel loader + sliding-index helper from the CSA sibling so
# there's a single source of truth for those concerns.
from .sparse_flash_attention import _build_sliding_indices, _load_kernel


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
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HCA / sliding attention via the vendored MindSpeed triton SFA kernel.

    Matches alloy's ``dsv4_hca.attention`` and ``dsv4_sliding.attention``
    dispatch contracts — same signature as
    :func:`sparse_flash_attention.triton`. Two routing paths:

      * ``compressed_seq_len == 0`` (sliding-only): topk is just the
        sliding window. ``key.shape[2] == S_sliding`` exactly.
      * ``compressed_seq_len  > 0`` (HCA): topk is sliding ++ per-query
        causal-compress indices: query at absolute position ``p`` sees
        compressed entry ``w`` only if ``w < (p + 1) //
        module.compressor.compress_rate``; later entries get ``-1``
        sentinel. ``key.shape[2] == S_sliding + compressed_seq_len``.

    Args:
        query:           [B, H, S, D] bf16
        key, value:      [B, 1, kv_len, D] bf16. ``kv_len = S_sliding``
                         for sliding-only; ``kv_len = S_sliding +
                         compressed_seq_len`` for HCA. key == value in
                         DSV4 (K=V).
        attention_mask:  ignored — the per-query mask is materialised
                         from ``sliding_window`` + ``position_ids`` +
                         compressed_seq_len.
        scaling:         softmax scale.
        sliding_window:  W, sliding-window width.
        s_aux:           [H] float32, per-head learnable sinks. DSV4
                         attention always carries these regardless of
                         layer type.
        csa_topk_idxs:   must be ``None`` (HCA / sliding don't have a
                         Lightning Indexer). Passed through the kwargs
                         channel for signature compatibility with the
                         CSA path; rejected with a clear error if not
                         None to catch caller-side wiring mistakes.
        compressed_seq_len: int T. 0 for sliding-only; >0 for HCA.
        position_ids:    [B, S] long. Required for HCA (``compressed_seq_len > 0``)
                         so the adapter can build the per-query causal
                         compress mask; ignored for sliding-only.
                         ``module.compressor.compress_rate`` is read for
                         the threshold.
        **kwargs:        ignored.

    Returns:
        ``(attn_output [B, S, H, D], None)`` — ``attn_weights`` is None
        because the fused kernel does not surface them. Same BSHD layout
        the alloy call site expects.
    """
    if s_aux is None:
        raise ValueError(
            "compressed_attention.triton requires sinks (s_aux); DSV4 "
            "attention always carries a per-head learnable sink. Got None."
        )
    if csa_topk_idxs is not None:
        raise ValueError(
            "compressed_attention.triton is for HCA / sliding layers and "
            "MUST NOT receive csa_topk_idxs (those come from the CSA-only "
            "Lightning Indexer). Got a non-None tensor — caller is "
            "probably routing through the wrong adapter; CSA layers "
            "should dispatch to sparse_flash_attention.triton instead."
        )
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(
            f"compressed_attention.triton requires sliding_window > 0; "
            f"got {sliding_window!r}."
        )
    if compressed_seq_len < 0:
        raise ValueError(
            f"compressed_seq_len must be >= 0; got {compressed_seq_len!r}."
        )

    SFA = _load_kernel()

    B, H, S, D = query.shape
    kv_len = key.shape[2]
    S_sliding = kv_len - compressed_seq_len
    if S_sliding < 0:
        raise ValueError(
            f"compressed_attention.triton: kv_len ({kv_len}) is shorter "
            f"than compressed_seq_len ({compressed_seq_len}); the alloy "
            f"call site should concat sliding+compressed onto key before "
            f"dispatch."
        )

    # BHSD -> SBHD for the kernel's preferred layout.
    q_sbnd = query.permute(2, 0, 1, 3).contiguous()  # [S, B, H, D]

    # Concatenated KV: alloy hands [B, 1, kv_len, D]. Drop the single
    # KV-head dim and permute to SBD. For sliding-only kv_len = S_sliding;
    # for HCA the sliding range comes first, compressed second, matching
    # alloy's ``torch.cat([sliding_kv, compressed_kv], dim=2)`` order.
    kv_sbd = key.squeeze(1).permute(1, 0, 2).contiguous()  # [kv_len, B, D]

    # ----- topk construction ------------------------------------------------
    # Sliding part: per-query causal window indices into [0, S_sliding).
    sliding_sw = _build_sliding_indices(S, sliding_window, query.device)  # [S, W']
    W_eff = sliding_sw.shape[-1]
    sliding_bsk = sliding_sw.unsqueeze(0).expand(B, S, W_eff)              # [B, S, W']

    if compressed_seq_len > 0:
        # HCA: per-query causal compress mask. Compressed entry `w` is
        # visible to query at absolute position `p` only if
        # `w < (p + 1) // compress_rate_hca` (window `w` covers source
        # tokens `[w*m', (w+1)*m')`, so the entire window must be in `p`'s
        # past). Later entries get a `-1` sentinel for the kernel to drop.
        # Matches alloy's HCA block_bias semantics (eq HF main
        # ``ac372e10f2`` 5/11).
        if position_ids is None:
            raise ValueError(
                "compressed_attention.triton: HCA mode (compressed_seq_len > 0) "
                "requires position_ids to build the per-query causal compress "
                "mask. alloy threads this via kwargs from "
                "DeepseekV4Attention.forward."
            )
        compressor = getattr(module, "compressor", None)
        compress_rate = getattr(compressor, "compress_rate", None)
        if compress_rate is None:
            raise ValueError(
                "compressed_attention.triton: HCA mode needs "
                "`module.compressor.compress_rate` (the m' divisor) to build the "
                "causal compress mask; got module.compressor=None or no "
                "compress_rate attr."
            )

        # causal_threshold[b, s] = number of compressed entries query at
        # position_ids[b, s] may see.
        causal_threshold = (position_ids + 1) // compress_rate              # [B, S]
        entry_w = torch.arange(
            compressed_seq_len, device=query.device, dtype=torch.int32,
        )                                                                  # [T]
        visible = entry_w.view(1, 1, -1) < causal_threshold.unsqueeze(-1).to(torch.int32)  # [B, S, T]
        cmp_bsk = torch.where(
            visible,
            entry_w.view(1, 1, -1) + S_sliding,
            torch.full_like(entry_w.view(1, 1, -1), -1).expand(B, S, compressed_seq_len),
        ).to(torch.int32)                                                  # [B, S, T]
        combined_bsk = torch.cat([sliding_bsk, cmp_bsk], dim=-1)           # [B, S, W'+T]
    else:
        # Sliding-only.
        combined_bsk = sliding_bsk                                          # [B, S, W']

    # Kernel layout: [Seq, Batch, TopK]
    topk_idxs = combined_bsk.permute(1, 0, 2).contiguous()

    # CONFIG_MAP check on the actually-constructed total width.
    total_topk = W_eff + compressed_seq_len
    from .kernels.sparse_flash_attention_triton import CONFIG_MAP
    if total_topk not in CONFIG_MAP:
        kind = "HCA" if compressed_seq_len > 0 else "sliding"
        breakdown = (
            f"min(seq_len={S}, sliding_window={sliding_window})={W_eff} + "
            f"compressed_seq_len={compressed_seq_len} = {total_topk}"
            if compressed_seq_len > 0
            else f"min(seq_len={S}, sliding_window={sliding_window})={W_eff} = {total_topk}"
        )
        raise ValueError(
            f"triton SFA kernel only supports total topk widths in "
            f"{sorted(CONFIG_MAP.keys())}; got ({kind}) {breakdown}. "
            f"Adjust the DSV4 config so the sum lands on a supported width, "
            f"or add a TilingBlockConfig entry to CONFIG_MAP."
        )

    # Sinks: kernel signature documents float32.
    sinks_f32 = s_aux.float() if s_aux.dtype != torch.float32 else s_aux

    out_sbnd = SFA.apply(q_sbnd, kv_sbd, sinks_f32, topk_idxs, float(scaling))

    # SBHD -> BSHD (alloy's call site immediately transposes to BHSD for RoPE).
    attn_output = out_sbnd.permute(1, 0, 2, 3).contiguous()  # [B, S, H, D]
    return attn_output, None


__all__ = ["triton"]
