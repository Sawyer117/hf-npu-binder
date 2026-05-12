"""Python entry for the vendored CANN ``aclnnSparseAttnSharedkv`` op.

Vendored from MindSpeed at::

    mindspeed/ops/npu_sparse_attn_shared_kv.py

Changes vs upstream:
  * Uses our :class:`AscendcOpBuilder` (in ``hf_npu_binder/shared/op_builder.py``)
    instead of MindSpeed's ``MindSpeedOpBuilder``. The two are
    structurally identical (subclass ``sources()``, get a cached
    ``load()`` that JIT-compiles the .cpp). Different ``Library``
    namespace ("hf_npu_binder" — but this op doesn't actually register a
    proto, so the namespace is unused at the dispatcher level).
  * Source path points to our vendored copy under
    ``deepseek_v4/csrc/npu_sparse_attn_shared_kv.cpp``.

API contract:

    npu_sparse_attn_shared_kv(
        query [S, B, N, D]  bfloat16,        # SBND, Megatron-native
        ori_kv [S_ori, B, D]  bfloat16,       # sliding window KV, single shared head
        cmp_kv [S_cmp, B, D]  bfloat16,       # compressed KV, single shared head
        cmp_sparse_indices [B, S, K]  int32,  # per-query topk picks into cmp_kv,
                                              # -1 marks invalid early-query slots
        sinks [N]  float32,                   # per-head learnable sink (gpt-oss style)
        softmax_scale: float,                 # typically 1 / sqrt(D)
        cmp_ratio: int,                       # compression rate; 4 for DSV4 CSA
        ori_mask_mode: int = 4,               # sliding-window-with-left-N
        cmp_mask_mode: int = 3,               # per-query-topk
        ori_win_left: int = 127,              # = sliding_window - 1 (e.g. 128-1)
        ori_win_right: int = 0,               # 0 = causal (no future)
    ) -> attn_output [S, B, N, D]

The underlying CANN op handles:
  * Sliding-window attention over ``ori_kv`` (mask_mode=4, win_left/right)
  * Per-query sparse attention over ``cmp_kv`` (mask_mode=3, picks indexed
    by ``cmp_sparse_indices``)
  * Combined-softmax with shared per-head sink
  * Backward pass for query, ori_kv, cmp_kv, sinks

Compile happens on first call (~10s; cached afterwards in
``~/.cache/torch_extensions/``). ``import`` of this module is cheap.
"""
from __future__ import annotations

import torch

from ..shared.op_builder import AscendcOpBuilder


class _NpuSparseAttnSharedKVOpBuilder(AscendcOpBuilder):
    OP_NAME = "npu_sparse_attn_shared_kv"

    def sources(self):
        return ["deepseek_v4/csrc/npu_sparse_attn_shared_kv.cpp"]


# Module-level singleton — first ``.load()`` triggers the JIT compile.
_op_builder = _NpuSparseAttnSharedKVOpBuilder()


class SparseAttnSharedKV(torch.autograd.Function):
    """Autograd Function around the CANN ``aclnnSparseAttnSharedkv``
    forward + backward. Saved-for-backward tensors mirror what the
    aclnnSparseAttnSharedkvGrad call needs.

    The forward returns only ``out`` to keep the autograd Function API
    standard; ``softmax_lse`` is computed (for backward) and saved in
    ``ctx.saved_tensors`` but not surfaced to callers. If a downstream
    pipeline needs lse, expose it via a second entry point.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        ori_kv,
        cmp_kv,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seq_len_q,
        max_seq_len_kv,
        topk,
        layout_q,
        layout_kv,
    ):
        op = _op_builder.load()
        empty_npu = torch.tensor([]).npu()

        metadata = op.npu_sparse_attn_shared_kv_metadata(
            cu_seq_lens_q if cu_seq_lens_q is not None else empty_npu,
            empty_npu,  # sequsedOriKv  (inference-side fields not used in training)
            empty_npu,  # sequsedCmpKv
            empty_npu,  # sequsedQ
            empty_npu,  # sequsedKv
            num_heads_q,
            num_heads_kv,
            head_dim,
            batch_size,
            max_seq_len_q,
            max_seq_len_kv,
            topk,  # oriTopk: not supported yet upstream — passed as same value
            topk,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            ori_kv is not None,
            cmp_kv is not None,
        )

        result, softmax_lse = op.npu_sparse_attn_shared_kv(
            query,
            ori_kv,
            cmp_kv,
            ori_sparse_indices,
            cmp_sparse_indices,
            None,  # oriBlockTable (inference-only)
            None,  # cmpBlockTable
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            None,  # sequsedQ
            None,  # sequsedKv
            sinks,
            metadata,
            softmax_scale,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            True,  # returnSoftmaxLse: needed for backward
        )

        ctx.save_for_backward(
            query, ori_kv, cmp_kv, result, softmax_lse,
            ori_sparse_indices, cmp_sparse_indices,
            cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv, sinks,
        )
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        return result

    @staticmethod
    def backward(ctx, grad_output):
        op = _op_builder.load()
        (query, ori_kv, cmp_kv, result, softmax_lse,
         ori_sparse_indices, cmp_sparse_indices,
         cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
         sinks) = ctx.saved_tensors

        query_grad, ori_kv_grad, cmp_kv_grad, sinks_grad = op.npu_sparse_attn_shared_kv_grad(
            query,
            ori_kv,
            cmp_kv,
            grad_output,
            result,
            softmax_lse,
            ori_sparse_indices,
            cmp_sparse_indices,
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            sinks,
            ctx.softmax_scale,
            ctx.cmp_ratio,
            ctx.ori_mask_mode,
            ctx.cmp_mask_mode,
            ctx.ori_win_left,
            ctx.ori_win_right,
            ctx.layout_q,
        )
        # Match forward arg count; gradients only for query / ori_kv / cmp_kv / sinks.
        return (
            query_grad, ori_kv_grad, cmp_kv_grad,
            None, None, None, None, None,
            sinks_grad,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None,
        )


def npu_sparse_attn_shared_kv(
    query: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor | None,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: int,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
) -> torch.Tensor:
    """Public entry. Inputs in Megatron-native SBND/SBD/BSK layouts.
    See module docstring for full contract."""
    cu_seq_lens_q = cu_seq_lens_ori_kv = cu_seq_lens_cmp_kv = None  # TND not supported
    ori_sparse_indices = None  # ori_kv uses band mode (sliding window), no sparse picks

    max_seq_len_q, batch_size, num_heads_q, head_dim = query.size()
    num_heads_kv = 1
    max_seq_len_kv = ori_kv.size(0)
    topk = 0 if cmp_ratio != 4 else cmp_sparse_indices.size(-1)
    layout_q = layout_kv = "BSND"

    # SBND -> BSND for the op's preferred internal layout.
    query = query.permute(1, 0, 2, 3).contiguous()
    # SBD -> BS1D (insert head dim).
    ori_kv = ori_kv.permute(1, 0, 2).unsqueeze(2).contiguous()
    cmp_kv = cmp_kv if cmp_kv is None else cmp_kv.permute(1, 0, 2).unsqueeze(2).contiguous()
    # BSK -> BS1K (insert head dim).
    cmp_sparse_indices = (
        None if cmp_ratio != 4 else cmp_sparse_indices.unsqueeze(2).contiguous()
    )

    output = SparseAttnSharedKV.apply(
        query, ori_kv, cmp_kv,
        cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
        ori_sparse_indices, cmp_sparse_indices, sinks,
        softmax_scale, cmp_ratio,
        ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
        num_heads_q, num_heads_kv, head_dim,
        batch_size, max_seq_len_q, max_seq_len_kv, topk,
        layout_q, layout_kv,
    )
    # BSND -> SBND (back to caller-visible Megatron layout).
    return output.transpose(0, 1).contiguous()


__all__ = ["npu_sparse_attn_shared_kv", "SparseAttnSharedKV"]
