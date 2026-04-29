"""qwen3_5_moe ``chunk_gated_delta_rule`` operator (prefill path).

Two backends:

  triton    pure triton — uses the 7 vendored kernels in ``./kernels/``
  flash     ascendc + triton hybrid — uses ``torch.ops.npu.npu_chunk_*`` for
            the heavy fwd_h / fwd_o / bwd kernels, vendored triton for the
            lighter prep ops (``chunk_local_cumsum``, ``solve_tril``, etc.)

Both backends share the contract that alloy's ``Qwen35GatedDeltaNet.forward``
calls with::

    fn(query, key, value, *,
       g, beta,
       initial_state, output_final_state, use_qk_l2norm_in_kernel)
        -> tuple[Tensor, Tensor | None]    # (core_attn_out, last_recurrent_state)

Both ports preserve the reference's autograd contract — a custom
``autograd.Function`` wraps the kernel calls. Each backend's Function class
is built lazily on first call (``_make_triton_fn`` / ``_make_flash_fn``)
and cached, so heavy imports (triton / torch_npu) only happen when the
backend is actually invoked. ``import hf_npu_binder`` stays CPU-safe.

Ports:
  triton  ← mindspeed_mm/fsdp/models/qwen3_5/chunk_gated_delta_rule.py
  flash   ← mindspeed_mm/fsdp/models/qwen3_5/flash_gated_delta_rule.py
"""
from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Caller-side helpers (pure torch, no kernel deps)
# ---------------------------------------------------------------------------
def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalisation. Pure torch so autograd tracks it natively — saves us
    from carrying ``q_rstd`` / ``k_rstd`` through the Function for backward.
    """
    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> None:
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q/k/v dtypes must match — got {q.dtype} / {k.dtype} / {v.dtype}"
        )
    if q.dtype == torch.float32:
        raise ValueError(
            "chunk_gated_delta_rule does not support float32. Use bfloat16."
        )
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta must be of shape [B, T, H], got {tuple(beta.shape)}"
        )


# ---------------------------------------------------------------------------
# Triton backend — pure triton via the 7 vendored kernels
# ---------------------------------------------------------------------------
_TritonFn = None  # populated on first call to triton()


def _make_triton_fn():
    """Build the triton-backend ``autograd.Function`` class lazily.

    All triton-touching imports happen inside this function body so that
    ``import hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule`` stays
    triton-free on a CPU dev box.
    """
    from .kernels.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from .kernels.chunk_o import (
        chunk_bwd_dqkwg,
        chunk_bwd_dv_local,
        chunk_fwd_o,
    )
    from .kernels.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from .kernels.cumsum import chunk_local_cumsum
    from .kernels.solve_tril import solve_tril
    from .kernels.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
    from .kernels.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd

    def _fwd(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size):
        g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=True)
        A = chunk_scaled_dot_kkt_fwd(
            k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
            output_dtype=torch.float32,
        )
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens)
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
        )
        o = chunk_fwd_o(
            q=q, k=k, v=v_new, h=h, g=g, scale=scale,
            cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        )
        return g, o, A, final_state

    def _bwd(q, k, v, g, beta, A, scale, initial_state, do, dht, cu_seqlens, chunk_size):
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        dv = chunk_bwd_dv_local(
            q=q, k=k, g=g, do=do, scale=scale,
            cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        )
        dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q, k=k, w=w, g=g, h0=initial_state, dht=dht, do=do, dv=dv,
            scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        )
        dq, dk, dw, dg = chunk_bwd_dqkwg(
            q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh,
            chunk_size=chunk_size, scale=scale, cu_seqlens=cu_seqlens,
        )
        dk2, dv, db, dg2 = prepare_wy_repr_bwd(
            k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv,
            cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        )
        dk.add_(dk2)
        dg.add_(dg2)
        if dg.dtype != torch.float32:
            raise ValueError(f"dg must be float32, got {dg.dtype}")
        dg = chunk_local_cumsum(
            dg, chunk_size=chunk_size, reverse=True,
            cu_seqlens=cu_seqlens, head_first=True,
        )
        return dq, dk, dv, db, dg, dh0

    class _ChunkGatedDeltaRuleFunction(torch.autograd.Function):
        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size):
            g, o, A, final_state = _fwd(
                q, k, v, g, beta, scale, initial_state,
                output_final_state, cu_seqlens, chunk_size,
            )
            ctx.save_for_backward(q, k, v, g, beta, A, initial_state, cu_seqlens)
            ctx.scale = scale
            ctx.chunk_size = chunk_size
            return o.to(q.dtype), final_state

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(ctx, do, dht):
            q, k, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
            dq, dk, dv, db, dg, dh0 = _bwd(
                q, k, v, g, beta, A, ctx.scale, initial_state, do, dht,
                cu_seqlens, ctx.chunk_size,
            )
            if initial_state is None:
                dh0 = None
            return (
                dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta),
                None, dh0, None, None, None,
            )

    return _ChunkGatedDeltaRuleFunction


def triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-triton chunk_gated_delta_rule (prefill path)."""
    global _TritonFn
    if _TritonFn is None:
        _TritonFn = _make_triton_fn()

    _validate(query, key, value, beta)

    # Caller-side l2norm — autograd tracks it natively, no need to plumb
    # rstd through the Function.
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    scale = key.shape[-1] ** -0.5
    chunk_size = 64
    cu_seqlens = None

    o, final_state = _TritonFn.apply(
        query, key, value, g, beta, scale, initial_state,
        output_final_state, cu_seqlens, chunk_size,
    )
    return o, final_state


# ---------------------------------------------------------------------------
# Flash backend — ascendc fwd_h / fwd_o + bwd kernels via torch.ops.npu,
# vendored triton kernels for the lighter prep ops.
# ---------------------------------------------------------------------------
_FlashFn = None  # populated on first call to flash()


def _make_flash_fn():
    """Build the flash-backend ``autograd.Function`` class lazily.

    Imports both triton (vendored kernels) and torch_npu (ascendc ops).
    """
    import math

    from .kernels.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from .kernels.cumsum import chunk_local_cumsum
    from .kernels.solve_tril import solve_tril
    from .kernels.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
    from .kernels.wy_fast import recompute_w_u_fwd

    # Helpers from the reference's flash_gated_delta_rule.py
    def _prepare_lens(cu_seqlens):
        return cu_seqlens[1:] - cu_seqlens[:-1]

    def _cdiv_torch(a, b):
        return (a + b - 1) // b

    def _prepare_chunk_indices(cu_seqlens, chunk_size):
        indices = torch.cat(
            [torch.arange(n) for n in _cdiv_torch(_prepare_lens(cu_seqlens), chunk_size).tolist()]
        )
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

    def _prepare_chunk_indices1(cu_seqlens_list, chunk_size):
        indices = []
        for i in range(len(cu_seqlens_list) - 1):
            start = cu_seqlens_list[i]
            end = cu_seqlens_list[i + 1]
            length = end - start
            if length <= 0:
                continue
            num_chunks = (length + chunk_size - 1) // chunk_size
            for chunk_id in range(num_chunks):
                indices.append(i)
                indices.append(chunk_id)
        return indices

    def _fwd(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size):
        g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
        A = chunk_scaled_dot_kkt_fwd(
            k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
            output_dtype=torch.float32,
        )
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens)

        chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

        # ascendc fwd_h expects head-first layout
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        w = w.transpose(1, 2).contiguous()
        u = u.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()

        h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state, cu_seqlens, chunk_indices,
            output_final_state, chunk_size,
        )
        o = torch.ops.npu.npu_chunk_fwd_o(
            q, k, v_new, h, g, cu_seqlens, chunk_indices, scale, chunk_size,
        )

        g = g.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
        return g, o, A, final_state

    def _bwd(q, k, v, g, beta, A, scale, initial_state, do, dht, cu_seqlens, chunk_size):
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens)
        chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

        # All ascendc bwd kernels want head-first
        w = w.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        do = do.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous().float()
        u = u.transpose(1, 2).contiguous()
        A_t = A.transpose(1, 2).contiguous()

        h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state, cu_seqlens, chunk_indices, False, chunk_size,
        )

        cu_seqlens1 = cu_seqlens.tolist() if cu_seqlens is not None else None
        chunk_indices1 = _prepare_chunk_indices1(cu_seqlens1, chunk_size) if cu_seqlens1 is not None else None

        dv = torch.ops.npu.npu_chunk_bwd_dv_local(
            q, k, do, g, g_gamma=None, A=A_t,
            cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices1,
            scale=scale, chunk_size=chunk_size,
        )
        dh, dh0, dv = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
            q, k, w, do, dv, g, gK=None, h0=None, dht=dht,
            cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices1,
            scale=scale, chunk_size=chunk_size,
        )
        dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
            q, k, v_new, g, h, do, dh, dv,
            cu_seqlens1, chunk_indices1, scale, chunk_size,
        )
        dq = dq.transpose(1, 2).contiguous()
        dk = dk.transpose(1, 2).contiguous()
        dg = dg.transpose(1, 2).contiguous()

        dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
            k, v_t, beta, A_t, dw, dv, g,
            cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices1, chunk_size=chunk_size,
        )
        dk2, dv, db, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
            k, v_t, beta, A_t, dA, dw, dv, g,
            cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices1, chunk_size=chunk_size,
        )
        dk2 = dk2.transpose(1, 2).contiguous()
        dv = dv.transpose(1, 2).contiguous()
        db = db.transpose(1, 2).contiguous()
        dg2 = dg2.transpose(1, 2).contiguous()

        dk.add_(dk2)
        dg.add_(dg2)
        if dg.dtype != torch.float32:
            raise ValueError(f"dg must be float32, got {dg.dtype}")
        dg = chunk_local_cumsum(
            dg, chunk_size=chunk_size, reverse=True,
            cu_seqlens=cu_seqlens, head_first=False,
        )
        return dq, dk, dv, db, dg, dh0

    class _FlashChunkGatedDeltaRuleFunction(torch.autograd.Function):
        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size):
            g, o, A, final_state = _fwd(
                q, k, v, g, beta, scale, initial_state,
                output_final_state, cu_seqlens, chunk_size,
            )
            ctx.save_for_backward(q, k, v, g, beta, A, initial_state, cu_seqlens)
            ctx.scale = scale
            ctx.chunk_size = chunk_size
            return o.to(q.dtype), final_state

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(ctx, do, dht):
            q, k, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
            dq, dk, dv, db, dg, dh0 = _bwd(
                q, k, v, g, beta, A, ctx.scale, initial_state, do, dht,
                cu_seqlens, ctx.chunk_size,
            )
            if initial_state is None:
                dh0 = None
            return (
                dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta),
                None, dh0, None, None, None,
            )

    return _FlashChunkGatedDeltaRuleFunction


def flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ascendc + triton hybrid chunk_gated_delta_rule (prefill path)."""
    global _FlashFn
    if _FlashFn is None:
        _FlashFn = _make_flash_fn()

    _validate(query, key, value, beta)

    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    scale = key.shape[-1] ** -0.5
    chunk_size = 64
    cu_seqlens = None

    o, final_state = _FlashFn.apply(
        query, key, value, g, beta, scale, initial_state,
        output_final_state, cu_seqlens, chunk_size,
    )
    return o, final_state
