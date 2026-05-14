"""Fused RMSNorm scaling-factor (rsqrt-only, no learnable weight).

Used by ``hyper_connection.triton`` to compute the per-row
``1 / sqrt(mean(x²) + eps)`` factor without applying it inside the
kernel — the caller multiplies the matmul output by ``rsqrt`` directly,
saving one materialisation of the normalised tensor. Mathematically
equivalent to the alloy ``DeepseekV4UnweightedRMSNorm`` body
(``x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)``)
when paired with the call sequence
``mix = F.linear(x, W.T) * rsqrt``.

Vendored from
``MindSpeed-LLM/mindspeed_llm/tasks/models/transformer/deepseek4/``:
``rmsnorm_without_weight_triton_kernel.py`` (kernel + forward/backward
launchers) and ``rmsnorm_without_weight.py`` (autograd Function
wrapper). Only changes vs upstream:
  * imports re-pointed to binder's vendored utils
    (:mod:`...shared.triton_utils`).
  * ``rmsnorm_without_weight`` renamed in this file to
    ``_rmsnorm_without_weight_fwd`` / ``_rmsnorm_without_weight_bwd``
    for clarity inside the autograd Function (the public entry point
    is ``rmsnorm_without_weight_triton`` at the bottom).

Heavy deps (``triton``, ``torch_npu``) imported lazily — top-level
import of this module is CPU-safe.
"""
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
# Vendored from MindSpeed-LLM 2026-05-15.
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from ...shared.triton_utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


if _TRITON_AVAILABLE:

    @triton.jit
    def _rmsnorm_without_weight_kernel(
        x_ptr, res_ptr, D: tl.constexpr, norm_eps: tl.constexpr, head: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        """Triton kernel for RMSNorm scaling factor forward pass."""
        pid = tl.program_id(0)
        offset_base = tl.arange(0, BLOCK_D)

        for idx in range(0, head):
            square_sum = 0.0
            for d in range(0, D, BLOCK_D):
                d_mask = (d + offset_base) < D
                offset = pid * head * D + idx * D + d + offset_base
                x = tl.load(x_ptr + offset, mask=d_mask, other=0.0)
                square_sum += tl.sum(x * x)

            mean = square_sum / D
            res = tl.rsqrt(mean + norm_eps)
            tl.store(res_ptr + pid * head + idx, res)


    @triton.jit
    def _rmsnorm_without_weight_backward_kernel(
        grad_res_ptr, x_ptr, res_ptr, grad_x_ptr, D: tl.constexpr, head: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        """Triton kernel for RMSNorm scaling factor backward pass."""
        pid = tl.program_id(0)
        offset_base = tl.arange(0, BLOCK_D)
        for idx in range(0, head):
            grad_res = tl.load(grad_res_ptr + pid * head + idx)
            res = tl.load(res_ptr + pid * head + idx)

            factor = (-1.0) * grad_res * (res * res * res) / D

            for d in range(0, D, BLOCK_D):
                d_mask = (d + offset_base) < D
                offset = pid * head * D + idx * D + d + offset_base
                x = tl.load(x_ptr + offset, mask=d_mask, other=0.0)
                grad_x = factor * x
                tl.store(grad_x_ptr + offset, grad_x, mask=d_mask)


def _rmsnorm_without_weight_fwd(x: torch.Tensor, norm_eps: float = 1e-6) -> torch.Tensor:
    """Launcher for the forward kernel. Returns the rsqrt scaling factor
    with shape ``[..., 1]`` (last-dim singleton for broadcast)."""
    x_shape = x.shape
    if len(x_shape) != 4 and len(x_shape) != 3:
        raise ValueError(
            f"rmsnorm_without_weight only supports 3-D or 4-D input; got shape {x_shape}"
        )
    D = x_shape[-1]
    batch_seq_size = x_shape[0] * x_shape[1]

    # Large-D fallback (kernel BLOCK_D capped at 16384).
    if D > 16384:
        x_square_mean = x.square().mean(dim=-1, keepdim=True)
        return torch.rsqrt(x_square_mean + norm_eps)

    if len(x_shape) == 4:
        res = torch.empty((x_shape[0], x_shape[1], x_shape[2], 1), dtype=x.dtype, device=x.device)
        head = x_shape[-2]
    else:
        res = torch.empty((x_shape[0], x_shape[1], 1), dtype=x.dtype, device=x.device)
        head = 1

    BLOCK_D = min(triton.next_power_of_2(D), 16384)
    _rmsnorm_without_weight_kernel[(batch_seq_size,)](x, res, D, norm_eps, head, BLOCK_D)
    return res


def _rmsnorm_without_weight_bwd(
    grad_res: torch.Tensor, x: torch.Tensor, res: torch.Tensor, norm_eps: float = 1e-6
) -> torch.Tensor:
    """Launcher for the backward kernel. Returns gradient w.r.t. x."""
    x_shape = x.shape
    if len(x_shape) != 4 and len(x_shape) != 3:
        raise ValueError(
            f"rmsnorm_without_weight only supports 3-D or 4-D input; got shape {x_shape}"
        )
    D = x_shape[-1]
    head = x_shape[-2] if len(x_shape) == 4 else 1

    if D > 16384:
        m_eps_pow32 = res ** 3
        grad_m = grad_res * (-0.5) * m_eps_pow32
        grad_x = grad_m * 2 * x / D
        return grad_x

    grad_x = torch.empty_like(x)
    batch_seq_size = x_shape[0] * x_shape[1]
    if batch_seq_size == 0 or D == 0:
        return grad_x

    BLOCK_D = min(triton.next_power_of_2(D), 16384)
    _rmsnorm_without_weight_backward_kernel[(batch_seq_size,)](
        grad_res, x, res, grad_x, D, head, BLOCK_D,
    )
    return grad_x


class _RMSNormWithoutWeightFunction(torch.autograd.Function):
    """Autograd Function around the forward / backward kernels.

    Mirrors MindSpeed-LLM's ``RMSNormWithoutWeightFunction`` exactly;
    only the imports differ (binder-local vs mindspeed_llm paths).
    """

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x: torch.Tensor, norm_eps: float = 1e-6):
        res = _rmsnorm_without_weight_fwd(x, norm_eps)
        ctx.save_for_backward(x, res)
        ctx.norm_eps = norm_eps
        return res

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, grad_res: torch.Tensor):
        x, res = ctx.saved_tensors
        grad_x = _rmsnorm_without_weight_bwd(grad_res, x, res, ctx.norm_eps)
        return grad_x, None


def rmsnorm_without_weight_triton(x: torch.Tensor, norm_eps: float = 1e-6) -> torch.Tensor:
    """Public entry. Returns the per-row rsqrt scaling factor of x.

    Equivalent to ``torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)``
    but fused into a single triton kernel pass. Callers typically use
    the result like ``out = (x @ W.T) * rsqrt`` instead of normalising
    x first and then projecting.

    Args:
        x: 3-D ``[B, S, D]`` or 4-D ``[B, S, H, D]``.
        norm_eps: small constant added inside the sqrt for stability.

    Returns:
        ``[B, S, 1]`` (3-D input) or ``[B, S, H, 1]`` (4-D input) — same
        leading dims, scalar per row.
    """
    return _RMSNormWithoutWeightFunction.apply(x, norm_eps)


__all__ = ["rmsnorm_without_weight_triton"]
