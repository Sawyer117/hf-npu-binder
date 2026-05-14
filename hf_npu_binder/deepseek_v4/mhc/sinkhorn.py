"""Autograd Function wrapper around the HC-Split Sinkhorn triton kernel.

Wraps :func:`sinkhorn_triton_kernel.hc_split_sinkhorn` (forward) and
:func:`sinkhorn_triton_kernel.hc_split_sinkhorn_backward` so the
sigmoid + sinkhorn-iter computation participates in autograd cleanly.

Public entry: :func:`hc_split_sinkhorn_triton`. Takes the mix tensor
(post matmul + scale by rsqrt) plus the per-class hc_scale / hc_base
parameters, returns ``(pre, post, comb)`` — the three mixing tensors
the MHC HyperConnection produces.

Constraint: ``hc_mult = 4`` (DSV4 paper config). Other values raise
inside the underlying kernel's docstring contract.

Vendored from MindSpeed-LLM
``mindspeed_llm/tasks/models/transformer/deepseek4/mhc/sinkhorn.py``.
Only change vs upstream: imports re-pointed at the binder-vendored
kernel + binder shared triton utils.
"""
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
# Vendored from MindSpeed-LLM 2026-05-15.
from __future__ import annotations

import torch

from .sinkhorn_triton_kernel import hc_split_sinkhorn, hc_split_sinkhorn_backward
from ...shared.triton_utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class HcSplitSinkhornFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ):
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps,
        )
        ctx.save_for_backward(mixes, hc_scale, hc_base)
        ctx.hc_mult = hc_mult
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps
        return pre, post, comb

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        grad_pre: torch.Tensor,
        grad_post: torch.Tensor,
        grad_comb: torch.Tensor,
    ):
        mixes, hc_scale, hc_base = ctx.saved_tensors
        grad_mixes, grad_scale, grad_base = hc_split_sinkhorn_backward(
            grad_pre, grad_post, grad_comb,
            mixes, hc_scale, hc_base,
            ctx.hc_mult, ctx.sinkhorn_iters, ctx.eps,
        )
        return grad_mixes, grad_scale, grad_base, None, None, None


@torch.compiler.disable
def hc_split_sinkhorn_triton(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-fused HC-Split Sinkhorn (sigmoid + iter normalisation).

    Args:
        mixes: ``[B, S, (2 + hc_mult) * hc_mult]`` post-matmul logits
            (== ``F.linear(input_norm(x), W) * rsqrt`` in the alloy
            HyperConnection structure).
        hc_scale: ``[3]`` per-class scale (pre / post / comb).
        hc_base: ``[(2 + hc_mult) * hc_mult]`` per-class bias.
        hc_mult: stream multiplicity. **MUST be 4** (kernel-hardcoded).
        sinkhorn_iters: row/column normalisation iteration count.
        eps: numeric stability constant.

    Returns:
        ``(pre [B, S, hc_mult], post [B, S, hc_mult], comb [B, S, hc_mult, hc_mult])``.
    """
    pre, post, comb = HcSplitSinkhornFunction.apply(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps,
    )
    return pre, post, comb


__all__ = ["hc_split_sinkhorn_triton", "HcSplitSinkhornFunction"]
