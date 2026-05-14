"""alloy-facing fast-path for the DSV4 MHC HyperConnection.

Drop-in replacement for alloy's :class:`_HyperConnection.forward`.
Composes the three vendored triton primitives:

  1. :func:`mhc.rmsnorm_without_weight.rmsnorm_without_weight_triton`
     — per-row ``rsqrt(mean(x²) + eps)`` factor.
  2. :func:`F.linear` (torch, the matmul stays standard fp32 on NPU's
     auto-fused linear).
  3. :func:`mhc.sinkhorn.hc_split_sinkhorn_triton`
     — sigmoid + sinkhorn-iter → ``(pre, post, comb)``.
  4. :func:`mhc.pre_bmm.hc_pre_bmm_forward`
     — ``(pre.unsqueeze(-1) * hidden_streams).sum(dim=2)`` fused.

Output contract: same triple ``(post, comb, collapsed)`` as the torch
reference in alloy's ``_HyperConnection.forward``. ``collapsed`` cast
back to ``hidden_streams.dtype`` (mirrors the binder MoE flash
dtype-contract — input-dtype in, input-dtype out for the residual
stream tensor).

**hc_mult = 4 constraint**: the vendored sinkhorn + pre_bmm kernels
are manually unrolled for 4 streams (DSV4 paper config). This entry
raises a clear ValueError for any other ``module.hc_mult`` so callers
hit the failure at the alloy bridge dispatch rather than deep inside
the triton kernel.

Math equivalence to alloy torch path:
  - alloy: ``flat = (x * rsqrt) @ W.T`` (norm first, then matmul)
  - here:  ``flat = (x @ W.T) * rsqrt`` (matmul first, then scale)
  Math-equivalent (rsqrt is per-row scalar; commutes through linear),
  but bf16 rounding order differs slightly → expect drift in noise floor.

Heavy deps (``triton`` / ``torch_npu``) imported lazily inside the
function body. This module is CPU-import-safe for the alloy bridge
registration step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def triton(
    module: torch.nn.Module,
    hidden_streams: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """alloy ``_HyperConnection.forward`` drop-in.

    Args:
        module: the alloy ``_HyperConnection`` instance — reads
            ``module.fn`` (mix Linear weight, shape
            ``[(2+hc_mult)*hc_mult, hc_mult*D]``), ``module.scale``
            (per-class scales, ``[3]``), ``module.base`` (per-class
            biases, ``[(2+hc_mult)*hc_mult]``), ``module.hc_mult`` (=
            4 required), ``module.hc_sinkhorn_iters``, ``module.hc_eps``.
        hidden_streams: ``[B, S, hc_mult, D]`` multi-stream residual
            input.

    Returns:
        ``(post, comb, collapsed)`` matching alloy torch:
        ``post  [B, S, hc_mult]``,
        ``comb  [B, S, hc_mult, hc_mult]`` (doubly-stochastic),
        ``collapsed [B, S, D]`` cast back to ``hidden_streams.dtype``.

    Raises:
        ValueError if ``module.hc_mult != 4``.
    """
    if module.hc_mult != 4:
        raise ValueError(
            f"MHC triton fast-path requires module.hc_mult=4 (DSV4 paper "
            f"config); got hc_mult={module.hc_mult}. The vendored MindSpeed "
            f"sinkhorn + pre_bmm kernels manually unroll 4 streams and "
            f"reject other counts. Use prefer='torch' for hc_mult != 4, "
            f"or set config.hc_mult=4 to match production DSV4."
        )

    # Lazy: ``triton`` + ``torch_npu`` only imported on first call.
    from .mhc.rmsnorm_without_weight import rmsnorm_without_weight_triton
    from .mhc.sinkhorn import hc_split_sinkhorn_triton
    from .mhc.pre_bmm import hc_pre_bmm_forward

    input_dtype = hidden_streams.dtype
    B, S, hc_mult, D = hidden_streams.shape

    # 1. Flatten streams + cast to fp32 for the norm/mix pipeline (same
    #    fp32 regime alloy's torch path uses — sinkhorn really wants
    #    fp32 for stable iterates).
    x_flat = hidden_streams.flatten(start_dim=2).float()           # [B, S, hc_mult*D]

    # 2. RMSNorm rsqrt (factor only, not the normalized x).
    rsqrt = rmsnorm_without_weight_triton(x_flat, module.hc_eps)   # [B, S, 1]

    # 3. Linear (mix matmul) + scale by rsqrt. Same algebra as alloy's
    #    ``flat = input_norm(x); mix = F.linear(flat, fn.float())`` —
    #    rsqrt is broadcast-scalar per row, commutes through linear.
    mix = F.linear(x_flat, module.fn.float()) * rsqrt              # [B, S, (2+hc_mult)*hc_mult]

    # 4. Sinkhorn pre / post / comb.
    pre, post, comb = hc_split_sinkhorn_triton(
        mix, module.scale, module.base,
        hc_mult=hc_mult,
        sinkhorn_iters=module.hc_sinkhorn_iters,
        eps=module.hc_eps,
    )

    # 5. pre × streams fused BMM. hc_pre_bmm_forward casts H to fp32
    #    internally and outputs fp32; cast back to input dtype to keep
    #    the residual-stream dtype contract.
    collapsed = hc_pre_bmm_forward(pre, hidden_streams).to(input_dtype)

    return post, comb, collapsed


__all__ = ["triton"]
