"""Grouped matmul primitive (GMM) for expert-routed MoE blocks.

Wraps ``torch_npu.npu_grouped_matmul`` with a custom autograd backward.
Reusable across MoE model families — ``qwen3_5_moe.experts`` is the first
caller; future ``mistral_moe`` / ``dbrx`` / etc. consume the same primitive.

Ported from ``mindspeed_mm/models/common/gmm.py`` (``GmmFunction`` /
``npu_group_gemm``).

Contract::

    fn(x, weight, group_list) -> Tensor

    x          : [total_tokens, in_dim]
    weight     : [num_experts, in_dim, out_dim]
    group_list : [num_experts]    cumulative or per-group token counts
                                  (matches torch_npu.npu_grouped_matmul's
                                   group_list_type=1)

    returns: [total_tokens, out_dim]

``torch_npu`` is imported lazily inside the autograd.Function so this
module loads on a CPU box. The Function is also instantiated lazily —
``flash`` only constructs / applies it on call, never at module load.
"""
from __future__ import annotations

import torch


def _make_gmm_function():
    """Build the autograd.Function class. Done inside a factory so the
    ``import torch_npu`` in ``forward`` only happens when ``flash`` is
    actually invoked, not at module load.
    """

    class _GmmFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
            import torch_npu  # lazy
            ctx.save_for_backward(x, weight)
            ctx.group_list = group_list
            fwd_output = torch_npu.npu_grouped_matmul(
                [x], [weight],
                bias=None,
                group_list=group_list,
                split_item=2,
                group_type=0,
                group_list_type=1,
            )[0]
            return fwd_output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            import torch_npu  # lazy
            input_tensor, weight = ctx.saved_tensors
            group_list = ctx.group_list

            weight_t = torch.transpose(weight, 1, 2)
            grad_input = torch_npu.npu_grouped_matmul(
                [grad_output], [weight_t],
                bias=None,
                group_list=group_list,
                split_item=2,
                group_type=0,
                group_list_type=1,
            )[0]

            grad_weight = torch_npu.npu_grouped_matmul(
                [input_tensor.T], [grad_output],
                bias=None,
                group_list=group_list,
                split_item=3,
                group_type=2,
                group_list_type=1,
            )[0]

            return grad_input, grad_weight, None

    return _GmmFunction


# Cached on first call so we don't rebuild the class every invocation.
_GmmFunction = None


def flash(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    global _GmmFunction
    if _GmmFunction is None:
        _GmmFunction = _make_gmm_function()
    return _GmmFunction.apply(x, weight, group_list)
