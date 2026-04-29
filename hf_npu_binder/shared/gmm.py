"""Grouped matmul primitive (GMM) for expert-routed MoE blocks.

Wraps ``torch_npu.npu_grouped_matmul`` plus a custom autograd backward.
Reusable across MoE model families — ``qwen3_5_moe.experts`` is the first
caller; future ``mistral_moe`` / ``dbrx`` / etc. consume the same primitive.

Reference: ``mindspeed_mm/models/common/gmm.py`` (``GmmFunction`` /
``npu_group_gemm``).

Contract::

    fn(x, weight, group_list) -> Tensor

    x          : [total_tokens, in_dim]    permuted token activations
    weight     : [num_experts, in_dim, out_dim]    expert weights, expert-major
    group_list : [num_experts]    cumulative or per-group token counts
                                  (matches torch_npu.npu_grouped_matmul's
                                   group_list_type)

    returns: [total_tokens, out_dim]

Heavy deps (``torch_npu``) are imported lazily inside the function body so
this module loads on a CPU box.
"""
from __future__ import annotations

import torch


def flash(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "hf_npu_binder.shared.gmm.flash: stub. "
        "Real autograd.Function wrapping torch_npu.npu_grouped_matmul coming next."
    )
