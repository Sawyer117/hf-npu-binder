"""qwen3_5_moe experts forward — GMM-based fused MoE path.

This is the **whole** experts forward — not just grouped matmul. Chains:

    npu_moe_token_permute      gather tokens by expert assignment
    shared.gmm.flash           gate_up projection (grouped matmul primitive)
    npu_swiglu                 fused gate * silu(up) activation
    shared.gmm.flash           down projection (second grouped matmul)
    npu_moe_token_unpermute    scatter tokens back, applying routing weights

The grouped-matmul primitive lives in ``hf_npu_binder.shared.gmm`` so other
MoE families (mistral_moe, dbrx, ...) can reuse the exact same wrapper +
custom backward without duplication. This file is purely the qwen3_5-specific
composition.

Reference: ``mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py``
``Qwen3_5MoeExperts.forward`` (NPU branch).

Contract — matches HuggingFace ``ALL_EXPERTS_FUNCTIONS`` entries::

    fn(self, hidden_states, top_k_index, top_k_weights) -> Tensor

``self`` is the experts ``nn.Module``; the function reads
``self.gate_up_proj`` / ``self.down_proj`` (both 3D
``[num_experts, *, *]``) directly.

Weight layout adaptation:
  * mindspeed reference stores ``gate_up_proj`` as ``[E, H, 2*I]``.
  * alloy / HF-canonical stores ``gate_up_proj`` as ``[E, 2*I, H]``.
  * ``shared.gmm.flash`` expects ``weight: [E, in, out]``.

This module branches on ``self.is_transposed`` (the HF flag set by
``@use_experts_implementation``):
  * ``is_transposed=False`` (HF-canonical / alloy default): permute on-the-fly.
  * ``is_transposed=True``: pass weights through directly.

Heavy deps (``torch_npu``) are imported lazily inside the function body so
this module loads on a CPU box.
"""
from __future__ import annotations

import torch

from ..shared.gmm import flash as gmm_flash


def _expert_weight_in_out(weight: torch.Tensor, is_transposed: bool) -> torch.Tensor:
    """Return weight in ``[E, in, out]`` form expected by ``gmm.flash``.

    HF's ``is_transposed=False`` means the stored layout is ``[E, out, in]``
    (output dimension first), which is the HF / alloy canonical form. We
    permute axes 1 and 2 to bring it into ``[E, in, out]``.
    """
    if not is_transposed:
        return weight.permute(0, 2, 1).contiguous()
    return weight


def flash(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    import torch_npu  # lazy: NPU-only

    # 1. Permute tokens so they're grouped by their assigned expert.
    selected_experts = top_k_index
    routing_weights = top_k_weights
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states,
        selected_experts.to(torch.int32),
    )

    # 2. group_list = how many tokens each expert handles, in order.
    tokens_per_expert = torch.histc(
        selected_experts,
        bins=self.num_experts,
        min=0,
        max=self.num_experts,
    )

    # 3. Gate+up projection (one fused GMM with concatenated [gate, up] weights).
    gate_up_weight = _expert_weight_in_out(self.gate_up_proj, self.is_transposed)
    intermediate = gmm_flash(permuted_hidden_states, gate_up_weight, tokens_per_expert)

    # 4. Fused SwiGLU: takes the concatenated [gate, up] tensor on the last
    #    dim and outputs silu(gate) * up.
    activated = torch_npu.npu_swiglu(intermediate, dim=-1)

    # 5. Down projection (second fused GMM).
    down_weight = _expert_weight_in_out(self.down_proj, self.is_transposed)
    output = gmm_flash(activated, down_weight, tokens_per_expert)

    # 6. Scatter tokens back to their original positions, applying the
    #    routing weights as part of the unpermute.
    final_hidden_states = torch_npu.npu_moe_token_unpermute(
        output.to(routing_weights.dtype),
        row_ids_map,
        probs=routing_weights,
    )
    return final_hidden_states
