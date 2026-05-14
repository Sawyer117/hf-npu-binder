"""Cross-family GMM-based MoE experts forward (the "flash" entry).

This is the **whole** experts forward — not just grouped matmul. Chains:

    npu_moe_token_permute      gather tokens by expert assignment
    shared.gmm.flash           gate_up projection (grouped matmul primitive)
    [optional clamp] +         per-experts-class clamp gate / up if
    npu_swiglu                 ``self.limit > 0`` (DSV4 swiglu_limit
                               semantics; HF's ``_apply_gate``).
    shared.gmm.flash           down projection (second grouped matmul)
    npu_moe_token_unpermute    scatter tokens back, applying routing weights

Used by every MoE family that wires through HuggingFace's
``@use_experts_implementation`` dispatch. Because
``ALL_EXPERTS_FUNCTIONS`` is a single global table keyed by intent
name (``"flash"`` / ``"grouped_mm"`` / ...), one entry must serve all
model families — branching on ``self.limit`` is how we keep the
fast path correct across architectures with different gate/up
semantics:

  * Qwen3.5-MoE: no ``self.limit`` (or 0). Plain ``npu_swiglu`` runs.
  * DSV4: ``self.limit = config.swiglu_limit`` (default 10.0). Gate
    one-sided ``clamp(max=limit)``; up two-sided
    ``clamp([-limit, limit])`` before ``npu_swiglu`` — matches HF's
    ``DeepseekV4Experts._apply_gate`` byte-for-byte.

References:
  * HF: ``transformers/models/deepseek_v4/modular_deepseek_v4.py``
    ``DeepseekV4Experts._apply_gate``.
  * MindSpeed reference: ``mindspeed_mm/.../qwen3_5_moe`` Qwen3_5MoeExperts.
    (Note: MindSpeed-LLM's ``fused_swiglu_with_limit`` has a separate
    bug — ``chunk(dim=0)`` — that we do NOT replicate. Tracked in
    ``model_gym/upstream_prs_pending.md``.)

Contract — matches HuggingFace ``ALL_EXPERTS_FUNCTIONS`` entries::

    fn(self, hidden_states, top_k_index, top_k_weights) -> Tensor

``self`` is the experts ``nn.Module``; the function reads
``self.gate_up_proj`` / ``self.down_proj`` (both 3D
``[num_experts, *, *]``) directly. Optional ``self.limit`` (float)
toggles the DSV4-style clamp on the activation.

Weight layout adaptation:
  * mindspeed reference stores ``gate_up_proj`` as ``[E, H, 2*I]``.
  * alloy / HF-canonical stores ``gate_up_proj`` as ``[E, 2*I, H]``.
  * ``shared.gmm.flash`` expects ``weight: [E, in, out]``.

This module branches on ``self.is_transposed`` (the HF flag set by
``@use_experts_implementation``):
  * ``is_transposed=False`` (HF-canonical / alloy default): permute on-the-fly.
  * ``is_transposed=True``: pass weights through directly.

Heavy deps (``torch_npu``) are imported lazily inside the function body
so this module loads on a CPU box.
"""
from __future__ import annotations

import torch

from .gmm import flash as gmm_flash


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

    # 4. Activation. If the experts class declares a positive ``self.limit``
    #    (DSV4's ``swiglu_limit``), apply HF ``_apply_gate``-style clamp on
    #    the gate / up halves before the fused npu_swiglu. Otherwise hand the
    #    raw tensor to npu_swiglu directly (Qwen3.5 path).
    limit = float(getattr(self, "limit", 0.0))
    if limit > 0.0:
        gate, up = intermediate.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        activated = torch_npu.npu_swiglu(
            torch.cat([gate, up], dim=-1), dim=-1,
        )
    else:
        activated = torch_npu.npu_swiglu(intermediate, dim=-1)

    # 5. Down projection (second fused GMM).
    down_weight = _expert_weight_in_out(self.down_proj, self.is_transposed)
    output = gmm_flash(activated, down_weight, tokens_per_expert)

    # 6. Scatter tokens back to their original positions, applying the
    #    routing weights as part of the unpermute. The ``output.to(...)`` /
    #    ``routing_weights`` pair runs the weighted-sum in routing_weights'
    #    dtype (typically fp32, since DSV4-style routers compute scores in
    #    fp32 for numerical stability) — equivalent to HF eager's
    #    ``F.linear(...) * top_k_weights`` mixed-dtype multiply. The
    #    ``.to(hidden_states.dtype)`` at the end mirrors HF eager's
    #    ``current.to(final.dtype)`` before index_add: it casts the final
    #    sum back to the residual stream dtype so callers see input-dtype
    #    in, input-dtype out (the contract every other experts impl in
    #    transformers/integrations/moe.py also satisfies). Without this
    #    cast, fp32 routing weights silently promote the entire residual
    #    stream layer-by-layer, eventually surfacing as
    #    aclnnGroupedMatmul x.dtype != weight.dtype crashes downstream.
    final_hidden_states = torch_npu.npu_moe_token_unpermute(
        output.to(routing_weights.dtype),
        row_ids_map,
        probs=routing_weights,
    )
    return final_hidden_states.to(hidden_states.dtype)


__all__ = ["flash", "_expert_weight_in_out"]
