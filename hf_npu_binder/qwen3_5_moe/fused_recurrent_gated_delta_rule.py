"""qwen3_5_moe ``fused_recurrent_gated_delta_rule`` operator (decode path).

Contract::

    fn(query, key, value, *,
       g, beta,
       initial_state, output_final_state, use_qk_l2norm_in_kernel)
        -> tuple[Tensor, Tensor | None]   # (core_attn_out, last_recurrent_state)

Sequential per-token recurrent update; counterpart to the chunked prefill op
in :mod:`.chunk_gated_delta_rule`.

Backend status
--------------
Real ``triton`` and ``flash`` ports (wrapping fla / mindspeed-lite recurrent
kernels) are not yet written. Until they are, both backends delegate to a
pure-torch reference (:func:`_torch_recurrent_gated_delta_rule`) which is
correct under the same contract. Decode is one token at a time so the
sequential loop in torch is acceptable in the interim — it just won't get
the speedup a fused kernel would provide.

Drop-in upgrade path: replace the body of :func:`triton` (or :func:`flash`)
with a real kernel call. Callers and the alloy bridge see the same function
signatures, no other code needs to change.
"""
from __future__ import annotations

import torch

from .chunk_gated_delta_rule import _l2norm


def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure-torch reference for the per-token gated delta-rule recurrence.

    Mirrors HF's ``Qwen3_5MoeGatedDeltaNet`` decode-path math: walk
    sequence_length tokens; at each step apply forget gate ``g``, blend
    in new ``v`` against current ``k`` weighted by ``beta``, query the
    updated state with ``q``. State + accum are kept in fp32 to dodge
    bf16 noise; final output cast back to input dtype.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``triton`` backend — currently a torch fallback.

    Triton kernel for fused_recurrent_gated_delta_rule not yet ported.
    Replace the body with the real triton call when one exists; the
    contract above stays the same.
    """
    return _torch_recurrent_gated_delta_rule(
        query, key, value, g, beta,
        initial_state, output_final_state, use_qk_l2norm_in_kernel,
    )


def flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``flash`` backend — currently a torch fallback.

    ascendc / fla wrap not yet implemented. Replace the body with the
    real call when one exists.
    """
    return _torch_recurrent_gated_delta_rule(
        query, key, value, g, beta,
        initial_state, output_final_state, use_qk_l2norm_in_kernel,
    )
