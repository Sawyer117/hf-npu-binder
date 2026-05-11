"""Vendored triton kernels for deepseek_v4 operators.

Pending port from ``MindSpeed-LLM/mindspeed_llm/tasks/models/transformer/
deepseek4/g2_attention_kernel.py``. The file there carries the full
``SparseFlashAttentionTriton`` autograd Function plus its fwd / bwd
kernels and tile configs; the port will copy it here with the
megatron-specific ``from megatron.training import get_args`` dependency
removed (the ``use_triton_sfa`` switch becomes a wrapper argument).
"""
