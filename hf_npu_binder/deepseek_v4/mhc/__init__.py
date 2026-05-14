"""MHC (Mixture of Head Clusters / HyperConnection) triton kernels.

Vendored from MindSpeed-LLM
``mindspeed_llm/tasks/models/transformer/deepseek4/mhc/`` and
``mindspeed_llm/tasks/models/transformer/deepseek4/rmsnorm_without_weight*.py``.
Used by the alloy-facing
:func:`hf_npu_binder.deepseek_v4.hyper_connection.triton` entry to
fast-path alloy's ``_HyperConnection.forward``.

Kernels:

  * :mod:`rmsnorm_without_weight` — fused RMSNorm rsqrt (no learnable
    scale). Returns only the scaling factor ``rsqrt(mean(x²) + eps)`` so
    callers can apply it to a matmul output instead of normalising x
    first (one less elementwise pass).

  * :mod:`sinkhorn` / :mod:`sinkhorn_triton_kernel` — fused
    sigmoid + sinkhorn-iter pre/post/comb computation. **Hardcoded
    hc_mult=4** (DSV4 paper config).

  * :mod:`pre_bmm` — 4D-stream BMM ``(pre.unsqueeze(-1) * x).sum(dim=2)``
    fused into a single kernel. **Hardcoded N=4** (= hc_mult=4).

The hc_mult=4 constraint comes from MindSpeed's kernels manually
unrolling the stream loop. alloy callers with ``config.hc_mult != 4``
have to stay on the torch path (the binder hyper_connection.triton
entry checks and raises a clear error).
"""
from __future__ import annotations

from . import rmsnorm_without_weight
from . import sinkhorn
from . import pre_bmm

__all__ = ["rmsnorm_without_weight", "sinkhorn", "pre_bmm"]
