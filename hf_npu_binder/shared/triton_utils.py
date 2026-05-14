"""Triton autograd helpers vendored from MindSpeed.

Two decorators that MindSpeed's triton autograd Functions rely on,
extracted from ``mindspeed/lite/ops/triton/utils.py`` and trimmed to
the binder's needs:

  * ``autocast_custom_fwd`` / ``autocast_custom_bwd`` — bind
    ``torch.amp.custom_fwd / custom_bwd`` to the current device type so
    autograd Functions integrate cleanly with mixed-precision autocast
    (the alloy modeling stack uses bf16 autocast in places).

  * ``input_guard`` — make every tensor input contiguous and run the
    forward inside a device context so multi-NPU setups dispatch to the
    right stream.

Heavy deps (``triton`` / ``torch_npu``) NOT required here — these are
pure Python decorators. The triton-kernel files that import them stay
lazy-NPU.

Vendored from ``MindSpeed/mindspeed/lite/ops/triton/utils.py`` (Songlin
Yang, Yu Zhang, Huawei Technologies Co., Ltd., 2023-2025) with these
changes vs upstream:

  * Removed AMD / x86 / Hopper / TF32 device discovery — binder only
    targets NPU + occasional CUDA dev boxes, so the device map is
    a small static table.
  * Removed ``check_pytorch_version`` branching — torch >= 2.4 is
    assumed (alloy's transformers floor is 5.7 which requires it).
  * Removed unrelated helpers (``tensor_cache``, ``prepare_lens``,
    ``prepare_chunk_indices``) — not used by the MHC kernels.
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable

import torch


def _resolve_device_type() -> str:
    """Pick the device type label that torch.amp.custom_fwd / custom_bwd
    should bind to. NPU first (alloy's primary target), CUDA fallback
    for dev boxes, CPU last (autocast is a no-op there but the calls
    have to bind to *some* device type)."""
    if hasattr(torch, "npu") and getattr(torch, "npu").is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_DEVICE = _resolve_device_type()


autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=_DEVICE)
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=_DEVICE)


def _custom_device_ctx(index: int | None):
    """Return a device-index context manager for the current device type.
    Idempotent: ``None`` index → no-op context (preserves caller's
    current device)."""
    if index is None:
        return contextlib.nullcontext()
    if _DEVICE == "npu":
        return torch.npu.device(index)
    if _DEVICE == "cuda":
        return torch.cuda.device(index)
    return contextlib.nullcontext()


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Decorator: make every tensor arg / kwarg contiguous, set the
    device context to match the first tensor's device, then call ``fn``.

    Triton kernels read tensor strides as ``constexpr``; non-contiguous
    inputs would either silently miscompute or rebuild the kernel for
    every new stride. Running inside a device-index context keeps
    multi-NPU dispatch on the right card without each kernel having to
    re-resolve.

    Mirrors MindSpeed's wrapper exactly — kept as a decorator (vs
    integrating into the autograd Functions) so we can drop it onto any
    triton-backed callable consistently.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        contiguous_args = tuple(
            (i if not isinstance(i, torch.Tensor) else i.contiguous()) for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in contiguous_args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in contiguous_kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        ctx = _custom_device_ctx(tensor.device.index if tensor is not None else None)
        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


__all__ = ["autocast_custom_fwd", "autocast_custom_bwd", "input_guard"]
