# hf-npu-binder

Optional fast-path implementations for HuggingFace-native models on Ascend
NPU (via `torch_npu` ascendc ops) and on triton/tilelang backends.

This package is consumer-agnostic. It exposes pure callables organised by
HuggingFace model family — `hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule.triton`
and friends. It does not register itself anywhere, patch any class, or import
any model package. Vanilla HF code, [alloy](https://github.com/Sawyer117/alloy),
mindspeed_mm, or any custom HF model package can wire these in on its own
terms.

## Status: skeleton only

Stubs raise `NotImplementedError`. Signatures are frozen; real triton /
ascendc kernels are coming next.

Currently shipped:

```
hf_npu_binder/
├── __init__.py
├── shared/                                   # cross-family NPU primitives
│   └── gmm.py                                #   grouped matmul (used by every MoE family)
└── qwen3_5_moe/
    ├── __init__.py
    ├── chunk_gated_delta_rule.py            # def triton(...) / def flash(...)
    ├── fused_recurrent_gated_delta_rule.py
    ├── causal_conv1d.py
    ├── experts.py                            # composite: permute + gmm + swiglu + gmm + unpermute
    └── kernels/                              # ported triton kernels land here
```

Family-specific operators (`qwen3_5_moe.experts.flash`) compose primitives
from `shared/` rather than each re-implementing them. Add a new MoE family
by creating `<family>/experts.py` that imports from `shared.gmm`.

## Install

```bash
pip install -e .
# pip install hf-npu-binder[npu]      # later, when torch_npu is needed
# pip install hf-npu-binder[triton]
```

## Use

Functions are plain callables. Heavy deps (`torch_npu`, `triton`) are
imported lazily inside the function body so the package loads on a CPU box.

```python
from hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule import flash as flash_chunk

# Wire it into your model however you like — replace a method, set an attr,
# register into a dispatch table, etc. The binder doesn't care.
```

For alloy users, the bridge module on the alloy side does the wiring:

```python
import alloy.integrations.hf_npu_binder as binder      # opt-in import; runs registration
binder.activate(model, prefer="flash")
```

## Layout convention

- One subpackage per HF model family (matching `transformers/models/<name>/`).
- One file per operator inside that family (matching the original HF function
  / class name where possible).
- One top-level function per backend, named after the backend (`triton`,
  `flash`, ...).

Real triton kernels (`chunk_delta_h.py`, `chunk_o.py`, ...) live in each
family's `kernels/` subdirectory and are imported lazily by the operator
files when their backend is invoked.
