<div align="center">

# hf-npu-binder

**Optional Ascend NPU, Triton-Ascend, and torch_npu fast paths for Hugging Face-native model code.**

`hf-npu-binder` ships backend callables. It does not patch models, register
global hooks on import, or depend on Alloy. Vanilla Hugging Face models, Alloy,
MindSpeed-MM, or a private model package can wire these functions into their own
dispatch tables.

[Design](#design) - [Support Matrix](#support-matrix) - [Install](#install) - [Use](#use) - [Defaults](#defaults) - [Layout](#layout) - [Validation](#validation)

</div>

---

## Design

The package is intentionally narrow. It is a binder between HF-shaped tensor
contracts and NPU-oriented kernels.

- **Consumer-agnostic.** Operator files expose plain functions such as
  `hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule.triton`. They do not import
  Alloy or patch Hugging Face classes.
- **Lazy heavy dependencies.** `torch_npu`, Triton-Ascend, and CANN JIT extension
  loading happen inside the backend function body, not at module import time. A
  CPU development box can import the package and run interface tests.
- **HF-native boundary.** Public entry points use the tensor layouts expected by
  HF / Alloy call sites. If a vendored MindSpeed kernel wants SBHD or SBND
  internally, the adapter handles the permutation at the boundary.
- **Correct fallback over silent speed claims.** Some backends are real kernels;
  some are torch fallbacks with the final signature already frozen. The support
  table below makes that explicit.

## Support Matrix

Status key:

- `Shipped`: real backend implementation is present.
- `Opt-in`: implementation exists but is not the conservative default.
- `Torch fallback`: function is callable and correct, but currently delegates to
  a torch reference rather than a fused kernel.
- `Planned`: contract is documented, implementation is not shipped here yet.

| Operator | Backend | Status | Requirements | Notes |
| --- | --- | --- | --- | --- |
| `shared.gmm.flash` | torch_npu | Shipped | `torch_npu` | Wraps `npu_grouped_matmul` with autograd; reusable grouped matmul primitive for MoE experts |
| `shared.moe_experts.flash` / `qwen3_5_moe.experts.flash` | torch_npu | Shipped | `torch_npu` | Full expert block: permute, GMM, SwiGLU, GMM, unpermute; handles Qwen3.5 and DSV4 clamp semantics |
| `qwen3_5_moe.chunk_gated_delta_rule.triton` | Triton-Ascend | Shipped | Triton-Ascend, bf16 tensors | Chunked prefill Gated DeltaNet path with vendored kernels and custom autograd |
| `qwen3_5_moe.chunk_gated_delta_rule.flash` | Triton fallback today | Opt-in | Triton-Ascend today; AscendC path kept behind a flag | Public `flash` entry delegates to `triton` until the AscendC chain is verified end-to-end |
| `qwen3_5_moe.fused_recurrent_gated_delta_rule.triton` | torch | Torch fallback | torch | Decode-path recurrent update; signature frozen for a future fused kernel |
| `qwen3_5_moe.fused_recurrent_gated_delta_rule.flash` | torch | Torch fallback | torch | Same contract as `triton`, no speedup yet |
| `qwen3_5_moe.causal_conv1d.triton` | torch | Torch fallback | torch | Decode rolling depthwise-conv update; signature matches HF / Alloy call site |
| `qwen3_5_moe.causal_conv1d.flash` | torch | Torch fallback | torch | Placeholder backend name with working torch behavior |
| `deepseek_v4.sparse_flash_attention.triton` | Triton-Ascend | Shipped, explicit opt-in | Triton-Ascend, supported topk width | CSA adapter: sliding window plus Lightning-Indexer picks; CONFIG_MAP widths are 128, 160, 640 |
| `deepseek_v4.sparse_flash_attention.ascendc` | AscendC / CANN aclnn | Opt-in | CANN with `aclnnSparseAttnSharedkv`, `torch_npu`, JIT build env | CSA fused path; first call JIT-compiles the C++ wrapper |
| `deepseek_v4.compressed_attention.triton` | Triton-Ascend | Shipped, explicit opt-in | Triton-Ascend, supported topk width | HCA and sliding-only adapter over the same vendored sparse attention kernel |
| `deepseek_v4.hyper_connection.triton` | Triton-Ascend | Shipped, explicit opt-in | Triton-Ascend, `hc_mult=4` | DSV4 MHC HyperConnection path; rejects non-paper stream counts clearly |
| DSV4 compressor / Lightning Indexer fused kernels | TBD | Planned | TBD | Contracts are tracked in `hf_npu_binder/deepseek_v4/README.md` |

The table is intentionally conservative. For DSV4 attention and MHC, `auto`
defaults currently prefer torch in Alloy because measured Triton-Ascend behavior
has not yet beaten eager `torch_npu` on both speed and precision across the target
shapes. Explicit opt-in remains available.

## Install

CPU-safe development install:

```bash
git clone https://github.com/Sawyer117/hf-npu-binder
cd hf-npu-binder
pip install -e .
```

NPU extras:

```bash
# torch_npu backend pieces
pip install -e .[npu]

# Triton-Ascend backend pieces
pip install -e .[triton] \
    --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

For a full Ascend development box setup, including CANN, matching torch /
`torch_npu`, Triton-Ascend, a source install of `transformers`, Alloy, and smoke
tests, see [INSTALL.md](INSTALL.md).

Important environment notes:

- `ASCEND_HOME_PATH` must be set before calling AscendC JIT ops. Source the CANN
  toolkit `set_env.sh` in the shell that launches Python.
- DSV4 CSA AscendC requires a CANN build that exposes
  `aclnnSparseAttnSharedkv`. Older CANN builds may import successfully but miss
  that host API.
- `transformers>=5.8.0.dev0` is expected for the newest DSV4 attention-interface
  surface; in practice this may mean installing `transformers` from source until
  a matching release is available.

## Use

Direct use is just a function call. Heavy imports happen only when the selected
backend is invoked.

```python
from hf_npu_binder.qwen3_5_moe.chunk_gated_delta_rule import triton

out, state = triton(
    query,
    key,
    value,
    g=g,
    beta=beta,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
```

For Alloy users, the bridge lives on the Alloy side so this package stays
consumer-agnostic:

```python
import alloy.integrations.hf_npu_binder as binder

# Registers binder functions into Alloy's implementation registry on import.
chosen = binder.activate(model, prefer="auto")
print(chosen)
```

Explicit per-module choices are supported:

```python
binder.activate(model, {
    "qwen3_5_gdn": "triton",
    "experts": "flash",
    "dsv4_csa": "ascendc",
    "dsv4_hca": "torch",
    "dsv4_mhc": "triton",
})
```

For vanilla HF or private model packages, register the function in your own
dispatch table, replace a method, or call it from a module forward. The binder
does not prescribe the integration strategy.

## Defaults

`hf_npu_binder.DEFAULTS` translates user-facing intents (`auto`, `flash`,
`triton`, `ascendc`, `torch`) into actual implementation names per operator.
Alloy reads this table when broadcasting a single preference across many modules.

Current policy:

| Intent area | `auto` choice | Reason |
| --- | --- | --- |
| Qwen3.5 GDN chunk prefill | `triton` | Real Triton backend is available and is the measured useful path |
| Qwen3.5 MoE experts | `flash` | `torch_npu` grouped-matmul expert path is available |
| Qwen3.5 recurrent / causal conv decode helpers | `triton` / `flash` names resolve, but functions currently torch-fallback | Contract stability before fused decode kernels land |
| DSV4 CSA / HCA / sliding attention | `torch` | Triton wrappers are available but not yet the conservative default for all target shapes |
| DSV4 MHC | `torch` | Triton path exists for `hc_mult=4`; opt-in until production-scale evidence changes the default |

This keeps `activate(model, "auto")` safe: it should not silently choose a
backend that is slower, numerically worse, or unavailable on common dev setups.
Users who know their environment can opt in to `triton` or `ascendc` explicitly.

## Layout

```text
hf_npu_binder/
├── __init__.py                         # DEFAULTS and package-level exports
├── shared/
│   ├── gmm.py                          # torch_npu grouped matmul primitive
│   ├── moe_experts.py                  # cross-family HF experts flash path
│   └── op_builder.py                   # AscendC / aclnn JIT extension builder
├── qwen3_5_moe/
│   ├── chunk_gated_delta_rule.py       # GDN prefill path
│   ├── fused_recurrent_gated_delta_rule.py
│   ├── causal_conv1d.py
│   ├── experts.py                      # legacy re-export of shared experts path
│   └── kernels/                        # vendored Triton kernels
└── deepseek_v4/
    ├── sparse_flash_attention.py       # CSA triton / ascendc adapters
    ├── compressed_attention.py         # HCA / sliding triton adapter
    ├── hyper_connection.py             # MHC triton adapter
    ├── ascendc_sparse_attn_shared_kv.py
    ├── csrc/                           # C++ JIT extension sources
    └── mhc/                            # MHC Triton primitives
```

Layout convention:

- One subpackage per HF model family, matching `transformers/models/<name>/`
  where practical.
- One file per operator, named after the HF function / class surface where
  possible.
- One top-level function per backend: `triton`, `flash`, `ascendc`, etc.
- Shared primitives belong under `shared/`; family modules compose them rather
  than reimplementing grouped matmul or expert routing.

## Validation

CPU-safe interface checks require ordinary `torch`, but should not require
`torch_npu`, Triton-Ascend, or CANN. They should import the package without
pulling `torch_npu` or Triton into `sys.modules`, and should verify function
signatures stay stable. The main entry point is:

```bash
PYTHONPATH=. python tests/test_kernels.py
```

NPU-only checks exercise real kernels and JIT paths when the Ascend environment is
installed and `ASCEND_HOME_PATH` is set:

```bash
PYTHONPATH=. python debug/test_ascendc_sparse_attn.py
PYTHONPATH=. python tests/test_kernels_on_npu.py
```

First AscendC calls compile a local extension under the PyTorch extension cache.
Subsequent runs reuse the compiled `.so` unless sources or build flags change.

## More Detail

- [INSTALL.md](INSTALL.md) gives the full NPU development setup.
- [hf_npu_binder/deepseek_v4/README.md](hf_npu_binder/deepseek_v4/README.md)
  documents DSV4 tensor layouts, MindSpeed-to-HF naming, CONFIG_MAP width
  constraints, and DSV4-specific backend policy.

## License

License is not yet chosen. Treat the code as source-available pending a formal
decision.
