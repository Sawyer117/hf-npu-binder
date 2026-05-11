# hf-npu-binder · deepseek_v4

NPU fast-path callables for HuggingFace `transformers.models.deepseek_v4`. Each
operator gets its own file with `triton(...)` / `flash(...)` entry points;
vendored triton kernels live under `kernels/`. Heavy imports (`triton`,
`torch_npu`) happen lazily inside each entry point so this package loads on a
CPU dev box.

## Tensor layout

All entry points accept and return tensors in HF-native **BHSD** =
`[batch, num_heads, seq, head_dim]`, matching what HF's eager / SDPA /
flash-attention paths use. Some triton kernels ported from MindSpeed-LLM
operate in Megatron's **SBHD** = `[seq, batch, num_heads, head_dim]`
internally — those entry points permute at the boundary so callers never
see the Megatron layout.

## NPU kernel ↔ HF / alloy module

The MindSpeed-LLM training stack uses different names than HF transformers
and alloy for the same algorithms. This table is the cross-reference:

| binder operator                  | MindSpeed-LLM source                             | HF / alloy module                                   | status      |
| -------------------------------- | ------------------------------------------------ | --------------------------------------------------- | ----------- |
| `sparse_flash_attention.triton`  | `g2_attention_kernel.SparseFlashAttentionTriton` | `dsv4_csa.attention` torch impl in alloy CSA layers | 🟡 scaffold |
| `compressed_attention.triton`    | `g2_attention_kernel.G2CoreAttention` torch path | `_eager_attention_with_sinks` in HCA/sliding layers | 🟡 planned  |
| `lightning_indexer.flash`        | `mindspeed.ops.npu_sparse_lightning_indexer_*`   | `DeepseekV4Indexer.forward`                         | 🟡 planned  |
| `compressor.triton`              | `compressor.Compressor`                          | `DeepseekV4{HCA,CSA}Compressor`                     | 🟡 planned  |
| `rmsnorm_no_weight.triton`       | `rmsnorm_without_weight_triton_kernel`           | `DeepseekV4UnweightedRMSNorm` (q_b_norm)            | 🟢 future   |
| `layernorm_gated.triton`         | `ops/triton/layernorm_gated`                     | gated RMSNorm in attention output                   | 🟢 future   |
| `mhc.triton`, `sinkhorn.triton`  | `mhc/mhc_triton`, `mhc/sinkhorn_triton_kernel`   | `use_mhc=True` HyperConnection (alloy spec pending) | 🟢 future   |

### Per-architecture notes

* **CSA layers** (`compress_ratio=4`, alloy `dsv4_csa_attention`): SDPA gets
  `topk_idxs` from the Lightning Indexer; the SFA kernel only does Q · K^T on
  the gated subset.
* **HCA layers** (`compress_ratio=128`, alloy `dsv4_hca_attention`): SDPA
  attends to sliding cache + heavily-compressed entries without indexer
  gating; uses the simpler core-attention kernel.
* **Sliding-only layers** (`compress_ratio=0`, alloy `dsv4_sliding_attention`):
  same core-attention kernel as HCA but without any compressed-KV
  concatenation; covered by the planned `compressed_attention` entry point.
* **MoE**, **embedding / lm_head / norms**: not yet wired here; precision /
  perf wins are smaller and the torch path is adequate.

## Wiring

End-users don't register anything themselves — the
`alloy.integrations.hf_npu_binder` bridge does it on import:

```python
import alloy.integrations.hf_npu_binder as bridge   # auto-registers
bridge.activate(model, prefer="auto")               # binder picks best per operator
```

For DSV4 CSA specifically, `bridge.activate(..., "auto")` currently sets
`config._dsv4_csa_implementation = "torch"` because the triton kernel
port from MindSpeed-LLM is still pending. Once
`sparse_flash_attention.triton` is implemented, bumping the entry in
`hf_npu_binder.DEFAULTS` from `{"auto": "torch", ...}` to
`{"auto": "triton", ...}` is the only change needed to flip every alloy
CSA layer to the NPU kernel — no alloy code edits required (that's the
point of the indirection).

Per-module manual override always works:

```python
config._dsv4_csa_implementation = "torch"   # or "triton" once the kernel ships
```

`hf_npu_binder.shared` carries cross-family primitives (e.g. `gmm`); inspect
`__init__.py` for the latest list.

## Validation

Each shipped operator has a paired test in
[`hf-npu-binder/tests/`](../../tests/) that asserts bit-exact equivalence
against the torch reference under fp32 and noise-floor equivalence under bf16
on real NPU.
