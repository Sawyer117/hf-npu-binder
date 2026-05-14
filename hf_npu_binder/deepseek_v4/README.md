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

| binder operator                  | MindSpeed-LLM source                             | HF / alloy module                                            | status      |
| -------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ | ----------- |
| `sparse_flash_attention.triton`  | `g2_attention_kernel.SparseFlashAttentionTriton` + Lightning-Indexer offset trick | `dsv4_csa.attention` in alloy CSA layers | ✅ shipped  |
| `sparse_flash_attention.ascendc` | `mindspeed.ops.npu_sparse_attn_shared_kv` (`aclnnSparseAttnSharedkv`) | same key on `dsv4_csa.attention`, fused single kernel | 🟡 opt-in (CANN ≥ 9.0.0 RC) |
| `compressed_attention.triton`    | same `SparseFlashAttentionTriton` kernel; topk built from sliding + all-compressed-positions (no indexer) | `dsv4_hca.attention` + `dsv4_sliding.attention` in alloy | ✅ shipped  |
| `lightning_indexer.flash`        | `mindspeed.ops.npu_sparse_lightning_indexer_*`   | `DeepseekV4Indexer.forward`                                  | 🟡 planned  |
| `compressor.triton`              | `compressor.Compressor`                          | `DeepseekV4{HCA,CSA}Compressor`                              | 🟡 planned  |
| `rmsnorm_no_weight.triton`       | `rmsnorm_without_weight_triton_kernel`           | `DeepseekV4UnweightedRMSNorm` (q_b_norm)                     | 🟢 future   |
| `layernorm_gated.triton`         | `ops/triton/layernorm_gated`                     | gated RMSNorm in attention output                            | 🟢 future   |
| `mhc.triton`, `sinkhorn.triton`  | `mhc/mhc_triton`, `mhc/sinkhorn_triton_kernel`   | `use_mhc=True` HyperConnection (alloy spec pending)          | 🟢 future   |

### Per-architecture notes

* **CSA layers** (`compress_ratio=4`, alloy `dsv4_csa_attention`): the SFA
  kernel attends over sliding-window indices ++ Lightning-Indexer picks.
  Backends: `sparse_flash_attention.triton` (auto), `.ascendc` (opt-in).
* **HCA layers** (`compress_ratio=128`, alloy `dsv4_hca_attention`): same
  kernel; topk is sliding-window indices ++ per-query *causal-compress*
  indices `[0, (p+1) // compress_rate_hca)` padded with `-1` sentinels to
  width `compressed_seq_len` — matches alloy's HCA `block_bias` over the
  compressed columns (and HF main `ac372e10f2`). Adapter takes
  `position_ids` via kwargs and reads `compress_rate` from
  `module.compressor`. Backend: `compressed_attention.triton`.
* **Sliding-only layers** (`compress_ratio=0`, alloy `dsv4_sliding_attention`):
  same kernel; topk is just the sliding window, no compressed range.
  Backend: `compressed_attention.triton`.
* **MoE**, **embedding / lm_head / norms**: not yet wired here; precision /
  perf wins are smaller and the torch path is adequate.

## CONFIG_MAP width constraint

The vendored SFA kernel autotunes against a fixed set of total topk widths
in `kernels/sparse_flash_attention_triton.py::CONFIG_MAP` —  `{128, 160, 640}`.
The adapter sums `sliding_window` with whatever compressed-range topk it
constructs and checks the result against this set; if it misses, you get a
clear `ValueError` pointing at the alloy config knobs to tweak.

Typical combinations that land:

  * Sliding-only with `sliding_window=128` → 128 ✓
  * CSA with `sliding_window=128 + index_topk=32` → 160 ✓
  * HCA at 4K with `sliding_window=128 + compressed=32` (4096/128) → 160 ✓
  * HCA at 64K with `sliding_window=128 + compressed=512` → 640 ✓
  * HCA at 8K..32K → misses (add a TilingBlockConfig entry to CONFIG_MAP or
    pad `sliding_window` so the sum hits a supported width).

## Wiring

End-users don't register anything themselves — the
`alloy.integrations.hf_npu_binder` bridge does it on import:

```python
import alloy.integrations.hf_npu_binder as bridge   # auto-registers
bridge.activate(model, prefer="auto")               # binder picks best per operator
```

`activate(..., "auto")` writes `_dsv4_{csa,hca,sliding}_implementation = "triton"`
on `model.config` (per `hf_npu_binder.DEFAULTS`), so every DSV4 attention
layer flips to the binder fast path without touching alloy code (that's the
point of the indirection).

Per-module manual override always works:

```python
config._dsv4_csa_implementation     = "ascendc"  # opt-in fused kernel
config._dsv4_hca_implementation     = "torch"    # bypass triton for one type
config._dsv4_sliding_implementation = "torch"
```

`hf_npu_binder.shared` carries cross-family primitives (e.g. `gmm`); inspect
`__init__.py` for the latest list.

## Validation

Each shipped operator has a paired test in
[`hf-npu-binder/tests/`](../../tests/) that asserts bit-exact equivalence
against the torch reference under fp32 and noise-floor equivalence under bf16
on real NPU.
