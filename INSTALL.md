# Installation

Step-by-step install for an NPU dev box. The binder package imports fine on a
CPU box (no torch_npu / triton-ascend / CANN needed) — the heavy deps below
are only required if you actually want to run an NPU backend.

## Prerequisites (system-level)

- **CANN toolkit >= 9.0.0** ([download](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)).
  Older CANN still loads but the `aclnnSparseAttnSharedkv` host API
  dispatches to a slower fallback. Source `set_env.sh` so
  `$ASCEND_HOME_PATH` is set:

  ```bash
  source ~/Ascend/ascend-toolkit/set_env.sh
  echo $ASCEND_HOME_PATH   # must be non-empty
  ```

- **gcc + dev headers** — needed for the JIT C++ extension build. NPU dev
  boxes always have these via the CANN install.

## Pip install (from scratch)

```bash
# 1. python env
conda create -n alloy-npu python=3.10 -y
conda activate alloy-npu

# 2. torch + torch_npu (torch 2.7.x or 2.9.x both work; pick a matching pair)
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.7.1

# 3. triton-ascend — NOT on default PyPI, install from the Ascend osinfra index
pip install triton-ascend==3.2.1 \
    --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

# 4. transformers — DSV4 CSA fast-path needs PR #45892, which is on main but
#    not in a tagged release yet. Install from source:
git clone https://github.com/huggingface/transformers
cd transformers
git checkout a1b77cca4e        # #45892 merge commit, or any main HEAD past it
pip install -e .
cd ..

# 5. alloy
git clone https://github.com/Sawyer117/alloy
cd alloy && pip install -e . && cd ..

# 6. hf-npu-binder (use the dsv4-csa-ascendc branch until it merges to main)
git clone https://github.com/Sawyer117/hf-npu-binder
cd hf-npu-binder
git checkout dsv4-csa-ascendc
pip install -e .
cd ..
```

## Sanity check

```bash
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print('npu_ok=', torch.npu.is_available())"
python -c "import triton.language.extra.cann.extension as al; print('triton-ascend ok')"
python -c "import transformers; print(transformers.__version__)"   # expect 5.8.0.dev0
python -c "import alloy, hf_npu_binder; print('alloy + binder ok')"
```

## Smoke test (NPU only)

```bash
cd hf-npu-binder
PYTHONPATH=. python debug/test_ascendc_sparse_attn.py
```

Four stages: JIT compile -> SBND-direct call -> BHSD adapter -> backward.
First run takes ~10s on stage 1 (one-time C++ extension build, cached under
`~/.cache/torch_extensions/`); subsequent runs skip the compile.

## Troubleshooting

- `ASCEND_HOME_PATH is unset` from `AscendcOpBuilder.load()` — you forgot
  to `source set_env.sh` before launching python.
- `ImportError: cannot import name 'extension' from triton.language.extra.cann`
  — you installed upstream PyPI `triton` instead of `triton-ascend`. Uninstall
  and reinstall from the osinfra index.
- pip refuses to resolve `transformers>=5.8.0.dev0` — make sure step 4 ran
  (source install of transformers main); the `.dev0` floor only matches a
  source / nightly install.
