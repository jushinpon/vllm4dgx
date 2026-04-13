Here is a comprehensive `README.md` draft for **vllm4dgx**.

````markdown
# vllm4dgx

A practical toolkit for installing, validating, snapshotting, comparing, and restoring a **working vLLM environment on NVIDIA DGX Spark / GB10**.

This repository contains four scripts that were built around a known-good installation flow:

- `install.sh` — one-shot installer for a working vLLM stack
- `smoke_test_vllm_model.sh` — post-install smoke test and lightweight serving helper
- `vllm_workable_info.pl` — snapshot the current working environment and compare future installs
- `restore_vllm_from_snapshot.pl` — repair a drifted/broken installation back toward the known-good state

The scripts assume a default installation path of:

```bash
/local_opt/vllm-install
````

and a default snapshot/info path of:

```bash
/local_opt/workable_llm_info
```

The installer pins:

* **vLLM commit:** `66a168a197ba214a5b70a74fa2e713c9eeb3251a`
* **Triton commit:** `4caa0328bf8df64896dd5f6fb9df41b0eb2e750a`
* **Python:** `3.12`

---

## Why this repository exists

On DGX Spark / GB10, upstream `vllm` installation can drift into a broken state because of:

* Triton version mismatch
* FlashInfer packaging/build issues
* vLLM CMake target wiring issues for Blackwell/SM100
* editable-install import path issues
* dependency drift after runtime package installation

This repository captures a **known workable configuration** and automates the fixes that were needed to make it run reliably. In particular, the installer patches `CMakeLists.txt` to add Blackwell-related architecture support and directly attaches:

```cmake
target_sources(_C PRIVATE
  csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu
)
```

to the `_C` extension target, which is critical for making `cutlass_moe_mm_sm100` resolve correctly in `vllm/_C.abi3.so`.

---

## Repository contents

### 1. `install.sh`

A one-shot installer for the full working environment. It:

* creates a Python 3.12 virtual environment with `uv`
* installs PyTorch `cu130`
* removes Torch’s Triton wheel
* builds **pinned Triton 3.5.0** from the tested commit
* installs FlashInfer from local source
* clones `vllm` at the tested commit
* patches `CMakeLists.txt` for SM100/SM120-related support
* patches `_C` target sources for `grouped_mm_c3x_sm100.cu`
* installs selected runtime dependencies
* restores pinned Triton after dependency drift
* rebuilds `vllm`
* verifies:

  * CUDA is available
  * Triton is pinned to `3.5.0`
  * `cutlass_moe_mm_sm100` is defined
  * `python -m vllm.entrypoints.openai.api_server --help` works

### 2. `smoke_test_vllm_model.sh`

A smoke test and serving helper that:

* activates the installed `.vllm` environment
* switches into the `vllm` source tree
* exports `PYTHONPATH` correctly
* verifies the native extension symbol
* optionally stops existing vLLM processes
* starts the API server
* waits for `/v1/models`
* optionally tests `/v1/chat/completions`
* can keep the server running after success

It also supports low-memory tuning through:

* `--gpu-memory-util`
* `--max-model-len`
* `--max-num-seqs`
* `--max-num-batched-tokens`
* `--enforce-eager`
* `--stop-existing`
* `--keep-server`

### 3. `vllm_workable_info.pl`

A snapshot and comparison tool. It can:

* save the current working environment under `/local_opt/workable_llm_info`
* record:

  * `pip freeze`
  * environment summary
  * `api_server --help`
  * symbol check
  * `CMakeLists.txt`
  * copies of `install.sh` and `smoke_test_vllm_model.sh`
* compare a future installation against the saved working snapshot

### 4. `restore_vllm_from_snapshot.pl`

A repair tool that uses the saved snapshot defaults and tries to restore a drifted installation by:

* forcing the working Python package pins
* restoring the CMake patch if missing
* restoring pinned Triton
* rebuilding `vllm`
* re-running final verification checks

---

## Default known-good environment

This repository is built around the following working package state:

* `torch == 2.11.0+cu130`
* `triton == 3.5.0+git4caa0328`
* `transformers == 4.56.0`
* `tokenizers == 0.22.2`
* `numpy == 2.2.6`
* `flashinfer-python == 0.6.7` (local-source workflow used in practice)
* `vllm` installed editable from the pinned commit

---

## Requirements

This repository assumes:

* NVIDIA DGX Spark / GB10
* CUDA 13.0 environment
* `git`
* `python3`
* `curl`
* `nvidia-smi`
* `nvcc`
* Linux shell environment with permission to write under `/local_opt`

If `uv` is not installed, `install.sh` will install it automatically. 

---

## Quick start

### 1. Fresh install

```bash
deactivate 2>/dev/null || true
rm -rf /local_opt/vllm-install
bash install.sh --install-dir /local_opt/vllm-install |& tee /home/install.log
```

Expected successful end state:

* final verification prints the correct `vllm` file path
* `cutlass_moe_mm_sm100` appears with `T`
* `api_server --help works`
* `Environment verification passed` 

### 2. Activate the environment

```bash
source /local_opt/vllm-install/.vllm/bin/activate
```

### 3. Run a smoke test

Basic:

```bash
bash smoke_test_vllm_model.sh \
  --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct
```

Low-memory mode:

```bash
bash smoke_test_vllm_model.sh \
  --stop-existing \
  --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
  --gpu-memory-util 0.15 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256 \
  --enforce-eager \
  --max-wait 600 \
  --test-chat \
  --keep-server
```

Keep server running after test:

```bash
bash smoke_test_vllm_model.sh \
  --stop-existing \
  --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
  --test-chat \
  --keep-server
```

These usage patterns are built into the script help text as examples. 

---

## Starting the API server manually

Once installation succeeds, a typical manual launch looks like:

```bash
source /local_opt/vllm-install/.vllm/bin/activate
cd /local_opt/vllm-install/vllm
export PYTHONPATH=/local_opt/vllm-install/vllm:${PYTHONPATH:-}

python -m vllm.entrypoints.openai.api_server \
  --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.85
```

Then test:

```bash
curl http://127.0.0.1:8000/v1/models
```

And a tiny chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Reply with exactly the word OK."}
    ],
    "max_tokens": 8,
    "temperature": 0
  }'
```

The smoke test script automates this flow.

---

## Saving the current working state

After you have a working installation, save a snapshot:

```bash
perl vllm_workable_info.pl snapshot
```

This stores the working information under:

```bash
/local_opt/workable_llm_info
```

with metadata, freeze output, environment summary, help output, symbol check, and copies of important files.

---

## Comparing a future installation

If you reinstall later into another path, compare it against the known-good snapshot:

```bash
perl vllm_workable_info.pl compare --target /path/
```

This checks:

* venv existence
* `_C.abi3.so` existence
* Triton version
* Transformers version
* Tokenizers version
* NumPy version
* CUDA availability
* `cutlass_moe_mm_sm100` symbol definition
* `api_server --help` health

---

## Restoring a broken installation

If a future installation drifts or breaks, run:

```bash
perl restore_vllm_from_snapshot.pl
```

Useful dry run:

```bash
perl restore_vllm_from_snapshot.pl --dry-run
```

What it repairs:

* package pins:

  * `numpy==2.2.6`
  * `transformers==4.56.0`
  * `tokenizers==0.22.2`
* missing CMake patch for `_C`
* pinned Triton reinstall
* `vllm` rebuild
* final verification checks

---

## Memory tuning notes

For lower GPU memory usage, use the smoke test with:

* lower `--gpu-memory-util`
* lower `--max-model-len`
* lower `--max-num-seqs`
* lower `--max-num-batched-tokens`
* `--enforce-eager`

Example:

```bash
bash smoke_test_vllm_model.sh \
  --stop-existing \
  --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
  --gpu-memory-util 0.10 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 128 \
  --enforce-eager \
  --max-wait 600 \
  --test-chat \
  --keep-server
```

Note that a 7B BF16 model still requires substantial GPU memory for model weights alone, so these options reduce KV cache and warmup overhead, not the base weight footprint. The smoke-test script was explicitly extended for this purpose.

---

## Troubleshooting

### 1. `api_server --help` fails

First ensure you are inside the correct source tree and `PYTHONPATH` is set:

```bash
source /local_opt/vllm-install/.vllm/bin/activate
cd /local_opt/vllm-install/vllm
export PYTHONPATH=/local_opt/vllm-install/vllm:${PYTHONPATH:-}
python -m vllm.entrypoints.openai.api_server --help
```

This matters because editable installs can otherwise import `vllm` incorrectly as a namespace package. The installer’s final verification was updated to handle exactly this case. 

### 2. Existing server prevents new settings from applying

Use:

```bash
bash smoke_test_vllm_model.sh --stop-existing ...
```

This kills any previous API server / EngineCore processes before launching with the new settings. 

### 3. Triton drift after installing more packages

Run:

```bash
perl restore_vllm_from_snapshot.pl
```

or at least compare:

```bash
perl vllm_workable_info.pl compare --target /local_opt/vllm-install
```

This repository is designed around restoring the tested Triton state.

### 4. Symbol unresolved in `_C.abi3.so`

Check:

```bash
nm -D /local_opt/vllm-install/vllm/vllm/_C.abi3.so | c++filt | grep cutlass_moe_mm_sm100
```

You want:

```text
T cutlass_moe_mm_sm100(...)
```

not `U`. The installer and restore script both enforce the relevant CMake patch.

---

## Suggested repository layout

```text
vllm4dgx/
├── install.sh
├── smoke_test_vllm_model.sh
├── vllm_workable_info.pl
├── restore_vllm_from_snapshot.pl
└── README.md
```

---

## Notes

This repository is intentionally opinionated and tuned for a **specific known-good DGX Spark / GB10 installation path**. It is not trying to be a generic upstream `vllm` installer. Its value is in preserving and reproducing a setup that was actually made to work end-to-end, including installation, verification, serving, snapshotting, comparison, and restore.

```

A good next step is to save this as `README.md` in the new repo and add your tested example model path near the top so users can copy-paste the smoke-test command directly.
```
