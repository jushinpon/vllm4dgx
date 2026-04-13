#!/bin/bash
################################################################################
# Final one-shot vLLM installer for NVIDIA DGX Spark / GB10
#
# Usage:
#   deactivate 2>/dev/null || true
#   rm -rf /local_opt/vllm-install
#   bash install.sh --install-dir /local_opt/vllm-install |& tee /home/install.log
################################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INSTALL_DIR="${PWD}/vllm-install"
VLLM_VERSION="66a168a197ba214a5b70a74fa2e713c9eeb3251a"
TRITON_VERSION="4caa0328bf8df64896dd5f6fb9df41b0eb2e750a"
PYTHON_VERSION="3.12"

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

print_header() {
  echo
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}$1${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo
}

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --install-dir DIR      Installation directory
  --vllm-version HASH    vLLM git commit
  --python-version VER   Python version
  --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --vllm-version)
      VLLM_VERSION="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

activate_env() {
  source "${INSTALL_DIR}/.vllm/bin/activate"
}

ensure_venv_pip() {
  activate_env
  if ! python -m pip --version >/dev/null 2>&1; then
    log_warning "pip missing in venv, bootstrapping with ensurepip"
    python -m ensurepip --upgrade
  fi
  python -m pip --version >/dev/null
}

assert_triton_pinned() {
  activate_env
  python - <<'PY'
import inspect
import triton
print("Triton version:", triton.__version__)
print("Triton path:", inspect.getfile(triton))
assert triton.__version__.startswith("3.5.0"), triton.__version__
PY
}

clean_vllm_build_tree() {
  cd "${INSTALL_DIR}/vllm"
  rm -rf build dist .pytest_cache .setuptools-cmake-build vllm.egg-info
  find . -name '*.so' -delete || true
  find . -name '*.o' -delete || true
  find . -name '__pycache__' -type d -prune -exec rm -rf {} + || true
}

print_header "Pre-flight checks"

for cmd in git python3 curl nvidia-smi; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log_error "Missing required command: $cmd"
    exit 1
  fi
done

if ! command -v nvcc >/dev/null 2>&1; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  else
    log_error "nvcc not found"
    exit 1
  fi
fi

if command -v uv >/dev/null 2>&1; then
  log_info "uv already installed: $(uv --version)"
else
  print_header "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)

log_info "Install directory : $INSTALL_DIR"
log_info "vLLM commit       : $VLLM_VERSION"
log_info "Triton commit     : $TRITON_VERSION"
log_info "Python version    : $PYTHON_VERSION"
log_info "Detected GPU      : $GPU_NAME"
log_info "CUDA version      : $CUDA_VERSION"

print_header "Creating virtual environment"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
uv venv .vllm --python "$PYTHON_VERSION" --seed
activate_env
ensure_venv_pip
python -V
python -m pip install --upgrade pip

print_header "Installing PyTorch"

uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()
PY

print_header "Replacing Torch Triton with pinned Triton"

python -m pip uninstall -y triton >/dev/null 2>&1 || true
uv pip uninstall triton >/dev/null 2>&1 || true

rm -rf "${INSTALL_DIR}/triton"
cd "$INSTALL_DIR"
git clone https://github.com/triton-lang/triton.git
cd "${INSTALL_DIR}/triton"
git checkout "$TRITON_VERSION"
git submodule update --init --recursive

python -m pip install \
  pip \
  cmake \
  ninja \
  pybind11 \
  wheel \
  packaging \
  "setuptools>=77.0.3,<80" \
  setuptools-scm

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CMAKE_BUILD_PARALLEL_LEVEL
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

python -m pip install --no-build-isolation -v . \
  2>&1 | tee "${INSTALL_DIR}/triton-build.log"

assert_triton_pinned

print_header "Installing base dependencies"

cd "$INSTALL_DIR"
python -m pip install \
  "setuptools>=77.0.3,<80" \
  setuptools-scm \
  wheel \
  packaging \
  ninja \
  pybind11 \
  nvidia-ml-py \
  tqdm \
  click \
  cuda-tile \
  einops \
  "nvidia-cudnn-frontend>=1.13.0" \
  "nvidia-cutlass-dsl>=4.4.2" \
  requests \
  tabulate \
  "apache-tvm-ffi>=0.1.6,<0.2,!=0.1.8,!=0.1.8.post0" \
  cloudpickle \
  msgspec \
  psutil \
  pyyaml \
  filelock \
  typing-extensions

assert_triton_pinned

print_header "Installing FlashInfer"

rm -rf "${INSTALL_DIR}/flashinfer"
git clone --recursive https://github.com/flashinfer-ai/flashinfer.git "${INSTALL_DIR}/flashinfer"
cd "${INSTALL_DIR}/flashinfer"
git submodule update --init --recursive || true

sed -i 's/^license = "Apache-2.0"$/license = {text = "Apache-2.0"}/' pyproject.toml || true
sed -i '/^license-files = /d' pyproject.toml || true

export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

python -m pip install --no-build-isolation --no-deps . \
  2>&1 | tee "${INSTALL_DIR}/flashinfer-build.log"

python - <<'PY'
import inspect
import flashinfer
print("FlashInfer path:", inspect.getfile(flashinfer))
PY

assert_triton_pinned

print_header "Cloning and patching vLLM"

rm -rf "${INSTALL_DIR}/vllm"
cd "$INSTALL_DIR"
git clone --recursive https://github.com/vllm-project/vllm.git
cd "${INSTALL_DIR}/vllm"
git checkout "$VLLM_VERSION"
git submodule update --init --recursive

sed -i 's/^license = "Apache-2.0"$/license = {text = "Apache-2.0"}/' pyproject.toml || true
sed -i '/^license-files = /d' pyproject.toml || true

if [[ -f use_existing_torch.py ]]; then
  python3 use_existing_torch.py
fi

python - <<'PY'
from pathlib import Path

p = Path("CMakeLists.txt")
text = p.read_text()

text = text.replace(
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")',
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")'
)

text = text.replace(
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a" "${CUDA_ARCHS}")',
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")'
)

anchor = 'target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)\n'
insert = anchor + '''
target_sources(_C PRIVATE
  csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu
)
'''

if 'target_sources(_C PRIVATE\n  csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu\n)' not in text:
    text = text.replace(anchor, insert, 1)

p.write_text(text)
print("Patched CMakeLists.txt")
PY

print_header "Initial vLLM build"

clean_vllm_build_tree
cd "${INSTALL_DIR}/vllm"

export TORCH_CUDA_ARCH_LIST=12.1a
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

python -m pip install --no-build-isolation --no-deps -e . \
  2>&1 | tee "${INSTALL_DIR}/vllm-build-initial.log"

print_header "Installing selected vLLM runtime dependencies"

python -m pip install \
  "numpy==2.2.6" \
  "transformers==4.56.0" \
  "tokenizers==0.22.2" \
  aiohttp \
  anthropic==0.71.0 \
  blake3 \
  cachetools \
  cbor2 \
  depyf==0.20.0 \
  diskcache==5.6.3 \
  "fastapi[standard]>=0.115.0" \
  "gguf>=0.13.0" \
  lark==1.2.2 \
  "llguidance>=0.7.11,<0.8.0" \
  lm-format-enforcer==0.11.3 \
  "mistral-common[audio,image]>=1.8.5" \
  numba==0.61.2 \
  "openai>=1.99.1" \
  "openai-harmony>=0.0.3" \
  "opencv-python-headless>=4.11.0" \
  outlines-core==0.2.11 \
  partial-json-parser \
  "prometheus-client>=0.18.0" \
  "prometheus-fastapi-instrumentator>=7.0.0" \
  protobuf \
  py-cpuinfo \
  pybase64 \
  "pydantic>=2.12.0" \
  python-json-logger \
  "pyzmq>=25.0.0" \
  regex \
  scipy \
  sentencepiece \
  setproctitle \
  six \
  "tiktoken>=0.6.0" \
  watchfiles \
  xgrammar==0.1.25 \
  2>&1 | tee "${INSTALL_DIR}/vllm-runtime-deps.log" || true

print_header "Restoring pinned Triton after dependency drift"

python -m pip uninstall -y triton >/dev/null 2>&1 || true
cd "${INSTALL_DIR}/triton"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

python -m pip install --no-build-isolation -v . \
  2>&1 | tee "${INSTALL_DIR}/triton-rebuild-final.log"

assert_triton_pinned

print_header "Final vLLM rebuild"

clean_vllm_build_tree
cd "${INSTALL_DIR}/vllm"

export TORCH_CUDA_ARCH_LIST=12.1a
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

python -m pip install --no-build-isolation --no-deps -e . \
  2>&1 | tee "${INSTALL_DIR}/vllm-build-final.log"

print_header "Final verification"

cd "${INSTALL_DIR}/vllm"
export PYTHONPATH="${INSTALL_DIR}/vllm:${PYTHONPATH:-}"

python - <<'PY'
import inspect
import torch
import triton
import flashinfer
import vllm

print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("triton path:", inspect.getfile(triton))
print("flashinfer path:", inspect.getfile(flashinfer))
print("vllm file:", getattr(vllm, "__file__", None))
print("vllm path:", list(getattr(vllm, "__path__", [])))
print("cuda available:", torch.cuda.is_available())

assert torch.cuda.is_available()
assert triton.__version__.startswith("3.5.0")
assert getattr(vllm, "__file__", None) is not None, "vllm imported as namespace package"
PY

nm -D "${INSTALL_DIR}/vllm/vllm/_C.abi3.so" | c++filt | grep -i cutlass_moe_mm_sm100

python -m vllm.entrypoints.openai.api_server --help >/tmp/vllm_help.out 2>/tmp/vllm_help.err || {
  echo "----- STDERR -----"
  cat /tmp/vllm_help.err
  exit 1
}

log_success "api_server --help works"
log_success "Environment verification passed"

print_header "Installation Complete"
log_success "Environment ready at: $INSTALL_DIR"
echo "Activate with:"
echo "  source $INSTALL_DIR/.vllm/bin/activate"
echo "Deactivate with:"
echo "  deactivate"