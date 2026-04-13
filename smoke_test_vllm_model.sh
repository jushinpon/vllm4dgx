#!/bin/bash
################################################################################
# Smoke test for the current DGX Spark / GB10 vLLM install
#
# Purpose:
#   - Verify the installed vLLM environment.
#   - Start a vLLM OpenAI-compatible API server.
#   - Test /v1/models.
#   - Optionally test /v1/chat/completions.
#
# Basic usage:
#   bash smoke_test_vllm_model.sh \
#     --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct
#
# Low-memory usage:
#   bash smoke_test_vllm_model.sh \
#     --stop-existing \
#     --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
#     --gpu-memory-util 0.15 \
#     --max-model-len 1024 \
#     --max-num-seqs 1 \
#     --max-num-batched-tokens 256 \
#     --enforce-eager \
#     --max-wait 600 \
#     --test-chat \
#     --keep-server
#
# Keep server running after test:
#   bash smoke_test_vllm_model.sh \
#     --stop-existing \
#     --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
#     --test-chat \
#     --keep-server
#
# Stop existing vLLM before applying new settings:
#   bash smoke_test_vllm_model.sh \
#     --stop-existing \
#     --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
#     --gpu-memory-util 0.15 \
#     --max-model-len 1024
################################################################################

set -u
set -o pipefail

INSTALL_DIR="/local_opt/vllm-install"
MODEL_PATH=""
HOST="127.0.0.1"
PORT="8000"
DTYPE="auto"
GPU_MEMORY_UTIL="0.30"
MAX_WAIT_SEC="600"
MAX_MODEL_LEN="4096"
MAX_NUM_SEQS="1"
MAX_NUM_BATCHED_TOKENS="1024"
TEST_CHAT=0
KEEP_SERVER=0
ENFORCE_EAGER=0
STOP_EXISTING=0
SERVER_PID=""
SERVER_LOG=""

show_help() {
  cat <<EOF
Usage:
  bash $0 --model MODEL_PATH [options]

Required:
  --model PATH_OR_HF_ID
      Model path or Hugging Face model ID.

Common options:
  --install-dir DIR
      vLLM install directory.
      Default: /local_opt/vllm-install

  --host HOST
      API server host.
      Default: 127.0.0.1

  --port PORT
      API server port.
      Default: 8000

  --test-chat
      Send a small /v1/chat/completions request after /v1/models works.

  --keep-server
      Keep the vLLM server running after the smoke test succeeds.

  --stop-existing
      Stop existing vLLM processes before starting a new server.
      Use this when changing memory settings.

Low-memory options:
  --gpu-memory-util FLOAT
      vLLM GPU memory utilization.
      Default: 0.30

  --max-model-len INT
      Maximum context length.
      Default: 4096

  --max-num-seqs INT
      Maximum concurrent sequences.
      Default: 1

  --max-num-batched-tokens INT
      Maximum batched tokens.
      Default: 1024

  --enforce-eager
      Disable CUDA graph capture to reduce memory overhead.

  --max-wait SEC
      Maximum time to wait for server startup.
      Default: 600

Examples:

  Basic test:
    bash $0 \\
      --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct

  Test and keep server running:
    bash $0 \\
      --stop-existing \\
      --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \\
      --test-chat \\
      --keep-server

  Low-memory test:
    bash $0 \\
      --stop-existing \\
      --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \\
      --gpu-memory-util 0.15 \\
      --max-model-len 1024 \\
      --max-num-seqs 1 \\
      --max-num-batched-tokens 256 \\
      --enforce-eager \\
      --max-wait 600 \\
      --test-chat \\
      --keep-server

  Very low-memory test:
    bash $0 \\
      --stop-existing \\
      --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \\
      --gpu-memory-util 0.10 \\
      --max-model-len 512 \\
      --max-num-seqs 1 \\
      --max-num-batched-tokens 128 \\
      --enforce-eager \\
      --max-wait 600 \\
      --test-chat \\
      --keep-server

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --gpu-memory-util) GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --max-wait) MAX_WAIT_SEC="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --test-chat) TEST_CHAT=1; shift ;;
    --keep-server) KEEP_SERVER=1; shift ;;
    --enforce-eager) ENFORCE_EAGER=1; shift ;;
    --stop-existing) STOP_EXISTING=1; shift ;;
    --help|-h) show_help; exit 0 ;;
    *) echo "[FAIL] Unknown option: $1"; show_help; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "[FAIL] --model is required"
  show_help
  exit 1
fi

cleanup() {
  if [[ "$KEEP_SERVER" -eq 1 ]]; then
    return 0
  fi

  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[INFO] Stopping vLLM server PID $SERVER_PID"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    sleep 3
    kill -9 "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

stop_existing_vllm() {
  echo "[INFO] Stopping existing vLLM processes first..."
  pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  sleep 3

  LEFT="$(ps -ef | grep -i vllm | grep -v grep || true)"
  if [[ -n "$LEFT" ]]; then
    echo "[WARN] Some vLLM-related processes remain:"
    echo "$LEFT"
  else
    echo "[PASS] Existing vLLM processes stopped"
  fi
}

if [[ "$STOP_EXISTING" -eq 1 ]]; then
  stop_existing_vllm
fi

source "$INSTALL_DIR/.vllm/bin/activate"
cd "$INSTALL_DIR/vllm"
export PYTHONPATH="$INSTALL_DIR/vllm:${PYTHONPATH:-}"

echo "========================================"
echo "vLLM Smoke Test"
echo "========================================"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Host: $HOST"
echo "[INFO] Port: $PORT"
echo "[INFO] GPU memory util: $GPU_MEMORY_UTIL"
echo "[INFO] Max model len: $MAX_MODEL_LEN"
echo "[INFO] Max num seqs: $MAX_NUM_SEQS"
echo "[INFO] Max batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "[INFO] Enforce eager: $ENFORCE_EAGER"
echo "[INFO] Keep server: $KEEP_SERVER"
echo "[INFO] Stop existing: $STOP_EXISTING"

python - <<'PY'
import torch, triton, flashinfer, vllm
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("cuda:", torch.cuda.is_available())
print("vllm:", getattr(vllm, "__file__", None))
assert torch.cuda.is_available()
assert triton.__version__.startswith("3.5.0")
assert getattr(vllm, "__file__", None) is not None
PY

nm -D "$INSTALL_DIR/vllm/vllm/_C.abi3.so" | c++filt | grep -i cutlass_moe_mm_sm100 | grep " T " >/dev/null || {
  echo "[FAIL] cutlass_moe_mm_sm100 is not defined"
  exit 1
}

if curl -sS "http://${HOST}:${PORT}/v1/models" >/tmp/vllm_models_resp.json 2>/dev/null; then
  echo "[PASS] Existing server is already responding"
else
  SERVER_LOG="$(mktemp /tmp/vllm_smoke_test.XXXXXX.log)"
  echo "[INFO] Server log: $SERVER_LOG"

  EXTRA_ARGS=()
  if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
    EXTRA_ARGS+=(--enforce-eager)
  fi

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    "${EXTRA_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &

  SERVER_PID=$!
  echo "[INFO] Started server PID: $SERVER_PID"

  START_TS=$(date +%s)

  while true; do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "[FAIL] Server exited early"
      tail -n 120 "$SERVER_LOG"
      exit 1
    fi

    if curl -sS "http://${HOST}:${PORT}/v1/models" >/tmp/vllm_models_resp.json 2>/dev/null; then
      echo "[PASS] /v1/models responded"
      break
    fi

    ELAPSED=$(($(date +%s) - START_TS))
    echo "[INFO] Waiting... ${ELAPSED}s"

    if [[ "$ELAPSED" -ge "$MAX_WAIT_SEC" ]]; then
      echo "[FAIL] Timeout after ${MAX_WAIT_SEC}s"
      tail -n 120 "$SERVER_LOG"
      exit 1
    fi

    sleep 5
  done
fi

cat /tmp/vllm_models_resp.json
echo

if [[ "$TEST_CHAT" -eq 1 ]]; then
  MODEL_ID="$(python - <<'PY'
import json
data = json.load(open("/tmp/vllm_models_resp.json"))
print(data["data"][0]["id"])
PY
)"

  cat >/tmp/vllm_chat_req.json <<EOF
{
  "model": "$MODEL_ID",
  "messages": [
    {"role": "user", "content": "Reply with exactly the word OK."}
  ],
  "max_tokens": 8,
  "temperature": 0
}
EOF

  curl -sS \
    -H "Content-Type: application/json" \
    -d @/tmp/vllm_chat_req.json \
    "http://${HOST}:${PORT}/v1/chat/completions" \
    >/tmp/vllm_chat_resp.json || {
      echo "[FAIL] Chat test failed"
      exit 1
    }

  cat /tmp/vllm_chat_resp.json
  echo
  echo "[PASS] Chat test completed"
else
  echo "[WARN] Skipping chat completion test. Use --test-chat to enable."
fi

echo "[PASS] Smoke test passed"

if [[ "$KEEP_SERVER" -eq 1 ]]; then
  echo "[INFO] Server kept running."
  echo "[INFO] Check with:"
  echo "curl http://${HOST}:${PORT}/v1/models"
fi