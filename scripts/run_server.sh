#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CONFIG="${1:-configs/server.yaml}"
ACTION="${2:-run}"
if [[ "$ACTION" != "run" && "$ACTION" != "index" ]]; then
  echo "Usage: bash scripts/run_server.sh [config] [run|index]" >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES in the terminal before running this script." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES

if [[ -z "${VLLM_BIN:-}" ]]; then
  echo "Set VLLM_BIN to the vLLM executable from the dedicated vLLM environment." >&2
  echo "Example: export VLLM_BIN=/path/to/vllm-env/bin/vllm" >&2
  exit 1
fi
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "VLLM_BIN is not executable: $VLLM_BIN" >&2
  exit 1
fi
export VLLM_BIN

# The host CUDA toolkit is 12.1, while the pinned vLLM environment uses the
# self-contained PyTorch cu128 wheels. Let their RPATH select the matching
# CUDA 12.8 libraries instead of preloading /usr/local/cuda/lib64.
unset LD_LIBRARY_PATH

# Never inherit another user's shared Hugging Face cache. MODEL_CACHE_DIR can
# still be set explicitly when a writable shared cache is intended.
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"
mkdir -p "$MODEL_CACHE_DIR/hub"
if [[ ! -w "$MODEL_CACHE_DIR/hub" ]]; then
  echo "Model cache is not writable: $MODEL_CACHE_DIR/hub" >&2
  exit 1
fi
export HF_HOME="$MODEL_CACHE_DIR"
export HF_HUB_CACHE="$MODEL_CACHE_DIR/hub"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE_DIR/hub"
unset TRANSFORMERS_CACHE

echo "Checking Python and vLLM environments before model startup"
python - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Use the Python 3.11 experiment environment; found {sys.version.split()[0]}")
try:
    graphrag = version("graphrag")
except PackageNotFoundError as exc:
    raise SystemExit("GraphRAG is missing; activate the experiment environment first") from exc
if graphrag != "3.1.2":
    raise SystemExit(f"Expected graphrag 3.1.2, found {graphrag}")
print(f"Experiment environment: Python {sys.version.split()[0]}, graphrag {graphrag}")
PY
python -m pip check
VLLM_PYTHON="$(sed -n '1s/^#!//p' "$VLLM_BIN")"
if [[ -z "$VLLM_PYTHON" || ! -x "$VLLM_PYTHON" ]]; then
  echo "Cannot resolve the Python interpreter from $VLLM_BIN" >&2
  exit 1
fi
"$VLLM_PYTHON" - <<'PY'
from importlib.metadata import version

expected = {"vllm": "0.10.1", "torch": "2.7.1+cu128"}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Expected dedicated vLLM environment {expected}, found {actual}")
print("vLLM packages: " + ", ".join(f"{name} {value}" for name, value in actual.items()))
PY
if ! VLLM_VERSION="$("$VLLM_BIN" --version 2>&1)"; then
  echo "vLLM compatibility check failed for $VLLM_BIN" >&2
  echo "$VLLM_VERSION" >&2
  exit 1
fi
echo "Model environment: $VLLM_VERSION ($VLLM_BIN)"
echo "Model cache: $MODEL_CACHE_DIR"

CHAT_GPU_MEMORY_UTILIZATION="${CHAT_GPU_MEMORY_UTILIZATION:-0.30}"
EMBEDDING_GPU_MEMORY_UTILIZATION="${EMBEDDING_GPU_MEMORY_UTILIZATION:-0.06}"

OUTPUT_DIR="$(python -c 'import sys; from rq1.config import load_settings; print(load_settings(sys.argv[1]).output)' "$CONFIG")"
MODEL_LOG_DIR="$OUTPUT_DIR/model_logs"
mkdir -p "$MODEL_LOG_DIR"

echo "Validating benchmark data before model startup"
python -m rq1.server validate-data --config "$CONFIG"

if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required so model workers can be stopped as one process group." >&2
  exit 1
fi

CHAT_PID=""
EMBEDDING_PID=""

stop_group() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  stop_group "$EMBEDDING_PID"
  stop_group "$CHAT_PID"
  echo "Model servers stopped; their GPU memory has been released."
}
trap cleanup EXIT INT TERM

echo "Starting chat and embedding models on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
setsid python -m rq1.server serve-chat --config "$CONFIG" --max-model-len 8192 --gpu-memory-utilization "$CHAT_GPU_MEMORY_UTILIZATION" \
  >"$MODEL_LOG_DIR/chat.log" 2>&1 &
CHAT_PID=$!
setsid python -m rq1.server serve-embedding --config "$CONFIG" --max-model-len 8192 --gpu-memory-utilization "$EMBEDDING_GPU_MEMORY_UTILIZATION" \
  >"$MODEL_LOG_DIR/embedding.log" 2>&1 &
EMBEDDING_PID=$!

python -m rq1.server "$ACTION" --config "$CONFIG" --wait-seconds 1800
