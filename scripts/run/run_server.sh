#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

CONFIG="${1:-configs/experiments/rq1_full.yaml}"
ACTION="${2:-run}"
if [[ "$ACTION" != "run" && "$ACTION" != "index" && "$ACTION" != "qa" ]]; then
  echo "Usage: bash scripts/run/run_server.sh [config] [run|index|qa]" >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES before running this script." >&2
  exit 1
fi
if [[ -z "${VLLM_BIN:-}" || ! -x "$VLLM_BIN" ]]; then
  echo "Set VLLM_BIN to an executable from the dedicated vLLM environment." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES VLLM_BIN

# Use the CUDA libraries packaged with the pinned PyTorch/vLLM environment.
unset LD_LIBRARY_PATH
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

python - <<'PY'
import sys
from importlib.metadata import version
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Use Python 3.11; found {sys.version.split()[0]}")
if version("graphrag") != "3.1.2":
    raise SystemExit(f"Expected graphrag 3.1.2, found {version('graphrag')}")
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
    raise SystemExit(f"Expected {expected}, found {actual}")
PY

python -m rq1.server validate-data --config "$CONFIG"
if [[ "$ACTION" == "qa" ]]; then
  python -m rq1.server validate-graphs --config "$CONFIG"
fi

OUTPUT_DIR="$(python -c 'import sys; from rq1.config import load_settings; print(load_settings(sys.argv[1]).output)' "$CONFIG")"
MODEL_LOG_DIR="$OUTPUT_DIR/model_logs"
mkdir -p "$MODEL_LOG_DIR"
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to stop model process groups cleanly." >&2
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
    kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  trap - EXIT INT TERM
  stop_group "$EMBEDDING_PID"
  stop_group "$CHAT_PID"
}
trap cleanup EXIT INT TERM

CHAT_GPU_MEMORY_UTILIZATION="${CHAT_GPU_MEMORY_UTILIZATION:-0.30}"
EMBEDDING_GPU_MEMORY_UTILIZATION="${EMBEDDING_GPU_MEMORY_UTILIZATION:-0.06}"
setsid python -m rq1.server serve-chat --config "$CONFIG" --max-model-len 8192 \
  --gpu-memory-utilization "$CHAT_GPU_MEMORY_UTILIZATION" >"$MODEL_LOG_DIR/chat.log" 2>&1 &
CHAT_PID=$!
setsid python -m rq1.server serve-embedding --config "$CONFIG" --max-model-len 8192 \
  --gpu-memory-utilization "$EMBEDDING_GPU_MEMORY_UTILIZATION" >"$MODEL_LOG_DIR/embedding.log" 2>&1 &
EMBEDDING_PID=$!

python -m rq1.server "$ACTION" --config "$CONFIG" --wait-seconds 1800
