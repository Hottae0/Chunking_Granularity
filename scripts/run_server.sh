#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CONFIG="${1:-configs/server.yaml}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES in the terminal before running this script." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES

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

python -m rq1.server run --config "$CONFIG" --wait-seconds 1800
