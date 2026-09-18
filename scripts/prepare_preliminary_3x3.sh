#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/server_preliminary_3x3.yaml}"
SOURCE="${2:-runs/_graph_store/novel5_seed42_qwen25_bge_m3}"

python scripts/prepare_subset.py \
  --config "$CONFIG" \
  --source "$SOURCE"
