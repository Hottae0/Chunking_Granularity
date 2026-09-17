#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/server_preliminary_3x3.yaml}"
SOURCE="${2:-runs/server_novel5_allq_8x8}"

python -m rq1.experiments.prepare_subset \
  --config "$CONFIG" \
  --source "$SOURCE"
