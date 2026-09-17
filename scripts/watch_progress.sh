#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/server_preliminary_3x3.yaml}"
INTERVAL="${2:-10}"

python scripts/progress.py --config "$CONFIG" --interval "$INTERVAL"
