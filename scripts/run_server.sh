#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
python -m rq1.server run --config "${1:-configs/server.yaml}"
