#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m rq1.experiments.run_grid --config configs/full.yaml
