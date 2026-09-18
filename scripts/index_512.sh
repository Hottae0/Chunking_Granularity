#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

exec bash scripts/run_server.sh configs/index_512.yaml index
