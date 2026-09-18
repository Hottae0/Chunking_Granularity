#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash scripts/run/run_server.sh configs/experiments/rq1_pilot.yaml run
