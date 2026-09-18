#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
from rq1.config import load_settings
from rq1.experiments.run_grid import validate_qa_graphs

settings = load_settings("configs/qa_3x3.yaml")
documents, questions = validate_qa_graphs(settings)
count = sum(map(len, questions.values()))
if settings.sizes != (128, 256, 512) or len(documents) != 5 or count != 513:
    raise SystemExit("Expected sizes [128,256,512], 5 novels and all 513 questions")
if settings.max_questions_per_document is not None:
    raise SystemExit("Question caps are not allowed for this experiment")
print(f"QA only: 9 cells × {count} questions = {9 * count} results")
PY

exec bash scripts/run_server.sh configs/qa_3x3.yaml qa
