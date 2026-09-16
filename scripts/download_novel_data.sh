#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import json
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install it with: "
        "python -m pip install huggingface_hub"
    ) from exc

repo_id = "GraphRAG-Bench/GraphRAG-Bench"
root = Path("data/GraphRAG-Bench")
files = (
    "Datasets/Corpus/novel.json",
    "Datasets/Questions/novel_questions.json",
)

for filename in files:
    print(f"Downloading {filename}")
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=root,
    )

corpus_path = root / files[0]
questions_path = root / files[1]
corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
questions = json.loads(questions_path.read_text(encoding="utf-8"))

if not corpus or not questions:
    raise SystemExit("Downloaded dataset is empty")
if not all({"corpus_name", "context"} <= set(row) for row in corpus):
    raise SystemExit("Unexpected Novel corpus schema")
if not all({"id", "source", "question", "answer"} <= set(row) for row in questions):
    raise SystemExit("Unexpected Novel question schema")

print(f"Ready: {corpus_path.resolve()} ({len(corpus)} novels)")
print(f"Ready: {questions_path.resolve()} ({len(questions)} questions)")
PY
