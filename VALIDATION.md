# Validation — 2026-09-18

## Automated checks

- Runtime: Python 3.11.16, Microsoft GraphRAG 3.1.2.
- `python -m unittest discover -s tests -v`: 25 tests passed.
- `python -m compileall -q src scripts tests`: passed.
- `python -m pip check`: no broken requirements.
- `bash -n scripts/download/*.sh scripts/run/*.sh`: passed.
- `git diff --check`: passed.

Tests cover source-offset chunking/alignment/provenance, GraphRAG-Bench loading, backend
availability, index compatibility, run/index locks, 64-cell synthetic execution, resume,
cache invalidation, QA-only behavior, prior-cell validation/linking, evidence rank/budget,
bootstrap analysis, model endpoint preflight, and secret-free cache metadata.

## Local data and index validation

- GraphRAG-Bench Novel: 20 documents and 2,010 questions in the raw files.
- RQ1 selection: seed 42, 5 documents, all 513 selected questions.
- Central index path: `runs/indexes/ms_graphrag/graphrag_bench`.
- E128, E256, and E512 each pass input/model/dimension/chunk/table validation and have
  `graph_complete.json`.
- Experiment configs resolve to the new dataset/backend config files and central index path.
- The 3×3 QA preflight correctly blocks because the configured four required 2×2 cells are
  absent from local storage. This prevents an accidental repeat of 2,052 prior QA calls.

## Historical artifact note

Before this refactor, the local 2×2 directory retained logs, summaries, and an old completion
marker, but `cells/`, `per_query_results.csv`, and `config_summary.csv` were already absent.
The directory was moved to the new experiment path without deleting the remaining evidence.
The old marker is named `results_complete.stale.json`, and `migration_status.json` records that
there are currently zero reusable cells.

## Limits

- No paid/model QA calls or official judge evaluation were run during this refactor.
- HippoRAG, MuSiQue, 2Wiki, and QASPER are explicit scaffolds, not completed integrations.
- GraphRAG indexing logs contain recoverable request failures described in the README; pipeline
  completion alone does not prove that every individual extraction/report request succeeded.
