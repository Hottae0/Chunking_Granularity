# Validation — 2026-09-16

- Runtime: Python 3.12, installed Microsoft GraphRAG 3.1.2.
- Unit and synthetic integration tests: 8 tests, including full 64-cell run (128 synthetic question evaluations, 8 graph fixtures), resume, loader, cache invalidation, novel-cluster RQ1 stability and wrong-document evidence rank/budget.
- Actual local GraphRAG-Bench Novel files read successfully: 20 documents, 2,010 questions. Types: Fact Retrieval 971; Complex Reasoning 610; Contextual Summarize 362; Creative Generation 67.
- Source-qualified IDs prevent an observed cross-novel duplicate ID from overwriting results.
- No real model indexing, QA, paid calls or official judge were run. This is implementation verification, not evidence for the research hypothesis.
- GraphRAG 3.1.2 is pinned for reproducibility; this is not a claim that it is the latest upstream release.

## Allocated GPU server workflow

- 11 unit/integration tests pass after adding server support.
- Separate completion/embedding endpoints and credentials; embedding dimensions propagate to GraphRAG vector-store schemas.
- Tested model command construction, preservation of GPU visibility, readiness probes with test clients, dimension mismatch rejection, and absence of embedding API keys in cache records.
- Server CLI help, shell syntax, and diff whitespace checks pass.
- Actual vLLM startup, CUDA compatibility, GPU memory capacity and real model inference have not been tested on this local machine.
