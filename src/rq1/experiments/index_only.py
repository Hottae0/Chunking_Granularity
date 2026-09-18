"""Build the configured extraction graphs without retrieval or QA."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from rq1.config import load_settings
from rq1.datasets import load_dataset
from rq1.backends.ms_graphrag.indexer import build_graph


def run(config: Path) -> Path:
    settings = load_settings(config)
    if settings.backend != "ms_graphrag":
        raise ValueError("Index-only execution requires backend: ms_graphrag")
    documents, questions = load_dataset(settings)
    os.environ["GRAPHRAG_API_KEY"] = settings.api_key
    os.environ["GRAPHRAG_EMBEDDING_API_KEY"] = settings.embedding_api_key
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Index only: sizes=%s, documents=%d; %d questions will not be executed",
                settings.sizes, len(documents), sum(map(len, questions.values())))
    graphs = [build_graph(settings, documents, size, logger) for size in settings.sizes]
    settings.output.mkdir(parents=True, exist_ok=True)
    report = settings.output / "index_complete.json"
    report.write_text(json.dumps({
        "config": str(Path(config).resolve()),
        "extraction_sizes": list(settings.sizes),
        "documents": [doc.name for doc in documents],
        "graphs": [str(graph.resolve()) for graph in graphs],
        "note": "Pipeline completion does not guarantee error-free extraction; inspect indexing-engine.log.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
