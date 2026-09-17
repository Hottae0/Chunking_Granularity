"""Prepare a smaller grid and reuse compatible graph work from a larger run."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rq1.config import load_settings
from rq1.data.graphrag_bench import load_novel
from rq1.experiments.cache import guard_run
from rq1.msgraphrag.indexer import REQUIRED_TABLES, _input_name


def _configured_models(raw: dict, section: str) -> set[str]:
    entries = raw.get(section) or {}
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid {section} in source graph settings")
    return {
        str(value["model"])
        for value in entries.values()
        if isinstance(value, dict) and value.get("model")
    }


def _validate_graph(source_graph: Path, extraction_size: int, documents) -> bool:
    """Validate data/model/chunk compatibility; return whether the graph is complete."""
    settings_path = source_graph / "settings.yaml"
    titles_path = source_graph / "input_titles.json"
    input_dir = source_graph / "input"
    for required in (settings_path, titles_path, input_dir):
        if not required.exists():
            raise ValueError(f"Cannot reuse {source_graph}: missing {required.name}")

    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    actual_size = int((raw.get("chunking") or {}).get("size", -1))
    if actual_size != extraction_size:
        raise ValueError(
            f"Cannot reuse {source_graph}: chunk size is {actual_size}, expected {extraction_size}"
        )

    expected_titles = {_input_name(doc.name): doc.name for doc in documents}
    actual_titles = json.loads(titles_path.read_text(encoding="utf-8"))
    if actual_titles != expected_titles:
        raise ValueError(f"Cannot reuse {source_graph}: selected documents differ")

    for doc in documents:
        input_path = input_dir / _input_name(doc.name)
        if not input_path.exists() or input_path.read_text(encoding="utf-8") != doc.text:
            raise ValueError(f"Cannot reuse {source_graph}: input text differs for {doc.name}")

    return (
        (source_graph / "graph_complete.json").exists()
        and all(
            (source_graph / "output" / f"{table}.parquet").exists()
            for table in REQUIRED_TABLES
        )
    )


def prepare(config: str | Path, source: str | Path, include_partial: bool = True) -> dict:
    settings = load_settings(config)
    documents, questions = load_novel(
        settings.corpus,
        settings.questions,
        settings.max_documents,
        settings.max_questions_per_document,
        settings.document_selection,
        settings.seed,
    )

    source = Path(source).resolve()
    target = settings.output.resolve()
    if source == target:
        raise ValueError("Source and target output directories must differ")
    if not source.exists():
        raise FileNotFoundError(f"Source run does not exist: {source}")

    target.mkdir(parents=True, exist_ok=True)
    fingerprint = guard_run(settings, documents, questions)
    graph_target = target / "graphs"
    graph_target.mkdir(exist_ok=True)

    imported: list[dict] = []
    for size in settings.sizes:
        source_graph = source / "graphs" / f"e{size}"
        target_graph = graph_target / f"e{size}"
        if target_graph.exists():
            imported.append({"extraction_size": size, "status": "already_present"})
            continue
        if not source_graph.exists():
            imported.append({"extraction_size": size, "status": "not_available"})
            continue

        complete = _validate_graph(source_graph, size, documents)
        if not complete and not include_partial:
            imported.append({"extraction_size": size, "status": "partial_skipped"})
            continue

        shutil.copytree(source_graph, target_graph)
        imported.append(
            {
                "extraction_size": size,
                "status": "imported_complete" if complete else "imported_partial",
            }
        )

    report = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config).resolve()),
        "source_output": str(source),
        "target_output": str(target),
        "sizes": list(settings.sizes),
        "documents": len(documents),
        "questions": sum(len(items) for items in questions.values()),
        "cache_fingerprint": fingerprint,
        "imports": imported,
    }
    (target / "subset_preparation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Reuse only completed graphs; skip interrupted GraphRAG caches",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.config, args.source, include_partial=not args.complete_only),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
