"""Validate or import graphs into the graph store configured for a run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rq1.config import load_settings
from rq1.data.graphrag_bench import load_novel
from rq1.experiments.cache import guard_run
from rq1.msgraphrag.indexer import validate_graph


def _validate_graph(source_graph: Path, extraction_size: int, documents, settings) -> bool:
    return validate_graph(source_graph, extraction_size, documents, settings)


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
    target = (settings.graph_store or (settings.output / "graphs")).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source graph store does not exist: {source}")
    if (source / "graphs").is_dir():
        source = (source / "graphs").resolve()

    settings.output.mkdir(parents=True, exist_ok=True)
    fingerprint = guard_run(settings, documents, questions)
    target.mkdir(parents=True, exist_ok=True)

    imported: list[dict] = []
    for size in settings.sizes:
        source_graph = source / f"e{size}"
        target_graph = target / f"e{size}"
        if source == target:
            if not source_graph.exists() or not any(source_graph.iterdir()):
                source_graph.mkdir(parents=True, exist_ok=True)
                imported.append({"extraction_size": size, "status": "store_empty"})
                continue
            complete = _validate_graph(source_graph, size, documents, settings)
            status = "store_complete" if complete else "store_partial"
            imported.append({"extraction_size": size, "status": status})
            continue
        if target_graph.is_symlink():
            if target_graph.resolve() != source_graph.resolve():
                raise ValueError(
                    f"{target_graph} points to {target_graph.resolve()}, expected {source_graph}"
                )
            imported.append({"extraction_size": size, "status": "already_linked"})
            continue
        if target_graph.exists():
            raise ValueError(
                f"{target_graph} is a copied directory. Remove the preliminary output "
                "directory and run preparation again to use space-saving graph links."
            )

        if source_graph.exists():
            complete = _validate_graph(source_graph, size, documents, settings)
            if not complete and not include_partial:
                imported.append({"extraction_size": size, "status": "partial_skipped"})
                continue
            status = "linked_complete" if complete else "linked_partial"
        else:
            source_graph.mkdir(parents=True)
            status = "linked_new"

        target_graph.symlink_to(source_graph.resolve(), target_is_directory=True)
        imported.append({"extraction_size": size, "status": status})

    report = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config).resolve()),
        "source_output": str(source),
        "target_graph_store": str(target),
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
    parser.add_argument("--source", type=Path, required=True, help="Graph store containing e<size> directories")
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
