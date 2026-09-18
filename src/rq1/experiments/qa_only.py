"""Validate indexes and reuse compatible completed cells for QA-only grids."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rq1.backends.ms_graphrag.indexer import graph_dir, validate_graph
from rq1.config import load_settings
from rq1.datasets import load_dataset
from rq1.experiments.cache import scientific_fingerprint


def validate_qa_indexes(settings):
    if settings.backend != "ms_graphrag":
        raise ValueError("QA-only execution currently requires backend: ms_graphrag")
    documents, questions = load_dataset(settings)
    for size in settings.sizes:
        root = graph_dir(settings.output, size, settings.index_store)
        if not validate_graph(root, size, documents, settings):
            raise ValueError(f"E{size} index is incomplete at {root}; finish indexing before QA")
    plan_reusable_cells(settings, documents, questions, require_configured=True)
    return documents, questions


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _valid_cell(cell: Path, e: int, r: int, question_ids: set[str], fingerprint: str) -> bool:
    results = cell / "per_query.csv"
    predictions = cell / "benchmark_predictions.json"
    identity = cell.parents[1] / "cache_identity.json"
    if not results.exists() or not predictions.exists() or not identity.exists():
        return False
    try:
        if json.loads(identity.read_text(encoding="utf-8")).get("scientific_fingerprint") != fingerprint:
            return False
    except (json.JSONDecodeError, OSError):
        return False
    rows = _rows(results)
    if len(rows) != len(question_ids) or {row.get("id") for row in rows} != question_ids:
        return False
    if any(row.get("status") != "ok" or int(row["g_E"]) != e or int(row["g_R"]) != r for row in rows):
        return False
    try:
        predicted = json.loads(predictions.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return len(predicted) == len(question_ids) and {str(row.get("id")) for row in predicted} == question_ids


def plan_reusable_cells(settings, documents, questions,
                        require_configured: bool = False) -> dict[str, Path]:
    question_ids = {q.id for group in questions.values() for q in group}
    fingerprint = scientific_fingerprint(settings, documents, questions)
    found = {}
    for e in settings.sizes:
        for r in settings.sizes:
            name = f"e{e}_r{r}"
            for source_root in settings.reuse_cells_from:
                source = source_root / "cells" / name
                if _valid_cell(source, e, r, question_ids, fingerprint):
                    found[name] = source.resolve()
                    break
    if require_configured:
        missing = sorted(set(settings.require_reuse_cells) - found.keys())
        if missing:
            raise ValueError(
                "Required prior QA cells are unavailable: " + ", ".join(missing)
                + ". Restore the 2x2 cell artifacts or remove require_reuse_cells to rerun them."
            )
    return found


def link_reusable_cells(settings, documents, questions) -> list[dict]:
    """Link immutable completed cells instead of copying or rerunning them."""
    found = plan_reusable_cells(settings, documents, questions, require_configured=True)
    cells = settings.output / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    report = []
    for e in settings.sizes:
        for r in settings.sizes:
            target = cells / f"e{e}_r{r}"
            if target.exists() or target.is_symlink():
                report.append({"cell": target.name, "status": "already_present"})
                continue
            if target.name in found:
                source = found[target.name]
                target.symlink_to(source, target_is_directory=True)
                report.append({"cell": target.name, "status": "linked", "source": str(source)})
            else:
                report.append({"cell": target.name, "status": "new"})
    (settings.output / "cell_reuse.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from rq1.experiments.run_grid import run
    print(run(args.config, qa_only=True))


if __name__ == "__main__":
    main()
