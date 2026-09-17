"""Live progress for graph construction and factorial QA evaluation."""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime
from pathlib import Path

from rq1.config import load_settings
from rq1.data.graphrag_bench import load_novel


STAGE_WORDS = (
    "workflow",
    "chunk",
    "extract",
    "summar",
    "cluster",
    "report",
    "embed",
    "community",
)


def _actual_graph(output: Path, size: int) -> Path:
    graph = output / "graphs" / f"e{size}"
    if graph.is_symlink():
        return graph.resolve()
    return graph


def _latest_stage(log_path: Path) -> str:
    if not log_path.exists():
        return "waiting"
    with log_path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end - 131072))
        text = stream.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        lower = line.lower()
        if any(word in lower for word in STAGE_WORDS):
            return line[-120:]
    return lines[-1][-120:] if lines else "starting"


def _cache_files(graph: Path) -> int:
    cache = graph / "cache"
    if not cache.exists():
        return 0
    return sum(path.is_file() for path in cache.rglob("*"))


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def snapshot(config: Path) -> str:
    settings = load_settings(config)
    documents, grouped = load_novel(
        settings.corpus,
        settings.questions,
        settings.max_documents,
        settings.max_questions_per_document,
        settings.document_selection,
        settings.seed,
    )
    questions = sum(len(items) for items in grouped.values())
    expected_cells = len(settings.sizes) ** 2
    expected_qa = questions * expected_cells

    graph_lines = []
    completed_graphs = 0
    for position, size in enumerate(settings.sizes, start=1):
        graph = _actual_graph(settings.output, size)
        if (graph / "graph_complete.json").exists():
            state = "COMPLETE"
            completed_graphs += 1
        elif graph.exists() and any(graph.iterdir()):
            state = "INDEXING/PARTIAL"
        else:
            state = "WAITING"
        graph_lines.append(
            f"  [{position}/{len(settings.sizes)}] e{size:<4} {state:<16} "
            f"cache_files={_cache_files(graph):<6} stage={_latest_stage(graph / 'index.log')}"
        )

    cells_root = settings.output / "cells"
    cell_dirs = list(cells_root.glob("e*_r*")) if cells_root.exists() else []
    completed_cells = sum(
        (cell / "benchmark_predictions.json").exists() for cell in cell_dirs
    )
    qa_rows = sum(_csv_rows(cell / "per_query.csv") for cell in cell_dirs)

    current_cell = "-"
    active = [
        cell for cell in cell_dirs
        if (cell / "per_query.csv").exists()
        and not (cell / "benchmark_predictions.json").exists()
    ]
    if active:
        current_cell = max(
            active, key=lambda cell: (cell / "per_query.csv").stat().st_mtime
        ).name

    disk_root = settings.output if settings.output.exists() else settings.project
    usage = shutil.disk_usage(disk_root)
    free_gib = usage.free / (1024 ** 3)

    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Mode: {settings.name}",
        f"Output: {settings.output.resolve()}",
        f"Disk free: {free_gib:.2f} GiB",
        "",
        f"Graphs: {completed_graphs}/{len(settings.sizes)}",
        *graph_lines,
        "",
        f"Cells: {completed_cells}/{expected_cells}  current={current_cell}",
        f"QA: {qa_rows}/{expected_qa}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/server_preliminary_3x3.yaml"),
    )
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    while True:
        if not args.once:
            print("\033[2J\033[H", end="")
        print(snapshot(args.config), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
