from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from rq1.config import load_settings
from rq1.experiments.run_grid import _mean, _read_csv, _write_csv
from rq1.plotting.heatmaps import heatmap


def run_official_eval(config: Path, benchmark_root: Path, judge_model: str,
                      embedding_model: str) -> None:
    """Run GraphRAG-Bench's official generation judge on every completed cell.

    Official answer_correctness is the primary benchmark-compatible score.
    EM and answer F1 remain transparent local QA complements.
    """
    settings = load_settings(config)
    root = benchmark_root.resolve()
    if not (root / "Evaluation" / "generation_eval.py").exists():
        raise FileNotFoundError("benchmark_root must contain Evaluation/generation_eval.py")
    env = os.environ.copy()
    env["LLM_API_KEY"] = settings.api_key
    for e in settings.sizes:
        for r in settings.sizes:
            cell = settings.output / "cells" / f"e{e}_r{r}"
            predictions = cell / "benchmark_predictions.json"
            result_path = cell / "official_generation_eval.json"
            if not predictions.exists():
                continue
            if not result_path.exists():
                command = [sys.executable, "-m", "Evaluation.generation_eval",
                           "--mode", "API", "--model", judge_model,
                           "--base_url", settings.base_url,
                           "--embedding_model", embedding_model,
                           "--data_file", str(predictions),
                           "--output_file", str(result_path), "--detailed_output"]
                with (cell / "official_eval.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
    per_query = _read_csv(settings.output / "per_query_results.csv")
    scored = {}
    for e in settings.sizes:
        for r in settings.sizes:
            path = settings.output / "cells" / f"e{e}_r{r}" / "official_generation_eval.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for question_type, group in data.items():
                for item in group.get("detailed", []):
                    scored[(e, r, str(item["id"]))] = item.get("metrics", {}).get("answer_correctness")
    for row in per_query:
        row["official_answer_correctness"] = scored.get((int(row["g_E"]), int(row["g_R"]), str(row["id"])))
    _write_csv(settings.output / "per_query_results.csv", per_query)
    summary = _read_csv(settings.output / "config_summary.csv")
    for row in summary:
        group = [x for x in per_query if int(x["g_E"]) == int(row["g_E"])
                 and int(x["g_R"]) == int(row["g_R"])]
        row["official_answer_correctness"] = _mean(group, "official_answer_correctness")
        row["official_scored_queries"] = sum(x.get("official_answer_correctness") not in (None, "") for x in group)
    _write_csv(settings.output / "config_summary.csv", summary)
    heatmap(summary, settings.sizes, "official_answer_correctness",
            settings.output / "qa_heatmap.png", "Official GraphRAG-Bench answer correctness")
    manifest_path = settings.output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["official_evaluation"] = {"judge_model": judge_model,
                                       "embedding_model": embedding_model,
                                       "scored_queries": len(scored),
                                       "metric": "answer_correctness"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--embedding-model", required=True,
                        help="Local path or HF ID for official BGE judge embedding")
    args = parser.parse_args()
    run_official_eval(args.config, args.benchmark_root,
                      args.judge_model, args.embedding_model)


if __name__ == "__main__":
    main()
