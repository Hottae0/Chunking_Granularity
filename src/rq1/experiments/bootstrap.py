from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def paired_bootstrap(rows: list[dict], metric: str = "answer_f1", repetitions: int = 2000,
                     seed: int = 42) -> dict:
    if repetitions < 1: raise ValueError("repetitions must be positive")
    by_cell = defaultdict(dict)
    for row in rows:
        if row.get("status") == "ok" and row.get(metric) not in (None, ""):
            by_cell[(int(row["g_E"]), int(row["g_R"]))][str(row["source"]), str(row["id"])] = float(row[metric])
    if not by_cell:
        raise ValueError("No valid observations")
    keys = sorted(set.intersection(*(set(x) for x in by_cell.values())))
    if not keys:
        raise ValueError("No paired questions across all configurations")
    diagonal = sorted(c for c in by_cell if c[0] == c[1])
    off_diagonal = sorted(c for c in by_cell if c[0] != c[1])
    if not diagonal or not off_diagonal:
        raise ValueError("Both diagonal and off-diagonal cells are required")
    def best(cells, sample):
        return max(cells, key=lambda cell: sum(by_cell[cell][k] for k in sample) / len(sample))
    chosen_d, chosen_o = best(diagonal, keys), best(off_diagonal, keys)
    def difference(sample, d, o):
        return sum(by_cell[o][k] - by_cell[d][k] for k in sample) / len(sample)
    observed = difference(keys, chosen_d, chosen_o)
    rng = random.Random(seed)
    deltas = []
    for _ in range(repetitions):
        sample = [rng.choice(keys) for _ in keys]
        # Reselect per bootstrap replicate to expose winner instability.
        d, o = best(diagonal, sample), best(off_diagonal, sample)
        deltas.append(difference(sample, d, o))
    deltas.sort()
    lower = deltas[int(0.025 * (len(deltas) - 1))]
    upper = deltas[int(0.975 * (len(deltas) - 1))]
    return {"metric": metric, "n_paired_questions": len(keys),
            "best_diagonal": list(chosen_d), "best_off_diagonal": list(chosen_o),
            "observed_off_minus_diagonal": observed, "ci_95": [lower, upper],
            "bootstrap_repetitions": repetitions, "seed": seed,
            "selection_note": "Exploratory same-data cell selection; report selection uncertainty and avoid causal claims."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="per_query_results.csv")
    parser.add_argument("--metric", default="answer_f1")
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = paired_bootstrap(rows, args.metric, args.repetitions)
    output = args.results.parent / "bootstrap_ci.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
