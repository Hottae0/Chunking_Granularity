"""RQ1 optimum stability analysis with novel-cluster bootstrap resampling."""
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def analyze(rows, sizes=None, metric="answer_f1", repetitions=2000, seed=42):
    if repetitions < 1:
        raise ValueError("Positive repetitions required")
    cells = defaultdict(dict)
    expected_keys = {(str(x['source']), str(x['id'])) for x in rows}
    for row in rows:
        if row.get('status') != 'ok' or row.get(metric) in (None, ''): continue
        value = float(row[metric])
        if not math.isfinite(value): continue
        cell = (int(row['g_E']), int(row['g_R']))
        key = (str(row['source']), str(row['id']))
        if key in cells[cell]: raise ValueError('Duplicate cell/question observation')
        cells[cell][key] = value
    if sizes is None:
        sizes = sorted({int(row['g_E']) for row in rows} | {int(row['g_R']) for row in rows})
    expected = {(e,r) for e in sizes for r in sizes}
    if set(cells) != expected or any(set(v) != expected_keys for v in cells.values()):
        return dict(status='incomplete', metric=metric,
                    reason='All cells must have all finite scored questions; no survivor-only selection')
    if len(sizes) < 2: return dict(status='unavailable', reason='At least two sizes required')
    ordered = sorted(cells)
    novels = sorted({k[0] for k in expected_keys})
    keys = sorted(expected_keys)
    by_novel = {n: [k for k in keys if k[0] == n] for n in novels}
    mean = lambda c, sample: sum(cells[c][k] for k in sample) / len(sample)
    best = lambda options, sample: max(options, key=lambda c: (mean(c, sample), -c[0], -c[1]))
    rng = random.Random(seed)
    def sample_books(books):
        return [k for _ in books for k in by_novel[rng.choice(books)]]
    counts = Counter()
    off_ties = []
    for _ in range(repetitions):
        sample = sample_books(novels)
        highest = max(mean(c, sample) for c in ordered)
        ties = [c for c in ordered if abs(mean(c, sample) - highest) < 1e-12]
        for c in ties: counts[c] += 1 / len(ties)
        off_ties.append(sum(c[0] != c[1] for c in ties) / len(ties))
    return dict(status='ok', metric=metric, seed=seed, repetitions=repetitions,
        weighting='question micro-average; resample whole novels',
        global_best=list(best(ordered, keys)),
        optimal_ties=[list(c) for c in ordered if abs(mean(c,keys)-max(mean(x,keys) for x in ordered)) < 1e-12],
        bootstrap_off_diagonal_mass=sum(off_ties)/repetitions,
        winner_frequency={f'e{c[0]}_r{c[1]}': counts[c]/repetitions for c in ordered},
        note='Exploratory joint optimum; frequency is not a p-value or independent stage optimum')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--metric', default='answer_f1')
    p.add_argument('--repetitions', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    with a.results.open(newline='') as f: rows=list(csv.DictReader(f))
    result=analyze(rows, metric=a.metric, repetitions=a.repetitions, seed=a.seed)
    output=a.results.parent / 'rq1_analysis.json'
    output.write_text(json.dumps(result, indent=2))
    print(output)

if __name__ == '__main__': main()
