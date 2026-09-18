from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rq1.core.alignment import Provenance, align_fact, overlap
from rq1.core.chunking import fixed_chunks
from rq1.eval.qa_metrics import qa_proxy
from rq1.experiments.bootstrap import paired_bootstrap
from rq1.experiments.run_grid import run
from rq1.config import load_settings


class SmokeTests(unittest.TestCase):
    def test_alignment_uses_source_span(self):
        text = "Alice gave Bob the map. Bob returned it."
        chunks = fixed_chunks("story", text, 4)
        fact = Provenance("r1", "story", 6, 14, "exact_quote")
        ids = align_fact(fact, chunks)
        self.assertTrue(ids)
        self.assertTrue(all(c.id in ids for c in chunks if overlap(6, 14, c.start, c.end)))

    def test_qa_and_paired_bootstrap(self):
        self.assertEqual(qa_proxy("Alice", "Alice")["answer_f1"], 1.0)
        rows = []
        for e in (4, 8):
            for r in (4, 8):
                for q in ("q1", "q2"):
                    rows.append({"g_E": e, "g_R": r, "source": "story", "id": q,
                                 "status": "ok", "answer_f1": 0.8 if e != r else 0.6})
        result = paired_bootstrap(rows, repetitions=100)
        self.assertAlmostEqual(result["observed_off_minus_diagonal"], 0.2)

    def test_synthetic_end_to_end(self):
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            path = project / "examples" / "synthetic" / "experiment.yaml"
            settings = replace(load_settings(path), output=Path(temporary)/"out", index_store=None)
            with patch('rq1.experiments.run_grid.load_settings', return_value=settings):
                output = run(path)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["completed_cells"], 4)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(Path(manifest["output_directory"]), output.resolve())
            self.assertIn("per_query_results.csv", manifest["result_files"])
            for file in ("per_query_results.csv", "config_summary.csv", "figures/qa_heatmap.png",
                         "figures/relation_heatmap.png", "figures/evidence_heatmap.png",
                         "figures/evidence_proxy_heatmap.png", "manifest.json", "near_optimal_cells.csv", "bootstrap_ci.json"):
                self.assertTrue((output / file).exists(), file)
            with (output / "per_query_results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            primary = {"official_answer_correctness", "qa_accuracy_proxy", "qa_em", "answer_f1"}
            diagnostic = {
                "relation_recall_proxy", "path_coverage_proxy",
                "evidence_recall_at_1", "evidence_recall_at_5",
                "evidence_recall_at_10", "fixed_budget_evidence_recall",
                "evidence_recall_at_1_proxy", "evidence_recall_at_5_proxy",
                "evidence_recall_at_10_proxy", "fixed_budget_evidence_recall_proxy",
                "evidence_span_precision", "evidence_span_recall", "evidence_span_f1",
            }
            self.assertTrue(primary | diagnostic <= set(rows[0]))
            with (output / "config_summary.csv").open(newline="", encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            self.assertTrue(primary | diagnostic <= set(summary[0]))
            with patch('rq1.experiments.run_grid.load_settings', return_value=settings):
                run(path)
            with (output / "per_query_results.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            self.assertTrue(all(r["status"] == "ok" for r in rows))


if __name__ == "__main__":
    unittest.main()
