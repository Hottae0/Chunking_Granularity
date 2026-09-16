from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rq1.alignment.provenance import Provenance, align_fact, overlap
from rq1.chunking.fixed import fixed_chunks
from rq1.eval.qa_metrics import qa_proxy
from rq1.experiments.bootstrap import paired_bootstrap
from rq1.experiments.run_grid import run


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
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            config = json.loads((project / "configs" / "synthetic.json").read_text(encoding="utf-8"))
            config["data"]["root"] = str(project / "examples" / "synthetic")
            config["data"]["output"] = str(Path(temporary) / "out")
            # Relative paths are resolved from the temporary project's parent.
            local = Path(temporary) / "configs"
            local.mkdir()
            path = local / "synthetic.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            output = run(path)
            self.assertEqual(json.loads((output / "run_manifest.json").read_text())["completed_cells"], 4)
            for file in ("per_query_results.csv", "config_summary.csv", "qa_heatmap.png",
                         "relation_recall_heatmap.png", "evidence_recall_heatmap.png",
                         "run_manifest.json", "near_optimal_cells.csv", "bootstrap_ci.json"):
                self.assertTrue((output / file).exists(), file)
            with (output / "per_query_results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            run(path)
            with (output / "per_query_results.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            self.assertTrue(all(r["status"] == "ok" for r in rows))


if __name__ == "__main__":
    unittest.main()
