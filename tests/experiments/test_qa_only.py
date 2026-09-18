import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rq1.config import load_settings
from rq1.experiments.run_grid import run
from rq1.experiments.qa_only import link_reusable_cells
from rq1.datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]


class QAOnlyTests(unittest.TestCase):
    def settings(self, output):
        return replace(load_settings(ROOT / 'examples/synthetic/experiment.yaml'),
                       backend='ms_graphrag', sizes=(128, 256, 512), output=output,
                       index_store=output / 'shared')

    def test_incomplete_graph_stops_before_build_or_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp) / 'qa')
            with patch('rq1.experiments.run_grid.load_settings', return_value=settings), \
                 patch('rq1.experiments.qa_only.validate_graph', side_effect=[True, True, False]), \
                 patch('rq1.experiments.run_grid.build_graph') as build, \
                 patch('rq1.experiments.run_grid.OfficialLocalSearch') as search:
                with self.assertRaisesRegex(ValueError, 'E512 index is incomplete'):
                    run(ROOT / 'configs/experiments/rq1_3x3.yaml', qa_only=True)
                build.assert_not_called()
                search.assert_not_called()
                self.assertFalse(settings.output.exists())

    def test_qa_only_runs_all_nine_cells_without_indexing(self):
        def adapt(root, docs, size, overlap, cell, cached):
            tables = {'relationships': SimpleNamespace(to_dict=lambda _: [])}
            return tables, cached, []

        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp) / 'qa')
            with patch('rq1.experiments.run_grid.load_settings', return_value=settings), \
                 patch('rq1.experiments.qa_only.validate_graph', return_value=True), \
                 patch('rq1.experiments.run_grid.build_graph') as build, \
                 patch('rq1.experiments.run_grid.adapt_view', side_effect=adapt), \
                 patch('rq1.experiments.run_grid.OfficialLocalSearch') as search, \
                 patch.dict('os.environ', {}):
                search.return_value.query.return_value = ('Bob', [0], 0.1)
                output = run(ROOT / 'configs/experiments/rq1_3x3.yaml', qa_only=True)
                build.assert_not_called()
                self.assertEqual(search.call_count, 9)
            with (output / 'per_query_results.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)  # Two synthetic questions per cell.
            self.assertTrue(all(row['status'] == 'ok' for row in rows))
            self.assertEqual({(int(row['g_E']), int(row['g_R'])) for row in rows},
                             {(e, r) for e in (128, 256, 512) for r in (128, 256, 512)})

    def test_completed_cell_is_linked_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'old'
            cell = source / 'cells/e4_r4'
            cell.mkdir(parents=True)
            settings = replace(load_settings(ROOT / 'examples/synthetic/experiment.yaml'),
                               output=root / 'new', reuse_cells_from=(source,))
            _, questions = load_dataset(settings)
            rows = [dict(id=q.id, g_E=4, g_R=4, status='ok')
                    for group in questions.values() for q in group]
            with (cell / 'per_query.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'g_E', 'g_R', 'status'])
                writer.writeheader(); writer.writerows(rows)
            (cell / 'benchmark_predictions.json').write_text(json.dumps([
                {'id': row['id']} for row in rows
            ]))
            report = link_reusable_cells(settings, questions)
            target = settings.output / 'cells/e4_r4'
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), cell.resolve())
            self.assertEqual(next(x for x in report if x['cell'] == 'e4_r4')['status'], 'linked')
