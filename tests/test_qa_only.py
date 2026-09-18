import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rq1.config import load_settings
from rq1.experiments.run_grid import run


ROOT = Path(__file__).resolve().parents[1]


class QAOnlyTests(unittest.TestCase):
    def settings(self, output):
        return replace(load_settings(ROOT / 'configs/synthetic.json'),
                       backend='official', sizes=(128, 256, 512), output=output,
                       graph_store=output / 'shared')

    def test_incomplete_graph_stops_before_build_or_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp) / 'qa')
            with patch('rq1.experiments.run_grid.load_settings', return_value=settings), \
                 patch('rq1.experiments.run_grid.validate_graph', side_effect=[True, True, False]), \
                 patch('rq1.experiments.run_grid.build_graph') as build, \
                 patch('rq1.experiments.run_grid.OfficialLocalSearch') as search:
                with self.assertRaisesRegex(ValueError, 'E512 graph is incomplete'):
                    run(ROOT / 'configs/qa_3x3.yaml', qa_only=True)
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
                 patch('rq1.experiments.run_grid.validate_graph', return_value=True), \
                 patch('rq1.experiments.run_grid.build_graph') as build, \
                 patch('rq1.experiments.run_grid.adapt_view', side_effect=adapt), \
                 patch('rq1.experiments.run_grid.OfficialLocalSearch') as search, \
                 patch.dict('os.environ', {}):
                search.return_value.query.return_value = ('Bob', [0], 0.1)
                output = run(ROOT / 'configs/qa_3x3.yaml', qa_only=True)
                build.assert_not_called()
                self.assertEqual(search.call_count, 9)
            with (output / 'per_query_results.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)  # Two synthetic questions per cell.
            self.assertTrue(all(row['status'] == 'ok' for row in rows))
            self.assertEqual({(int(row['g_E']), int(row['g_R'])) for row in rows},
                             {(e, r) for e in (128, 256, 512) for r in (128, 256, 512)})
