import fcntl
import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rq1.config import load_settings
from rq1.experiments.index_only import run
from rq1.msgraphrag.indexer import build_graph


ROOT = Path(__file__).resolve().parents[1]


class IndexOnlyTests(unittest.TestCase):
    def test_only_requested_graph_is_built_without_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(load_settings(ROOT / 'configs/synthetic.json'),
                               backend='official', sizes=(512,), output=Path(tmp),
                               graph_store=Path(tmp) / 'graphs')
            with patch('rq1.experiments.index_only.load_settings', return_value=settings), \
                 patch('rq1.experiments.index_only.build_graph', return_value=Path(tmp) / 'graphs/e512') as build, \
                 patch('rq1.experiments.run_grid.run') as grid, \
                 patch.dict('os.environ', {}):
                report = run(ROOT / 'configs/index_512.yaml')
            build.assert_called_once()
            self.assertEqual(build.call_args.args[2], 512)
            grid.assert_not_called()
            self.assertTrue(report.exists())
            self.assertFalse((Path(tmp) / 'cells').exists())

    def test_shared_graph_lock_prevents_concurrent_indexing(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(load_settings(ROOT / 'configs/synthetic.json'),
                               graph_store=Path(tmp))
            graph = Path(tmp) / 'e512'
            graph.mkdir()
            with (graph / '.index.lock').open('a+') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('rq1.msgraphrag.indexer._build_graph') as inner:
                    with self.assertRaisesRegex(RuntimeError, 'Another process'):
                        build_graph(settings, [], 512, logging.getLogger(__name__))
                    inner.assert_not_called()
            with patch('rq1.msgraphrag.indexer._build_graph', return_value=graph):
                self.assertEqual(build_graph(settings, [], 512, logging.getLogger(__name__)), graph)
