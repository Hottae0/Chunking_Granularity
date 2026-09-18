import unittest

from rq1.datasets.graphrag_bench import relation_statements


class GraphRAGBenchTests(unittest.TestCase):
    def test_nested_relation_triples_are_preserved(self):
        self.assertEqual(relation_statements([["A", "knows", "B"]]), ("(A, knows, B)",))
