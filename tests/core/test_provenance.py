import unittest

from rq1.core.provenance import fact_provenance


class ProvenanceTests(unittest.TestCase):
    def test_exact_entity_quote_narrows_extraction_span(self):
        rows = fact_provenance("e", "doc", "before Alice after", ["u"], {"u": (0, 18)}, "Alice")
        self.assertEqual((rows[0].start, rows[0].end, rows[0].precision), (7, 12, "exact_quote"))
