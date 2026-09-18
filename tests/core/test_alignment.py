import unittest

from rq1.core.alignment import Provenance, align_fact
from rq1.core.types import Chunk


class AlignmentTests(unittest.TestCase):
    def test_alignment_requires_same_document_and_overlap(self):
        fact = Provenance("r", "a", 5, 10, "exact_quote")
        chunks = [Chunk("hit", "a", "", 8, 12, 1), Chunk("wrong", "b", "", 5, 10, 1)]
        self.assertEqual(align_fact(fact, chunks), ["hit"])
