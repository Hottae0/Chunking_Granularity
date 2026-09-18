import unittest

from rq1.core.chunking import fixed_chunks


class ChunkingTests(unittest.TestCase):
    def test_fixed_chunks_preserve_source_offsets(self):
        text = "Alpha βeta gamma delta"
        chunks = fixed_chunks("doc", text, 2)
        self.assertTrue(chunks)
        self.assertTrue(all(text[item.start:item.end] == item.text for item in chunks))
