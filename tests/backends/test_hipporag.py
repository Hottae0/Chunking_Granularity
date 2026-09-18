import unittest

from rq1.backends.registry import get_backend


class HippoRAGScaffoldTests(unittest.TestCase):
    def test_unimplemented_backend_fails_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "scaffolded"):
            get_backend("hipporag")
