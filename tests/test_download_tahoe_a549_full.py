"""Run with Python's standard-library unittest; no test-framework dependency."""
import importlib.util
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa

spec = importlib.util.spec_from_file_location("download_tahoe", Path(__file__).parents[1] / "scripts/download_tahoe_a549_full.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TahoeExtractionTests(unittest.TestCase):
    def assert_rejected(self, genes, counts):
        lookup = np.full(8, -1, dtype=np.int32)
        lookup[3] = 0
        with self.assertRaises(ValueError):
            module.make_csr(pa.table({"genes": genes, "expressions": counts}), lookup, 1)

    def test_csr_preserves_counts_and_gene_identity(self):
        data = pa.table({"genes": [[1, 7, 3], [1, 3]], "expressions": [[-2., 5., 2.], [-2., 4.]]})
        lookup = np.full(8, -1, dtype=np.int32)
        lookup[[3, 7]] = [0, 1]
        actual = module.make_csr(data, lookup, 2)
        np.testing.assert_array_equal(actual.toarray(), [[2, 5], [4, 0]])

    def test_wrong_cls_rejected(self):
        self.assert_rejected([[1, 3]], [[-1., 2.]])

    def test_negative_count_rejected(self):
        self.assert_rejected([[1, 3]], [[-2., -1.]])

    def test_fractional_count_rejected(self):
        self.assert_rejected([[1, 3]], [[-2., 1.5]])

    def test_length_mismatch_rejected(self):
        self.assert_rejected([[1, 3]], [[-2.]])

    def test_duplicate_token_rejected(self):
        self.assert_rejected([[1, 3, 3]], [[-2., 1., 2.]])

    def test_unknown_token_rejected(self):
        self.assert_rejected([[1, 5]], [[-2., 1.]])

    def test_strict_hash_detects_equal_size_equal_mtime_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(module, "BASE", root):
                path = root / "source.parquet"
                path.write_bytes(b"abc")
                record = {"source_sha256": hashlib.sha256(b"abc").hexdigest(),
                          "files": [module.file_record(path)]}
                entry = {"lfs": {"oid": record["source_sha256"]}}
                stamp = path.stat().st_mtime_ns
                path.write_bytes(b"adc")
                os.utime(path, ns=(stamp, stamp))
                self.assertTrue(module.checkpoint_valid(record, entry))
                self.assertFalse(module.checkpoint_valid(record, entry, verify_hashes=True))


if __name__ == "__main__":
    unittest.main()
