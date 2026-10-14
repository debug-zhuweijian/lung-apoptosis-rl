"""Bounded CSR mmap reuse contracts on tiny CPU-only synthetic shards."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.lung.data.common import check_shard, sha256
from src.lung.data.sparse_cache import CSRCache, _inside_resolved


class CSRMappingReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.values = np.array([[0, 7, 2], [3, 0, 9], [5, 4, 0]], dtype=np.int32)
        for index in range(3):
            sparse.save_npz(self.root / f"s{index}.npz", sparse.csr_matrix(self.values + index))

    def cache(self, limit=2, directory="cache"):
        cache = CSRCache(self.root, directory, max_open_mappings=limit)
        self.addCleanup(cache.close)
        return cache

    def test_repeat_load_reuses_arrays_and_manifest_but_audits_source(self):
        cache = self.cache()
        first = cache.load("s0.npz")
        with patch("src.lung.data.sparse_cache.np.load", wraps=np.load) as load, \
                patch.object(Path, "read_text", side_effect=AssertionError("manifest must be reused")), \
                patch("src.lung.data.sparse_cache.check_shard", wraps=check_shard) as audit, \
                patch("src.lung.data.sparse_cache._inside_resolved", wraps=_inside_resolved) as resolve:
            second = cache.load("s0.npz")
            third = cache.load("s0.npz")
        self.assertEqual(load.call_count, 0)
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(resolve.call_count, 2)  # Only the cache target, no leaf re-resolves.
        self.assertIs(first.data, second.data)
        self.assertIs(second.indices, third.indices)
        np.testing.assert_array_equal(third[[2, 0, 2], :].toarray(), self.values[[2, 0, 2], :])
        for array in (third.data, third.indices, third.indptr):
            self.assertIsInstance(array, np.memmap)
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            third.data[0] = -1

    def test_returned_wrapper_changes_do_not_poison_reused_arrays(self):
        cache = self.cache()
        first = cache.load("s0.npz")
        first.data = np.zeros_like(first.data)
        first._shape = (3, 100)
        np.testing.assert_array_equal(cache.load("s0.npz").toarray(), self.values)

    def test_default_remains_uncached_and_limit_is_validated(self):
        cache = CSRCache(self.root, "default")
        self.addCleanup(cache.close)
        cache.load("s0.npz")
        with patch("src.lung.data.sparse_cache.np.load", wraps=np.load) as load:
            cache.load("s0.npz")
        self.assertEqual(load.call_count, 3)
        for value in (-1, 1.5, True, "2"):
            with self.subTest(limit=value), self.assertRaises(ValueError):
                CSRCache(self.root, max_open_mappings=value)

    def test_lru_eviction_preserves_external_views_and_reopens_evicted_source(self):
        cache = self.cache(limit=2)
        retained = cache.load("s0.npz")
        cache.load("s1.npz")
        cache.load("s0.npz")
        cache.load("s2.npz")
        self.assertEqual(len(cache._mappings), 2)
        with patch("src.lung.data.sparse_cache.np.load", wraps=np.load) as load:
            cache.load("s0.npz")
            self.assertEqual(load.call_count, 0)
            cache.load("s1.npz")
        self.assertEqual(load.call_count, 3)
        np.testing.assert_array_equal(retained.toarray(), self.values)

    def test_clear_and_close_only_release_owned_references(self):
        cache = self.cache()
        retained = cache.load("s0.npz")
        cache.clear()
        self.assertFalse(cache._mappings)
        np.testing.assert_array_equal(retained.toarray(), self.values)
        with patch("src.lung.data.sparse_cache.np.load", wraps=np.load) as load:
            cache.load("s0.npz")
        self.assertEqual(load.call_count, 4)
        cache.close()
        cache.close()
        np.testing.assert_array_equal(retained.toarray(), self.values)

    def test_source_mutation_refreshes_mapping_and_strict_audit_still_rejects(self):
        cache = self.cache()
        cache.load("s0.npz")
        sparse.save_npz(self.root / "s0.npz", sparse.csr_matrix(self.values + 100))
        np.testing.assert_array_equal(cache.load("s0.npz").toarray(), self.values + 100)
        stat = (self.root / "s1.npz").stat()
        audit = {"csr_bytes": stat.st_size, "csr_mtime_ns": stat.st_mtime_ns,
                 "csr_sha256": sha256(self.root / "s1.npz")}
        cache.load("s1.npz", audit)
        with (self.root / "s1.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since"):
            cache.load("s1.npz", audit)

    def test_manifest_and_array_mutation_invalidate_and_revalidate(self):
        cache = self.cache()
        cache.load("s0.npz")
        manifest_path = next(cache.directory.glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        bad_shape = {**manifest, "shape": [3, 4]}
        manifest_path.write_text(json.dumps(bad_shape))
        with self.assertRaisesRegex(ValueError, "shape differs"):
            cache.load("s0.npz")
        manifest_path.write_text(json.dumps(manifest))
        cache.load("s0.npz")
        data_path = manifest_path.parent / "data.npy"
        data = np.load(data_path).copy()
        data[0] += 1
        # Windows forbids truncating an active mapped file; fixed-length writes
        # still exercise mutation detection while the mapping remains resident.
        with data_path.open("r+b") as stream:
            stream.seek(-data.nbytes, os.SEEK_END)
            stream.write(data.tobytes())
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.load("s0.npz")

    def directory_link(self, path, destination):
        try:
            path.symlink_to(destination, target_is_directory=True)
            return
        except OSError as error:
            if os.name != "nt":
                raise error
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(destination)],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            self.skipTest("Directory symlink/junction creation unavailable")

    def test_cross_load_directory_link_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary)
            for field in ("directory", "target"):
                with self.subTest(replaced=field):
                    cache = self.cache(directory=f"cache-{field}")
                    cache.load("s0.npz")
                    target = next(cache.directory.glob("*/manifest.json")).parent
                    path = cache.directory if field == "directory" else target
                    destination = outside / field
                    try:
                        path.rename(destination)
                    except PermissionError:
                        self.skipTest("OS disallows moving an active mapped directory")
                    try:
                        self.directory_link(path, destination)
                        with self.assertRaisesRegex(ValueError, "inside the project"):
                            cache.load("s0.npz")
                    finally:
                        cache.clear()
                        if path.is_symlink():
                            path.unlink()
                        elif path.exists():
                            path.rmdir()  # Remove the junction itself, never its target.

    def test_initial_internal_directory_link_is_rechecked_after_retargeting(self):
        physical = self.root / "physical-cache"
        physical.mkdir()
        alias = self.root / "cache-alias"
        self.directory_link(alias, physical)
        try:
            cache = self.cache(directory="cache-alias")
            cache.load("s0.npz")
            if alias.is_symlink():
                alias.unlink()
            else:
                alias.rmdir()
            with tempfile.TemporaryDirectory() as temporary:
                self.directory_link(alias, Path(temporary))
                try:
                    with self.assertRaisesRegex(ValueError, "inside the project"):
                        cache.load("s0.npz")
                finally:
                    if alias.is_symlink():
                        alias.unlink()
                    else:
                        alias.rmdir()
        finally:
            if alias.is_symlink():
                alias.unlink()
            elif alias.exists():
                alias.rmdir()

    def test_cross_load_leaf_link_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary)
            probe = self.root / "symlink-probe"
            try:
                probe.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("File symlink creation unavailable")
            probe.unlink()
            for field in ("manifest.json", "data.npy", "indices.npy", "indptr.npy"):
                with self.subTest(replaced=field):
                    cache = self.cache(directory=f"cache-{field}")
                    cache.load("s0.npz")
                    path = next(cache.directory.glob("*/manifest.json")).parent / field
                    destination = outside / field
                    try:
                        path.rename(destination)
                    except PermissionError:
                        self.skipTest("OS disallows moving an active mapped file")
                    try:
                        path.symlink_to(destination)
                        with self.assertRaisesRegex(ValueError, "inside the project"):
                            cache.load("s0.npz")
                    finally:
                        path.unlink()
                        cache.clear()


if __name__ == "__main__":
    unittest.main()
