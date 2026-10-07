"""Synthetic CPU cache contract checks; no training or biological validation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse

from src.lung.data import TrainingData, prepare_training
from src.lung.data.common import sha256
from src.lung.data.reader import SparseRows
from tests.test_training_data_v1 import tiny_bundle


class SparseCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.values = np.array([[0, 7, 2], [3, 0, 9], [5, 4, 0]], dtype=np.int32)
        for index in range(3):
            sparse.save_npz(self.root / f"s{index}.npz", sparse.csr_matrix(self.values + index))
        self.rows = pd.DataFrame({"matrix_path": ["s2.npz", "s0.npz", "s2.npz", "s1.npz"],
                                  "matrix_row": [2, 0, 2, 1]})

    def test_lossless_order_duplicates_readonly_and_one_decode_per_source(self):
        reader = SparseRows(self.root, [2, 0, 2], cache_size=1, max_cache_bytes=268435456)
        self.addCleanup(reader.cache.clear)
        original = {p.name: sha256(p) for p in self.root.glob("*.npz")}
        with patch("scipy.sparse.load_npz", wraps=sparse.load_npz) as load:
            first = reader.read(self.rows)
            second = reader.read(self.rows)
            self.assertEqual(load.call_count, 3)
        expected = np.stack([(self.values + shard)[row, [2, 0, 2]]
                             for shard, row in ((2, 2), (0, 0), (2, 2), (1, 1))])
        np.testing.assert_array_equal(first.toarray(), expected)
        np.testing.assert_array_equal(second.toarray(), expected)
        self.assertEqual(first.dtype, self.values.dtype)
        self.assertEqual(original, {p.name: sha256(p) for p in self.root.glob("*.npz")})
        matrix = reader.disk_cache.load("s1.npz")
        self.assertIsInstance(matrix.data, np.memmap)
        self.assertFalse(matrix.data.flags.writeable)
        del matrix
        self.assertLessEqual(len(reader.cache), 1)

    def test_byte_limit_and_reopen_without_npz(self):
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        reader.read(self.rows)
        self.assertEqual(reader.cache_bytes, 0)
        self.assertEqual(len(reader.cache), 0)
        with patch("scipy.sparse.load_npz", side_effect=AssertionError("must reuse disk cache")):
            other = SparseRows(self.root, [0], max_cache_bytes=0)
            np.testing.assert_array_equal(other.read(self.rows).toarray(), reader.read(self.rows).toarray())

    def test_zero_heap_cache_audits_source_once_per_load(self):
        from src.lung.data.common import check_shard
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        rows = self.rows.iloc[[1]]
        reader.read(rows)
        with patch("src.lung.data.reader.inside", side_effect=AssertionError("duplicate source resolution")), \
                patch("src.lung.data.reader.check_shard", side_effect=AssertionError("duplicate source audit")), \
                patch("src.lung.data.reader.file_identity", side_effect=AssertionError("duplicate source identity")), \
                patch("src.lung.data.sparse_cache.check_shard", wraps=check_shard) as audit:
            for _ in range(2):
                np.testing.assert_array_equal(reader.read(rows).toarray(), self.values[[0], :1])
            self.assertEqual(audit.call_count, 2)
        self.assertEqual(reader.cache_bytes, 0)
        self.assertFalse(reader.cache)
        with self.assertRaisesRegex(ValueError, "gene space"):
            SparseRows(self.root, [3], max_cache_bytes=0).read(rows)

    def test_heap_and_nonpersistent_source_changes_invalidate(self):
        rows = self.rows.iloc[[1]]
        for disk, budget in ((False, 0), (False, 1024), (True, 1024)):
            with self.subTest(use_disk_cache=disk, max_cache_bytes=budget):
                sparse.save_npz(self.root / "s0.npz", sparse.csr_matrix(self.values))
                reader = SparseRows(self.root, [0], max_cache_bytes=budget, use_disk_cache=disk)
                self.assertEqual(reader.read(rows)[0, 0], 0)
                sparse.save_npz(self.root / "s0.npz", sparse.csr_matrix(self.values + 100))
                self.assertEqual(reader.read(rows)[0, 0], 100)
                reader.cache.clear()

    def test_cross_load_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary)
            probe = self.root / "symlink-probe"
            try:
                probe.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Symlink creation unavailable: {error}")
            probe.unlink()
            for field in ("directory", "target", "manifest.json", "data.npy", "indices.npy", "indptr.npy"):
                with self.subTest(replaced=field):
                    reader = SparseRows(self.root, [0], cache_dir=f"cache-{field}", max_cache_bytes=0)
                    rows = self.rows.iloc[[1]]
                    reader.read(rows)
                    manifest = next(reader.disk_cache.directory.glob("*/manifest.json"))
                    path = (reader.disk_cache.directory if field == "directory" else
                            manifest.parent if field == "target" else manifest.parent / field)
                    destination = outside / field
                    is_directory = path.is_dir()
                    path.rename(destination)
                    path.symlink_to(destination, target_is_directory=is_directory)
                    try:
                        with self.assertRaisesRegex(ValueError, "inside the project"):
                            reader.read(rows)
                    finally:
                        path.unlink()

    def test_source_change_creates_new_version(self):
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        rows = self.rows.iloc[[1]]
        reader.read(rows)
        sparse.save_npz(self.root / "s0.npz", sparse.csr_matrix(self.values + 100))
        self.assertEqual(reader.read(rows)[0, 0], 100)
        self.assertEqual(len(list(reader.disk_cache.directory.glob("*/manifest.json"))), 2)

    def test_bad_checksum_and_shape_fail_loudly(self):
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        reader.read(self.rows.iloc[[1]])
        manifest_path = next(reader.disk_cache.directory.glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        manifest["shape"][1] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "shape differs"):
            reader.read(self.rows.iloc[[1]])
        manifest["shape"][1] -= 1
        manifest_path.write_text(json.dumps(manifest))
        data_path = manifest_path.parent / "data.npy"
        data = np.load(data_path).copy()
        data[0] += 1
        np.save(data_path, data)
        with self.assertRaisesRegex(ValueError, "checksum"):
            reader.read(self.rows.iloc[[1]])

    def test_paths_registry_hash_and_rows_fail_closed(self):
        with self.assertRaises(ValueError):
            SparseRows(self.root, [0], cache_dir="../outside")
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        with self.assertRaises(ValueError):
            reader.read(self.rows.assign(matrix_path="../outside.npz"))
        with self.assertRaisesRegex(ValueError, "out of range"):
            reader.read(self.rows.assign(matrix_row=-1))
        stat = (self.root / "s0.npz").stat()
        registry = {"s0.npz": {"csr_bytes": stat.st_size, "csr_mtime_ns": stat.st_mtime_ns,
                                "csr_sha256": "0" * 64}}
        audited = SparseRows(self.root, [0], registry=registry)
        with self.assertRaisesRegex(ValueError, "not registered"):
            audited.read(self.rows.iloc[[0]])
        with self.assertRaisesRegex(ValueError, "hash differs"):
            audited.read(self.rows.iloc[[1]])

    def test_atomic_failure_never_publishes_partial_cache(self):
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        with patch("src.lung.data.sparse_cache.np.save", side_effect=OSError("simulated write failure")):
            with self.assertRaisesRegex(OSError, "write failure"):
                reader.read(self.rows.iloc[[1]])
        self.assertFalse(any(path.is_dir() for path in reader.disk_cache.directory.iterdir()))
        reader.read(self.rows.iloc[[1]])
        self.assertEqual(len(list(reader.disk_cache.directory.glob("*/manifest.json"))), 1)

    def test_explicit_prepare_is_deduplicated_and_reports_progress(self):
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        events = []
        with patch("scipy.sparse.load_npz", wraps=sparse.load_npz) as load:
            self.assertEqual(reader.prepare_cache(["s1.npz", "s1.npz"], lambda *event: events.append(event)), 1)
            self.assertEqual(load.call_count, 1)
        self.assertEqual(events, [(1, 1, "s1.npz")])

    def test_existing_build_lock_is_not_silently_ignored(self):
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        from src.lung.data.common import digest_json
        from src.lung.data.sparse_cache import SCHEMA
        digest = sha256(self.root / "s0.npz")
        key = digest_json({"schema": SCHEMA, "source_sha256": digest})
        reader.disk_cache.directory.mkdir(parents=True)
        with FileLock(reader.disk_cache.directory / (key + ".lock"), timeout=0):
            with self.assertRaisesRegex(RuntimeError, "build locked"):
                reader.read(self.rows.iloc[[1]])
        self.assertEqual(list(reader.disk_cache.directory.glob("*/manifest.json")), [])

    def test_stale_lock_file_does_not_block_new_builder(self):
        from src.lung.data.common import digest_json
        from src.lung.data.sparse_cache import SCHEMA
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        key = digest_json({"schema": SCHEMA, "source_sha256": sha256(self.root / "s0.npz")})
        reader.disk_cache.directory.mkdir(parents=True)
        (reader.disk_cache.directory / (key + ".lock")).touch()
        orphan = reader.disk_cache.directory / (key + ".tmp-interrupted")
        orphan.mkdir()
        (orphan / "partial.npy").write_bytes(b"interrupted data")
        np.testing.assert_array_equal(reader.read(self.rows.iloc[[1]]).toarray(), self.values[[0], :1])
        self.assertTrue((orphan / "partial.npy").exists())
        self.assertEqual(len(list(reader.disk_cache.directory.glob("*/manifest.json"))), 1)

    def test_abrupt_builder_exit_releases_os_lock(self):
        from src.lung.data.common import digest_json
        from src.lung.data.sparse_cache import SCHEMA
        reader = SparseRows(self.root, [0], max_cache_bytes=0)
        key = digest_json({"schema": SCHEMA, "source_sha256": sha256(self.root / "s0.npz")})
        reader.disk_cache.directory.mkdir(parents=True)
        lock_path = reader.disk_cache.directory / (key + ".lock")
        code = ("import os,sys; from filelock import FileLock; "
                "lock=FileLock(sys.argv[1], timeout=0); lock.acquire(); os._exit(0)")
        result = subprocess.run([sys.executable, "-c", code, str(lock_path)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(lock_path.exists())
        np.testing.assert_array_equal(reader.read(self.rows.iloc[[1]]).toarray(), self.values[[0], :1])

    def test_audited_source_mutation_and_wrong_dimension_are_rejected(self):
        stat = (self.root / "s0.npz").stat()
        audit = {"csr_bytes": stat.st_size, "csr_mtime_ns": stat.st_mtime_ns,
                 "csr_sha256_from_checkpoint": sha256(self.root / "s0.npz"), "source_rows": 4}
        reader = SparseRows(self.root, [0], registry={"s0.npz": audit}, max_cache_bytes=0)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            reader.read(self.rows.iloc[[1]])
        audit["source_rows"] = 3
        reader.read(self.rows.iloc[[1]])
        with (self.root / "s0.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since"):
            reader.read(self.rows.iloc[[1]])


class RoleSamplingCacheTests(unittest.TestCase):
    def test_multiple_control_pools_keep_scalar_rng_order(self):
        reader = TrainingData.__new__(TrainingData)
        reader.allowed_roles = frozenset(["fit"])
        reader._control_ids = {"treatment": ("a", "b")}
        reader._present_conditions = {"fit": frozenset(["a", "b"])}
        pools = [pd.DataFrame({"cell_id": ["a0", "a1"]}),
                 pd.DataFrame({"cell_id": ["b0", "b1", "b2", "b3"]})]
        oracle = np.random.default_rng(888)
        actual = np.random.default_rng(888)
        picks = oracle.integers(2, size=31)
        expected = pd.concat([pools[i].iloc[[oracle.integers(len(pools[i]))]]
                              for i in picks], ignore_index=True)
        with patch.object(reader, "_cell_metadata", side_effect=lambda name, role: pools[0 if name == "a" else 1]):
            result = reader._control_rows("treatment", 31, actual, "fit")
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(oracle.integers(1000000), actual.integers(1000000))

    def test_sampling_rng_exactly_matches_original_and_permissions_remain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_bundle(root)
            prepare_training(root)
            reader = TrainingData(root, row_cache_options={"max_cache_bytes": 0})
            condition = reader.condition_ids()[0]
            rng = np.random.default_rng(718)
            target_pool = reader.cell_metadata(condition)
            target = target_pool.iloc[rng.integers(len(target_pool), size=19)]
            control_ids = reader.controls.loc[reader.controls.condition_id.eq(condition), "control_condition_id"]
            present = set(reader.role_counts.loc[reader.role_counts.role.eq("fit"), "condition_id"])
            pools = [reader.cell_metadata(item) for item in control_ids if item in present]
            picks = rng.integers(len(pools), size=23)
            controls = pd.concat([pools[i].iloc[[rng.integers(len(pools[i]))]] for i in picks], ignore_index=True)
            batch = reader.sample_response(condition, target_n=19, control_n=23,
                                           rng=np.random.default_rng(718))
            np.testing.assert_array_equal(batch["target_cell_ids"], target.cell_id.to_numpy())
            np.testing.assert_array_equal(batch["control_cell_ids"], controls.cell_id.to_numpy())
            np.testing.assert_array_equal(batch["target"], reader.normalize(reader.rows.read(target)))
            np.testing.assert_array_equal(batch["control"], reader.normalize(reader.rows.read(controls)))
            public = reader.cell_metadata(condition)
            public.iloc[0, public.columns.get_loc("cell_id")] = "MUTATED"
            self.assertNotEqual(reader.cell_metadata(condition).iloc[0].cell_id, "MUTATED")
            listed = reader.condition_ids()
            listed.clear()
            self.assertTrue(reader.condition_ids())
            with self.assertRaises(PermissionError):
                reader.sample_response(condition, role="final_test")
            with self.assertRaises(PermissionError):
                reader.sample_response(condition, role="selection")


if __name__ == "__main__":
    unittest.main()
