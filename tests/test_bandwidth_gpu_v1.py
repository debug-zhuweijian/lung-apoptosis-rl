"""CPU/CUDA bandwidth equivalence on synthetic, isolated source fixtures."""

import hashlib
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.lung.data.bandwidth import _pair_distances, build_bandwidths


def bandwidth_fixture(root, *, cells_per_condition=140, identical=False):
    """Interleave conditions across three shards to expose repeated reads."""
    assets, bundle = root / "assets", root / "prepared"
    (assets / "fit_cache").mkdir(parents=True)
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"status": "complete", "tables": {}}), encoding="utf-8")
    rng = np.random.default_rng(31)
    size, genes = 4 * cells_per_condition + 3, 13
    counts = (np.ones((size, genes), dtype=np.int32) if identical else
              rng.poisson(2., (size, genes)).astype(np.int32) + 1)
    # Invalid common-library cells and R rows must never enter pair selection.
    invalid = []
    if not identical and cells_per_condition > 1:
        counts[0] = 0
        invalid = ["cell00000"]
    rows, locations = [], {}
    for shard in range(3):
        indices = np.arange(shard, size, 3)
        sparse.save_npz(root / f"source{shard}.npz", sparse.csr_matrix(counts[indices]))
        locations.update({int(original): (shard, row) for row, original in enumerate(indices)})
    for index in range(size):
        shard, row = locations[index]
        rows.append({"cell_id": f"cell{index:05d}",
                     "condition_id": f"condition{index // cells_per_condition}",
                     "matrix_path": f"source{shard}.npz", "matrix_row": row,
                     "role": "fit" if index < 4 * cells_per_condition else "residual_fit"})
    frame = pd.DataFrame(rows)
    frame.to_parquet(assets / "train.parquet", index=False)
    pd.DataFrame({"tahoe_column": np.arange(genes)}).to_parquet(assets / "genes.parquet", index=False)
    mean, scale = rng.normal(size=genes), rng.uniform(.1, 3., size=genes)
    np.savez(assets / "normalization.npz", mean=mean, scale=scale)
    (assets / "fit_cache" / "index.json").write_text(json.dumps({
        "invalid_zero_common_library_cell_ids": invalid}), encoding="utf-8")
    return assets, frame, counts, mean, scale, invalid


def reference_pairs(frame, counts, mean, scale, invalid):
    """Literal SHA ordering + original per-pair NumPy definition."""
    hashed = lambda value: int(hashlib.sha256(value.encode()).hexdigest(), 16)
    fitting = frame.loc[frame.role.eq("fit") & ~frame.cell_id.isin(invalid)]
    candidates = []
    for condition, group in fitting.groupby("condition_id", sort=True):
        cells = sorted(group.cell_id, key=hashed)[:128]
        for first, second in combinations(sorted(cells), 2):
            candidates.append((hashed(f"{condition}|{first}|{second}"), condition, first, second))
    selected = sorted(candidates)[:4096]
    selected.sort(key=lambda item: item[1])  # Stable SHA order inside each condition.
    normalized = {}
    for row in fitting.itertuples():
        raw = counts[int(row.cell_id.removeprefix("cell"))].astype(np.float64)
        normalized[row.cell_id] = ((np.log1p(raw * (10000. / raw.sum())) - mean) / scale
                                  ).astype(np.float32).astype(np.float64)
    return [(condition, first, second,
             float(np.mean((normalized[first] - normalized[second]) ** 2)))
            for _, condition, first, second in selected]


class BandwidthGpuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_cpu_matches_sha_selection_and_loads_each_source_shard_once(self):
        assets, frame, counts, mean, scale, invalid = bandwidth_fixture(self.root)
        calls = []
        original = sparse.load_npz
        def load(path):
            calls.append(Path(path).name)
            return original(path)
        events = []
        with patch("src.lung.data.reader.sparse.load_npz", side_effect=load):
            report = build_bandwidths(self.root, assets, {"bundle": "prepared"},
                                      lambda *event: events.append(event), device="cpu")
        expected = pd.DataFrame(reference_pairs(frame, counts, mean, scale, invalid),
                                columns=["condition_id", "cell_id_1", "cell_id_2",
                                         "distance_squared_per_gene"])
        actual = pd.read_parquet(assets / "bandwidth_pairs.parquet")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        self.assertEqual(report["pairs"], 4096)
        self.assertEqual(sorted(calls), [f"source{i}.npz" for i in range(3)])
        self.assertEqual(list(assets.glob("mmd_selected_*")), [])
        for phase in ("mmd_cell_selection", "mmd_pair_selection", "mmd_expression_cache", "mmd_bandwidth"):
            updates = [(done, total) for name, done, total in events if name == phase]
            self.assertTrue(updates, phase)
            self.assertEqual(updates[0][0], 0)
            self.assertEqual(updates[-1][0], updates[-1][1])
            self.assertTrue(all(0 <= done <= total for done, total in updates))
            self.assertEqual([done for done, _ in updates], sorted(done for done, _ in updates))

    def test_identical_expression_and_empty_pairs_use_original_median_fallback(self):
        for number in (1, 4):
            with self.subTest(cells_per_condition=number):
                root = self.root / str(number)
                root.mkdir()
                assets, *_ = bandwidth_fixture(root, cells_per_condition=number, identical=True)
                report = build_bandwidths(root, assets, {"bundle": "prepared"})
                self.assertEqual(report["positive_pairs"], 0)
                self.assertEqual(report["median"], 1.)
                self.assertEqual(report["bandwidths"], [.25, .5, 1., 2., 4.])
                self.assertEqual(report["pairs"], 4 * number * (number - 1) // 2)
                self.assertEqual(list(assets.glob("mmd_selected_*")), [])

    def test_temporary_expression_is_cleaned_after_failure(self):
        assets, *_ = bandwidth_fixture(self.root, cells_per_condition=4)
        with patch("src.lung.data.bandwidth._pair_distances", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                build_bandwidths(self.root, assets, {"bundle": "prepared"})
        self.assertEqual(list(assets.glob("mmd_selected_*")), [])
        self.assertFalse((assets / "bandwidths.npz").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_preserves_full_gene_distance_and_pair_identity(self):
        assets, *_ = bandwidth_fixture(self.root, cells_per_condition=8)
        cpu = build_bandwidths(self.root, assets, {"bundle": "prepared"}, device="cpu")
        cpu_pairs = pd.read_parquet(assets / "bandwidth_pairs.parquet")
        cuda = build_bandwidths(self.root, assets, {"bundle": "prepared"}, device="cuda:0")
        cuda_pairs = pd.read_parquet(assets / "bandwidth_pairs.parquet")
        pd.testing.assert_frame_equal(cpu_pairs.iloc[:, :3], cuda_pairs.iloc[:, :3], check_exact=True)
        np.testing.assert_allclose(cuda_pairs.iloc[:, 3], cpu_pairs.iloc[:, 3], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cuda["bandwidths"], cpu["bandwidths"], rtol=1e-12, atol=1e-12)
        values = np.random.default_rng(4).normal(size=(5, 32461)).astype(np.float32)
        pairs = [(0, 4), (1, 3), (2, 2), (4, 0)]
        expected = _pair_distances(values, pairs, "cpu", block_size=2)
        actual = _pair_distances(values, pairs, "cuda:0", block_size=2)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(actual[2], 0.)
        self.assertAlmostEqual(actual[0], actual[3], places=14)
        identical_root = self.root / "identical"
        identical_root.mkdir()
        identical_assets, *_ = bandwidth_fixture(identical_root, cells_per_condition=4, identical=True)
        fallback = build_bandwidths(identical_root, identical_assets, {"bundle": "prepared"},
                                    device="cuda:0")
        self.assertEqual(fallback["positive_pairs"], 0)
        self.assertEqual(fallback["median"], 1.)
        self.assertEqual(fallback["bandwidths"], [.25, .5, 1., 2., 4.])


if __name__ == "__main__":
    unittest.main()
