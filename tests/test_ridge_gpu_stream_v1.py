"""CUDA full-population means: independent arithmetic and reader boundaries."""

import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.pilot_ridge_data import mean_plan, stream_means
from test_pilot_ridge_v1 import tiny_data

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA numerical execution")


def dense_reference(counts, mean, scale, pools, n_pools):
    """Literal dense definition, independent of the sparse-deviation algorithm."""
    dense = counts.toarray().astype(np.float64)
    coordinates = ((np.log1p(dense * (10000. / dense.sum(1))[:, None]) - mean) / scale)
    quantized = coordinates.astype(np.float32)
    return np.stack([quantized[pools == i].mean(0, dtype=np.float64) for i in range(n_pools)])


def accumulate(counts, pools, mean, scale, chunks):
    from lung.engine.ridge_stream_cuda import CudaPopulationAccumulator, pack_counts

    n_pools = int(pools.max()) + 1
    accumulator = CudaPopulationAccumulator(n_pools, mean, scale, device="cuda:0")
    start = 0
    for size in chunks:
        stop = start + size
        packed = pack_counts(counts[start:stop], pools[start:stop])
        assert packed.shape == (size, counts.shape[1])
        for array in (packed.offsets, packed.columns, packed.values, packed.pools):
            assert array.device.type == "cpu" and array.is_pinned()
        accumulator.add(packed)
        start = stop
    assert start == counts.shape[0]
    result = accumulator.means(np.bincount(pools, minlength=n_pools))
    assert result.dtype == torch.float64 and result.device.type == "cuda"
    return result.cpu().numpy()


def test_sparse_means_include_zeros_frozen_scaling_and_fp32_quantization():
    counts = sparse.csr_matrix(np.array([
        [1., 0., 3., 0., 17.], [0., 7., 0., 9., 0.],
        [13., 0., 0., 2., 1.], [0., 3., 5., 0., 0.],
        [9., 0., 2., 0., 11.], [0., 0., 7., 1., 0.],
        [5., 2., 0., 0., 0.],
    ], dtype=np.float64))
    pools = np.array([1, 0, 2, 1, 0, 2, 1], dtype=np.int64)
    mean = np.array([.123456789, 3.234567891, -1.876543219, 8.123456789, .456789123])
    scale = np.array([.123456789, .7, 2.34567891, .111111113, 3.141592653])
    expected = dense_reference(counts, mean, scale, pools, 3)
    dense = counts.toarray()
    unquantized = (np.log1p(dense * (10000. / dense.sum(1))[:, None]) - mean) / scale
    unquantized_means = np.stack([unquantized[pools == i].mean(0) for i in range(3)])
    assert np.max(np.abs(expected - unquantized_means)) > 1e-7
    for chunks in ([7], [2, 1, 4], [1] * 7):
        actual = accumulate(counts, pools, mean, scale, chunks)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-12)


def test_full_32461_gene_axis_is_retained():
    n_genes = 32461
    counts = sparse.csr_matrix((np.array([1., 2., 7., 13., 19., 31.]),
                               (np.array([0, 0, 1, 2, 3, 3]),
                                np.array([0, n_genes - 1, 147, 30001, 1, n_genes - 1]))),
                              shape=(4, n_genes))
    pools = np.array([0, 1, 0, 1], dtype=np.int64)
    mean = np.linspace(-2.12345, 8.45678, n_genes)
    scale = np.linspace(.3, 2.7, n_genes)
    actual = accumulate(counts, pools, mean, scale, [1, 2, 1])
    assert actual.shape == (2, n_genes)
    np.testing.assert_allclose(actual, dense_reference(counts, mean, scale, pools, 2),
                               rtol=0., atol=2e-12)


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, -np.inf])
def test_invalid_counts_fail_explicitly(bad):
    from lung.engine.ridge_stream_cuda import CudaPopulationAccumulator, pack_counts

    counts = sparse.csr_matrix([[bad, 0.]])
    with pytest.raises(ValueError, match="(?i)zero|library|finite|nonnegative|invalid"):
        accumulator = CudaPopulationAccumulator(1, np.zeros(2), np.ones(2), device="cuda:0")
        accumulator.add(pack_counts(counts, np.array([0], dtype=np.int64)))
        accumulator.means(np.array([1]))


def test_unsorted_unique_csr_matches_shared_standardized_without_mutating_input():
    from lung.data.preprocess import standardized

    counts = sparse.csr_matrix((np.array([3., 1., 2., 4., 7., 2.]),
                               np.array([2, 0, 1, 3, 0, 1]), np.array([0, 3, 6])),
                              shape=(2, 4))
    assert not counts.has_sorted_indices
    before = (counts.data.copy(), counts.indices.copy(), counts.indptr.copy())
    pools = np.array([0, 0], dtype=np.int64)
    mean, scale = np.array([.7, -.3, 2., 1.]), np.array([.4, 1.3, 2.1, .7])
    expected = standardized(counts, mean, scale).mean(0, dtype=np.float64, keepdims=True)
    np.testing.assert_allclose(accumulate(counts, pools, mean, scale, [2]),
                               expected, rtol=0., atol=2e-12)
    for actual, original in zip((counts.data, counts.indices, counts.indptr), before):
        np.testing.assert_array_equal(actual, original)


def test_duplicate_csr_rejected_without_changing_frozen_coordinate_semantics():
    from lung.data.preprocess import standardized
    from lung.engine.ridge_stream_cuda import pack_counts

    counts = sparse.csr_matrix((np.array([1., 2., 3.], dtype=np.float64),
                               np.array([0, 0, 1]), np.array([0, 3])), shape=(1, 2))
    before = (counts.data.copy(), counts.indices.copy(), counts.indptr.copy())
    frozen = standardized(counts, np.zeros(2), np.ones(2))
    merged = counts.copy()
    merged.sum_duplicates()
    merged_coordinates = standardized(merged, np.zeros(2), np.ones(2))
    # Actual shared reader logs stored entries before the dense duplicate sum.
    # Canonicalizing first would silently change these frozen coordinates.
    np.testing.assert_allclose(frozen, [[15.531208992, 8.517393112]], rtol=0., atol=1e-8)
    np.testing.assert_allclose(merged_coordinates, [[8.517393112, 8.517393112]],
                               rtol=0., atol=1e-8)
    assert frozen[0, 0] != merged_coordinates[0, 0]
    with pytest.raises(ValueError, match="DUPLICATE_CSR_COORDINATES"):
        pack_counts(counts, np.array([0], dtype=np.int64))
    for actual, original in zip((counts.data, counts.indices, counts.indptr), before):
        np.testing.assert_array_equal(actual, original)

def test_negative_duplicate_cannot_be_hidden_by_positive_sum():
    from lung.engine.ridge_stream_cuda import pack_counts

    counts = sparse.csr_matrix((np.array([-1., 3.]), np.array([0, 0]), np.array([0, 2])),
                              shape=(1, 2))
    with pytest.raises(ValueError, match="(?i)nonnegative|invalid"):
        pack_counts(counts, np.array([0], dtype=np.int64))


def assert_same_population(actual, expected):
    for key in ("pool_mean", "control_mean", "target_mean"):
        assert actual[key].dtype == np.float64
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-7, atol=2e-7)
    for key in ("pool_n", "pool_condition_ids", "condition_ids", "group_ids"):
        np.testing.assert_array_equal(actual[key], expected[key])
    actual_meta = json.loads(actual["metadata_json"].item())
    expected_meta = json.loads(expected["metadata_json"].item())
    for key in ("role", "data_identity", "plan_sha256", "ordered_cell_read_sha256",
                "unique_cells", "standardized_input_dtype", "accumulation_dtype",
                "control_summary_scope"):
        assert actual_meta[key] == expected_meta[key]
    assert "cuda" in actual_meta["numeric_backend"].lower()


@pytest.mark.parametrize("role,reader_role", [("T", "fit"), ("V", "selection")])
def test_stream_matches_cpu_and_reads_same_role_once_without_cpu_normalize(tiny_data, role, reader_role):
    plan = mean_plan(tiny_data, role)
    expected = stream_means(tiny_data, plan, batch_size=7, device="cpu")
    original_read = tiny_data.reader.rows.read
    observations = []
    lock = threading.Lock()
    workers = set()

    def checked_read(rows):
        assert lock.acquire(blocking=False), "SparseRows was accessed concurrently"
        try:
            assert set(rows.role) == {reader_role}
            observations.extend(rows.cell_id.tolist())
            workers.add(threading.current_thread())
            return original_read(rows)
        finally:
            lock.release()

    sequences = []
    for prefetch in (False, True):
        observations.clear()
        workers.clear()
        with patch.object(tiny_data.reader.rows, "read", side_effect=checked_read), patch.object(
                tiny_data.reader, "normalize", side_effect=AssertionError("CUDA must normalize on GPU")):
            actual = stream_means(tiny_data, plan, batch_size=7, device="cuda:0", prefetch=prefetch)
        assert_same_population(actual, expected)
        metadata = json.loads(actual["metadata_json"].item())
        assert metadata["prefetch_batches"] == int(prefetch)
        if prefetch:
            assert len(workers) == 1 and threading.current_thread() not in workers
        else:
            assert workers == {threading.current_thread()}
        assert len(observations) == len(set(observations)) == sum(plan["pool_counts"].values())
        sequences.append(list(observations))
        assert all(not worker.is_alive() for worker in workers if worker is not threading.current_thread())
    assert sequences[0] == sequences[1]


@pytest.mark.parametrize("prefetch", [False, True])
def test_cuda_keeps_duplicate_and_role_guards(tiny_data, prefetch):
    plan = mean_plan(tiny_data, "T")
    row = tiny_data.reader.cell_metadata(plan["conditions"][0]["condition_id"], role="fit").iloc[0]
    for seen in ({"seen_physical": {(str(row.matrix_path), int(row.matrix_row))}},
                 {"seen_cells": {row.cell_id}}):
        with pytest.raises(ValueError, match="DUPLICATE_PHYSICAL_CELL"):
            stream_means(tiny_data, plan, batch_size=7, device="cuda:0", prefetch=prefetch, **seen)
    with patch.object(tiny_data.reader.rows, "read", side_effect=AssertionError("No E expression")):
        with pytest.raises(PermissionError, match="ONLY_T_V"):
            stream_means(tiny_data, {**plan, "role": "E"}, device="cuda:0", prefetch=prefetch)
        with patch.object(tiny_data.reader, "allowed_roles", frozenset({"fit", "selection", "final_test"})):
            with pytest.raises(PermissionError, match="T_V_ONLY"):
                stream_means(tiny_data, plan, device="cuda:0", prefetch=prefetch)


@pytest.mark.parametrize("failure_origin", ["reader", "consumer"])
def test_prefetch_worker_stops_after_error_and_fresh_run_matches(tiny_data, failure_origin):
    plan = mean_plan(tiny_data, "T")
    original_read = tiny_data.reader.rows.read
    workers, calls = set(), []

    def failing_read(rows):
        workers.add(threading.current_thread())
        calls.append(len(rows))
        if failure_origin == "reader" and len(calls) == 2:
            raise RuntimeError("synthetic reader failure")
        return original_read(rows)

    def failing_progress(label, processed, total):
        if failure_origin == "consumer" and processed > 0:
            raise RuntimeError("synthetic consumer failure")

    with patch.object(tiny_data.reader.rows, "read", side_effect=failing_read):
        with pytest.raises(RuntimeError, match=f"synthetic {failure_origin} failure"):
            stream_means(tiny_data, plan, batch_size=7, device="cuda:0", prefetch=True,
                         progress=failing_progress)
    assert workers
    assert all(not worker.is_alive() for worker in workers if worker is not threading.current_thread())
    actual = stream_means(tiny_data, plan, batch_size=7, device="cuda:0", prefetch=True)
    expected = stream_means(tiny_data, plan, batch_size=7, device="cpu")
    assert_same_population(actual, expected)
