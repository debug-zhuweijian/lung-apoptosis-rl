"""CUDA preprocessing equivalence, integrity and interrupted preparation tests."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
import torch

from src.lung.data.common import sha256
from src.lung.data.gpu_preprocess import cross_moment_cuda, weighted_moments_cuda
from src.lung.data.preprocess import exact_neighbor_graph, weighted_moments
from src.lung.data.stream_cache import build_stream_cache
from src.lung.data import prepare_training
from test_training_data_v1 import tiny_bundle


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def fixture():
    rng = np.random.default_rng(457)
    x = rng.integers(0, 7, (73, 41)).astype(np.float64)
    x[rng.random(x.shape) < .7] = 0
    x[:, 0] = 0
    x[:, 1] = 3
    x[:, 5] = x[:, 3]
    w = rng.uniform(.1, 1, len(x))
    w /= w.sum()
    matrix = sparse.csr_matrix(x)
    def parts():
        for start in range(0, len(x), 17):
            yield matrix[start:start+17], w[start:start+17]
    return matrix, w, parts


@CUDA
def test_cuda_statistics_and_cross_moments_are_float64():
    matrix, weights, parts = fixture()
    cpu = weighted_moments(parts(), matrix.shape[1])
    gpu = weighted_moments_cuda(parts(), matrix.shape[1], 'cuda:0')
    for a, b in zip(cpu, gpu):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    cross = cross_moment_cuda(parts(), 7, 39, matrix.shape[1], 'cuda:0')
    assert cross.dtype == torch.float64 and cross.is_cuda
    expected = (matrix[:, 7:39].T @ matrix.multiply(weights[:, None])).toarray()
    np.testing.assert_allclose(cross.cpu().numpy().T, expected, atol=1e-12, rtol=1e-12)


@CUDA
def test_cuda_graph_matches_cpu_ties_zero_variance_and_resume(tmp_path):
    matrix, weights, parts = fixture()
    mean, _, sd, _ = weighted_moments(parts(), matrix.shape[1])
    ids = np.array([f'gene_{i:03d}' for i in range(matrix.shape[1])])[::-1].copy()
    expected = exact_neighbor_graph(parts, mean, sd, ids, neighbors=6, block_size=7)
    # Model the user's existing first CPU block and interrupted CUDA continuation.
    np.savez(tmp_path / 'neighbors_000000.npz', neighbors=expected[:7])
    old_hash = sha256(tmp_path / 'neighbors_000000.npz')
    progress = []
    result = exact_neighbor_graph(parts, mean, sd, ids, neighbors=6, block_size=7,
        output_dir=tmp_path, device='cuda:0', gpu_width=21, total_parts=5,
        progress=lambda *args: progress.append(args))
    np.testing.assert_array_equal(result, expected)
    assert old_hash == sha256(tmp_path / 'neighbors_000000.npz')
    assert any('cuda' in stage and done == 0 for stage, done, _ in progress)
    assert any('cuda' in stage and 0 < done < total for stage, done, total in progress)
    def should_not_read():
        raise AssertionError('Complete graph must be reused')
    resumed = exact_neighbor_graph(should_not_read, mean, sd, ids, neighbors=6, block_size=7,
                                   output_dir=tmp_path, device='cuda:0', gpu_width=14)
    np.testing.assert_array_equal(resumed, result)


@CUDA
def test_partial_gpu_graph_continues_after_keyboard_interrupt(tmp_path):
    matrix, _, parts = fixture()
    mean, _, sd, _ = weighted_moments(parts(), matrix.shape[1])
    ids = [f'g{i:03d}' for i in range(matrix.shape[1])]
    def interrupt(stage, done, total):
        if stage == 'gene_graph_save_1_of_2' and done == 7:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        exact_neighbor_graph(parts, mean, sd, ids, neighbors=3, block_size=7,
                             output_dir=tmp_path, device='cuda:0', gpu_width=21, progress=interrupt)
    assert (tmp_path / 'neighbors_000000.npz').exists()
    resumed = exact_neighbor_graph(parts, mean, sd, ids, neighbors=3, block_size=7,
                                  output_dir=tmp_path, device='cuda:0', gpu_width=14)
    expected = exact_neighbor_graph(parts, mean, sd, ids, neighbors=3, block_size=7)
    np.testing.assert_array_equal(resumed, expected)


def test_lossless_stream_cache_reuses_arrays_and_detects_corruption(tmp_path):
    matrix, _, _ = fixture()
    sparse.save_npz(tmp_path / 'rows.npz', matrix)
    np.save(tmp_path / 'conditions.npy', np.array(['a'] * len(matrix.indptr[:-1])))
    index = {'parts': [{'matrix': 'rows.npz', 'conditions': 'conditions.npy',
                        'rows': matrix.shape[0]}]}
    progress = []
    parts = build_stream_cache(tmp_path, index, {'a': .125}, lambda *a: progress.append(a))
    loaded, w = next(iter(parts()))
    np.testing.assert_array_equal(loaded.toarray(), matrix.toarray())
    np.testing.assert_array_equal(w, np.full(matrix.shape[0], .125))
    with patch('src.lung.data.stream_cache.sparse.load_npz', side_effect=AssertionError('No decompression')):
        loaded2, _ = next(iter(build_stream_cache(tmp_path, index, {'a': .125})()))
        np.testing.assert_array_equal(loaded2.toarray(), matrix.toarray())
    with pytest.raises(ValueError, match='IDENTITY'):
        build_stream_cache(tmp_path, index, {'a': .25})
    del loaded, loaded2, w  # release Windows mmap handles before corruption check
    path = tmp_path / 'fit_cache_stream' / 'part_000000_data.npy'
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='HASH'):
        build_stream_cache(tmp_path, index, {'a': .125})
    assert progress[0] == ('gpu_stream_cache', 0, 1)


@CUDA
def test_tiny_full_preparation_cuda_keeps_frozen_dataset_configuration(tmp_path):
    tiny_bundle(tmp_path)
    assets = prepare_training(tmp_path, build_preprocessing=False)
    old = json.loads((assets / 'manifest.json').read_text())
    original = sha256(tmp_path / 'source.npz')
    events = []
    prepare_training(tmp_path, device='cuda:0', progress=lambda *args: events.append(args))
    new = json.loads((assets / 'manifest.json').read_text())
    assert new['status'] == 'complete'
    assert new['signature'] == old['signature']
    assert new['preprocessing']['execution_device'] == 'cuda:0'
    assert new['preprocessing']['statistics_dtype'] == 'float64'
    assert sha256(tmp_path / 'source.npz') == original
    assert not (assets / '.prepare.lock').exists()
    assert any(stage == 'normalization_cuda' for stage, *_ in events)


@CUDA
def test_legacy_normalization_preserved_and_receipt_reused(tmp_path):
    tiny_bundle(tmp_path)
    assets = prepare_training(tmp_path)
    norm_hash = sha256(assets / 'normalization.npz')
    from src.lung.data.preprocess import build_preprocessing
    from src.lung.data.common import dataset_config
    build_preprocessing(tmp_path, assets, dataset_config(tmp_path), device='cuda:0')
    assert sha256(assets / 'normalization.npz') == norm_hash
    with patch('src.lung.data.gpu_preprocess.weighted_moments_cuda', side_effect=AssertionError('No refit')):
        build_preprocessing(tmp_path, assets, dataset_config(tmp_path), device='cuda:0')


@CUDA
def test_invalid_weights_and_corrupt_graph_rejected(tmp_path):
    matrix, weights, parts = fixture()
    mean, _, sd, _ = weighted_moments(parts(), matrix.shape[1])
    with pytest.raises(ValueError, match='weights must sum'):
        cross_moment_cuda([(matrix, weights * 2)], 0, 7, matrix.shape[1], 'cuda:0')
    bad = np.zeros((7, 4), dtype=np.int64)
    np.savez(tmp_path / 'neighbors_000000.npz', neighbors=bad)
    with pytest.raises(ValueError, match='Incompatible'):
        exact_neighbor_graph(parts, mean, sd, np.arange(matrix.shape[1]), 3, 7,
                             output_dir=tmp_path, device='cuda:0')
