"""Independent CPU-only checks of CUDA preprocessing cache identity guards.

Cached graph checks replace only the CUDA device resolver and never execute
CUDA arithmetic; numerical GPU equivalence is covered in the companion tests.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
import torch

from src.lung.data.common import sha256
from src.lung.data.gpu_preprocess import exact_neighbor_graph_cuda, _refine_close_rows
from src.lung.data.preprocess import _normalization, exact_neighbor_graph, weighted_moments
from src.lung.data.stream_cache import build_stream_cache


@pytest.fixture
def cache_fixture(tmp_path):
    matrices = [sparse.csr_matrix(np.array([[1., 0., 2.], [2., 1., 0.]])),
                sparse.csr_matrix(np.array([[0., 3., 1.], [1., 0., 4.]]))]
    records = []
    for ordinal, matrix in enumerate(matrices):
        matrix_path = tmp_path / f'rows_{ordinal}.npz'
        condition_path = tmp_path / f'conditions_{ordinal}.npy'
        sparse.save_npz(matrix_path, matrix)
        np.save(condition_path, np.array(['a', 'a']), allow_pickle=False)
        records.append({'matrix': matrix_path.name, 'conditions': condition_path.name,
                        'rows': 2, 'matrix_sha256': sha256(matrix_path),
                        'conditions_sha256': sha256(condition_path)})
    index = {'parts': records, 'role': 'fit', 'valid_condition_counts': {'a': 4}}
    weights = {'a': .25}
    def parts():
        for matrix in matrices:
            yield matrix, np.full(2, .25, dtype=np.float64)
    return tmp_path, index, weights, parts


def test_valid_receipt_from_another_part_cannot_be_reused(cache_fixture):
    root, index, weights, _ = cache_fixture
    build_stream_cache(root, index, weights)
    folder = root / 'fit_cache_stream'
    first = folder / 'part_000000.json'
    second = folder / 'part_000001.json'
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match='GPU_STREAM_CACHE_IDENTITY_MISMATCH'):
        build_stream_cache(root, index, weights)


def test_part_array_cannot_point_to_another_valid_part(cache_fixture):
    root, index, weights, _ = cache_fixture
    build_stream_cache(root, index, weights)
    folder = root / 'fit_cache_stream'
    first = json.loads((folder / 'part_000000.json').read_text())
    second_path = folder / 'part_000001.json'
    second = json.loads(second_path.read_text())
    second['arrays']['data'] = first['arrays']['data']
    second_path.write_text(json.dumps(second))
    with pytest.raises(ValueError, match='GPU_STREAM_CACHE_PART_PATH_MISMATCH'):
        build_stream_cache(root, index, weights)


@pytest.mark.parametrize('change', ['weights', 'genes', 'source_index'])
def test_normalization_receipt_rejects_changed_inputs(cache_fixture, change):
    root, index, weights, parts = cache_fixture
    genes = np.array(['gene_a', 'gene_b', 'gene_c'])
    _normalization(root, parts, index, weights, genes, 'cpu', None)
    if change == 'weights':
        weights = {'a': .5}
    elif change == 'genes':
        genes = genes[::-1]
    else:
        index = {**index, 'role': 'other_fit_version'}
    with patch('src.lung.data.preprocess.weighted_moments',
               side_effect=AssertionError('Identity change must fail before recomputation')):
        with pytest.raises(ValueError, match='NORMALIZATION_CACHE_IDENTITY_MISMATCH'):
            _normalization(root, parts, index, weights, genes, 'cpu', None)


def test_normalization_receipt_rejects_valid_but_changed_npz(cache_fixture):
    root, index, weights, parts = cache_fixture
    genes = np.array(['gene_a', 'gene_b', 'gene_c'])
    mean, scale, sd, mass = _normalization(root, parts, index, weights, genes, 'cpu', None)
    np.savez(root / 'normalization.npz', mean=mean + .001, scale=scale,
             pearson_sd=sd, fit_weight_sum=mass, gene_ids=genes)
    with pytest.raises(ValueError, match='NORMALIZATION_CACHE_IDENTITY_MISMATCH'):
        _normalization(root, parts, index, weights, genes, 'cpu', None)


@pytest.fixture
def saved_graph(cache_fixture):
    root, _, _, parts = cache_fixture
    mean, _, sd, _ = weighted_moments(parts(), 3)
    genes = np.array(['gene_c', 'gene_a', 'gene_b'])
    folder = root / 'graph'
    graph = exact_neighbor_graph(parts, mean, sd, genes, neighbors=2,
                                 block_size=1, output_dir=folder)
    return folder, parts, mean, sd, genes, graph


def adopt_without_gpu(saved_graph, **overrides):
    folder, parts, mean, sd, genes, _ = saved_graph
    args = {'mean': mean, 'sd': sd, 'gene_ids': genes,
            'input_identity': 'frozen_T_v1'}
    args.update(overrides)
    with patch('src.lung.data.gpu_preprocess.cuda_device', return_value=torch.device('cpu')):
        with patch('src.lung.data.gpu_preprocess.cross_moment_cuda',
                   side_effect=AssertionError('Complete cache must not recompute')):
            return exact_neighbor_graph_cuda(parts, neighbors=2, block_size=1,
                                             output_dir=folder, **args)


def test_legacy_graph_adoption_records_its_actual_evidence(saved_graph):
    result = adopt_without_gpu(saved_graph)
    np.testing.assert_array_equal(result, saved_graph[-1])
    receipt = json.loads((saved_graph[0] / 'cuda_receipt.json').read_text())
    assert {item['verification'] for item in receipt['blocks'].values()} == {'legacy_structure_adopted'}


@pytest.mark.parametrize('change', ['mean', 'sd', 'gene_ids', 'input_identity'])
def test_graph_receipt_rejects_changed_inputs_before_reuse(saved_graph, change):
    adopt_without_gpu(saved_graph)
    replacements = {'mean': saved_graph[2] + .001,
                    'sd': saved_graph[3] + .001,
                    'gene_ids': saved_graph[4][::-1],
                    'input_identity': 'different_T_v2'}
    with pytest.raises(ValueError, match='GPU_GRAPH_INPUT_IDENTITY_MISMATCH'):
        adopt_without_gpu(saved_graph, **{change: replacements[change]})


def test_graph_receipt_rejects_structurally_valid_changed_neighbors(saved_graph):
    adopt_without_gpu(saved_graph)
    path = saved_graph[0] / 'neighbors_000000.npz'
    with np.load(path, allow_pickle=False) as values:
        changed = values['neighbors'].copy()
    changed[:, 1:] = changed[:, 1:][:, ::-1]
    np.savez(path, neighbors=changed)
    with pytest.raises(ValueError, match='GPU_GRAPH_BLOCK_HASH_MISMATCH'):
        adopt_without_gpu(saved_graph)


def test_graph_receipt_rejects_missing_recorded_block(saved_graph):
    adopt_without_gpu(saved_graph)
    (saved_graph[0] / 'neighbors_000000.npz').unlink()
    with pytest.raises(ValueError, match='GPU_GRAPH_RECORDED_BLOCK_MISSING'):
        adopt_without_gpu(saved_graph)


def _cpu_selector_probe(values, weights, fake_cross, neighbors, gene_ids):
    """Exercise production selection/refinement using injected CPU cross values."""
    matrix = sparse.csr_matrix(values)
    def parts():
        for start in range(0, len(values), 17):
            yield matrix[start:start + 17], weights[start:start + 17]
    mean, _, sd, _ = weighted_moments(parts(), values.shape[1])
    expected = exact_neighbor_graph(parts, mean, sd, gene_ids, neighbors=neighbors, block_size=7)
    def injected_cross(_parts, start, stop, _genes, device, *args, **kwargs):
        assert torch.device(device).type == 'cpu'
        assert kwargs.get('return_metadata') is True
        metadata = {'max_rows': min(17, len(values)), 'part_count': (len(values) + 16) // 17,
                    'total_rows': len(values)}
        return torch.from_numpy(fake_cross[:, start:stop].copy()), metadata
    with patch('src.lung.data.gpu_preprocess.cuda_device', return_value=torch.device('cpu')):
        with patch('src.lung.data.gpu_preprocess.cross_moment_cuda', side_effect=injected_cross):
            with patch('src.lung.data.gpu_preprocess._refine_close_rows', wraps=_refine_close_rows) as refine:
                actual = exact_neighbor_graph_cuda(parts, mean, sd, gene_ids,
                    neighbors=neighbors, block_size=7, gpu_width=values.shape[1])
                called_queries = [np.asarray(sorted(call.args[1]))
                                  if isinstance(call.args[1], dict) else call.args[1]
                                  for call in refine.call_args_list]
    np.testing.assert_array_equal(actual, expected)
    return mean, sd, called_queries


def test_zero_gpu_scores_with_nonzero_sd_are_rechecked_on_cpu():
    values = np.array([[1., 1., 1., 0.], [1., 0., 0., 0.],
                       [0., 1., 0., 0.], [0., 0., 1., 0.]])
    weights = np.array([.25 + 1e-16, .25 - 1e-16, .25, .25])
    matrix = sparse.csr_matrix(values)
    mean, _, sd, _ = weighted_moments([(matrix, weights)], 4)
    cpu_cross = (matrix.T @ matrix.multiply(weights[:, None])).toarray()
    fake_cross = mean[:, None] * mean[None, :]
    np.fill_diagonal(fake_cross, np.diag(cpu_cross))
    # The injected GPU-like result has exact zero nonself scores, while the
    # original CPU result has nonzero ~2e-16 scores that alter ID-only order.
    assert np.any(cpu_cross - fake_cross != 0)
    _, _, query_batches = _cpu_selector_probe(values, weights, fake_cross, 2,
                                             np.array(['z', 'y', 'a', 'b']))
    assert len(query_batches) == 1
    np.testing.assert_array_equal(query_batches[0], np.flatnonzero(sd > 0))


def test_near_constant_cross_roundoff_uses_original_cpu_ranking():
    rng = np.random.default_rng(5432)
    values = 1 + rng.normal(0, 1e-6, size=(83, 41))
    weights = rng.uniform(.1, 1, len(values))
    weights /= weights.sum()
    matrix = sparse.csr_matrix(values)
    cpu_cross = sum((matrix[i:i + 17].T @ matrix[i:i + 17].multiply(weights[i:i + 17, None])).toarray()
                    for i in range(0, len(values), 17))
    fake_cross = cpu_cross + rng.choice([-1., 1.], size=cpu_cross.shape) * np.finfo(np.float64).eps
    _, _, query_batches = _cpu_selector_probe(values, weights, fake_cross, 6,
                                             np.array([f'g{i:03d}' for i in range(41)]))
    assert len(query_batches) == 1
    assert len(query_batches[0]) > 0


def test_candidate_only_refinement_matches_original_across_query_groups():
    rng = np.random.default_rng(8931)
    values = rng.poisson(.4, size=(123, 91)).astype(np.float64)
    values[:, 3] = values[:, 4]
    values[:, 5] = 0
    weights = rng.uniform(.1, 1, len(values))
    weights /= weights.sum()
    matrix = sparse.csr_matrix(values)
    def parts():
        for start in range(0, len(values), 9):
            yield matrix[start:start + 9], weights[start:start + 9]
    mean, _, sd, _ = weighted_moments(parts(), values.shape[1])
    ids = np.array([f'g{i:03d}' for i in range(values.shape[1])])[::-1]
    expected = exact_neighbor_graph(parts, mean, sd, ids, neighbors=6, block_size=17)
    candidates = {q: np.unique(np.r_[expected[q, 1:], (q + 7) % 91, (q + 13) % 91])
                  for q in range(91)}
    refined = _refine_close_rows(parts, candidates, mean, sd, ids, 6)
    np.testing.assert_array_equal(np.stack([refined[q] for q in range(91)]), expected)
