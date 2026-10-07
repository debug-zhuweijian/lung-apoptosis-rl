"""Bounded real-cache CPU/CUDA comparison, without preparing production assets."""

from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
from scipy import sparse
import torch

from lung.data.common import atomic_json, inside, sha256
from lung.data.preprocess import weighted_moments, exact_neighbor_graph
from lung.data.stream_cache import build_stream_cache
from lung.data.gpu_preprocess import cross_moment_cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', type=int, default=8)
    parser.add_argument('--output', default='logs/environment_checks/gpu_preprocessing')
    args = parser.parse_args()
    if not 1 <= args.parts <= 32:
        raise ValueError('This bounded check supports 1..32 parts')
    assets = ROOT / '08-datasets/LungCancer_A549_Integration/training-v1'
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((assets / 'fit_cache/index.json').read_text(encoding='utf-8'))
    indices = np.linspace(0, len(source['parts']) - 1, args.parts, dtype=int)
    selected = [source['parts'][i] for i in indices]
    for part in selected:
        for key in ('matrix', 'conditions'):
            if sha256(inside(assets, part[key])) != part[key + '_sha256']:
                raise ValueError('Source cache changed')
    # Equal-per-cell weights are a benchmark fixture, not production T weights.
    rows = sum(part['rows'] for part in selected)
    mapping = {}
    for part in selected:
        for condition in np.load(inside(assets, part['conditions']), allow_pickle=False):
            mapping[str(condition)] = 1. / rows
    def compressed_parts():
        for part in selected:
            yield sparse.load_npz(inside(assets, part['matrix'])), np.full(part['rows'], 1. / rows)
    # Isolate the acceleration cache; never alter the user's training-v1.
    temporary_index = {'parts': [{**part, 'matrix': str(inside(assets, part['matrix'])),
                                 'conditions': str(inside(assets, part['conditions']))}
                                for part in selected]}
    # build_stream_cache deliberately confines input paths to its assets root;
    # copy only the tiny selected benchmark fixtures into this check directory.
    import shutil
    bounded = output / 'fixture'
    bounded.mkdir(exist_ok=True)
    for i, part in enumerate(selected):
        for key, suffix in (('matrix', '.npz'), ('conditions', '.npy')):
            path = bounded / f'{key}_{i:04d}{suffix}'
            if not path.exists():
                shutil.copy2(inside(assets, part[key]), path)
            if sha256(path) != part[key + '_sha256']:
                raise ValueError('Benchmark fixture changed')
            temporary_index['parts'][i][key] = path.name
    t0 = time.perf_counter()
    streams = build_stream_cache(bounded, temporary_index, mapping)
    prepare_seconds = time.perf_counter() - t0
    mean, _, sd, _ = weighted_moments(compressed_parts(), 32461)
    ids = np.asarray([f'g{i:06d}' for i in range(32461)])
    torch.cuda.init()
    torch.cuda.synchronize()
    report = {'check': 'bounded_real_expression_full_gene_graph', 'formal_training': False,
              'optimizer_steps': 0, 'genes': 32461, 'parts': args.parts, 'rows': rows,
              'weights': 'benchmark_equal_cell_not_production_T_weights',
              'stream_cache_build_seconds': prepare_seconds,
              'device': torch.cuda.get_device_name(0)}
    for backend in ('cpu', 'cuda:0'):
        if backend.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        graph = exact_neighbor_graph(compressed_parts if backend == 'cpu' else streams,
             mean, sd, ids, block_size=128, device=backend, total_parts=args.parts)
        if backend.startswith('cuda'):
            torch.cuda.synchronize()
        report[backend + '_seconds'] = time.perf_counter() - start
        if backend == 'cpu':
            expected = graph
            np.save(output / 'cpu_graph.npy', graph)
        else:
            np.save(output / 'gpu_graph.npy', graph)
            report['graph_equal'] = bool(np.array_equal(expected, graph))
            report['different_rows'] = int(np.any(expected != graph, axis=1).sum())
            report['peak_gpu_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps({backend: report[backend + '_seconds']}, ensure_ascii=False), flush=True)
    cpu_cross = sum((x[:, :128].T @ x.multiply(w[:, None])).toarray() for x, w in compressed_parts())
    gpu_cross = cross_moment_cuda(streams(), 0, 128, 32461, 'cuda:0').cpu().numpy().T
    report['max_abs_cross_error'] = float(np.max(np.abs(cpu_cross - gpu_cross)))
    report['observed_speedup'] = report['cpu_seconds'] / report['cuda:0_seconds']
    atomic_json(output / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['graph_equal']:
        raise AssertionError('Investigate graph ordering before adopting faster execution')


if __name__ == '__main__':
    main()
