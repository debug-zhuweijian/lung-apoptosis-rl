"""Lossless, memory-mapped T cache for repeated GPU graph passes."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .common import atomic_json, digest_json, inside, sha256


def prefetched(items, load, workers=2):
    """Bounded lookahead, preserving part order and floating-point reduction order."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(workers):
            item = next(iterator, None)
            if item is not None:
                pending.append(executor.submit(load, item))
        while pending:
            result = pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(executor.submit(load, item))
            yield result


def _atomic_array(path, value):
    temporary = path.with_suffix('.tmp.npy')
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def build_stream_cache(assets, index, weights, progress=None):
    """Decompress each verified CSR part once, retaining exact float64 values.

    The original compressed cache stays intact. A per-part receipt is published
    last, so interruption never presents partially written arrays as complete.
    Existing receipts are hash checked before any mmap is consumed.
    """
    folder = assets / 'fit_cache_stream'
    folder.mkdir(exist_ok=True)
    identity = digest_json({'index': index, 'weights': weights, 'schema': 1})
    total = len(index['parts'])
    if progress:
        progress('gpu_stream_cache', 0, total)
    records = []
    for number, part in enumerate(index['parts']):
        stem = folder / f'part_{number:06d}'
        receipt = stem.with_suffix('.json')
        part_identity = digest_json({'ordinal': number, 'part': part})
        if receipt.exists():
            record = json.loads(receipt.read_text(encoding='utf-8'))
            if (record['identity'] != identity or record.get('part_identity') != part_identity
                    or record['shape'][0] != part['rows']
                    or set(record['arrays']) != {'data', 'indices', 'indptr', 'weights'}):
                raise ValueError('GPU_STREAM_CACHE_IDENTITY_MISMATCH')
            for key, item in record['arrays'].items():
                if item['path'] != stem.name + f'_{key}.npy':
                    raise ValueError('GPU_STREAM_CACHE_PART_PATH_MISMATCH')
                path = inside(folder, item['path'])
                if not path.exists() or sha256(path) != item['sha256']:
                    raise ValueError('GPU_STREAM_CACHE_HASH_MISMATCH')
        else:
            matrix = sparse.load_npz(inside(assets, part['matrix'])).tocsr()
            ids = np.load(inside(assets, part['conditions']), allow_pickle=False)
            if matrix.shape[0] != part['rows'] or len(ids) != part['rows']:
                raise ValueError('GPU_STREAM_CACHE_ROW_MISMATCH')
            arrays = {'data': matrix.data, 'indices': matrix.indices,
                      'indptr': matrix.indptr,
                      'weights': np.asarray([weights[key] for key in ids], dtype=np.float64)}
            files = {}
            for key, value in arrays.items():
                path = stem.with_name(stem.name + f'_{key}.npy')
                _atomic_array(path, value)
                files[key] = {'path': path.name, 'sha256': sha256(path)}
            record = {'identity': identity, 'part_identity': part_identity,
                      'shape': list(matrix.shape), 'arrays': files}
            atomic_json(receipt, record)
        records.append(record)
        if progress:
            progress('gpu_stream_cache', number + 1, total)
    return StreamParts(folder, records)


class StreamParts:
    def __init__(self, folder, records):
        self.folder, self.records = Path(folder), records
        self.part_count = len(records)

    def _read(self, record):
        arrays = {key: np.load(inside(self.folder, item['path']), mmap_mode='r', allow_pickle=False)
                  for key, item in record['arrays'].items()}
        matrix = sparse.csr_matrix((arrays['data'], arrays['indices'], arrays['indptr']),
                                   shape=tuple(record['shape']), copy=False)
        return matrix, arrays['weights']

    def __call__(self):
        return prefetched(self.records, self._read)
