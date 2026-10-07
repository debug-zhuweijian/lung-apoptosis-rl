"""Float64 CUDA weighted statistics and exact, cell-level Pearson graph.

GPU execution changes the tiling, not the estimator, gene axis, fitting rows,
or tie rule. CPU and CUDA reductions need not be bitwise identical.
"""

from pathlib import Path
import json

import numpy as np
import torch

from .common import atomic_json, digest_json, sha256


def cuda_device(device):
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA_PREPROCESSING_UNAVAILABLE: no silent CPU fallback')
    return selected


def csr_tensor(matrix, device):
    matrix = matrix.tocsr()
    matrix.check_format(full_check=True)
    # copy avoids exposing a read-only mmap to torch; only sparse arrays move.
    return torch.sparse_csr_tensor(
        torch.tensor(matrix.indptr, device=device, dtype=torch.int64),
        torch.tensor(matrix.indices, device=device, dtype=torch.int64),
        torch.tensor(matrix.data, device=device, dtype=torch.float64),
        size=matrix.shape, device=device, dtype=torch.float64, check_invariants=False)


@torch.no_grad()
def weighted_moments_cuda(parts, n_genes, device, progress=None, total_parts=None):
    device = cuda_device(device)
    first = torch.zeros(n_genes, dtype=torch.float64, device=device)
    second = torch.zeros_like(first)
    mass = 0.
    if progress:
        progress('normalization_cuda', 0, total_parts or 1)
    for number, (values, weights) in enumerate(parts, 1):
        weights = np.asarray(weights, dtype=np.float64)
        if (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError('Invalid sample weight')
        transpose = csr_tensor(values.T.tocsr(), device)
        weight = torch.tensor(weights, device=device)[:, None]
        first.add_(torch.sparse.mm(transpose, weight).squeeze(1))
        squared = torch.sparse_csr_tensor(transpose.crow_indices(), transpose.col_indices(),
                                          transpose.values().square(), size=transpose.shape,
                                          device=device, dtype=torch.float64, check_invariants=False)
        second.add_(torch.sparse.mm(squared, weight).squeeze(1))
        mass += weights.sum()
        if progress:
            progress('normalization_cuda', number, total_parts or number)
    if mass <= 0:
        raise ValueError('No usable fitting rows')
    mean = first / mass
    variance = (second / mass - mean.square()).clamp_min_(0)
    variance[variance < 1e-14] = 0
    sd = variance.sqrt()
    return mean.cpu().numpy(), sd.clamp_min(.1).cpu().numpy(), sd.cpu().numpy(), float(mass)


def gpu_block_width(n_genes, saved_block_size, device):
    """At most 4096 queries; reserve ample space for desktop use and CUDA work."""
    free, _ = torch.cuda.mem_get_info(device)
    # Cross accumulator + possible library workspace, with >=1GiB spare.
    usable = max(0, min(free * .6, free - 1024**3))
    width = int(usable / (n_genes * 8 * 2))
    width = min(4096, width, n_genes)
    if width < min(saved_block_size, n_genes):
        raise RuntimeError('INSUFFICIENT_GPU_MEMORY_FOR_EXACT_GRAPH: close other GPU workloads')
    return max(saved_block_size, (width // saved_block_size) * saved_block_size)


def _load_block(path, start, stop, genes, degree):
    with np.load(path, allow_pickle=False) as saved:
        block = saved['neighbors']
    if (block.shape != (stop - start, degree) or not np.issubdtype(block.dtype, np.integer)
            or (block < 0).any() or (block >= genes).any()
            or not np.array_equal(block[:, 0], np.arange(start, stop))
            or (np.diff(np.sort(block, axis=1), axis=1) == 0).any()):
        raise ValueError('Incompatible partial graph block')
    return block


def _pending_spans(starts, n_genes, block_size, gpu_width):
    spans = []
    for start in starts:
        stop = min(start + block_size, n_genes)
        if spans and spans[-1][1] == start and stop - spans[-1][0] <= gpu_width:
            spans[-1] = (spans[-1][0], stop)
        else:
            spans.append((start, stop))
    return spans


@torch.no_grad()
def cross_moment_cuda(parts, start, stop, n_genes, device, progress=None, total_parts=None,
                      *, return_metadata=False):
    """Return [all genes, query genes] sum_i w_i x_i x_i^T, in float64."""
    cross = torch.zeros((n_genes, stop - start), device=device, dtype=torch.float64)
    mass = 0.
    part_count, max_rows, total_rows = 0, 0, 0
    for number, (values, weights) in enumerate(parts, 1):
        part_count, max_rows = number, max(max_rows, values.shape[0])
        total_rows += values.shape[0]
        weights = np.asarray(weights, dtype=np.float64)
        if (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError('Invalid sample weight')
        if not np.isfinite(values.data).all() or (values.data < 0).any():
            raise ValueError('Graph requires finite nonnegative log expression')
        transpose = csr_tensor(values.T.tocsr(), device)
        right = torch.tensor(values[:, start:stop].toarray(), device=device, dtype=torch.float64)
        right.mul_(torch.tensor(weights, device=device)[:, None])
        # In-place SpMM avoids a second dense product and add per cached part.
        torch.addmm(cross, transpose, right, beta=1., alpha=1., out=cross)
        mass += float(weights.sum())
        if progress:
            progress(number, total_parts or number)
    if not np.isclose(mass, 1., atol=1e-9):
        raise ValueError('Graph fitting weights must sum to one')
    return (cross, {'max_rows': max_rows, 'part_count': part_count,
                    'total_rows': total_rows}) if return_metadata else cross


def _refine_close_rows(part_factory, candidates, mean, sd, gene_ids, neighbors, progress=None,
                       total_parts=None):
    """Re-evaluate only candidates that can reach top-k under the error bound.

    Groups of queries share a small union of possible candidates. Source parts
    are visited once; no full G-column CPU product is needed for ordinary ties.
    """
    queries = sorted(candidates)
    groups = []
    for start in range(0, len(queries), 64):
        selected = np.asarray(queries[start:start+64], dtype=np.int64)
        columns = np.unique(np.concatenate([candidates[q] for q in selected]))
        groups.append((selected, columns, np.zeros((len(selected), len(columns)), dtype=np.float64)))
    mass = 0.
    for number, (values, weights) in enumerate(part_factory(), 1):
        columns = values.tocsc()
        weighted = values.multiply(weights[:, None]).tocsc()
        for selected, candidate_columns, cross in groups:
            cross += (columns[:, selected].T @ weighted[:, candidate_columns]).toarray()
        mass += float(np.sum(weights))
        if progress:
            progress(number, total_parts or number)
    if not np.isclose(mass, 1., atol=1e-9):
        raise ValueError('Graph fitting weights must sum to one')
    blocks = {}
    for selected, columns, cross in groups:
        cross -= mean[selected, None] * mean[columns][None, :]
        divisor = sd[selected, None] * sd[columns][None, :]
        corr = np.divide(cross, divisor, out=np.zeros_like(cross), where=divisor > 0)
        np.clip(corr, -1., 1., out=corr)
        for local, query in enumerate(selected):
            eligible = np.asarray(candidates[query])
            eligible = eligible[np.argsort(gene_ids[eligible], kind='stable')]
            positions = np.searchsorted(columns, eligible)
            order = np.argsort(-np.abs(corr[local, positions]), kind='stable')
            blocks[int(query)] = np.r_[query, eligible[order[:neighbors]]]
    return blocks


@torch.no_grad()
def exact_neighbor_graph_cuda(part_factory, mean, sd, gene_ids, neighbors=32,
                              block_size=128, output_dir=None, progress=None,
                              *, device='cuda:0', gpu_width=None, total_parts=None,
                              input_identity=None):
    device = cuda_device(device)
    ids = np.asarray(gene_ids).astype(str)
    genes = len(ids)
    if len(set(ids)) != genes or not 0 < neighbors < genes or block_size < 1:
        raise ValueError('Unique genes and 0 < neighbors < gene_count are required')
    output = np.empty((genes, neighbors + 1), dtype=np.int64)
    directory = Path(output_dir) if output_dir is not None else None
    identity = digest_json({'source': input_identity, 'genes': ids.tolist(),
                            'mean': np.asarray(mean).tolist(), 'sd': np.asarray(sd).tolist(),
                            'neighbors': neighbors, 'block_size': block_size,
                            'definition': 'exact_cell_weighted_pearson_id_ties_v1',
                            'ranking_policy': 'fp64_total_rows_bound_cpu_candidate_refine_v1'})
    receipts = {}
    receipt_path = directory / 'cuda_receipt.json' if directory else None
    if receipt_path is not None and receipt_path.exists():
        receipts = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipts['identity'] != identity:
            raise ValueError('GPU_GRAPH_INPUT_IDENTITY_MISMATCH')
    records = dict(receipts.get('blocks', {}))
    pending = []
    for start in range(0, genes, block_size):
        stop = min(start + block_size, genes)
        path = directory / f'neighbors_{start:06d}.npz' if directory else None
        if path is not None and path.exists():
            if path.name in records and sha256(path) != records[path.name]['sha256']:
                raise ValueError('GPU_GRAPH_BLOCK_HASH_MISMATCH')
            output[start:stop] = _load_block(path, start, stop, genes, neighbors + 1)
            if path.name not in records:
                with np.load(path, allow_pickle=False) as saved:
                    backend = str(saved['backend']) if 'backend' in saved.files else 'legacy_cpu'
                    if backend != 'legacy_cpu' and (
                            'input_identity' not in saved.files or str(saved['input_identity']) != identity):
                        raise ValueError('GPU_GRAPH_ORPHAN_IDENTITY_MISMATCH')
                # Old CPU blocks are adopted explicitly. A CUDA block whose
                # receipt was interrupted must carry the matching identity.
                records[path.name] = {'sha256': sha256(path), 'verification':
                    'legacy_structure_adopted' if backend == 'legacy_cpu' else 'cuda_receipt_recovered'}
        else:
            if path is not None and path.name in records:
                raise ValueError('GPU_GRAPH_RECORDED_BLOCK_MISSING')
            pending.append(start)
    if receipt_path is not None:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt_path, {'identity': identity, 'blocks': records})
    if not pending:
        if progress:
            progress('gene_graph', genes, genes)
        return output
    width = gpu_width or gpu_block_width(genes, block_size, device)
    if width < min(block_size, genes):
        raise ValueError('GPU block must accommodate a saved block')
    spans = _pending_spans(pending, genes, block_size, width)
    total_parts = total_parts or getattr(part_factory, 'part_count', None)
    mu = torch.tensor(mean, dtype=torch.float64, device=device)
    sigma = torch.tensor(sd, dtype=torch.float64, device=device)
    by_id = torch.tensor(np.argsort(ids, kind='stable'), device=device)
    for index, (start, stop) in enumerate(spans, 1):
        phase = f'gene_graph_cuda_{index}_of_{len(spans)}'
        if progress:
            progress(phase, 0, total_parts or 1)
        callback = (lambda done, total: progress(phase, done, total)) if progress else None
        cross, reduction = cross_moment_cuda(part_factory(), start, stop, genes, device, callback,
                                              total_parts, return_metadata=True)
        phase = f'gene_graph_select_cuda_{index}_of_{len(spans)}'
        if progress:
            progress(phase, 0, stop - start)
        chunks, ambiguous = [], {}
        # Conservatively cover both reductions and centering/division. Using
        # a bound only routes work for exact CPU re-evaluation; it never rounds
        # scores or declares numerically close correlations to be tied.
        unit = np.finfo(np.float64).eps / 2
        # Use total rows: no assumption about vendor SpMM's internal reduction tree.
        depth = reduction['total_rows'] + reduction['part_count'] + 8
        gamma = depth * unit / (1 - depth * unit)
        rounding = 8 * gamma / (1 - gamma)
        for first in range(start, stop, block_size):
            last = min(first + block_size, stop)
            corr = cross[:, first-start:last-start].T.contiguous()
            center = mu[first:last, None] * mu[None, :]
            divisor = sigma[first:last, None] * sigma[None, :]
            error_bound = torch.where(divisor > 0,
                rounding * (corr.abs() + center.abs()) / divisor.clamp_min(torch.finfo(torch.float64).tiny)
                + 16 * unit, 0.)
            bound = error_bound.amax(dim=1)
            corr.sub_(center)
            corr = torch.where(divisor > 0, corr / divisor.clamp_min(torch.finfo(torch.float64).tiny), 0.)
            corr.clamp_(-1., 1.).abs_()
            row = torch.arange(last - first, device=device)
            own = torch.arange(first, last, device=device)
            corr[row, own] = -torch.inf
            # Stable sort after ID ordering gives exact ID ties, without jitter.
            order = torch.argsort(corr[:, by_id], dim=1, descending=True, stable=True)[:, :neighbors+1]
            ranked = corr.gather(1, by_id[order])
            gaps = ranked[:, :-1] - ranked[:, 1:]
            close = (gaps <= 2 * bound[:, None]).any(dim=1) & (sigma[first:last] > 0)
            close_indices = torch.nonzero(close, as_tuple=False).flatten()
            thresholds = ranked[close_indices, neighbors-1] - 2 * bound[close_indices]
            possible = (corr[close_indices] >= thresholds[:, None]).cpu().numpy()
            for position, local in enumerate(close_indices.cpu().tolist()):
                ambiguous[first + local] = np.flatnonzero(possible[position])
            block = torch.cat((own[:, None], by_id[order[:, :neighbors]]), dim=1).cpu().numpy()
            chunks.append((first, last, block))
            if progress:
                progress(phase, last - start, stop - start)
        del cross, corr, divisor, error_bound
        if ambiguous:
            phase = f'gene_graph_close_rank_check_{index}_of_{len(spans)}'
            if progress:
                progress(phase, 0, total_parts or 1)
            callback = (lambda done, total: progress(phase, done, total)) if progress else None
            refined = _refine_close_rows(part_factory, ambiguous, np.asarray(mean),
                np.asarray(sd), ids, neighbors, callback, total_parts)
        else:
            refined = {}
        phase = f'gene_graph_save_{index}_of_{len(spans)}'
        if progress:
            progress(phase, 0, stop - start)
        for first, last, block in chunks:
            for query in range(first, last):
                if query in refined:
                    block[query-first] = refined[query]
            output[first:last] = block
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f'neighbors_{first:06d}.npz'
                temporary = path.with_suffix('.tmp.npz')
                np.savez(temporary, neighbors=block, backend='cuda_float64', input_identity=identity)
                temporary.replace(path)
                records[path.name] = {'sha256': sha256(path), 'verification': 'cuda_computed',
                    'cpu_refined_rows': sum(query in refined for query in range(first, last))}
                atomic_json(receipt_path, {'identity': identity, 'blocks': records})
            if progress:
                progress(phase, last - start, stop - start)
    if progress:
        progress('gene_graph', genes, genes)
    return output
