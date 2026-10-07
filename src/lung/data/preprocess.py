"""T-only normalization and exact, bounded-memory Pearson neighborhoods.

The full correlation matrix is never allocated. Normalized T rows are cached
as CSR parts on disk; each feature block accumulates exact weighted moments.
CUDA uses larger compute tiles and a lossless mmap cache. The estimator stays
cell-level and exact (up to floating-point reduction), without subsampling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from .common import atomic_json, check_shard, digest_json, inside, sha256, source_shard_registry


def log_normalize(counts: sparse.csr_matrix):
    matrix = counts.astype(np.float64, copy=True).tocsr()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    total = np.asarray(matrix.sum(axis=1)).ravel()
    valid = total > 0
    factor = np.divide(10000., total, out=np.zeros_like(total), where=valid)
    matrix = matrix.multiply(factor[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix, valid


def standardized(counts, mean, scale):
    normalized, valid = log_normalize(counts)
    if not valid.all():
        raise ValueError("ZERO_COMMON_GENE_LIBRARY: unusable expression row")
    return ((normalized.toarray() - mean) / scale).astype(np.float32)


def weighted_moments(parts, n_genes, progress=None, total_parts=None):
    """Parts yield (CSR log-normalized rows, positive sample weights)."""
    first, second = np.zeros(n_genes), np.zeros(n_genes)
    mass = 0.
    if progress:
        progress('normalization_cpu', 0, total_parts or 1)
    for number, (values, weights) in enumerate(parts, 1):
        weights = np.asarray(weights, dtype=np.float64)
        if (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError("Invalid sample weight")
        first += np.asarray(values.T @ weights).ravel()
        second += np.asarray(values.power(2).T @ weights).ravel()
        mass += weights.sum()
        if progress:
            progress('normalization_cpu', number, total_parts or number)
    if mass <= 0:
        raise ValueError("No usable fitting rows")
    mean = first / mass
    variance = np.maximum(second / mass - mean ** 2, 0.)
    variance[variance < 1e-14] = 0.
    sd = np.sqrt(variance)
    return mean, np.maximum(sd, .1), sd, float(mass)


def exact_neighbor_graph(part_factory, mean, sd, gene_ids, neighbors=32,
                         block_size=128, output_dir=None, progress=None,
                         *, device=None, gpu_width=None, total_parts=None, input_identity=None):
    """Exact directed top-|Pearson| graph, self first, ID-based ties.

    part_factory can be replayed per feature block. Its weights must sum to 1.
    Stored blocks allow a cancelled first preparation to resume.
    """
    if device is not None and str(device).startswith('cuda'):
        from .gpu_preprocess import exact_neighbor_graph_cuda
        return exact_neighbor_graph_cuda(part_factory, mean, sd, gene_ids, neighbors,
            block_size, output_dir, progress, device=device, gpu_width=gpu_width,
            total_parts=total_parts, input_identity=input_identity)
    gene_ids = np.asarray(gene_ids).astype(str)
    n_genes = len(gene_ids)
    if len(set(gene_ids)) != n_genes or not 0 < neighbors < n_genes:
        raise ValueError("Unique genes and 0 < neighbors < gene_count are required")
    by_id = np.argsort(gene_ids, kind="stable")
    output = np.empty((n_genes, neighbors + 1), dtype=np.int64)
    if progress:
        progress('gene_graph_cpu', 0, n_genes)
    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        saved = Path(output_dir) / f"neighbors_{start:06d}.npz" if output_dir else None
        if saved is not None and saved.exists():
            block = np.load(saved)["neighbors"]
            if block.shape != (stop - start, neighbors + 1):
                raise ValueError("Incompatible partial graph block")
        else:
            cross = np.zeros((stop - start, n_genes), dtype=np.float64)
            mass = 0.
            for values, weights in part_factory():
                cross += (values[:, start:stop].T @ values.multiply(weights[:, None])).toarray()
                mass += float(np.sum(weights))
            if not np.isclose(mass, 1., atol=1e-9):
                raise ValueError("Graph fitting weights must sum to one")
            cross -= mean[start:stop, None] * mean[None, :]
            divisor = sd[start:stop, None] * sd[None, :]
            correlation = np.divide(cross, divisor, out=np.zeros_like(cross), where=divisor > 0)
            correlation = np.clip(correlation, -1., 1.)
            block = np.empty((stop - start, neighbors + 1), dtype=np.int64)
            for local, gene in enumerate(range(start, stop)):
                candidates = by_id[by_id != gene]
                order = np.argsort(-np.abs(correlation[local, candidates]), kind="stable")
                block[local] = np.r_[gene, candidates[order[:neighbors]]]
            if saved is not None:
                saved.parent.mkdir(parents=True, exist_ok=True)
                temporary = saved.with_suffix(".tmp.npz")
                np.savez(temporary, neighbors=block)
                temporary.replace(saved)
        output[start:stop] = block
        if progress:
            progress("gene_graph_cpu", stop, n_genes)
    return output


def _cache_training_rows(root, assets, genes, config, progress):
    """Visit source rows once; retain only fit/T log-expression CSR rows."""
    cache = assets / "fit_cache"
    cache.mkdir(exist_ok=True)
    from src.lung_data import PreparedLungData
    registry = source_shard_registry(PreparedLungData(root, config["bundle"]))
    index_path = cache / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        total_parts = len(index['parts'])
        if progress:
            progress('verify_fit_cache', 0, total_parts)
        for number, part in enumerate(index["parts"], 1):
            for key in ("matrix", "conditions"):
                path = inside(assets, part[key])
                if not path.exists() or sha256(path) != part[key + "_sha256"]:
                    raise ValueError("Prepared fitting cache hash differs")
            if progress:
                progress('verify_fit_cache', number, total_parts)
        return index
    columns = genes["tahoe_column"].to_numpy(dtype=np.int64)
    train_path = assets / "train.parquet"
    parts, invalid_ids, counts = [], [], {}
    total = pq.ParquetFile(train_path).metadata.num_rows
    if progress:
        progress('fit_expression_cache', 0, total)
    visited, cached_path, cached_matrix = 0, None, None
    use_columns = ["cell_id", "matrix_path", "matrix_row", "condition_id", "role"]
    for record in pq.ParquetFile(train_path).iter_batches(batch_size=65536, columns=use_columns):
        rows = record.to_pandas()
        visited += len(rows)
        fit = rows.loc[rows["role"].eq("fit")]
        for matrix_path, subset in fit.groupby("matrix_path", sort=False):
            if cached_path != matrix_path:
                source_path = inside(root, matrix_path)
                if registry and matrix_path not in registry:
                    raise ValueError("Matrix is not registered in the source audit")
                check_shard(source_path, registry.get(matrix_path))
                cached_matrix = sparse.load_npz(source_path).tocsr()
                cached_path = matrix_path
            raw = cached_matrix[subset["matrix_row"].to_numpy(dtype=np.int64)][:, columns]
            values, valid = log_normalize(raw)
            invalid_ids.extend(subset.loc[~valid, "cell_id"].to_list())
            subset = subset.loc[valid]
            if subset.empty:
                continue
            values = values[valid]
            for condition, number in subset["condition_id"].value_counts().items():
                counts[condition] = counts.get(condition, 0) + int(number)
            for start in range(0, len(subset), config["stream_batch_size"]):
                stop = start + config["stream_batch_size"]
                part_id = len(parts)
                matrix_file = cache / f"rows_{part_id:06d}.npz"
                conditions_file = cache / f"conditions_{part_id:06d}.npy"
                sparse.save_npz(matrix_file, values[start:stop])
                ids = subset.iloc[start:stop]["condition_id"].to_numpy(dtype=str)
                np.save(conditions_file, ids, allow_pickle=False)
                parts.append({"matrix": matrix_file.relative_to(assets).as_posix(),
                              "conditions": conditions_file.relative_to(assets).as_posix(),
                              "matrix_sha256": sha256(matrix_file),
                              "conditions_sha256": sha256(conditions_file), "rows": len(ids)})
        if progress:
            progress("fit_expression_cache", visited, total)
    index = {"role": "fit", "parts": parts, "valid_condition_counts": counts,
             "invalid_zero_common_library_cell_ids": invalid_ids,
             "normalization": "log1p(10000 * common_counts / common_library_size)"}
    atomic_json(index_path, index)
    return index


def _normalization(assets, parts, index, weights, gene_ids, device, progress):
    """Reuse bound statistics; verify an older interrupted run before adoption."""
    identity = digest_json({'index': index, 'weights': weights, 'genes': list(gene_ids),
                            'definition': 'weighted_cell_moments_v1'})
    path, receipt = assets / 'normalization.npz', assets / 'normalization_receipt.json'
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding='utf-8'))
        if record['identity'] != identity or record['sha256'] != sha256(path):
            raise ValueError('NORMALIZATION_CACHE_IDENTITY_MISMATCH')
        with np.load(path, allow_pickle=False) as saved:
            return saved['mean'], saved['scale'], saved['pearson_sd'], float(saved['fit_weight_sum'])
    if str(device).startswith('cuda'):
        from .gpu_preprocess import weighted_moments_cuda
        result = weighted_moments_cuda(parts(), len(gene_ids), device, progress, len(index['parts']))
    else:
        result = weighted_moments(parts(), len(gene_ids), progress, len(index['parts']))
    mean, scale, sd, mass = result
    if not np.isclose(mass, 1., atol=1e-9):
        raise ValueError('T normalization weights do not sum to one')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            old = saved['mean'], saved['scale'], saved['pearson_sd'], float(saved['fit_weight_sum'])
            if (not np.array_equal(saved['gene_ids'], gene_ids) or
                    any(not np.allclose(a, b, atol=1e-11, rtol=1e-10) for a, b in zip(old, result))):
                raise ValueError('LEGACY_NORMALIZATION_VERIFICATION_FAILED')
            result = old  # Preserve original CPU statistics and existing graph blocks.
    else:
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, mean=mean, scale=scale, pearson_sd=sd,
                 gene_ids=np.asarray(gene_ids, dtype=str), fit_weight_sum=mass)
        temporary.replace(path)
    atomic_json(receipt, {'identity': identity, 'sha256': sha256(path),
                          'verification_device': str(device), 'dtype': 'float64'})
    return result


def build_preprocessing(root: Path, assets: Path, config: dict, progress=None, *, device=None):
    genes = pd.read_parquet(assets / "genes.parquet")
    conditions = pd.read_parquet(assets / "conditions.parquet")
    index = _cache_training_rows(root, assets, genes, config, progress)
    counts = index["valid_condition_counts"]
    declared = set(pd.read_parquet(assets / "role_condition_counts.parquet")
                   .query("role == 'fit'")["condition_id"])
    if set(counts) != declared:
        raise ValueError("NO_USABLE_FIT_CELLS in one or more frozen T conditions")
    source = conditions.set_index("condition_id")["source_id"].to_dict()
    by_source = {}
    for condition in counts:
        by_source.setdefault(source[condition], []).append(condition)
    weights = {condition: 1. / len(by_source) / len(by_source[source[condition]]) / count
               for condition, count in counts.items()}

    def compressed_parts():
        for part in index["parts"]:
            matrix = sparse.load_npz(inside(assets, part["matrix"]))
            ids = np.load(inside(assets, part["conditions"]), allow_pickle=False)
            weight = np.fromiter((weights[item] for item in ids), dtype=np.float64)
            yield matrix, weight

    device = str(device or 'cpu')
    parts = compressed_parts
    if device.startswith('cuda'):
        from .gpu_preprocess import cuda_device
        from .stream_cache import build_stream_cache
        cuda_device(device)
        parts = build_stream_cache(assets, index, weights, progress)
    ids = genes['ensembl_id'].to_numpy(dtype=str)
    mean, scale, sd, mass = _normalization(assets, parts, index, weights, ids, device, progress)
    graph = exact_neighbor_graph(parts, mean, sd, genes["ensembl_id"],
                                 neighbors=config["graph_neighbors"],
                                 block_size=config["graph_block_size"],
                                 output_dir=assets / "graph_blocks", progress=progress,
                                 device=device, total_parts=len(index['parts']),
                                 input_identity=digest_json({'index': index, 'weights': weights}))
    temporary = assets / 'neighbors.tmp.npz'
    np.savez(temporary, neighbors=graph, gene_ids=ids)
    temporary.replace(assets / 'neighbors.npz')
    return {"fitted_role": "fit", "gene_count": len(genes), "valid_cells": sum(counts.values()),
            "conditions": len(counts), "sources": len(by_source),
            "graph": "exact_cell_level_condition_equal_weighted_Pearson",
            "invalid_zero_common_library_rows": len(index["invalid_zero_common_library_cell_ids"]),
            "normalization": "source_equal_condition_equal_cell_equal",
            "execution_device": device, "statistics_dtype": "float64",
            "graph_compute": "cuda_sparse_addmm" if device.startswith('cuda') else "scipy_sparse",
            "graph_cache": "lossless_mmap" if device.startswith('cuda') else "compressed_csr",
            "replicates": "each_current_source_condition_is_one_source_sample_unit"}
