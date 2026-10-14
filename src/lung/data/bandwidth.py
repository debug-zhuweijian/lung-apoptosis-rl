"""Freeze the architecture's deterministic, T-only MMD bandwidth."""

from __future__ import annotations

import hashlib
import heapq
import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .reader import training_rows
from .preprocess import standardized


def _hash(value):
    return int(hashlib.sha256(value.encode()).hexdigest(), 16)


def _keep_smallest(heap, entry, count):
    if len(heap) < count:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _pair_distances(values, pair_indices, device=None, *, block_size=64):
    """Full-gene squared distances after the original float32 rounding.

    At most ``block_size`` pair differences are live on the device. The
    float64 reduction preserves the original estimator, including its /G
    scale; this does not use a norm/dot-product expansion with cancellation.
    """
    target = torch.device("cpu" if device is None else device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("MMD bandwidth supports CPU or CUDA computation")
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    result = np.empty(len(pairs), dtype=np.float64)
    if target.type == "cpu":
        values64 = np.asarray(values, dtype=np.float64)
        for start in range(0, len(pairs), block_size):
            block = pairs[start:start + block_size]
            difference = values64[block[:, 0]] - values64[block[:, 1]]
            result[start:start + len(block)] = np.mean(difference ** 2, axis=1)
    else:
        values64 = torch.as_tensor(values, device=target, dtype=torch.float64)
        with torch.no_grad():
            for start in range(0, len(pairs), block_size):
                block = torch.as_tensor(pairs[start:start + block_size], device=target)
                difference = values64[block[:, 0]] - values64[block[:, 1]]
                result[start:start + len(block)] = difference.square_().mean(dim=1).cpu().numpy()
    return result


def _selected_expression_cache(reader, rows, mean, scale, destination, progress):
    """Visit each original shard once, with bounded dense row batches."""
    if progress:
        progress("mmd_expression_cache", 0, len(rows))
    done = 0
    for _, group in rows.groupby("matrix_path", sort=True):
        for start in range(0, len(group), 128):
            batch = group.iloc[start:start + 128]
            # SparseRows retains the current source shard between row batches.
            destination[batch.index.to_numpy()] = standardized(reader.read(batch), mean, scale)
            done += len(batch)
            if progress:
                progress("mmd_expression_cache", done, len(rows))


def build_bandwidths(root, assets, config, progress=None, *, device=None):
    """Select first 128 SHA cells/condition, then first 4096 SHA pairs globally.

    A pair's cell IDs are lexicographically ordered before its SHA is computed.
    Distances use all standardized common genes and the documented /G scale.
    """
    by_condition = {}
    cache = json.loads((assets / "fit_cache" / "index.json").read_text(encoding="utf-8"))
    invalid = set(cache["invalid_zero_common_library_cell_ids"])
    columns = ["cell_id", "condition_id", "matrix_path", "matrix_row", "role"]
    training = pq.ParquetFile(assets / "train.parquet")
    total, seen = training.metadata.num_rows, 0
    if progress:
        progress("mmd_cell_selection", 0, total)
    for batch in training.iter_batches(batch_size=65536, columns=columns):
        frame = batch.to_pandas()
        for row in frame.loc[frame["role"].eq("fit")].itertuples(index=False):
            if row.cell_id in invalid:
                continue
            heap = by_condition.setdefault(row.condition_id, [])
            entry = (-_hash(row.cell_id), row.cell_id, row.matrix_path, int(row.matrix_row))
            _keep_smallest(heap, entry, 128)
        seen += len(frame)
        if progress:
            progress("mmd_cell_selection", seen, total)
    pair_heap = []
    if progress:
        progress("mmd_pair_selection", 0, len(by_condition))
    for number, (condition, cells) in enumerate(sorted(by_condition.items()), 1):
        ids = sorted(item[1] for item in cells)
        for first, second in combinations(ids, 2):
            key = f"{condition}|{first}|{second}"
            _keep_smallest(pair_heap, (-_hash(key), condition, first, second), 4096)
        if progress:
            progress("mmd_pair_selection", number, len(by_condition))
    selected = {}
    for _, condition, first, second in sorted(pair_heap, reverse=True):
        selected.setdefault(condition, []).append((first, second))
    genes = pd.read_parquet(assets / "genes.parquet")
    with np.load(assets / "normalization.npz", allow_pickle=False) as norm:
        mean, scale = norm["mean"], norm["scale"]
    reader = training_rows(root, assets, config, genes, cache_size=1)
    distances, pair_rows = [], []
    records = []
    for condition, pairs in sorted(selected.items()):
        lookup = {item[1]: item for item in by_condition[condition]}
        used = sorted({item for pair in pairs for item in pair})
        records.extend({"condition_id": condition, "cell_id": cell, "matrix_path": lookup[cell][2],
                        "matrix_row": lookup[cell][3]} for cell in used)
    if records:
        rows = pd.DataFrame(records)
        locations = {(row.condition_id, row.cell_id): row.Index for row in rows.itertuples()}
        # At most 8192 selected cells: 1.06 GB temporary disk at 32461 genes.
        # A unique directory avoids overwriting any cache left by a hard stop.
        with TemporaryDirectory(prefix="mmd_selected_", dir=assets) as temporary:
            stored = np.memmap(Path(temporary) / "expression.bin", mode="w+", dtype=np.float32,
                               shape=(len(rows), len(genes)))
            try:
                _selected_expression_cache(reader, rows, mean, scale, stored, progress)
                if progress:
                    progress("mmd_bandwidth", 0, len(pair_heap))
                for condition, pairs in sorted(selected.items()):
                    used = sorted({cell for pair in pairs for cell in pair})
                    positions = {cell: index for index, cell in enumerate(used)}
                    # Fancy indexing copies only this condition (at most 128 cells).
                    values = stored[[locations[(condition, cell)] for cell in used]]
                    indices = [(positions[first], positions[second]) for first, second in pairs]
                    result = _pair_distances(values, indices, device)
                    for (first, second), distance in zip(pairs, result):
                        distance = float(distance)
                        if distance > 0:
                            distances.append(distance)
                        pair_rows.append((condition, first, second, distance))
                    if progress:
                        progress("mmd_bandwidth", len(pair_rows), len(pair_heap))
            finally:
                # Explicit close is required before TemporaryDirectory cleanup on Windows.
                stored._mmap.close()
    elif progress:
        progress("mmd_expression_cache", 0, 0)
        progress("mmd_bandwidth", 0, 0)
    median = float(np.median(distances)) if distances else 1.
    bandwidths = median * np.asarray([.25, .5, 1., 2., 4.], dtype=np.float64)
    np.savez(assets / "bandwidths.npz", bandwidths=bandwidths,
             median_positive_squared_distance=median, selected_pair_count=len(pair_rows))
    pd.DataFrame(pair_rows, columns=["condition_id", "cell_id_1", "cell_id_2", "distance_squared_per_gene"]
                 ).to_parquet(assets / "bandwidth_pairs.parquet", index=False)
    return {"role": "fit", "pairs": len(pair_rows), "positive_pairs": len(distances),
            "median": median, "bandwidths": bandwidths.tolist(),
            "pair_identity": "lexicographic_cell_IDs",
            "selection": "first_128_SHA_cells_per_condition_then_first_4096_SHA_pairs"}
