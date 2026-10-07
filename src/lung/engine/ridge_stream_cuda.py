"""Sparse CUDA population means in the frozen float32 expression coordinates.

Structural zeros have the same standardized value in every cell. Summing that
baseline plus sparse deviations avoids a cells-by-genes dense host allocation.
All normalization and expression reductions run on CUDA in float64, with the
same intervening float32 rounding as the shared training reader.
"""

from dataclasses import dataclass

import numpy as np
import torch

from lung.data.gpu_preprocess import cuda_device


@dataclass(frozen=True)
class PinnedBatch:
    offsets: torch.Tensor
    columns: torch.Tensor
    values: torch.Tensor
    pools: torch.Tensor
    shape: tuple


def pack_counts(counts, pool_indices):
    """One CPU producer selects CSR rows; only sparse arrays are pinned/uploaded."""
    matrix = counts.tocsr()
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format:
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("Counts must be finite and nonnegative")
        matrix = matrix.copy()
        matrix.sort_indices()
        # The frozen CPU reader logs stored duplicate entries separately before
        # densifying. Summing duplicate counts here would change its coordinates.
        if not matrix.has_canonical_format:
            raise ValueError("DUPLICATE_CSR_COORDINATES: frozen normalization cannot be changed")
    pools = np.asarray(pool_indices)
    if pools.shape != (matrix.shape[0],) or not np.issubdtype(pools.dtype, np.integer):
        raise ValueError("RIDGE_POOL_INDICES_SHAPE_OR_DTYPE")

    def pinned(array):
        return torch.from_numpy(np.ascontiguousarray(array)).pin_memory()

    return PinnedBatch(pinned(matrix.indptr), pinned(matrix.indices), pinned(matrix.data),
                       pinned(pools.astype(np.int64, copy=False)), matrix.shape)


class CudaPopulationAccumulator:
    def __init__(self, n_pools, mean, scale, device="cuda:0"):
        self.device = cuda_device(device)
        self.mean = torch.as_tensor(mean, dtype=torch.float64, device=self.device)
        self.scale = torch.as_tensor(scale, dtype=torch.float64, device=self.device)
        if (n_pools < 1 or self.mean.ndim != 1 or self.mean.numel() < 1
                or self.mean.shape != self.scale.shape or not torch.isfinite(self.mean).all()
                or not torch.isfinite(self.scale).all() or not (self.scale > 0).all()):
            raise ValueError("INVALID_FROZEN_NORMALIZATION")
        self.n_pools = n_pools
        self.n_genes = self.mean.numel()
        self.zero = (-self.mean / self.scale).float()
        self.deviations = torch.zeros((n_pools, self.n_genes), dtype=torch.float64, device=self.device)
        self.rows_per_pool = torch.zeros(n_pools, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def add(self, batch):
        if batch.shape[1] != self.n_genes or batch.shape[0] < 1:
            raise ValueError("INVALID_FULL_GENE_OBSERVATION")
        # Transfer compact counts in their source dtype; convert on the GPU.
        offsets = batch.offsets.to(self.device, non_blocking=True).long()
        columns = batch.columns.to(self.device, non_blocking=True).long()
        values = batch.values.to(self.device, non_blocking=True).double()
        pools = batch.pools.to(self.device, non_blocking=True)
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Counts must be finite and nonnegative")
        if (pools < 0).any() or (pools >= self.n_pools).any():
            raise ValueError("RIDGE_POOL_INDEX_OUT_OF_RANGE")
        total = torch.segment_reduce(values, "sum", offsets=offsets)
        if not (total > 0).all():
            raise ValueError("ZERO_COMMON_GENE_LIBRARY: unusable expression row")
        if not torch.isfinite(total).all():
            raise ValueError("NONFINITE_COMMON_GENE_LIBRARY")
        rows = torch.repeat_interleave(torch.arange(batch.shape[0], device=self.device),
            offsets[1:] - offsets[:-1], output_size=values.numel())
        logged = torch.log1p(values * (10000. / total)[rows])
        normalized = ((logged - self.mean[columns]) / self.scale[columns]).float()
        if not torch.isfinite(normalized).all() or not torch.isfinite(self.zero).all():
            raise ValueError("INVALID_FULL_GENE_OBSERVATION")
        # Crucial: round each coordinate to float32 BEFORE float64 subtraction.
        delta = normalized.double() - self.zero[columns].double()
        flat_indices = pools[rows] * self.n_genes + columns
        self.deviations.view(-1).index_add_(0, flat_indices, delta)
        self.rows_per_pool.add_(torch.bincount(pools, minlength=self.n_pools))

    @torch.no_grad()
    def means(self, counts):
        expected = torch.as_tensor(counts, dtype=torch.int64, device=self.device)
        if (expected.shape != self.rows_per_pool.shape or not (expected > 0).all()
                or not torch.equal(expected, self.rows_per_pool)):
            raise ValueError("RIDGE_FULL_POPULATION_COUNT_MISMATCH")
        mass = expected[:, None].double()
        result = (self.deviations + mass * self.zero.double()[None, :]) / mass
        if not torch.isfinite(result).all():
            raise ValueError("INVALID_FULL_GENE_OBSERVATION")
        return result
