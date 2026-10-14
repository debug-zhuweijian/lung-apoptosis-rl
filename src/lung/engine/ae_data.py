"""Bounded AE data supply; speculative reads never advance saved random streams."""

import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from ..data.reader import SparseRows


class AEDataSupply:
    """Own the AE-only mapped reader and fixed expression cache for one stage."""

    def __init__(self, data, batch):
        self.data, self.batch = data, batch
        self.reader = self.expressions = None
        original = getattr(data, "reader", None)
        if original is not None and hasattr(original, "_response_rows"):
            from ..data.ae_expression_cache import NormalizedExpressionCache
            self.reader = copy.copy(original)
            self.reader._metadata_cache = OrderedDict(original._metadata_cache)
            rows = original.rows
            self.reader.rows = rows.clone_for_ae(max_open_mappings=64) if hasattr(rows, "clone_for_ae") else SparseRows(
                rows.root, rows.columns, cache_size=rows.cache_size, registry=rows.registry,
                cache_dir=rows.disk_cache.directory if rows.disk_cache else None,
                max_cache_bytes=rows.max_cache_bytes, use_disk_cache=rows.disk_cache is not None,
                max_open_mappings=64)
            self.expressions = NormalizedExpressionCache(self.reader)

    def sample_group(self, keys, rng, role="T"):
        if self.reader is None:
            return [self.data.sample_response(key, self.batch, rng, role=role) for key in keys]
        return self.reader.sample_ae_group(
            keys, self.batch, rng, role={"T": "fit", "V": "selection"}[role],
            expression_cache=self.expressions)

    def groups(self, *, enabled=True):
        return AEGroupPrefetch(self, enabled=enabled)

    def close(self):
        if self.expressions is not None:
            self.expressions.close()
        if self.reader is not None:
            if hasattr(self.reader.rows, "close"):
                self.reader.rows.close()
            elif self.reader.rows.disk_cache is not None:
                self.reader.rows.disk_cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AEGroupPrefetch:
    """One producer and at most one pending group; checkpoints see consumed RNG only."""

    def __init__(self, supply, *, enabled=True):
        self.supply = supply
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ae-reader") if enabled else None
        self.future, self.keys = None, None

    def _load(self, keys, rng):
        batches = self.supply.sample_group(keys, rng)
        return batches, copy.deepcopy(rng.bit_generator.state)

    def queue(self, keys, rng):
        if self.pool is None or not keys:
            return
        if self.future is not None:
            raise RuntimeError("AE_PREFETCH_ALREADY_QUEUED")
        self.keys = tuple(keys)
        self.future = self.pool.submit(self._load, self.keys, copy.deepcopy(rng))

    def consume(self, keys, rng):
        if self.pool is None:
            return self.supply.sample_group(keys, rng)
        if self.future is None:
            self.queue(keys, rng)
        if tuple(keys) != self.keys:
            raise ValueError("AE_PREFETCH_CONDITION_ORDER_MISMATCH")
        batches, state = self.future.result()
        rng.bit_generator.state = state
        self.future, self.keys = None, None
        return batches

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.pool, self.future, self.keys = None, None, None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
