"""One CPU reader worker, one speculative group, private NumPy random streams."""
import copy
from concurrent.futures import ThreadPoolExecutor


class ResponsePrefetch:
    def __init__(self, data, cfg, enabled=False):
        self.data, self.cfg = data, cfg
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="response-reader") if enabled else None
        self.future, self.keys = None, None

    def _load(self, keys, rng, mmd_rng):
        grouped = getattr(self.data, "sample_response_group", None)
        if callable(grouped):
            batches = grouped(keys, self.cfg, rng, mmd_rng, role="T")
            return batches, copy.deepcopy(rng.bit_generator.state), copy.deepcopy(mmd_rng.bit_generator.state)
        batches = []
        for key in keys:
            first = self.data.sample_response(key, self.cfg["ot_samples"], rng, role="T")
            second = (self.data.sample_response(key, self.cfg["mmd_samples"], mmd_rng, role="T")
                      if self.cfg.get("mmd_weight", 1.) else None)
            batches.append((first, second))
        return batches, copy.deepcopy(rng.bit_generator.state), copy.deepcopy(mmd_rng.bit_generator.state)

    def queue(self, keys, rng, mmd_rng):
        if self.pool is None or not keys:
            return
        if self.future is not None:
            raise RuntimeError("PREFETCH_ALREADY_QUEUED")
        self.keys = tuple(keys)
        # No torch RNG, CUDA work, or shared generator mutation in the worker.
        self.future = self.pool.submit(self._load, self.keys, copy.deepcopy(rng), copy.deepcopy(mmd_rng))

    def consume(self, keys, rng, mmd_rng):
        if self.pool is None:
            return None
        if self.future is None:
            self.queue(keys, rng, mmd_rng)
        if tuple(keys) != self.keys:
            raise ValueError("PREFETCH_CONDITION_ORDER_MISMATCH")
        future = self.future
        if future is None:
            raise RuntimeError("PREFETCH_NOT_QUEUED")
        batches, numpy_state, mmd_state = future.result()
        # Only consumption advances checkpoint-visible states.
        rng.bit_generator.state = numpy_state
        mmd_rng.bit_generator.state = mmd_state
        self.future, self.keys = None, None
        return batches

    def drain(self):
        # Join before validation touches reader caches; leave the result unconsumed.
        if self.future is not None:
            self.future.result()

    def discard(self):
        """Join and drop a queued load WITHOUT committing its captured streams.

        Used when the effective training stage changes mid-epoch: the queued
        group was drawn under the old stage config, so committing its captured
        RNG states would roll back streams the main thread has since advanced
        (rework C2). The wasted draw costs IO only.
        """
        if self.future is not None:
            self.future.result()
            self.future, self.keys = None, None

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.future, self.keys = None, None
