"""Phase timing for research runs: host monotonic clocks plus CUDA events.

Host durations use ``time.monotonic``. GPU intervals use CUDA events recorded
on the current stream and are read only when the caller flushes, so no per-op
synchronisation is introduced. Host-side data waits are recorded separately
from transfers: a blocking queue must never be reported as transfer time.
Nothing here starts work, touches data files or changes numerics; it only
measures what the caller already executes.
"""

import time
from collections import OrderedDict


class _Pending:
    __slots__ = ("name", "host_seconds", "start", "end")

    def __init__(self, name, host_seconds, start=None, end=None):
        self.name = name
        self.host_seconds = host_seconds
        self.start = start
        self.end = end


class PhaseTimer:
    """Accumulate host (and optionally CUDA) durations per named phase."""

    def __init__(self, device=None):
        self.device = None
        if device is not None and str(device).startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                self.device = torch.device(device)
        self.host = OrderedDict()   # name -> seconds
        self.cuda = OrderedDict()   # name -> seconds (event-measured)
        self.calls = OrderedDict()  # name -> count
        self.pending = []
        self.waiting = {}           # name -> seconds of pure consumer waiting

    def phase(self, name):
        """Time one caller-owned segment; CUDA spans are read at flush()."""
        import contextlib

        @contextlib.contextmanager
        def span():
            begin = time.monotonic()
            start = end = None
            if self.device is not None:
                import torch
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            try:
                yield
            finally:
                if end is not None:
                    end.record()
                self.pending.append(_Pending(name, time.monotonic() - begin, start, end))
        return span()

    def record_wait(self, name, seconds):
        """Host-side waiting (queue/queue.result), never counted as transfer."""
        if seconds < 0:
            raise ValueError("NEGATIVE_WAIT_TIME")
        self.waiting[name] = self.waiting.get(name, 0.) + float(seconds)

    def flush(self):
        """Synchronise pending CUDA events once per caller-chosen boundary."""
        for item in self.pending:
            self.host[item.name] = self.host.get(item.name, 0.) + item.host_seconds
            self.calls[item.name] = self.calls.get(item.name, 0) + 1
            if item.start is not None:
                item.end.synchronize()
                self.cuda[item.name] = self.cuda.get(item.name, 0.) + \
                    item.start.elapsed_time(item.end) / 1000.
        self.pending.clear()

    def summary(self):
        self.flush()
        return {"host_seconds": dict(self.host), "cuda_seconds": dict(self.cuda),
                "calls": dict(self.calls), "wait_seconds": dict(self.waiting)}


class H2DAccounting:
    """Separate transfer time from unhidden waiting around host->device copies."""

    def __init__(self):
        self.transfers = 0
        self.transfer_seconds = 0.
        self.wait_seconds = 0.

    def record(self, *, wait_seconds, transfer_seconds):
        if wait_seconds < 0 or transfer_seconds < 0:
            raise ValueError("NEGATIVE_H2D_TIME")
        self.transfers += 1
        self.wait_seconds += float(wait_seconds)
        self.transfer_seconds += float(transfer_seconds)

    def summary(self):
        return {"transfers": self.transfers,
                "transfer_seconds": self.transfer_seconds,
                "unhidden_wait_seconds": self.wait_seconds}


class ThroughputCounter:
    """Conditions and cells processed against wall clock; no stopping power."""

    def __init__(self):
        self.started = time.monotonic()
        self.conditions = 0
        self.cells = 0
        self.updates = 0

    def update_done(self, conditions, cells):
        if conditions < 0 or cells < 0:
            raise ValueError("NEGATIVE_THROUGHPUT_COUNT")
        self.updates += 1
        self.conditions += int(conditions)
        self.cells += int(cells)

    def summary(self):
        elapsed = max(time.monotonic() - self.started, 1e-9)
        return {"wall_seconds": elapsed, "updates": self.updates,
                "conditions": self.conditions, "cells": self.cells,
                "conditions_per_second": self.conditions / elapsed,
                "cells_per_second": self.cells / elapsed}
