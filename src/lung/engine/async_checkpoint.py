"""Background checkpoint writing (implementation task W5, save half; rework R7).

The training thread owns mutation; at an update boundary it takes a consistent
snapshot of genuinely CPU-resident tensors, then a single writer thread
serialises, fsyncs and atomically replaces the destination. Three steps are
tracked separately:

- ``computed_step``  optimizer steps actually executed;
- ``snapshot_step``  steps whose consistent clone was handed to the writer;
- ``durable_step``   steps whose file completed the atomic replace.

Members note: batched-member runs submit one job per member with the SAME
group step number; the queue is FIFO, so ``durable_step == step`` only after
the last member file of that group has been replaced.

A writer failure is stored and re-raised on the next submit/flush: background
saving never swallows IO errors. ``close`` drains and joins the writer even
when flushing raises, then propagates the original error. No exception-window
guarantee is claimed beyond what durable_step reports.
"""

import queue
import threading


def _to_cpu(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_cpu(item) for item in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    return value


def clone_snapshot(model, optimizer=None):
    """Independent CPU-resident snapshot taken at an update boundary.

    The returned mapping shares no storage with the live model/optimizer, so
    queued snapshots never hold GPU memory and mutation during serialisation is
    impossible (rework R7: the previous version kept GPU copies).
    """
    import copy

    snapshot = {"model": _to_cpu(model.state_dict())}
    if optimizer is not None:
        snapshot["optimizer"] = _to_cpu(copy.deepcopy(optimizer.state_dict()))
    return snapshot


class BackgroundSaver:
    """One writer thread, bounded queue, escalating failures."""

    def __init__(self, max_queue=2):
        if not isinstance(max_queue, int) or max_queue < 1:
            raise ValueError("BackgroundSaver queue depth must be positive")
        self._queue = queue.Queue(maxsize=max_queue)
        self._failure = None
        self._lock = threading.Lock()
        self._pending_steps = {}  # step -> jobs still required for the group (S10)
        self.computed_step = None
        self.snapshot_step = None
        self.durable_step = None
        self._thread = threading.Thread(target=self._run, name="checkpoint-writer", daemon=True)
        self._closing = False
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                step, writer = item
                writer()
                with self._lock:
                    # A step becomes durable only when ALL files of its group
                    # have been replaced; one member finishing first must not
                    # mark the whole group durable (rework S10).
                    remaining = self._pending_steps.get(step, 0) - 1
                    if remaining > 0:
                        self._pending_steps[step] = remaining
                    else:
                        self._pending_steps.pop(step, None)
                        if self.durable_step is None or step >= self.durable_step:
                            self.durable_step = step
            except BaseException as error:  # escalated, never swallowed
                with self._lock:
                    if self._failure is None:
                        self._failure = error
            finally:
                self._queue.task_done()

    def _check_failure(self):
        with self._lock:
            if self._failure is not None:
                raise RuntimeError(f"BACKGROUND_SAVE_FAILED: {self._failure}")

    def submit(self, step, writer, *, group=1):
        """Queue one serialisation job for an already-cloned CPU snapshot.

        ``group`` is the number of jobs sharing this step number (e.g. one per
        ensemble member); durable_step advances only when all of them complete.
        """
        if self._closing:
            raise RuntimeError("BACKGROUND_SAVE_CLOSED")
        if not isinstance(group, int) or group < 1:
            raise ValueError("BACKGROUND_SAVE_GROUP_SIZE")
        self._check_failure()
        with self._lock:
            self._pending_steps.setdefault(step, group)
        self._queue.put((step, writer))  # blocks when full: explicit backpressure
        with self._lock:
            self.snapshot_step = step

    def observe_computed(self, step):
        self.computed_step = step

    def flush(self):
        self._queue.join()
        self._check_failure()

    def close(self):
        """Drain and join the writer even when flushing raises (rework R7)."""
        if self._closing:
            return
        error = None
        try:
            self.flush()
        except BaseException as caught:  # propagate AFTER joining the thread
            error = caught
        finally:
            self._closing = True
            self._queue.put(None)
            self._thread.join()
        if error is not None:
            raise error
        self._check_failure()

    def status(self):
        with self._lock:
            return {"computed_step": self.computed_step,
                    "snapshot_step": self.snapshot_step,
                    "durable_step": self.durable_step,
                    "pending_group_jobs": dict(self._pending_steps),
                    "failure": repr(self._failure) if self._failure is not None else None}
