"""Bounded spawn workers for independent response ensemble members."""

import contextlib
import csv
import json
import math
import multiprocessing
import queue
import time
import traceback
import uuid
from pathlib import Path

from .checkpoint import atomic_json


class MemberMonitor:
    """Member-local metrics and progress; never opens a dashboard or shared manifest."""

    def __init__(self, config, weights, logdir, records):
        self.config = {**config, "progress": True}
        self.records = records
        self.weights, self.logdir = Path(weights), Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.logdir / "metrics.csv").open("w", encoding="utf-8", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["stage", "axis", "step", "metric", "value", "seconds"])
        self.started = time.monotonic()

    def progress(self, iterable=None, **kwargs):
        from tqdm.auto import tqdm
        return tqdm(iterable, **{**kwargs, "disable": True})

    def record(self, stage, step, metrics, axis="epoch"):
        clean = {}
        for name, value in metrics.items():
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"NONFINITE_METRIC: {stage}/{name}={value}")
            clean[name] = value
            self.csv.writerow([stage, axis, step, name, value, round(time.monotonic()-self.started, 3)])
        self.csv_file.flush()
        self.records.put((stage, step, clean, axis))

    def say(self, message):
        print(message, flush=True)

    def close(self):
        self.csv_file.close()


def _train_member(member, data_identity, config, weights, logdir, resume, records):
    import torch
    from .data import open_data
    from .response import _train_response_members

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    status_path = logdir / "worker_status.json"
    with (logdir / "console.log").open("a", encoding="utf-8") as console:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            try:
                torch.set_num_threads(config["response"].get("member_cpu_threads", 4))
                atomic_json(status_path, {"member": member, "status": "opening_data"})
                data = open_data(config, "response", prepare=False)
                if data.identity != data_identity:
                    raise ValueError("PARALLEL_MEMBER_DATA_IDENTITY_MISMATCH")
                monitor = MemberMonitor(config, weights, logdir, records)
                try:
                    atomic_json(status_path, {"member": member, "status": "training"})
                    paths = _train_response_members(data, config, monitor, resume=resume,
                                                    member_indices=(member,))
                finally:
                    monitor.close()
                records.close()
                records.join_thread()
                atomic_json(status_path, {"member": member, "status": "complete", "path": paths[0]})
            except BaseException as error:
                records.cancel_join_thread()
                records.close()
                traceback.print_exc()
                atomic_json(status_path, {"member": member, "status": "failed",
                                         "error": f"{type(error).__name__}: {error}"})
                # multiprocessing otherwise prints this traceback again outside
                # redirected stderr, interleaving the parent's single progress bar.
                raise SystemExit(1) from error


def _read_status(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _drain_records(records, monitor, limit=None):
    consumed = 0
    while limit is None or consumed < limit:
        try:
            stage, step, metrics, axis = records.get_nowait()
        except queue.Empty:
            return
        monitor.record(stage, step, metrics, axis=axis)
        consumed += 1


def train_parallel_members(data_identity, config, monitor, resume=False):
    """One CUDA context/optimizer/RNG per member; no model or reader is pickled."""
    count = config["response"]["members"]
    workers = min(config["response"]["member_workers"], count)
    context = multiprocessing.get_context("spawn")
    records = context.Queue(maxsize=64)
    segment = uuid.uuid4().hex
    logs = monitor.logdir / "response_members" / segment
    logs.mkdir(parents=True, exist_ok=True)
    monitor.say(f"响应成员并行进程数：{workers}；独立成员日志：{logs}；指标由父进程归并至原 CSV / TensorBoard。")
    active, paths, next_member = {}, [None] * count, 0
    bar = monitor.progress(total=count, desc="响应成员", unit="member", position=0,
                           dynamic_ncols=False, ncols=110, mininterval=2,
                           bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}{postfix}]")
    try:
        while next_member < count or active:
            _drain_records(records, monitor, limit=128)
            for member, (process, folder) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                status = _read_status(folder / "worker_status.json")
                if process.exitcode != 0 or status.get("status") != "complete":
                    raise RuntimeError(f"RESPONSE_MEMBER_FAILED: member={member+1}, "
                        f"exitcode={process.exitcode}, error={status.get('error', 'missing completion')}; "
                        f"log={folder / 'console.log'}")
                paths[member] = status["path"]
                process.close()
                del active[member]
                bar.update(1)
            while next_member < count and len(active) < workers:
                member = next_member
                folder = logs / f"member_{member+1}"
                process = context.Process(target=_train_member,
                    args=(member, data_identity, config, str(monitor.weights), str(folder), resume, records),
                    name=f"response-member-{member+1}")
                active[member] = (process, folder)
                process.start()
                next_member += 1
            labels = []
            for member, (_, folder) in active.items():
                state = _read_status(folder / "response_progress.json")
                rate = state.get("conditions_per_second")
                speed = f" {rate:.2f}条件/s" if isinstance(rate, (int, float)) else ""
                labels.append(f"P{member+1} E{state.get('epoch', '-')} "
                              f"{state.get('conditions_done', 0)}/{state.get('conditions_total', '?')}{speed}")
            bar.set_postfix_str(" | ".join(labels), refresh=True)
            if active:
                time.sleep(2)
        _drain_records(records, monitor)
        return paths
    finally:
        # Explicit processes permit immediate cleanup on failure/Ctrl-C on Python 3.13.
        # A ProcessPoolExecutor context manager would wait for unfinished training.
        for process, _ in active.values():
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process, _ in active.values():
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
        records.cancel_join_thread()
        records.close()
        bar.close()
