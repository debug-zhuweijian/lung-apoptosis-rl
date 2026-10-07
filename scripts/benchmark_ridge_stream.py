"""Bounded real-fit subset throughput probe; no model training or asset writes."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import pandas as pd
from pyarrow.compute import field
import pyarrow.dataset as ds
import torch

from lung.data.reader import SparseRows
from lung.data.stream_cache import prefetched
from lung.engine.config import file_hash, project_path, read_yaml, verify_interpreter
from lung.engine.data import EngineData
from lung.engine.pilot_ridge_data import mean_plan
from lung.engine.ridge_stream_cuda import CudaPopulationAccumulator, pack_counts


def subset(data, limit):
    plan = mean_plan(data, "T")
    names = list(plan["pool_counts"])
    scanner = ds.dataset(data.assets_dir / "train.parquet", format="parquet").scanner(
        columns=["cell_id", "condition_id", "matrix_path", "matrix_row", "role"],
        filter=(field("role") == "fit") & field("condition_id").isin(names),
        batch_size=min(limit, 8192), use_threads=False)
    parts, remaining = [], limit
    for batch in scanner.to_batches():
        frame = batch.slice(0, remaining).to_pandas()
        parts.append(frame)
        remaining -= len(frame)
        if remaining == 0:
            break
    rows = pd.concat(parts, ignore_index=True)
    if rows.empty or not rows.role.eq("fit").all() or not rows.condition_id.isin(names).all():
        raise ValueError("PROBE_REQUIRES_PLANNED_FIT_ROWS")
    if rows.cell_id.duplicated().any() or rows.duplicated(["matrix_path", "matrix_row"]).any():
        raise ValueError("DUPLICATE_PHYSICAL_CELL")
    serialized = rows.to_json(orient="records")
    assert serialized is not None
    return rows, hashlib.sha256(serialized.encode()).hexdigest()


@torch.no_grad()
def measure(data, rows, batch_size, method, device):
    # Each trial starts with a fresh, independent two-shard LRU, including warmup.
    original = data.reader.rows
    reader = SparseRows(original.root, original.columns, registry=original.registry)
    names = sorted(rows.condition_id.unique())
    positions = {key: i for i, key in enumerate(names)}
    indices = np.asarray([positions[key] for key in rows.condition_id], dtype=np.int64)
    counts = np.bincount(indices, minlength=len(names))
    timings = {key: 0. for key in ("read_s", "pack_s", "cpu_normalize_s", "consumer_wait_s", "submit_s")}
    events, order = [], []
    csr_checks = []

    def load(start):
        tick = perf_counter()
        raw = reader.read(rows.iloc[start:start + batch_size])
        timings["read_s"] += perf_counter() - tick
        csr_checks.append(bool(raw.has_canonical_format))
        idx = indices[start:start + batch_size]
        tick = perf_counter()
        value = pack_counts(raw, idx) if method == "new_gpu_sparse" else raw
        timings["pack_s"] += perf_counter() - tick
        return start, idx, value

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    tick = perf_counter()
    accumulator = CudaPopulationAccumulator(len(names), data.mu, data.scale, device) if method == "new_gpu_sparse" else None
    sums = torch.zeros((len(names), data.n_genes), dtype=torch.float64, device=device) if accumulator is None else None
    starts = range(0, len(rows), batch_size)
    loaded = prefetched(starts, load, workers=1) if accumulator is not None else (load(i) for i in starts)
    with closing(loaded):
        while True:
            waiting = perf_counter()
            try:
                start, idx, raw = next(loaded)
            except StopIteration:
                break
            timings["consumer_wait_s"] += perf_counter() - waiting
            order.append(start)
            normalized = None
            if accumulator is None:
                cpu_tick = perf_counter()
                normalized = data.reader.normalize(raw)
                timings["cpu_normalize_s"] += perf_counter() - cpu_tick
                if normalized.shape != (len(idx), 32461) or not np.isfinite(normalized).all():
                    raise ValueError("INVALID_FULL_GENE_OBSERVATION")
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            submitted = perf_counter()
            begin.record()
            if accumulator is None:
                assert sums is not None and normalized is not None
                sums.index_add_(0, torch.as_tensor(idx, device=device),
                                torch.as_tensor(normalized, dtype=torch.float64, device=device))
            else:
                accumulator.add(raw)
            end.record()
            timings["submit_s"] += perf_counter() - submitted
            events.append((begin, end))
    if accumulator is not None:
        means = accumulator.means(counts)
    else:
        assert sums is not None
        means = sums / torch.as_tensor(counts[:, None], device=device)
    result = means.cpu().numpy()
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - tick
    if order != list(starts) or result.shape != (len(names), 32461) or not np.isfinite(result).all():
        raise ValueError("PROBE_ORDER_AXIS_OR_FINITE_CHECK_FAILED")
    return result, {"method": method, "batch_size": batch_size, "total_s": elapsed,
        "cells_per_s": len(rows) / elapsed, **timings,
        "gpu_event_compute_and_transfer_s": sum(a.elapsed_time(b) for a, b in events) / 1000.,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "order_verified": True,
        "all_input_csr_canonical_no_duplicates": all(csr_checks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_ridge.yaml")
    parser.add_argument("--max-cells", type=int, default=8192)
    parser.add_argument("--batches", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=".omo/evidence/ridge-gpu-20260926/benchmark.json")
    args = parser.parse_args(argv)
    if not 1 <= args.max_cells <= 65536 or not 3 <= args.repeats <= 10:
        parser.error("require 1 <= max-cells <= 65536 and 3 <= repeats <= 10")
    if not args.batches or any(not 1 <= batch <= 8192 for batch in args.batches):
        parser.error("batch sizes must be in [1, 8192]")
    output = project_path(args.output)
    if not output.is_relative_to(ROOT / ".omo" / "evidence") or output.exists():
        parser.error("output must be a new file inside .omo/evidence")
    verify_interpreter()
    config = read_yaml(project_path(args.config))
    code_hashes = {name: file_hash(ROOT / name) for name in (
        "scripts/benchmark_ridge_stream.py", "src/lung/engine/ridge_stream_cuda.py",
        "src/lung/data/reader.py", "src/lung/data/preprocess.py", "src/lung/data/stream_cache.py")}
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("PROBE_REQUIRES_CUDA")
    torch.cuda.set_device(device)
    init = perf_counter()
    data = EngineData(config, roles=("T", "V"), allow_test=False, require_preprocessing=True)
    if data.n_genes != 32461:
        raise ValueError("FULL_32461_GENE_AXIS_REQUIRED")
    rows, digest = subset(data, args.max_cells)
    metadata_s = perf_counter() - init
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = ["old_cpu_normalize_gpu_sum", "new_gpu_sparse"]
    report = {"scope": "subset throughput probe/no model training", "cells": len(rows),
        "full_gene_axis": data.n_genes, "role": "fit", "ordered_rows_sha256": digest,
        "unique_cell_ids": True, "unique_physical_rows": True,
        "pools": int(rows.condition_id.nunique()), "source_shards": int(rows.matrix_path.nunique()),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "python": sys.executable, "torch": torch.__version__, "argv": sys.argv,
        "data_identity": data.identity, "metadata_setup_s": metadata_s,
        "timing_scope": "Fresh SparseRows cache; read, pack/normalize, transfer, aggregate, final mean copy included. Shared manifest verification and subset metadata scan reported separately.",
        "timing_caveat": "Producer read/pack overlaps consumer/GPU. CPU wait includes blocking next(); submit is host wall time; GPU events include transfers and host synchronization gaps. These components cannot be summed. OS file cache is not flushed; prefix subset is not full-population throughput.",
        "tolerances": {"atol": 1e-7, "rtol": 1e-6}, "trials": [], "status": "running"}
    def save():
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save()
    reference = None
    try:
        for method in methods:
            measure(data, rows.iloc[:min(128, len(rows))], 128, method, device)
        for batch in args.batches:
            for repeat in range(args.repeats):
                for method in methods if repeat % 2 == 0 else methods[::-1]:
                    result, trial = measure(data, rows, batch, method, device)
                    if reference is None:
                        reference = result.copy()
                    delta = np.abs(result - reference)
                    nonzero = np.abs(reference) > 1e-7
                    trial.update(mean_shape=list(result.shape),
                        max_rel_mean_diff_nonzero=float(np.max(delta[nonzero] / np.abs(reference[nonzero]))) if nonzero.any() else 0.,
                        max_abs_mean_diff_nearzero=float(np.max(delta[~nonzero])) if (~nonzero).any() else 0.,
                        nearzero_coordinates=int((~nonzero).sum()))
                    trial.update(repeat=repeat, max_abs_mean_diff=float(np.max(delta)),
                                 numeric_pass=bool(np.allclose(result, reference, atol=1e-7, rtol=1e-6)))
                    report["trials"].append(trial)
                    save()
                    print(json.dumps(trial), flush=True)
                    if not trial["numeric_pass"]:
                        raise ValueError("NUMERIC_COMPARISON_FAILED: fixed tolerances; no speed recommendation")
        report["summary"] = [{"batch_size": batch, "method": method,
            "median_total_s": float(np.median([t["total_s"] for t in report["trials"] if t["batch_size"] == batch and t["method"] == method]))}
            for batch in args.batches for method in methods]
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["code_sha256"] = code_hashes
        report["code_unchanged_during_probe"] = all(file_hash(ROOT / name) == value for name, value in code_hashes.items())
        if not report["code_unchanged_during_probe"]:
            report.update(status="failed", error="CODE_CHANGED_DURING_PROBE")
        save()
    if report["status"] != "passed":
        raise ValueError(report["status"])
    print(f"subset throughput probe/no model training: {output}", flush=True)


if __name__ == "__main__":
    main()
