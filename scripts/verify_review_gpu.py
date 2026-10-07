"""Bounded, no-optimizer full-gene CUDA numerical and memory verification."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lung.engine.checkpoint import atomic_json, seed_all
from lung.engine.config import implementation_identity
from lung.timecourse.auxiliary import population_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    seed_all(20260927)
    torch.set_num_threads(4)
    rng = np.random.default_rng(20260927)
    genes = 32461
    predicted = rng.normal(0, .1, (512, genes)).astype(np.float32)
    observed = rng.normal(.01, .1, (769, genes)).astype(np.float32)
    bandwidth = .1 * genes**.5
    device = torch.device("cuda:0")
    x = torch.from_numpy(predicted).to(device)
    free, total = torch.cuda.mem_get_info(device)
    required = 8 * (predicted.size + observed.size)
    if free // 4 < required:
        raise RuntimeError("GPU_BUSY_FOR_CACHED_REFERENCE_COMPARISON; no other processes were changed")

    def measure(budget):
        population_error(x, observed, bandwidth, device_cache_bytes=budget)
        torch.cuda.synchronize()
        wall, peaks, values = [], [], []
        for _ in range(3):
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            result = population_error(x, observed, bandwidth, device_cache_bytes=budget)
            torch.cuda.synchronize()
            wall.append(time.perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated(device) - base)
            values.append(result)
        return {"wall_seconds": wall, "median_wall_seconds": statistics.median(wall),
                "peak_extra_allocated_bytes": max(peaks), "result": values[-1]}

    cached = measure(1024**3)
    bounded = measure(256 * 1024**2)
    # A CPU reference uses exactly the same full cells, gene axis and bandwidth.
    cpu = population_error(predicted, observed, bandwidth, device_cache_bytes=0)
    errors = {key: abs(bounded["result"][key] - cpu[key]) for key in ("mean_mse", "mmd_v", "error")}
    for key, error in errors.items():
        if error > 1e-11 or abs(cached["result"][key] - bounded["result"][key]) > 1e-11:
            raise AssertionError(f"FULL_GENE_NUMERICAL_MISMATCH: {key}: {error}")
    if bounded["peak_extra_allocated_bytes"] >= cached["peak_extra_allocated_bytes"]:
        raise AssertionError("BOUNDED_PATH_DID_NOT_REDUCE_EXTRA_DEVICE_ALLOCATION")
    report = {"scope": "synthetic_full_gene_metric_not_training_or_biological_accuracy",
        "source_identity": implementation_identity(), "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "genes": genes, "predicted_cells": len(predicted),
        "observed_cells": len(observed), "dtype": "FP64", "optimizer_steps": 0,
        "warmups_per_path": 1, "repetitions": 3, "device_free_bytes_before": free,
        "device_total_bytes": total, "full_FP64_input_bytes": required,
        "cached": cached, "bounded": bounded, "cpu_reference": cpu,
        "absolute_CPU_GPU_errors": errors,
        "memory_ratio": bounded["peak_extra_allocated_bytes"] / cached["peak_extra_allocated_bytes"],
        "whole_model_residency_verified": False, "passed": True}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
