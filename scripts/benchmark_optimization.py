"""Bounded real-T diagnostic. Never steps an optimizer or saves model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from lung.engine import response
from lung.engine.config import load_config, merge, implementation_identity
from lung.engine.data import open_data
from lung.engine.checkpoint import seed_all
from lung.models import ModelConfig, build_response_model


def batch_digest(batch):
    digest = hashlib.sha256()
    for name in ("control", "target", "control_cell_ids", "target_cell_ids"):
        if name not in batch:
            continue
        value = np.asarray(batch[name])
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes() if value.dtype.kind != "O" else json.dumps(value.tolist()).encode())
    return digest.hexdigest()


def sample_pass(data, keys, label):
    rng, rows = np.random.default_rng(81723), []
    for key in keys:
        start = time.perf_counter()
        batch = data.sample_response(key, 128, rng, "T")
        elapsed = time.perf_counter() - start
        rows.append({"condition": key, "seconds": elapsed, "batch_sha256": batch_digest(batch)})
        print(f"{label}: {key} {elapsed:.4f}s", flush=True)
    return {"rows": rows, "mean_seconds": statistics.mean(row["seconds"] for row in rows)}


class FrozenBatch:
    """Reuses measured real expressions only for the compute-only comparison."""
    def __init__(self, data, batch):
        self.data = data
        values = [batch] if isinstance(batch, dict) else batch
        self.batches = {value["condition_id"]: value for value in values}

    def __getattr__(self, name):
        return getattr(self.data, name)

    def sample_response(self, condition_id, n, rng, role):
        if role != "T" or condition_id not in self.batches:
            raise ValueError("DIAGNOSTIC_T_CONDITION_ONLY")
        batch = self.batches[condition_id]
        selected = {key: value for key, value in batch.items() if key not in ("control", "target")}
        for key in ("control", "target"):
            array = batch[key]
            selected[key] = array[rng.choice(len(array), size=n, replace=n > len(array))]
        return selected


def compute_pass(config, data, keys):
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_GPU_DIAGNOSTIC")
    torch.set_num_threads(4)
    seed_all(81723)
    model = build_response_model(ModelConfig(**config["response"]["model"]),
                                 torch.as_tensor(data.neighbors, device=device)).to(device)
    model.train()
    key = keys[0]
    fixed = FrozenBatch(data, [data.sample_response(cid, 128, np.random.default_rng(17), "T") for cid in keys[:2]])
    rows = []
    for mode in ("replay", "direct"):
        times, memories = [], []
        metrics = {}
        for repeat in range(4):
            seed_all(812)
            rng, mmd_rng, generator = response.response_rngs(812, device)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            metrics = response.train_condition(model, fixed, key, rng,
                {**config["response"], "mmd_backward": mode}, device, 1,
                mmd_rng=mmd_rng, mmd_generator=generator)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                       if parameter.grad is not None):
                raise FloatingPointError("NONFINITE_DIAGNOSTIC_GRADIENT")
            if repeat:
                times.append(elapsed)
                memories.append(torch.cuda.max_memory_allocated())
        rows.append({"mode": mode, "seconds": times, "median_seconds": statistics.median(times),
                     "max_allocated_bytes": max(memories), "metrics": metrics})
        print(f"cached compute {mode}: {rows[-1]['median_seconds']:.4f}s", flush=True)
    grouped = []
    if len(keys) >= 2:
        from lung.engine.response_training import train_conditions
        gradient_reference = None
        for width in (1, 2):
            times, memories = [], []
            metrics = []
            for repeat in range(4):
                seed_all(813)
                rng, mmd_rng, generator = response.response_rngs(813, device)
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                metrics = train_conditions(model, fixed, keys[:2], rng,
                    {**config["response"], "mmd_backward": "direct", "condition_microbatch": width},
                    device, divisor=8, mmd_rng=mmd_rng, mmd_generator=generator)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if repeat:
                    times.append(elapsed)
                    memories.append(torch.cuda.max_memory_allocated())
            gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                         if parameter.grad is not None}
            error = 0.
            if gradient_reference is None:
                gradient_reference = gradients
            else:
                if set(gradients) != set(gradient_reference):
                    raise AssertionError("BATCHED_PARAMETER_GRADIENT_SET_CHANGED")
                for name in gradients:
                    torch.testing.assert_close(gradients[name], gradient_reference[name], rtol=3e-5, atol=3e-6)
                    error = max(error, float((gradients[name] - gradient_reference[name]).abs().max()))
            grouped.append({"condition_microbatch": width, "conditions": 2, "seconds": times,
                            "median_seconds": statistics.median(times), "max_allocated_bytes": max(memories),
                            "gradient_max_absolute_error": error, "metrics": metrics})
            print(f"two-condition group width={width}: {grouped[-1]['median_seconds']:.4f}s", flush=True)
    model.zero_grad(set_to_none=True)
    model_dtype = str(next(model.parameters()).dtype)
    del model
    return {"device": torch.cuda.get_device_name(), "dtype": model_dtype,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cases": rows, "condition_batches": grouped,
            "scope": "cached_real_condition_forward_backward_without_optimizer_or_validation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cache-bytes", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis_outputs/optimization_v4/benchmark.json")
    args = parser.parse_args()
    if not 1 <= args.conditions <= 8:
        raise ValueError("Bounded diagnostic requires 1 to 8 conditions")
    config = load_config()
    if args.cache_bytes is not None:
        if args.cache_bytes < 0:
            raise ValueError("cache-bytes must be nonnegative")
        config = merge(config, {"data_loading": {"row_cache": {"max_cache_bytes": args.cache_bytes}}})
    identity = implementation_identity()
    start = time.perf_counter()
    legacy = open_data(merge(config, {"data_loading": {"row_cache": {"use_disk_cache": False}}}),
                       "response", prepare=False)
    keys = list(legacy.iter_response_epoch("T", config["response"]["seed"]))[:args.conditions]
    before = sample_pass(legacy, keys, "NPZ baseline")
    del legacy
    cached = open_data(config, "response", prepare=False)
    first = sample_pass(cached, keys, "cache first pass")
    warm = sample_pass(cached, keys, "cache reuse")
    hashes = lambda result: [row["batch_sha256"] for row in result["rows"]]
    if not hashes(before) == hashes(first) == hashes(warm):
        raise AssertionError("CACHE_CHANGED_SAMPLED_EXPRESSIONS")
    report = {"scope": "bounded_real_T_only", "formal_training": False,
        "optimizer_updates": 0, "weights_saved": False, "source_identity": identity,
        "data_identity": cached.identity, "conditions": keys, "expressions_equal": True,
        "response_config": config["response"],
        "cache_options": config.get("data_loading", {}).get("row_cache", {}),
        "data": {"npz_baseline": before, "cache_first_pass": first, "cache_reuse": warm},
        "first_pass_note": "includes any required conversion and first-use checks; existing cache may be reused"}
    if args.gpu:
        report["compute"] = compute_pass(config, cached, keys)
    report["total_seconds"] = time.perf_counter() - start
    if implementation_identity() != identity:
        raise RuntimeError("SOURCE_CHANGED_DURING_DIAGNOSTIC")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
