"""Bounded real AE loader and update comparison; never save training weights.

Use an immutable local AE checkpoint. Source checks cover only the selected
condition/control pools. Source/CSR first-use checks and data initialization are
reported separately from hot sampling; no full-atlas checksum pass is requested.
"""

# pyright: reportMissingImports=false

import argparse
import copy
import gc
from pathlib import Path
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
EVIDENCE = ROOT / ".omo/evidence/ae-loader-20261008-154732"

import numpy as np
import pandas as pd
import psutil
import torch

from lung.data.reader import TrainingData
from lung.data.ae_expression_cache import NormalizedExpressionCache
from lung.engine.ae_data import AEDataSupply
from lung.engine.checkpoint import atomic_json, load_checkpoint, restore_rng
from lung.engine.config import config_hash, file_hash, implementation_identity, project_path
from lung.engine.data import EngineData, ROLES
from lung.engine.modular_config import load_modular_config
from lung.engine.numerics import PrecisionPolicy, policy_autocast
from lung.engine.response import tensor
from lung.engine.response_training import batch_tensor
from lung.models import ModelConfig
from lung.models.latent_staged import SharedAutoencoder


def generator(state):
    value = np.random.default_rng()
    value.bit_generator.state = copy.deepcopy(state)
    return value


def resources():
    memory = psutil.Process().memory_info()
    return {"available_ram_bytes": psutil.virtual_memory().available,
            "process_rss_bytes": memory.rss,
            "process_peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
            "system_swap_used_bytes": psutil.swap_memory().used}


def open_real_data(config):
    # Per-source audits stay enabled. Frozen full-role asset hashes are not
    # scanned by this bounded diagnostic; formal entry points are unchanged.
    def selected_reader(*args, **kwargs):
        return TrainingData(*args, **kwargs, verify_hashes=False)

    with patch("lung.data.TrainingData", selected_reader):
        data = EngineData(config, roles=("T", "V"))
    if data.synthetic or data.n_genes != 32461:
        raise ValueError("BENCHMARK_REQUIRES_REAL_FULL_GENE_DATA")
    return data


def resume_plan(config, data, saved, epoch, cursor):
    position = saved.get("extra", {}).get("resume")
    if saved.get("metadata", {}).get("stage") != "latent_ae" or not position:
        raise ValueError("BENCHMARK_REQUIRES_IN_EPOCH_AE_CHECKPOINT")
    if saved["epoch"] != epoch or position["next_group_start"] != cursor:
        raise ValueError("BENCHMARK_CHECKPOINT_CURSOR_MISMATCH")
    keys = list(data.iter_response_epoch("T", config["seed"] + 100000 + epoch))
    if position["keys"] != keys or position["epoch"] != epoch:
        raise ValueError("BENCHMARK_CHECKPOINT_CONDITION_ORDER_MISMATCH")
    if saved["metadata"].get("data_identity") != data.identity:
        raise ValueError("BENCHMARK_CHECKPOINT_DATA_MISMATCH")
    return keys[cursor:], copy.deepcopy(position["numpy_rng"])


def prime_selected_sources(data, supply, plans):
    """Exclude first source/CSR checks for selected full condition pools."""
    tick = perf_counter()
    names, cells = set(), 0
    for role, keys, _ in plans:
        internal = ROLES[role]
        pools = set(keys)
        for key in keys:
            pools.update(data.reader._control_ids[key])
        for key in sorted(pools):
            if key not in data.reader._present_conditions[internal]:
                continue
            rows = data.reader.role_cells(key, internal)
            cells += len(rows)
            names.update(rows.matrix_path)
    caches = (data.reader.rows.disk_cache, supply.reader.rows.disk_cache)
    for number, name in enumerate(sorted(names), 1):
        for cache in caches:
            cache.load(name, data.reader.rows.registry.get(name))
        if number % 100 == 0 or number == len(names):
            print(f"selected source warmup {number}/{len(names)}", flush=True)
    return {"seconds": perf_counter() - tick, "source_shards": len(names),
            "condition_pool_rows": cells, "included_in_hot_timings": False,
            "scope": "selected_conditions_and_same_role_controls_only"}


def sample(loader, keys, role, state):
    rng = generator(state)
    tick = perf_counter()
    batches = loader(keys, rng, role)
    return batches, rng.bit_generator.state, perf_counter() - tick


def compare_batches(before, after, rng_before, rng_after):
    if len(before) != len(after):
        raise AssertionError("AE_BATCH_COUNT_CHANGED")
    differences, duplicates = [], 0
    for old, new in zip(before, after):
        if old["condition"]["condition_id"] != new["condition"]["condition_id"]:
            raise AssertionError("AE_CONDITION_ORDER_CHANGED")
        for side in ("control", "target"):
            ids = side + "_cell_ids"
            np.testing.assert_array_equal(old[ids], new[ids])
            np.testing.assert_array_equal(old[side], new[side])
            differences.append(float(np.max(np.abs(old[side] - new[side]))))
            duplicates += len(old[ids]) - len(np.unique(old[ids]))
    if config_hash(rng_before) != config_hash(rng_after):
        raise AssertionError("AE_NUMPY_RNG_END_CHANGED")
    return {"cell_ids_and_order_equal": True, "expressions_elementwise_equal": True,
            "max_absolute_error": max(differences, default=0.),
            "numpy_rng_end_equal": True, "numpy_rng_end_sha256": config_hash(rng_after),
            "sampled_duplicate_cells": duplicates}


def forced_duplicates(data, supply, key, role, state, batch):
    control, target = data.reader._response_rows(key, batch, generator(state), ROLES[role])
    rows = pd.concat((target.iloc[[0]], control, target, control.iloc[[0]],
                      target.iloc[[0]], control.iloc[[0]]), ignore_index=True)
    reference = data.reader.normalize(data.reader.rows.read(rows))
    cached = supply.expressions.read(rows, ROLES[role])
    np.testing.assert_array_equal(reference, cached)
    return {"rows": len(rows), "forced_duplicate_order_equal": True,
            "max_absolute_error": float(np.max(np.abs(reference - cached)))}


def cache_inventory(directory):
    files = list(directory.rglob("*")) if directory.exists() else []
    return {"condition_manifests": sum(item.name == "manifest.json" for item in files),
            "bytes": sum(item.stat().st_size for item in files if item.is_file())}


def data_benchmark(data, supply, keys, role, state, batch, repeats):
    legacy = lambda group, rng, selected_role: [
        data.sample_response(key, batch, rng, role=selected_role) for key in group]
    optimized = lambda group, rng, selected_role: supply.sample_group(group, rng, role=selected_role)
    baseline, old_state, old_warm = sample(legacy, keys, role, state)
    inventory = cache_inventory(supply.expressions.directory)
    first, first_state, first_seconds = sample(optimized, keys, role, state)
    equality = compare_batches(baseline, first, old_state, first_state)
    after = cache_inventory(supply.expressions.directory)
    del first
    warmed, warm_state, warm_seconds = sample(optimized, keys, role, state)
    compare_batches(baseline, warmed, old_state, warm_state)
    del warmed
    times = {"legacy": [], "optimized": []}
    for repeat in range(repeats):
        order = ("legacy", "optimized") if repeat % 2 == 0 else ("optimized", "legacy")
        for name in order:
            drawn, end, seconds = sample(legacy if name == "legacy" else optimized, keys, role, state)
            compare_batches(baseline, drawn, old_state, end)
            times[name].append(seconds)
            del drawn
        print(f"{role} hot read {repeat + 1}/{repeats}: "
              f"legacy={times['legacy'][-1]:.3f}s optimized={times['optimized'][-1]:.3f}s", flush=True)
    equality["forced_repeated_rows"] = forced_duplicates(data, supply, keys[0], role, state, batch)
    medians = {name: statistics.median(values) for name, values in times.items()}
    return {"conditions": keys, "role": role, "conditions_per_group": len(keys),
            "warmup_seconds": {"legacy": old_warm, "optimized": warm_seconds},
            "normalized_cache_first_read_seconds": first_seconds,
            "cache_before": inventory, "cache_after": after,
            "new_condition_caches": after["condition_manifests"] - inventory["condition_manifests"],
            "hot_seconds": times, "hot_median_seconds": medians,
            "hot_speedup": medians["legacy"] / medians["optimized"], "equality": equality}


def load_optimizer(optimizer, saved):
    # Torch 2.13 casts moments to CUDA, but a noncapturable AdamW CPU step
    # aliases the source tensor. Clone those scalars only, not all CPU moments.
    source = {**saved["optimizer"], "state": {
        key: {name: value.clone() if name == "step" and isinstance(value, torch.Tensor) else value
              for name, value in entry.items()} for key, entry in saved["optimizer"]["state"].items()}}
    optimizer.load_state_dict(source)


def update_arm(config, data, supply, saved, keys, state, batch, optimized, warm_expression):
    device = torch.device(config["device"])
    policy = PrecisionPolicy.from_config(config)
    model = SharedAutoencoder(ModelConfig(**saved["metadata"]["model_config"]),
                              tensor(data.neighbors, device, torch.long)).to(device)
    model.precision_policy = policy
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["research"]["latent"]["ae"]["lr"],
                                  weight_decay=config["response"]["weight_decay"])
    load_optimizer(optimizer, saved)
    policy.apply()
    model.train()
    # Warm forward/backward, clipping and AdamW outside timing, then reload the
    # exact checkpoint weights/optimizer/RNG before each measured update.
    with policy_autocast(model, device):
        model.reconstruction_loss(tensor(warm_expression, device)).float().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["response"]["grad_clip"],
                                   error_if_nonfinite=True)
    optimizer.step()
    model.load_state_dict(saved["model"])
    load_optimizer(optimizer, saved)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    restore_rng(saved["rng"])
    policy.apply()
    rng = generator(state)
    resources_before = resources()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    tick, read_seconds = perf_counter(), 0.
    if optimized:
        read_tick = perf_counter()
        batches = supply.sample_group(keys, rng, role="T")
        read_seconds += perf_counter() - read_tick
    else:
        def sequential():
            nonlocal read_seconds
            for key in keys:
                read_tick = perf_counter()
                drawn = data.sample_response(key, batch, rng, role="T")
                read_seconds += perf_counter() - read_tick
                yield drawn
        batches = sequential()
    optimizer.zero_grad(set_to_none=True)
    losses, x, loss = [], None, None
    for drawn in batches:
        x = batch_tensor(np.concatenate((drawn["control"], drawn["target"])), device,
                         pin_memory=optimized)
        with policy_autocast(model, device):
            loss = model.reconstruction_loss(x).float()
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_AE_BENCHMARK_LOSS")
        (loss / len(keys)).backward()
        losses.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["response"]["grad_clip"],
                                   error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    seconds = perf_counter() - tick
    result = {"end_to_end_seconds": seconds, "data_prepare_wall_seconds": read_seconds,
              "condition_losses": losses, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "numpy_rng_end_sha256": config_hash(rng.bit_generator.state),
              "torch_rng_end_sha256": config_hash(torch.get_rng_state().tolist()),
              "cuda_rng_end_sha256": config_hash([item.tolist() for item in torch.cuda.get_rng_state_all()]),
              "resources_before": resources_before, "resources_after": resources()}
    parameters = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    moments = {key: {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                     for name, value in entry.items()} for key, entry in optimizer.state_dict()["state"].items()}
    del optimizer, model, batches, x, loss
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU {'optimized' if optimized else 'legacy'} update: {seconds:.3f}s "
          f"data={read_seconds:.3f}s", flush=True)
    return result, parameters, moments


def gpu_benchmark(config, data, supply, saved, keys, state, batch, measured_output):
    if not torch.cuda.is_available() or torch.device(config["device"]).type != "cuda":
        raise RuntimeError("GPU_BENCHMARK_REQUIRES_CUDA")
    if len(keys) != config["response"]["conditions_per_update"]:
        raise ValueError("GPU_BENCHMARK_REQUIRES_ONE_FULL_UPDATE_GROUP")
    original_steps = {key: value["step"].clone() for key, value in saved["optimizer"]["state"].items()}
    print(f"GPU cache warmup for {len(keys)} actual checkpoint conditions", flush=True)
    tick = perf_counter()
    warmed, end, _ = sample(lambda group, rng, role: supply.sample_group(group, rng, role=role),
                            keys, "T", state)
    cache_warmup_seconds = perf_counter() - tick
    warm_expression = np.concatenate((warmed[0]["control"], warmed[0]["target"]))
    del warmed
    legacy, old_parameters, old_moments = update_arm(
        config, data, supply, saved, keys, state, batch, False, warm_expression)
    optimized, new_parameters, new_moments = update_arm(
        config, data, supply, saved, keys, state, batch, True, warm_expression)
    atomic_json(measured_output, {"status": "measured_pending_equivalence_checks",
        "scope": "one_real_full_gene_update_per_arm", "legacy": legacy, "optimized": optimized,
        "cache_warmup_seconds_excluded": cache_warmup_seconds, "weights_saved": False})
    np.testing.assert_array_equal(legacy["condition_losses"], optimized["condition_losses"])
    if set(old_parameters) != set(new_parameters) or set(old_moments) != set(new_moments):
        raise AssertionError("AE_STATE_KEYS_CHANGED")
    parameter_error = 0.
    for name in old_parameters:
        old, new = old_parameters[name], new_parameters[name]
        if old.layout != torch.strided:
            old, new = old.to_sparse_coo().coalesce(), new.to_sparse_coo().coalesce()
            np.testing.assert_array_equal(old.indices().numpy(), new.indices().numpy())
            old, new = old.values(), new.values()
        torch.testing.assert_close(old, new, rtol=0., atol=0.)
        parameter_error = max(parameter_error, float((old - new).abs().max()) if old.numel() else 0.)
    for key, entries in old_moments.items():
        for name, value in entries.items():
            other = new_moments[key][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0., atol=0.)
            elif value != other:
                raise AssertionError("AE_OPTIMIZER_STATE_CHANGED")
    for name in ("numpy_rng_end_sha256", "torch_rng_end_sha256", "cuda_rng_end_sha256"):
        if legacy[name] != optimized[name]:
            raise AssertionError("AE_UPDATE_RANDOM_STREAM_CHANGED")
    for key, value in original_steps.items():
        torch.testing.assert_close(value, saved["optimizer"]["state"][key]["step"], rtol=0., atol=0.)
    return {"scope": "one_real_full_gene_update_per_arm_with_identical_checkpoint_state",
            "device": torch.cuda.get_device_name(), "precision": PrecisionPolicy.from_config(config).identity(),
            "conditions": keys, "conditions_per_update": len(keys), "optimizer_updates_per_arm": 1,
            "gpu_kernel_warmup_updates_per_arm_excluded": 1,
            "checkpoint_state_restored_after_kernel_warmup": True,
            "normalized_cache_warmup_seconds_excluded": cache_warmup_seconds,
            "legacy": legacy, "optimized": optimized,
            "end_to_end_speedup": legacy["end_to_end_seconds"] / optimized["end_to_end_seconds"],
            "parameters_and_optimizer_state_equal": True, "parameter_max_absolute_error": parameter_error,
            "source_optimizer_steps_unchanged": True,
            "prefetch_overlap_measured": False, "weights_saved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/modular_research.yaml")
    parser.add_argument("--conditions", type=int, default=2)
    parser.add_argument("--groups16", action="store_true", help="Use 16 conditions for loader timing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-epoch", type=int, default=83)
    parser.add_argument("--expected-cursor", type=int, default=448)
    parser.add_argument("--gpu", action="store_true", help="Compare one 16-condition AE update per arm")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "ae-data-benchmark.json")
    args = parser.parse_args()
    count = 16 if args.groups16 else args.conditions
    if not 1 <= count <= 16 or not 1 <= args.repeats <= 10:
        raise ValueError("BOUNDED_BENCHMARK_REQUIRES_1_TO_16_CONDITIONS_AND_1_TO_10_REPEATS")
    output = args.output.resolve()
    if not output.is_relative_to(EVIDENCE.resolve()):
        raise ValueError("BENCHMARK_OUTPUT_MUST_REMAIN_IN_EVIDENCE_DIRECTORY")
    report = {"status": "running", "scope": "bounded_real_T_and_V_full_32461_genes",
              "synthetic": False, "formal_training": False, "weights_saved": False,
              "full_asset_hash_pass": False, "command": [sys.executable, *sys.argv],
              "resources_at_start": resources(),
              "limits": ["Hot readings reuse selected-source audit state and OS page cache.",
                         "Sequential reference uses current raw CSR reader and normalization.",
                         "A single GPU update does not measure inter-group prefetch overlap.",
                         "No full epoch, convergence, or biological effect is evaluated."]}
    source = implementation_identity()
    report["implementation_identity"] = source
    supply, data = None, None
    tick = perf_counter()
    try:
        config = load_modular_config(args.config)
        if config.get("synthetic"):
            raise ValueError("SYNTHETIC_CONFIG_FORBIDDEN")
        checkpoint = project_path(args.checkpoint)
        saved = load_checkpoint(checkpoint)
        checkpoint_hash = file_hash(checkpoint)
        data = open_real_data(config)
        remaining, train_state = resume_plan(config, data, saved, args.expected_epoch, args.expected_cursor)
        width = config["response"]["conditions_per_update"] if args.gpu else count
        if len(remaining) < max(count, width):
            raise ValueError("CHECKPOINT_HAS_TOO_FEW_REMAINING_CONDITIONS")
        if saved["metadata"].get("precision") != PrecisionPolicy.from_config(config).identity():
            raise ValueError("CHECKPOINT_PRECISION_POLICY_MISMATCH")
        batch = int(config["research"]["latent"]["ae"]["batch"])
        supply = AEDataSupply(data, batch)
        supply.expressions.close()
        supply.expressions = NormalizedExpressionCache(supply.reader, directory=output.parent / "benchmark-cache")
        validation_seed = 20260923 + 998
        val_keys = data.condition_ids("V")[:count]
        val_state = np.random.default_rng(validation_seed).bit_generator.state
        report.update({"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                       "checkpoint_epoch": saved["epoch"], "next_group_start": args.expected_cursor,
                       "data_identity": data.identity, "n_genes": data.n_genes, "cells_per_side": batch,
                       "normalized_cache_directory": str(supply.expressions.directory),
                       "data_initialization_seconds_excluded": perf_counter() - tick})
        report["source_first_use_checks"] = prime_selected_sources(
            data, supply, (("T", remaining[:max(count, width)], train_state), ("V", val_keys, val_state)))
        report["T"] = data_benchmark(data, supply, remaining[:count], "T", train_state, batch, args.repeats)
        atomic_json(output, report)
        report["V"] = data_benchmark(data, supply, val_keys, "V", val_state, batch, args.repeats)
        atomic_json(output, report)
        if args.gpu:
            report["gpu_update"] = gpu_benchmark(config, data, supply, saved, remaining[:width], train_state,
                                                  batch, output.parent / "ae-update-measured.json")
        if file_hash(checkpoint) != checkpoint_hash:
            raise RuntimeError("CHECKPOINT_CHANGED_DURING_BENCHMARK")
        if implementation_identity() != source:
            raise RuntimeError("SOURCE_CHANGED_DURING_BENCHMARK")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if supply is not None:
            supply.close()
        if data is not None and data.reader.rows.disk_cache is not None:
            data.reader.rows.disk_cache.close()
        report["total_wall_seconds"] = perf_counter() - tick
        report["resources_at_end"] = resources()
        atomic_json(output, report)
        print(f"{report['status']}: {output}", flush=True)


if __name__ == "__main__":
    main()
