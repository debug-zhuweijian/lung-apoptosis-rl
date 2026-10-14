"""Measure complete real AE epochs through the production loop in isolation."""

# pyright: reportMissingImports=false

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch

from benchmark_ae_data_supply import open_real_data, resources
from lung.engine import latent_pipeline
from lung.engine.ae_data import AEGroupPrefetch
from lung.engine.checkpoint import atomic_json, load_checkpoint
from lung.engine.config import file_hash, implementation_identity, project_path
from lung.engine.modular_config import load_modular_config, stage_config


class BenchmarkComplete(Exception):
    pass


class EpochMonitor:
    def __init__(self, directory, report, target):
        self.experiment = directory
        self.weights = directory / "weights"
        self.report = report
        self.target = target
        self.epoch_started = perf_counter()
        self.group_started = self.epoch_started
        self.fetch_wait = 0.
        self.last_cursor = report["source_cursor"]
        self.validation_seconds = 0.

    def say(self, message):
        print(message, flush=True)

    def record(self, stage, epoch, metrics):
        torch.cuda.synchronize()
        now = perf_counter()
        row = {"epoch": epoch, "seconds": now - self.epoch_started,
               "validation_seconds": self.validation_seconds, "metrics": metrics,
               "scope": "resume_remainder_and_initialization" if not self.report["epochs"] else "full_epoch"}
        groups = [g for g in self.report["groups"] if g["epoch_completed_after_this_round"] == epoch]
        row.update(updates=len(groups), conditions=sum(g["conditions"] for g in groups),
                   fetch_wait_seconds=sum(g["fetch_wait_seconds"] for g in groups),
                   checkpoint_seconds=sum(g["checkpoint_seconds"] for g in groups))
        if row["scope"] == "full_epoch":
            row["conditions_per_second"] = row["conditions"] / row["seconds"]
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite production AE metrics")
        self.report["epochs"].append(row)
        atomic_json(self.experiment / "report.json", self.report)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        self.epoch_started = now
        self.validation_seconds = 0.
        self.last_cursor = 0
        if len(self.report["epochs"]) >= self.target + 1:
            raise BenchmarkComplete()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--full-epochs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.full_epochs < 1:
        raise ValueError("At least one complete epoch is required")
    directory = project_path(args.output_dir)
    if not directory.is_relative_to((ROOT / ".omo/evidence").resolve()) or directory.exists():
        raise ValueError("Use a fresh directory inside project evidence")
    checkpoint = project_path(args.checkpoint)
    saved = load_checkpoint(checkpoint)
    position = saved["extra"].get("resume")
    if not position or position["epoch"] != saved["epoch"]:
        raise ValueError("An immutable in-epoch AE checkpoint is required")
    config = stage_config(load_modular_config(), "ae", {})
    if config.get("synthetic") or not torch.cuda.is_available():
        raise ValueError("Real CUDA data and the existing stage configuration are required")
    checkpoint_hash = file_hash(checkpoint)
    source = implementation_identity()
    directory.mkdir(parents=True)
    report = {"status": "running", "formal_training": False,
              "full_asset_hash_pass": False,
              "scope": "production_AE_loop_with_prefetch_validation_and_checkpoint_saves",
              "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "implementation_identity": source, "checkpoint": str(checkpoint),
              "checkpoint_sha256": checkpoint_hash, "source_epoch": saved["epoch"],
              "source_cursor": position["next_group_start"],
              "source_optimizer_step": saved["counters"]["optimizer_step"],
              "requested_full_epochs": args.full_epochs, "epochs": [], "groups": [],
              "resources_before": resources(), "device": torch.cuda.get_device_name(),
              "production_latest_updated": False,
              "limits": ["The first resumed remainder includes initialization and source/cache first-use costs.",
                         "Complete epochs use the unchanged scientific configuration and production loop.",
                         "Measured epoch duration does not predict early-stopping time or model efficacy."]}
    monitor = EpochMonitor(directory, report, args.full_epochs)
    folder = monitor.weights / "latent_ae"
    folder.mkdir(parents=True)
    shutil.copy2(checkpoint, folder / "last.pt")
    best = checkpoint.with_name("best.pt")
    if best.is_file():
        shutil.copy2(best, folder / "best.pt")
    del saved
    atomic_json(directory / "report.json", report)
    original_consume = AEGroupPrefetch.consume
    original_save = latent_pipeline.atomic_save
    original_validation = latent_pipeline._ae_validation

    def consume(prefetch, keys, rng):
        monitor.group_started = perf_counter()
        value = original_consume(prefetch, keys, rng)
        monitor.fetch_wait = perf_counter() - monitor.group_started
        return value

    def save(path, payload):
        tick = perf_counter()
        original_save(path, payload)
        position = payload.get("extra", {}).get("resume")
        if position is not None and Path(path).name == "last.pt":
            now = perf_counter()
            cursor = position["next_group_start"]
            row = {"epoch_completed_after_this_round": position["epoch"] + 1,
                   "optimizer_step": payload["counters"]["optimizer_step"],
                   "conditions": cursor - monitor.last_cursor,
                   "fetch_wait_seconds": monitor.fetch_wait,
                   "checkpoint_seconds": now - tick,
                   "group_including_checkpoint_seconds": now - monitor.group_started}
            report["groups"].append(row)
            monitor.last_cursor = cursor
            atomic_json(directory / "report.json", report)
            print(f"AE production update {row['optimizer_step']}: "
                  f"{row['group_including_checkpoint_seconds']:.2f}s, "
                  f"fetch wait {row['fetch_wait_seconds']:.2f}s", flush=True)

    def validation(*values, **kwargs):
        torch.cuda.synchronize()
        tick = perf_counter()
        result = original_validation(*values, **kwargs)
        torch.cuda.synchronize()
        monitor.validation_seconds = perf_counter() - tick
        return result

    data = None
    tick = perf_counter()
    try:
        data = open_real_data(config)
        report.update(data_identity=data.identity, n_genes=data.n_genes,
                      data_initialization_seconds=perf_counter() - tick,
                      conditions_per_full_epoch=len(data.condition_ids("T")),
                      validation_conditions=len(data.condition_ids("V")))
        with patch.object(AEGroupPrefetch, "consume", consume), \
             patch.object(latent_pipeline, "atomic_save", save), \
             patch.object(latent_pipeline, "_ae_validation", validation):
            try:
                latent_pipeline.run_latent_ae(config, monitor, data)
            except BenchmarkComplete:
                pass
        if len(report["epochs"]) < args.full_epochs + 1:
            raise RuntimeError("Training stopped before the requested complete epochs were measured")
        if any(row["conditions"] != report["conditions_per_full_epoch"] for row in report["epochs"][1:]):
            raise RuntimeError("A measured epoch omitted conditions")
        if implementation_identity() != source or file_hash(checkpoint) != checkpoint_hash:
            raise RuntimeError("Source or production checkpoint changed during measurement")
        report.update(status="passed", production_checkpoint_unchanged=True)
    except Exception as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if data is not None and data.reader.rows.disk_cache is not None:
            data.reader.rows.disk_cache.close()
        report.update(total_seconds=perf_counter() - tick, resources_after=resources())
        atomic_json(directory / "report.json", report)
        print(f"{report['status']}: {directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
