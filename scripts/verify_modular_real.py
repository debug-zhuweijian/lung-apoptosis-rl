"""Explicitly bounded real-data engineering verification; never a training launcher.

Without --run this file prints its scope and performs no data or model work.
The registered full numerical audit is never replaced by a small-sample pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.data import TrainingData
from lung.engine.checkpoint import atomic_json, atomic_save, load_checkpoint
from lung.engine.config import config_hash, file_hash, merge, verify_interpreter
from lung.engine.data import EngineData
from lung.engine.modular_budget import StagePaused


class BoundedRows:
    """Reuse only already-read rows from the explicitly registered small view."""

    def __init__(self, source, frames):
        self.source = source
        self.registered = pd.concat(list(frames.values()), ignore_index=True).set_index("cell_id", drop=False)
        if self.registered.index.duplicated().any():
            raise ValueError("BOUNDED_VIEW_DUPLICATE_CELL_IDS")
        self.cached = {}
        self.source_reads = self.source_cells_read = 0

    def read(self, rows):
        ids = rows["cell_id"].astype(str).tolist()
        if not ids or any(cell not in self.registered.index for cell in ids):
            raise PermissionError("BOUNDED_VIEW_CELL_NOT_REGISTERED")
        expected = self.registered.loc[ids]
        for column in ("matrix_path", "matrix_row", "condition_id", "role"):
            if column in rows and column in expected:
                if rows[column].astype(str).tolist() != expected[column].astype(str).tolist():
                    raise PermissionError("BOUNDED_VIEW_ROW_LOCATION_MISMATCH")
        missing = list(dict.fromkeys(cell for cell in ids if cell not in self.cached))
        if missing:
            counts = self.source.read(self.registered.loc[missing].reset_index(drop=True))
            if len(missing) != counts.shape[0]:
                raise ValueError("BOUNDED_VIEW_SOURCE_ROW_COUNT_MISMATCH")
            self.cached = {**self.cached, **{cell: counts[index:index + 1]
                                           for index, cell in enumerate(missing)}}
            self.source_reads += 1
            self.source_cells_read += len(missing)
        pieces = [self.cached[cell] for cell in ids]
        return sparse.vstack(pieces, format="csr") if sparse.issparse(pieces[0]) else np.concatenate(pieces)


class BoundedRealReader(TrainingData):
    """Use TrainingData's sampling laws with fixed subsets of its original roles."""

    def __init__(self, source, train_conditions, val_conditions, cells_per_condition, state_fold):
        self._source, self.state_fold = source, state_fold
        self.allowed_roles = frozenset(("fit", "selection", "calibration"))
        self.allow_test = False
        self.gene_ids = source.gene_ids.copy()
        self.n_genes = source.n_genes
        self.assets_path = source.assets_path
        self.manifest = source.manifest
        self._treated, self._frames, control_rows = {}, {}, []
        for role, count in (("fit", train_conditions), ("selection", val_conditions)):
            candidates = sorted(source.condition_ids(role))
            if len(candidates) < count:
                raise ValueError(f"BOUNDED_VIEW_CONDITIONS_INSUFFICIENT: {role}")
            self._treated[role] = tuple(candidates[:count])
            present = frozenset(source.condition_ids(role, include_controls=True))
            controls = source.controls.loc[source.controls.condition_id.isin(self._treated[role])]
            controls = controls.loc[controls.control_condition_id.isin(present)].copy()
            if not set(self._treated[role]) <= set(controls.condition_id):
                raise ValueError(f"BOUNDED_VIEW_CONTROL_MISSING: {role}")
            control_rows.append(controls)
            keys = sorted(set(self._treated[role]) | set(controls.control_condition_id))
            for condition in keys:
                original = source.role_cells(condition, role)
                order = [name for name in ("matrix_path", "matrix_row", "cell_id") if name in original]
                selected = original.sort_values(order, kind="stable").head(cells_per_condition).copy()
                if selected.empty:
                    raise ValueError(f"BOUNDED_VIEW_EMPTY_CONDITION: {role}/{condition}")
                self._frames[role, condition] = selected.reset_index(drop=True)
        self.controls = pd.concat(control_rows, ignore_index=True).drop_duplicates()
        self._control_ids = {key: tuple(frame.control_condition_id)
                             for key, frame in self.controls.groupby("condition_id", sort=False)}
        self._present_conditions = {role: frozenset(key for r, key in self._frames if r == role)
                                    for role in self._treated}
        selected_ids = {key for _, key in self._frames}
        self.conditions = source.conditions.loc[source.conditions.condition_id.isin(selected_ids)].copy()
        self.condition_table = self.conditions.set_index("condition_id", drop=False)
        self.rows = BoundedRows(source.rows, self._frames)
        self.state_rows = pd.read_parquet(self.assets_path / "tidyi_splits.parquet",
            columns=["cell_id", "fold", "role", "state_class"],
            filters=[("fold", "==", state_fold), ("role", "in", ["fit", "selection", "calibration"])])
        if any(self.state_rows.loc[self.state_rows.role.eq(role)].empty for role in self.allowed_roles):
            raise ValueError("BOUNDED_VIEW_STATE_ROLE_MISSING")

    def condition_ids(self, role="fit", *, include_controls=False):
        if role not in self._treated:
            raise PermissionError("BOUNDED_VIEW_RESPONSE_ROLE_FORBIDDEN")
        return sorted(self._present_conditions[role]) if include_controls else list(self._treated[role])

    def _is_response(self, condition_id, role):
        return condition_id in self.condition_ids(role)

    def _cell_metadata(self, condition_id, role):
        self.condition_ids(role)
        if (role, condition_id) not in self._frames:
            raise PermissionError("BOUNDED_VIEW_CONDITION_ROLE_MISMATCH")
        return self._frames[role, condition_id].copy()

    def normalize(self, counts):
        return self._source.normalize(counts)

    def action_features(self, condition_id):
        if condition_id not in self.condition_table.index:
            raise PermissionError("BOUNDED_VIEW_ACTION_NOT_REGISTERED")
        return self._source.action_features(condition_id)

    def state_arrays(self, fold="1_to_8", role="fit"):
        if fold != self.state_fold or role not in self.allowed_roles:
            raise PermissionError("BOUNDED_VIEW_STATE_ROLE_FORBIDDEN")
        values, labels = self._source.state_arrays(fold=fold, role=role)
        registered = self.state_rows.loc[self.state_rows.role.eq(role)]
        if len(values) != len(registered) or not np.array_equal(labels, registered.state_class.to_numpy()):
            raise ValueError("BOUNDED_VIEW_STATE_LABELS_CHANGED")
        return values, labels

    def registration(self):
        return {"selection_rule": "first_conditions_then_matrix_path_row_cell_id_without_outcome_access",
                "treated_conditions": {role: list(values) for role, values in self._treated.items()},
                "cells": {role: {condition: frame.cell_id.astype(str).tolist()
                                 for (r, condition), frame in self._frames.items() if r == role}
                          for role in self._treated},
                "state_fold": self.state_fold,
                "state_rows": self.state_rows.to_dict("records")}


class BoundedRealData(EngineData):
    synthetic = False
    is_bounded_real_view = True

    def __init__(self, source, train_conditions=16, val_conditions=1, cells_per_condition=128,
                 state_fold="1_to_8"):
        if source.synthetic or source.n_genes != 32461:
            raise ValueError("BOUNDED_REAL_SOURCE_REQUIRES_FULL_32461_GENES")
        if any(type(value) is not int or value < 1 for value in (train_conditions, val_conditions, cells_per_condition)):
            raise ValueError("BOUNDED_VIEW_COUNTS_INVALID")
        if cells_per_condition > 128:
            raise ValueError("BOUNDED_VIEW_CELL_CAP_EXCEEDS_128")
        self.source_identity = source.identity
        self.reader = BoundedRealReader(source.reader, train_conditions, val_conditions,
                                        cells_per_condition, state_fold)
        self.selection = {"protocol": "modular-bounded-real-v1", "source_identity": source.identity,
                          "normalization_signature": source.manifest.get("signature"),
                          "gene_axis_sha256": config_hash(source.reader.gene_ids.tolist()),
                          "limits": {"T": train_conditions, "V": val_conditions, "cells": cells_per_condition},
                          **self.reader.registration()}
        self.identity = self.view_identity = config_hash(self.selection)
        self.manifest = {**source.manifest, **self.evidence, "view_registration": self.selection}
        self.mu, self.scale, self.neighbors = source.mu, source.scale, source.neighbors
        self.n_genes, self.assets_dir, self.bandwidth = source.n_genes, source.assets_dir, source.bandwidth

    @property
    def evidence(self):
        return {"synthetic": False, "is_bounded_real_view": True,
                "source_identity": self.source_identity, "view_identity": self.view_identity,
                "formal_training": False, "scope": "bounded_engineering_verification_only"}


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run", action="store_true", help="Execute the explicitly bounded verification")
    command.add_argument("--config", default="configs/modular_research.yaml")
    command.add_argument("--run-id")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--max-seconds", type=float, default=600.)
    command.add_argument("--train-conditions", type=int, default=16)
    command.add_argument("--val-conditions", type=int, default=1)
    command.add_argument("--cells-per-condition", type=int, default=128)
    command.add_argument("--ode-samples", type=int, choices=(0, 1, 2), default=2)
    command.add_argument("--no-progress", action="store_true")
    return command


def verification_config(base, arguments):
    if base.get("synthetic", False):
        raise ValueError("BOUNDED_REAL_CONFIG_CANNOT_BE_SYNTHETIC")
    return merge(base, {"workflow": "modular_stage", "modular_stage": "bounded_real_verification",
        "stage": "response", "epochs": 1, "synthetic": False, "auto_prepare": False,
        "device": arguments.device, "early_stopping": False, "dashboard": False,
        "open_browser": False, "progress": not arguments.no_progress,
        "verification": {"is_bounded_real_view": True, "formal_training": False},
        "response": {"members": 1, "member_workers": 1, "ot_samples": 128, "cfm_samples": 128,
                     "microbatch": 128, "conditions_per_update": arguments.train_conditions,
                     "inference_ode_steps": 128},
        "research": {"latent": {"d_model": 256, "latent_dim": 256,
                                  "ae": {"batch": 128}, "refine": {"enabled": False}}}})


def _finite_tensors(value):
    if torch.is_tensor(value):
        values = value.coalesce().values() if value.is_sparse else value
        return bool(torch.isfinite(values).all())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(item) for item in value)
    return True


def checkpoint_evidence(path, data, expected_updates=None):
    """Tag only this new run's checkpoints before their hashes become dependencies."""
    path = Path(path)
    for candidate in path.parent.glob("*.pt"):
        if candidate.name == "temperature-last.pt":
            continue  # This is calibrator progress, not a published state classifier.
        saved = load_checkpoint(candidate)
        if saved.get("metadata", {}).get("data_identity") != data.identity:
            raise ValueError("BOUNDED_CHECKPOINT_DATA_IDENTITY_MISMATCH")
        atomic_save(candidate, {**saved, "metadata": {**saved["metadata"], **data.evidence}})
    saved = load_checkpoint(path)
    counters = saved.get("counters", {})
    updates = counters.get("optimizer_step", 0)
    finite = all(_finite_tensors(saved.get(key, {})) for key in ("model", "shared", "optimizer"))
    if updates < 1 or not finite or (expected_updates is not None and updates != expected_updates):
        raise ValueError("BOUNDED_CHECKPOINT_UPDATE_OR_FINITE_CHECK_FAILED")
    if saved.get("extra", {}).get("budget_exhausted", False):
        raise StagePaused("budget_exhausted", path, stage=saved["metadata"]["stage"], step=updates)
    return {**data.evidence, "path": str(path), "sha256": file_hash(path), "optimizer_steps": updates,
            "epoch": saved["epoch"], "finite": finite, "metadata": saved["metadata"],
            "validation": saved.get("extra", {}).get("validation")}


def _cuda_memory(device):
    if torch.device(device).type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device), "free": free, "total": total}


def execute_verification(arguments):
    from lung.engine.modular_config import load_modular_config
    from lung.engine.latent_pipeline import run_latent_ae, run_latent_cache, run_latent_cfm
    from lung.engine.auxiliary import train_pool, train_state, load_auxiliary, verify_auxiliary_dependencies
    from lung.engine.representation_cache import load_representation_cache
    from lung.engine.response_research import ResearchBudget
    from lung.monitoring import RunMonitor

    verify_interpreter()
    if torch.device(arguments.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE: no silent CPU fallback")
    run_id = arguments.run_id or ("modular-real-verification-" +
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    if Path(run_id).name != run_id or not run_id.startswith("modular-real-verification-"):
        raise ValueError("BOUNDED_VERIFICATION_RUN_ID_INVALID")
    if any((ROOT / folder / run_id).exists() for folder in ("experiments", "checkpoints", "logs")):
        raise FileExistsError("Verification run exists; choose a new run_id without overwriting it")
    started = time.perf_counter()
    config = verification_config(load_modular_config(arguments.config), arguments)
    source = EngineData(config, roles=("T", "V", "C"))
    data = BoundedRealData(source, arguments.train_conditions, arguments.val_conditions,
                           arguments.cells_per_condition, config["state_fold"])
    config = {**config, "verification": data.evidence}
    monitor = RunMonitor(config, run_id=run_id)
    monitor.update_status("running", **data.evidence)
    report_path = monitor.experiment / "verification_report.json"
    atomic_json(monitor.experiment / "bounded_real_view.json", {**data.evidence, **data.selection})
    budget = ResearchBudget(arguments.max_seconds, monitor.experiment / "verification_budget.json")
    budget.consumed = time.perf_counter() - started
    monitor.stop_check = budget.check
    pipeline, stages = {}, []
    report = {**data.evidence, "run_id": run_id, "n_genes": data.n_genes, "stages": stages,
        "config": config, "prepare_seconds": budget.consumed,
        "gates": {"1": {"passed": False, "reason": "No audited World or Actor/Critic update in this script"},
                  "6": {"passed": False, "reason": "Full real numerical audit and World scoring remain unverified"}},
        "formal_dataset_reusable": False, "population_audit_run": False, "world_or_rl_started": False}

    def persist():
        atomic_json(report_path, {**report, "stages": stages, "pipeline": pipeline,
                    "elapsed_seconds": time.perf_counter() - started, "budget": budget.summary(),
                    "source_reads": data.reader.rows.source_reads,
                    "unique_source_cells_read": data.reader.rows.source_cells_read})

    def run_stage(name, action):
        if budget.check(0, 0):
            raise StagePaused("budget_exhausted", stage=name)
        gc.collect()
        if torch.device(config["device"]).type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(config["device"])
            torch.cuda.reset_peak_memory_stats(config["device"])
        before = time.perf_counter()
        record = {**data.evidence, "stage": name, "status": "running"}
        try:
            result = action()
            record.update(status=result.get("status", "complete"), result=result)
            return result
        except BaseException as error:
            record.update(status="paused" if isinstance(error, StagePaused) else "failed",
                          error_type=type(error).__name__, error=str(error))
            if isinstance(error, StagePaused):
                record.update(checkpoint_path=getattr(error, "checkpoint_path"), step=getattr(error, "step"))
            raise
        finally:
            if torch.device(config["device"]).type == "cuda":
                torch.cuda.synchronize(config["device"])
            record.update(seconds=time.perf_counter() - before, gpu_memory=_cuda_memory(config["device"]))
            stages.append(record)
            persist()

    def train_ae():
        result = run_latent_ae(config, monitor, data, budget)
        return checkpoint_evidence(result["weights"], data, expected_updates=1)

    def cache_h():
        result = run_latent_cache(config, monitor, data, budget)
        cached = load_representation_cache(result["directory"], "cpu", expected_data_identity=data.identity)
        if not all(bool(torch.isfinite(value).all()) for value in cached.h.values()):
            raise ValueError("BOUNDED_CACHE_NONFINITE")
        atomic_json(Path(result["cache"]), {**cached.manifest, **data.evidence})
        return {**data.evidence, **result, "summary": cached.summary(), "finite": True}

    def train_cfm():
        result = run_latent_cfm(config, monitor, data, budget)
        return checkpoint_evidence(result["weights"], data, expected_updates=1)

    def auxiliary(name, function):
        result = function(data, config, monitor, pipeline)
        evidence = checkpoint_evidence(result[name], data)
        verify_auxiliary_dependencies(result[name], pipeline)
        load_auxiliary(result[name], data, config["device"])
        return evidence

    def partial_ode():
        from lung.engine.diagnostics import resolve_ode_steps
        monitor.stop_check = lambda step, epoch: budget.check(step, epoch) or step >= arguments.ode_samples
        try:
            resolve_ode_steps(data, config, monitor, pipeline)
            raise RuntimeError("UNEXPECTED_FULL_ODE_AUDIT: bounded sample cap did not pause")
        except StagePaused as pause:
            directory = Path(pause.checkpoint_path)
            samples = list(directory.glob("member-*/steps-*/sample-*.pt")) if directory.is_dir() else []
            for path in samples:
                saved = torch.load(path, map_location="cpu", weights_only=False)
                atomic_save(path, {**saved, **data.evidence})
            return {**data.evidence, "status": "partial", "passed": False, "completed_samples": len(samples),
                    "checkpoint_path": pause.checkpoint_path,
                    "registered_samples_per_member": 256, "registered_step_grid": [128, 256, 512, 1024],
                    "registered_tolerance": .001, "reason": "bounded verification sample cap or budget"}
        finally:
            monitor.stop_check = budget.check

    status, code = "bounded_components_verified_audit_pending", 0
    try:
        pipeline["representation"] = run_stage("ae", train_ae)["path"]
        cached = run_stage("cache", cache_h)
        pipeline["representation_cache"] = cached["directory"]
        config["research"]["latent"] = {**config["research"]["latent"],
            "ae_checkpoint": pipeline["representation"], "cache_dir": cached["directory"]}
        pipeline["response"] = [run_stage("response", train_cfm)["path"]]
        pipeline["pool"] = run_stage("pool", lambda: auxiliary("pool", train_pool))["path"]
        pipeline["state"] = run_stage("state", lambda: auxiliary("state", train_state))["path"]
        if arguments.ode_samples:
            run_stage("ode_audit_partial", partial_ode)
        report["bounded_component_chain_passed"] = True
    except StagePaused as pause:
        status, code = "budget_exhausted", 2
        report["pause"] = {**data.evidence, "reason": pause.reason, "stage": pause.stage,
                           "checkpoint_path": pause.checkpoint_path, "step": pause.step}
    except KeyboardInterrupt:
        status, code = "interrupted", 130
        report["error"] = {"type": "KeyboardInterrupt", "message": "Bounded verification interrupted"}
    except Exception as error:
        status, code = "failed", 1
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        budget.pause()
        report["status"] = status
        persist()
        atomic_json(monitor.experiment / "bounded_pipeline.json", {**data.evidence, "artifacts": pipeline})
        monitor.close(status, **data.evidence)
    print(json.dumps({"status": status, "report": str(report_path), **data.evidence}, ensure_ascii=False))
    return code


def main(argv=None):
    arguments = parser().parse_args(argv)
    if not math.isfinite(arguments.max_seconds) or arguments.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive and finite")
    if min(arguments.train_conditions, arguments.val_conditions, arguments.cells_per_condition) < 1:
        raise ValueError("View limits must be positive")
    if arguments.cells_per_condition > 128:
        raise ValueError("--cells-per-condition may not exceed 128")
    if not arguments.run:
        print(json.dumps({"status": "dry_run", "requires": "--run", "synthetic": False,
            "is_bounded_real_view": True, "n_genes": 32461, "T_conditions": arguments.train_conditions,
            "V_conditions": arguments.val_conditions, "max_cells_per_condition": arguments.cells_per_condition,
            "samples_per_update_condition": 128, "AE_CFM_updates": 1,
            "max_ode_samples": arguments.ode_samples, "full_audit_or_rl": False,
            "max_seconds": arguments.max_seconds, "formal_training": False}, ensure_ascii=False))
        return 0
    return execute_verification(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
