"""Explicitly bounded real-data engineering verification; never a training launcher.

Without --run this file prints its scope and performs no data or model work.
The registered full numerical audit is never replaced by a small-sample pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
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
    command.add_argument("--world-actor-probe", action="store_true",
                         help="Isolated real-data World/DAAC gradient probe; never grants formal readiness")
    command.add_argument("--component-run", help="Reuse a prior bounded component run, without retraining")
    command.add_argument("--probe-population", type=int, choices=(16, 32, 64, 128), default=16)
    command.add_argument("--probe-repeats", type=int, choices=(1, 2, 4), default=2)
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


def existing_components(run_id, config):
    if Path(run_id).name != run_id or not run_id.startswith("modular-real-verification-"):
        raise ValueError("BOUNDED_COMPONENT_RUN_ID_INVALID")
    directory = ROOT / "experiments" / run_id
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("BOUNDED_COMPONENT_RUN_PATH_ESCAPE")
    report_path, view_path = directory / "verification_report.json", directory / "bounded_real_view.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    view = json.loads(view_path.read_text(encoding="utf-8"))
    prior = json.loads((directory / "bounded_pipeline.json").read_text(encoding="utf-8"))
    if (report.get("synthetic") is not False or not report.get("is_bounded_real_view")
            or report.get("bounded_component_chain_passed") is not True or report.get("n_genes") != 32461):
        raise ValueError("BOUNDED_COMPONENT_CHAIN_NOT_VERIFIED")
    for key in ("task", "history_support", "state_fold", "reward"):
        if report["config"].get(key) != config.get(key):
            raise ValueError(f"BOUNDED_COMPONENT_SCIENTIFIC_CONFIG_CHANGED: {key}")
    if report["config"].get("research", {}).get("precision") != config.get("research", {}).get("precision"):
        raise ValueError("BOUNDED_COMPONENT_PRECISION_CHANGED")
    artifacts = {name: prior["artifacts"][name]
                 for name in ("representation", "representation_cache", "response", "pool", "state")}
    by_stage = {row["stage"]: row.get("result", {}) for row in report["stages"]}
    for name, stage in (("representation", "ae"), ("response", "response"), ("pool", "pool"), ("state", "state")):
        paths = artifacts[name] if isinstance(artifacts[name], list) else [artifacts[name]]
        if len(paths) != 1:
            raise ValueError("BOUNDED_COMPONENT_PROBE_REQUIRES_SINGLE_MEMBER")
        path = Path(paths[0]).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != by_stage[stage].get("sha256"):
            raise ValueError("BOUNDED_COMPONENT_HASH_OR_PATH_MISMATCH")
    cache = Path(artifacts["representation_cache"]).resolve()
    if not cache.is_relative_to(ROOT) or not (cache / "manifest.json").is_file():
        raise ValueError("BOUNDED_COMPONENT_CACHE_MISSING")
    return artifacts, view, {"run_id": run_id, "report_sha256": file_hash(report_path),
        "source_identity": prior["source_identity"], "view_identity": prior["view_identity"],
        "checkpoint_hashes": {stage: by_stage[stage]["sha256"] for stage in ("ae", "response", "pool", "state")}}


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(repr((name, tuple(value.shape), value.dtype, value.layout)).encode())
        arrays = (value,) if value.layout == torch.strided else (
            value.to_sparse_coo().coalesce().indices(), value.to_sparse_coo().coalesce().values())
        for array in arrays:
            digest.update(array.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def world_actor_probe(data, config, pipeline, monitor, arguments, budget):
    """No adapter readiness override: isolated interfaces and one private update."""
    from contextlib import nullcontext
    from dataclasses import asdict, replace
    from lung.engine.decision import daac_settings
    from lung.engine.potential import uniform_completion
    from lung.engine.world import BiologicalWorld
    from lung.engine.diagnostics import ode_audit_batch
    from lung.engine.numerics import PrecisionPolicy
    from lung.models import rbf_mmd
    from lung.rl import DAACTrainer, RLUnavailable, Transition, stream_seed
    from lung.rl.environment import shape_reward
    from lung.engine.response import tensor

    probe_config = {**config, "population": arguments.probe_population}
    if config["reward"].get("gamma") != 1. or config["reward"].get("shaping") != "frozen_reference_potential_difference":
        raise ValueError("BOUNDED_PROBE_REWARD_NOT_SUPPORTED")
    world = BiologicalWorld(data, probe_config, pipeline, require_resolution=False)
    try:
        world.adapter.validate_ready(require_potential=True)
    except RLUnavailable as unavailable:
        if unavailable.code != "POPULATION_RESOLUTION_UNRESOLVED":
            raise
        production_gate = unavailable.as_dict()
    else:
        raise ValueError("BOUNDED_PROBE_MUST_REMAIN_FORMALLY_UNAVAILABLE")
    frozen = {"representation": getattr(world.representation, "shared", world.representation),
              "pool": world.pool, "state": world.classifier,
              **{f"response_{i}": getattr(member, "member", member) for i, member in enumerate(world.members)}}
    if any(parameter.requires_grad for model in frozen.values() for parameter in model.parameters()):
        raise ValueError("BOUNDED_PROBE_WORLD_NOT_FROZEN")
    frozen_hashes = {name: module_hash(model) for name, model in frozen.items()}
    n, seed = arguments.probe_population, config["seed"]

    def expression(condition, role, controls=False):
        keys = data.reader._control_ids[condition] if controls else (condition,)
        rows = pd.concat([data.reader.role_cells(key, role) for key in keys], ignore_index=True)
        rows = rows.drop_duplicates("cell_id")
        available = len(rows)
        rows = rows.head(n)
        if rows.empty or (controls and len(rows) != n):
            raise ValueError("BOUNDED_PROBE_UNIQUE_POPULATION_UNAVAILABLE")
        return data.reader.normalize(data.reader.rows.read(rows)), rows.cell_id.astype(str).tolist(), available

    def fixed_root(condition, role):
        values, ids, _ = expression(condition, {"T": "fit", "V": "selection"}[role], controls=True)
        return world.root(condition, role, expression=values, gene_ids=data.reader.gene_ids,
                          data_identity=data.identity), values, ids

    def check():
        if budget.check(0, 0):
            raise StagePaused("budget_exhausted", stage="world_actor_probe")

    check()
    train_condition = data.condition_ids("T")[0]
    root, _, control_ids = fixed_root(train_condition, "T")
    planning_seed = stream_seed(seed, "isolated-real-probe", "planning")
    state = world.observe(root, (), planning_seed)
    mask = world.protocol.legal_mask(world.registry, ())
    if int(mask.sum()) < 2:
        raise ValueError("BOUNDED_PROBE_REQUIRES_TWO_LEGAL_ACTIONS")
    reference_requests = []
    for repeat in range(arguments.probe_repeats):
        reference_seed = stream_seed(seed, "isolated-real-probe", "reference-policy", repeat)
        sequence = uniform_completion(world.adapter, (), torch.Generator().manual_seed(reference_seed))
        score_seed = stream_seed(seed, "isolated-real-probe", "reference-score", repeat)
        reference_requests.append((sequence, (score_seed,)))
    reference_scores = world.score_many(root, reference_requests)
    phi_current = float(torch.cat(reference_scores).mean())
    learner = DAACTrainer(world.registry, world.adapter.versions,
        replace(daac_settings(config), capacity=2, batch_size=2, warmup_transitions=1, actor_delay=1),
        config["device"], stream_seed(seed, "isolated-real-probe", "learner"))
    actor_before = module_hash(learner.actor)
    critic_before = [module_hash(model) for model in learner.critics]
    remaining, records, metrics = mask.clone(), [], {}
    generator = torch.Generator().manual_seed(stream_seed(seed, "isolated-real-probe", "behavior"))
    for index in range(2):
        check()
        behavior = remaining.float() / remaining.sum()
        action = int(torch.multinomial(behavior, 1, generator=generator))
        remaining[action] = False
        sequence = (world.registry.candidates[action].action_id,)
        world.protocol.validate_complete(world.registry, sequence)
        score_seeds = tuple(stream_seed(seed, "isolated-real-probe", "feedback", index, repeat)
                            for repeat in range(arguments.probe_repeats))
        scores = world.score(root, sequence, score_seeds)
        base = float(scores.mean())
        transition = Transition(state, mask, behavior, action, base, phi_current, 0.,
            shape_reward(base, phi_current, 0., True), None, None, True, world.adapter.versions,
            f"isolated-real-probe-{index}", config["task"], planning_seed, score_seeds, 0,
            evidence={**data.evidence, "formal_world_ready": False,
                      "response_is_model_simulation": True, "reference_phi_is_fixed_single_root_mc": True})
        metrics = learner.observe(transition)
        records.append({"sequence": sequence, "scoring_seeds": score_seeds,
                        "base_reward": base, "shaped_reward": transition.reward})
    changed = {"actor": module_hash(learner.actor) != actor_before,
               "critics": [module_hash(model) != old for model, old in zip(learner.critics, critic_before)]}
    if (learner.actor_step != 1 or learner.critic_step != 1 or not changed["actor"]
            or not all(changed["critics"]) or not all(math.isfinite(float(value)) for value in metrics.values())
            or not _finite_tensors(learner.state_dict())):
        raise ValueError("BOUNDED_PROBE_ACTOR_CRITIC_UPDATE_FAILED")
    check()
    val_condition = data.condition_ids("V")[0]
    val_root, control, val_control_ids = fixed_root(val_condition, "V")
    features = data.action_features(val_condition)
    action_id = "a-" + config_hash((features["group_id"], str(features["dose"]), str(features["duration"])))[:24]
    sequence, val_seed = (action_id,), stream_seed(seed, "isolated-real-probe", "heldout-response")
    world.protocol.validate_complete(world.registry, sequence)
    predicted, _ = world.propagate(val_root, sequence, val_seed)
    target, target_ids, target_available = expression(val_condition, "selection")
    target = tensor(target, world.device)
    indexed = [(index, {**features, "control": control[index:index + 1]}) for index in range(2)]
    numeric_seeds = tuple(stream_seed(seed, "isolated-real-probe", "batch-score", index) for index in range(2))
    numeric_checks = {}
    for mode, policy in (("production", None), ("explicit_bf16", PrecisionPolicy("bf16"))):
        check()
        dtypes = {}

        def observe_dtype(module, inputs, output):
            if torch.is_tensor(output):
                key = str(output.dtype)
                dtypes[key] = dtypes.get(key, 0) + 1

        hooks = [module.register_forward_hook(observe_dtype) for model in frozen.values()
                 for module in model.modules() if isinstance(module, torch.nn.Linear)]

        def timed(action):
            if world.device.type == "cuda":
                torch.cuda.synchronize(world.device)
            before = time.perf_counter()
            with policy.autocast(world.device) if policy is not None else nullcontext():
                result = action()
            if world.device.type == "cuda":
                torch.cuda.synchronize(world.device)
            return result, time.perf_counter() - before

        score_one = lambda: sum((world._score_seed_batch(val_root, sequence, [value], n, world.steps)
                                 for value in numeric_seeds), [])
        score_batch = lambda: world._score_seed_batch(val_root, sequence, list(numeric_seeds), n, world.steps)
        ode_one = lambda: ode_audit_batch(world.reference, data, indexed, device=world.device,
                                          member=0, steps=world.steps, batch_size=1)
        ode_batch = lambda: ode_audit_batch(world.reference, data, indexed, device=world.device,
                                            member=0, steps=world.steps, batch_size=2)
        try:
            for action in (score_one, score_batch, ode_one, ode_batch):
                timed(action)  # Same-layout warmup; no scalar score-cache lookup.
            scalar, scalar_seconds = timed(score_one)
            batched, batched_seconds = timed(score_batch)
            single_y, single_seconds = timed(ode_one)
            batch_y, batch_seconds = timed(ode_batch)
        finally:
            for hook in hooks:
                hook.remove()
        difference = np.abs(np.asarray(scalar) - np.asarray(batched))
        maes = [float((one - grouped).abs().mean()) for one, grouped in zip(single_y, batch_y)]
        numeric_checks[mode] = {"scope": "two_fixed_V_samples_two_fixed_seeds_not_formal_audit",
            "dense_output_dtypes": dtypes, "manual_bf16_autocast": policy is not None,
            "bf16_active": bool(policy and policy.bf16_active(world.device)), "score_cache_bypassed": True,
            "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
            "score": {"seeds": numeric_seeds, "single": scalar, "batch": batched,
                "max_abs_difference": float(difference.max()), "mean_abs_difference": float(difference.mean()),
                "tolerance": .01, "within_tolerance": bool(difference.max() <= .01),
                "single_seconds": scalar_seconds, "batch_seconds": batched_seconds},
            "ode": {"sample_indices": [0, 1], "sample_mae": maes, "tolerance": .001,
                "within_tolerance": bool(max(maes) <= .001),
                "single_seconds": single_seconds, "batch_seconds": batch_seconds}}
    train_mean = np.zeros(data.n_genes, dtype=np.float64)
    train_conditions, train_cell_counts = data.condition_ids("T"), {}
    for condition in train_conditions:
        check()
        values, ids, available = expression(condition, "fit")
        train_cell_counts[condition] = {"available_unique": available, "used_unique": len(ids)}
        train_mean += values.mean(0) / len(train_conditions)
    mean_baseline = tensor(train_mean, world.device)[None].expand(n, -1)
    comparison = {name: {"mean_mse": float((values.mean(0) - target.mean(0)).square().mean()),
        "mmd_biased": float(rbf_mmd(values, target, bandwidth=data.bandwidth, unbiased=False))}
        for name, values in (("world", predicted), ("same_plate_control", tensor(control, world.device)),
                             ("T_treated_mean", mean_baseline))}
    if not all(math.isfinite(value) for row in comparison.values() for value in row.values()):
        raise ValueError("BOUNDED_PROBE_HELDOUT_METRICS_NONFINITE")
    after = {name: module_hash(model) for name, model in frozen.items()}
    if after != frozen_hashes or any(parameter.grad is not None for model in frozen.values() for parameter in model.parameters()):
        raise ValueError("BOUNDED_PROBE_FROZEN_WORLD_CHANGED")
    path = monitor.weights / "engineering_probe" / "actor_critic_update.pt"
    atomic_save(path, {"schema_version": 1, "metadata": {**data.evidence,
        "stage": "engineering_daac_gradient_probe", "formal_world_ready": False,
        "biological_identity": world.biological_identity, "formal_pipeline_reusable": False},
        "learner": learner.state_dict(), "reference_phi": phi_current, "frozen_parameter_hashes": frozen_hashes})
    return {**data.evidence, "status": "complete", "formal_world_ready": False,
        "formal_pipeline_reusable": False, "production_gate": production_gate,
        "protocol": {"name": "single_T_root_real_model_gradient_probe_v1", "population": n,
            "ode_steps": world.steps, "scoring_repeats": arguments.probe_repeats, "n_genes": data.n_genes,
            "independent_experimental_roots": 1, "simulated_transitions": 2, "reward": config["reward"],
            "precision": config["research"]["precision"], "reference_policy": "uniform_complete_legal_actions"},
        "learner_config": asdict(learner.config), "batch_numeric_checks": numeric_checks,
        "train_condition": train_condition, "control_cell_ids": control_ids, "planning_seed": planning_seed,
        "reference_requests": reference_requests, "reference_scores": [row.tolist() for row in reference_scores],
        "fixed_single_root_MC_phi": phi_current, "production_phi_network_trained": False,
        "transitions": records, "optimizer_steps": {"actor": learner.actor_step, "critic": learner.critic_step},
        "parameter_changes": changed, "finite_losses": metrics, "checkpoint": str(path),
        "frozen_parameter_hashes_before": frozen_hashes, "frozen_parameter_hashes_after": after,
        "tidyi_label_counts": {role: data.reader.state_rows.loc[data.reader.state_rows.role.eq(role), "state_class"].value_counts().to_dict()
                               for role in ("fit", "selection", "calibration")},
        "heldout_response": {"condition": val_condition, "role": "V", "target_cell_ids": target_ids,
            "target_cell_count": len(target_ids), "target_available_unique": target_available,
            "control_cell_ids": val_control_ids, "generation_seed": val_seed, "metrics": comparison,
            "mean_fit_role": "T", "mean_fit_cell_counts": train_cell_counts,
            "mean_condition_weighting": "equal", "scope": "one_bounded_condition_not_generalization_or_apoptosis_efficacy"}}


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
    reused = existing_components(arguments.component_run, config) if arguments.component_run else None
    source = EngineData(config, roles=("T", "V", "C"))
    limits = reused[1]["limits"] if reused else {
        "T": arguments.train_conditions, "V": arguments.val_conditions, "cells": arguments.cells_per_condition}
    data = BoundedRealData(source, limits["T"], limits["V"], limits["cells"], config["state_fold"])
    if reused and (data.identity != reused[2]["view_identity"]
                   or any(data.selection[key] != reused[1].get(key) for key in data.selection)):
        raise ValueError("BOUNDED_COMPONENT_VIEW_CHANGED: source, normalization, genes or selected cells differ")
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
    if arguments.world_actor_probe:
        report.update(formal_world_ready=False, formal_rl_started=False)
        report["gates"]["1"] = {"passed": False,
            "reason": "Isolated unaudited World/gradient probe cannot grant formal World or Actor/Critic readiness"}
    if reused:
        pipeline = dict(reused[0])
        report["component_lineage"] = {**reused[2], "current_source_identity": source.identity,
            "current_view_identity": data.identity, "existing_components_rewritten": False}

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
        if reused:
            run_stage("components_reused", lambda: {**data.evidence, "status": "reused", **report["component_lineage"]})
        else:
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
        if arguments.world_actor_probe:
            report.update(world_or_rl_started=True, engineering_gradient_probe_started=True)
            report["engineering_probe"] = run_stage("world_actor_probe",
                lambda: world_actor_probe(data, config, pipeline, monitor, arguments, budget))
            status = "bounded_world_actor_probe_verified_formal_audit_pending"
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
    if arguments.component_run and not arguments.world_actor_probe:
        raise ValueError("--component-run requires --world-actor-probe")
    if not arguments.run:
        print(json.dumps({"status": "dry_run", "requires": "--run", "synthetic": False,
            "is_bounded_real_view": True, "n_genes": 32461, "T_conditions": arguments.train_conditions,
            "V_conditions": arguments.val_conditions, "max_cells_per_condition": arguments.cells_per_condition,
            "samples_per_update_condition": 128, "AE_CFM_updates": 1,
            "max_ode_samples": arguments.ode_samples, "full_audit_or_rl": False,
             "world_actor_probe": arguments.world_actor_probe, "component_run": arguments.component_run,
             "probe_population": arguments.probe_population, "probe_repeats": arguments.probe_repeats,
             "formal_world_ready": False,
            "max_seconds": arguments.max_seconds, "formal_training": False}, ensure_ascii=False))
        return 0
    return execute_verification(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
