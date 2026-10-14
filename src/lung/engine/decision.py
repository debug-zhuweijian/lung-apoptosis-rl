"""Five-seed DA-AC / DLPA epoch engines, boundary checkpoints and independent V scores."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from lung.rl import (DAACConfig, DAACTrainer, DLPAConfig, DLPAPlanner, ModelRollout,
                     ProposalDistiller, critic_mc_diagnostic, generate_fixed_sequences,
                     reevaluate_fixed_sequences, stream_seed)
from .checkpoint import atomic_json, atomic_save, load_checkpoint, restore_rng, rng_state
from .config import StageUnavailable, config_hash, file_hash, project_path
from .world import BiologicalWorld
from .training_control import TrainingControl


def daac_settings(config):
    c = config["daac"]
    return DAACConfig(capacity=c.get("replay_capacity", 250000), batch_size=c.get("batch_size", 256),
        warmup_transitions=c.get("warmup_transitions", 1000), actor_delay=c.get("actor_delay", 2),
        actor_lr=c.get("actor_lr", 1e-4), critic_lr=c.get("critic_lr", 3e-4),
        polyak=c.get("polyak", .005), epsilon_start=c.get("epsilon_start", .2),
        epsilon_end=c.get("epsilon_end", .02), epsilon_steps=c.get("epsilon_decay_steps", 50000),
        grad_clip=c.get("grad_clip", 1.), critic_type=c.get("critic_type", "affine"),
        actor_type=c.get("actor_type", "conditional"))


def dlpa_settings(config):
    c = config["dlpa"]
    return DLPAConfig(rounds=c.get("iterations", 6), candidates_per_round=c.get("candidates", 256),
        elite_count=c.get("elites", 64), score_repeats=config.get("scoring_repeats", 32),
        temperature=c.get("temperature", .1), update_weight=1-c.get("old_weight", .2),
        uniform_mix=c.get("exploration", .02))


def _new_epoch():
    return {"phase": "collect", "unit": 0, "update": 0, "base": [], "shaped": [],
            "length": [], "search": [], "kl": [], "updates": [], "actions": {}}


def has_policy_update(stage, counters):
    """Warmup or Critic-only updates are not completed policy learning."""
    if stage == "daac":
        return counters.get("actor_step", 0) > 0 and counters.get("critic_step", 0) > 0
    return counters.get("distill_step", 0) > 0


def _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata,
                   filename="last.pt", control=None, runtime=None):
    model = learner.actor if stage == "daac" else learner.proposal
    learner_state = learner.state_dict()
    counters = learner_state.get("counters", {"distill_step": learner.distill_step} if stage == "dlpa" else {})
    if planner is not None:
        counters = {**counters, "root_count": planner.root_count, "plan_iteration": planner.plan_iteration}
    if control is not None:
        control.checkpoint_saved()
    atomic_save(folder / filename, {"schema_version": 1, "model": model.state_dict(),
        "learner": learner_state, "planner": planner.state_dict() if planner is not None else None,
        "epoch": epoch, "identity": identity, "metadata": metadata, "counters": counters,
        "rng": rng_state(), "extra": {"best": best, "cursor": cursor,
            **({"decision_runtime": runtime.state_dict()} if runtime is not None else {}),
            **({"training_control": control.state_dict()} if control is not None else {})},
        "resume_boundary": "completed_episode" if stage == "daac" else "completed_root_or_distill_update"})


def _collect_daac(learner, world, root, seed, cursor, monitor, name, repeats):
    env = ModelRollout(world.adapter, root, seed, score_repeats=repeats)
    base_return = shaped_return = 0.
    length = 0
    while not env.done:
        action, probabilities = learner.select(env.state, env.legal_mask)
        transition = env.step(action, probabilities, policy_step=learner.actor_step)
        metrics = learner.observe(transition)
        base_return += transition.base_reward
        shaped_return += transition.reward
        candidate = world.registry.candidates[action]
        aid = candidate.action_id
        cursor["actions"][aid] = cursor["actions"].get(aid, 0) + 1
        length += int(not candidate.is_stop)
        cursor["updates"].append(metrics)
        monitor.record(name, learner.total_transitions,
                       {**metrics, "base_reward": transition.base_reward,
                        "shaped_reward": transition.reward}, axis="env_step")
    cursor["base"].append(base_return)
    cursor["shaped"].append(shaped_return)
    cursor["length"].append(length)


def _behavior_metrics(registry, counts):
    total = sum(counts.values())
    if not total:
        return {}
    result = {"behavior/stop_fraction": sum(counts.get(a.action_id, 0)
              for a in registry.candidates if a.is_stop)/total}
    for drug, ids in registry.drug_indices.items():
        nd = sum(counts.get(registry.candidates[i].action_id, 0) for i in ids)
        result[f"behavior/drug/{drug}"] = nd/total
        if not nd or registry.candidates[ids[0]].is_stop:
            continue
        for dose in registry.dose_groups[drug]:
            group = registry.group_indices[drug, dose]
            nq = sum(counts.get(registry.candidates[i].action_id, 0) for i in group)
            result[f"behavior/dose/{drug}/{dose}"] = nq/nd
            if nq:
                for index in group:
                    candidate = registry.candidates[index]
                    result[f"behavior/time/{drug}/{dose}/{candidate.time_id}"] = counts.get(candidate.action_id, 0)/nq
    return result


def _epoch_metrics(stage, cursor, learner, registry):
    result = {}
    for field, key in (("base", "base_return"), ("shaped", "shaped_return"), ("length", "sequence_length"),
                       ("search", "search_score"), ("kl", "distill_kl")):
        if cursor[field]:
            result[f"train/{key}"] = float(np.mean(cursor[field]))
    if stage == "daac":
        for key in ("critic_loss_1", "critic_loss_2", "actor_loss"):
            values = [row[key] for row in cursor["updates"] if key in row]
            if values:
                result[f"train/{key}"] = float(np.mean(values))
        result.update(epsilon=learner.epsilon, env_step=learner.total_transitions,
                      critic_step=learner.critic_step, actor_step=learner.actor_step,
                      episodes=learner.episodes)
        result.update(_behavior_metrics(registry, cursor["actions"]))
    else:
        result["distill_step"] = learner.distill_step
    return result


def validate_decision(stage, learner, world, config, monitor, seed, epoch):
    roots = world.roots("V")[:config[stage].get("validation_roots", 32)]
    if not roots:
        raise StageUnavailable("DECISION_SELECTION_UNAVAILABLE: V is empty")
    repeats = config.get("decision_validation_repeats", config.get("scoring_repeats", 32))
    candidate_count = config.get("validation_candidates", 8 if config.get("synthetic") else 256)
    report, saved_rng, diagnostics = [], rng_state(), {}
    try:
        for index, root in enumerate(monitor.progress(roots, desc=f"{stage} independent V evaluation")):
            run_seed = stream_seed(config.get("seed", 20260923), "V", stage, seed, epoch, root.root_id)
            if stage == "daac":
                plans = generate_fixed_sequences(learner.actor, world.adapter, root, candidate_count, run_seed)
            else:
                planner = DLPAPlanner(world.adapter, learner.proposal, dlpa_settings(config))
                plans = planner.plan(root, run_seed).sequences
            selection = reevaluate_fixed_sequences(world.adapter, root, plans, run_seed,
                repeats=repeats, namespace="training-validation-select")
            winner = min(selection.means, key=lambda u: (-selection.means[u], u))
            # Report a fresh winner audit, rather than the maximum noisy search/selection score.
            audit = reevaluate_fixed_sequences(world.adapter, root, [winner], run_seed,
                repeats=repeats, namespace="training-validation-audit")
            report.append({"root_id": root.root_id, "role": "V", "sequence": list(winner),
                           "score": audit.means[winner], "mc_standard_error": audit.standard_errors[winner],
                           "n_candidates": len(plans), "repeats": repeats,
                           "scope": "training_validation_mc_not_final_precision_certificate"})
            if stage == "daac" and index == 0:
                state = world.adapter.observe(root, (), stream_seed(run_seed, "q-diagnostic-state"))
                phi = world.adapter.potential(state)
                mc = critic_mc_diagnostic(learner.actor, world.adapter, root, winner[0], phi,
                    seed=stream_seed(run_seed, "q-diagnostic"),
                    repeats=config.get("diagnostic_repeats", 2 if config.get("synthetic") else 64))
                with torch.no_grad():
                    ai = world.registry.by_id[winner[0]]
                    if learner.config.critic_type == "affine":
                        predicted = float(learner.critics[0]([state])[0, ai])
                    else:
                        p = torch.zeros(1, len(world.registry), device=learner.device)
                        p[0, ai] = 1
                        predicted = float(learner.critics[0].distribution_value([state], p)[0])
                diagnostics["val/q_mc_abs_error"] = abs(predicted - mc["shaped_target"])
    finally:
        restore_rng(saved_rng)
    path = monitor.experiment / stage / f"seed_{seed}" / f"validation_epoch_{epoch+1}.json"
    atomic_json(path, {"task": config["task"], "biological_identity": world.biological_identity,
                       "synthetic": config.get("synthetic", False), "rows": report,
                       "scope": "training_validation_mc"})
    return {"val/score": float(np.mean([row["score"] for row in report])), **diagnostics}


def train_decision(stage, data, config, monitor, pipeline, resume=False):
    if stage not in {"daac", "dlpa"}:
        raise ValueError("Expected a DA-AC or DLPA stage")
    if resume and config[stage].get("init_from"):
        raise ValueError("INIT_FROM_RESUME_MUTUALLY_EXCLUSIVE")
    if config[stage].get("init_from") and (stage != "daac" or not config[stage].get("actor_schedule")):
        raise ValueError("ACTOR_INITIALIZATION_REQUIRES_STAGED_DAAC")
    world = BiologicalWorld(data, config, pipeline)
    world.adapter.validate_ready(require_potential=stage == "daac")
    roots = world.roots("T")
    if not roots:
        raise StageUnavailable("DECISION_TRAINING_UNAVAILABLE: T is empty")
    if not world.roots("V"):
        raise StageUnavailable("DECISION_SELECTION_UNAVAILABLE: V is empty")
    paths = []
    seeds = config.get("policy_seeds", [0] if config.get("synthetic") else [0, 1, 2, 3, 4])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Policy seeds must be nonempty and unique")
    for policy_seed in seeds:
        paths.append(_train_seed(stage, data, config, monitor, world, roots, policy_seed, resume))
    result = {**pipeline, stage: paths}
    if stage == "daac" and config[stage].get("actor_schedule"):
        result["daac_best"] = [str(Path(path).with_name("best.pt")) for path in paths
                               if Path(path).with_name("best.pt").is_file()]
    return result


def _train_seed(stage, data, config, monitor, world, roots, seed, resume):
    if stage == "daac" and config[stage].get("actor_schedule"):
        return _train_scheduled_daac(data, config, monitor, world, roots, seed, resume)
    c, name = config[stage], f"{stage}/seed_{seed}"
    folder = monitor.weights / stage / f"seed_{seed}"
    count = c.get("episodes_per_epoch", 128) if stage == "daac" else c.get("roots_per_epoch", 64)
    if count < 1:
        raise ValueError("Each decision epoch must collect positive complete units")
    seed_base = stream_seed(config.get("seed", 20260923), stage, seed)
    if stage == "daac":
        learner = DAACTrainer(world.registry, world.adapter.versions, daac_settings(config), config["device"], seed_base)
        planner = None
    else:
        learner = ProposalDistiller(world.registry, world.adapter.versions, config["device"], seed_base,
                                     c.get("lr", 1e-4), c.get("grad_clip", 1.))
        planner = DLPAPlanner(world.adapter, learner.proposal, dlpa_settings(config))
    settings = {"learner": asdict(daac_settings(config)) if stage == "daac" else asdict(dlpa_settings(config)),
                "stage_config": c, "scoring_repeats": config.get("scoring_repeats", 32),
                "base_seed": config.get("seed", 20260923), "policy_seed": seed,
                "validation_candidates": config.get("validation_candidates", 8 if config.get("synthetic") else 256),
                "validation_repeats": config.get("decision_validation_repeats", config.get("scoring_repeats", 32))}
    identity = config_hash({"stage": stage, "data": data.identity, "versions": world.adapter.versions,
                            "network": asdict(world.registry.spec), "settings": settings})
    metadata = {"stage": stage, "data_identity": data.identity, "biological_identity": world.biological_identity,
                "synthetic": config.get("synthetic", False), "config": config,
                "policy_seed": seed, "network_spec": asdict(world.registry.spec), "settings": settings,
                "train_role": "T", "selection_role": "V", "task_id": world.adapter.task_id,
                "versions": world.adapter.versions, "evidence": world.adapter.evidence}
    epoch, cursor, best = 0, _new_epoch(), -float("inf")
    control_config = {**config, "training_schedule": {"mode": "full"},
        "training_control": {"checkpoint_every_updates": c.get("checkpoint_units", 16),
                             "checkpoint_every_seconds": 300, **config.get("training_control", {})}}
    control = TrainingControl(control_config, config["epochs"])
    if resume and (folder / "last.pt").exists():
        saved = load_checkpoint(folder / "last.pt", identity)
        learner.load_state_dict(saved["learner"])
        if planner is not None:
            planner.load_state_dict(saved["planner"])
        epoch, cursor, best = saved["epoch"], saved["extra"]["cursor"], saved["extra"]["best"]
        restore_rng(saved["rng"])
        control = TrainingControl(control_config, config["epochs"], saved["extra"].get("training_control"))
    else:
        _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, control=control)
    while epoch < config["epochs"] and not control.stopped:
        if cursor["phase"] == "collect":
            if stage == "dlpa" and cursor["unit"] == 0:
                learner.begin_epoch()
            root_indices = np.random.default_rng(stream_seed(seed_base, epoch, "roots")).integers(len(roots), size=count)
            bar = monitor.progress(total=count, initial=cursor["unit"], desc=f"{name} epoch {epoch+1}")
            for unit in range(cursor["unit"], count):
                root = roots[int(root_indices[unit])]
                draw_seed = stream_seed(seed_base, epoch, unit, "training-unit")
                if stage == "daac":
                    _collect_daac(learner, world, root, draw_seed, cursor, monitor, name, config.get("scoring_repeats", 32))
                    bar.set_postfix(base=f"{cursor['base'][-1]:.4f}", critic=learner.critic_step)
                else:
                    result = planner.plan(root, draw_seed, progress=lambda metrics:
                        monitor.record(name, metrics["plan_iteration"], metrics, axis="plan_iteration"))
                    learner.add(result)
                    cursor["search"].append(max(result.search_scores.values()))
                    bar.set_postfix(search=f"{cursor['search'][-1]:.4f}", roots=planner.root_count)
                cursor["unit"] = unit+1
                bar.update()
                control.advance(epoch, learner.critic_step if stage == "daac" else learner.distill_step)
                if control.checkpoint_due():
                    _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, control=control)
            bar.close()
            cursor["phase"] = "validate" if stage == "daac" else "distill"
            _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, control=control)
        if cursor["phase"] == "distill":
            updates = c.get("distill_updates", 200)
            for update in monitor.progress(range(cursor["update"], updates), desc=f"{name} joint distillation"):
                metrics = learner.update(c.get("batch_size", 256))
                cursor["kl"].append(metrics["distill_kl"])
                cursor["update"] = update+1
                monitor.record(name, learner.distill_step, metrics, axis="distill_step")
                control.advance(epoch, learner.distill_step)
                if control.checkpoint_due():
                    _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, control=control)
            cursor["phase"] = "validate"
            _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, control=control)
        control.advance(epoch + 1, learner.critic_step if stage == "daac" else learner.distill_step)
        validation = (validate_decision(stage, learner, world, config, monitor, seed, epoch)
                      if control.validation_due() or best == -float("inf") or control.stopped else {})
        metrics = {**_epoch_metrics(stage, cursor, learner, world.registry), **validation}
        trained = has_policy_update(stage, {"actor_step": learner.actor_step, "critic_step": learner.critic_step} if stage == "daac"
                                     else {"distill_step": learner.distill_step})
        improved = trained and "val/score" in validation and validation["val/score"] > best
        if improved:
            best = validation["val/score"]
        epoch += 1
        control.advance(epoch, learner.critic_step if stage == "daac" else learner.distill_step)
        if trained and "val/score" in validation:
            control.observe(-validation["val/score"])
        cursor = _new_epoch()
        _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, control=control)
        if improved:
            _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, "best.pt", control=control)
        if epoch % config.get("save_period", 10) == 0:
            _boundary_save(folder, stage, learner, planner, epoch, cursor, best, identity, metadata, f"epoch_{epoch}.pt", control=control)
        monitor.record(name, epoch, metrics)
    final = folder / "best.pt"
    if not final.exists():
        raise StageUnavailable(f"TRAINED_DECISION_REQUIRED: {name}")
    if not has_policy_update(stage, load_checkpoint(final)["counters"]):
        raise StageUnavailable(f"TRAINED_POLICY_UPDATE_REQUIRED: {name}")
    return str(final)


class _DecisionRuntime:
    """Serializable cumulative wall time, independent of Actor stopping criteria."""

    def __init__(self, saved=None):
        self.values = {"schema_version": 1, "status": "running", "reason": None,
            "elapsed_seconds": 0.,
            "stage_seconds": {stage: 0. for stage in ("warmup", "drug", "dose", "joint")},
            "validation_seconds": 0., "checkpoint_seconds": 0., "environment_units": 0,
            "training_feedback_calls": 0, "training_mc_draws": 0}
        if saved is not None:
            if saved.get("schema_version") != 1:
                raise ValueError("DECISION_RUNTIME_SCHEMA_MISMATCH")
            # Legacy budget fields are deliberately not carried into new checkpoints.
            self.values.update({key: saved[key] for key in self.values if key in saved})
        self.started = time.monotonic()
        if self.values["status"] != "complete":
            self.values.update(status="running", reason=None)

    def stop(self, reason):
        self.values.update(status="complete" if reason == "actor_schedule_complete" else "paused", reason=reason)

    def state_dict(self):
        return {**self.values, "elapsed_seconds": self.values["elapsed_seconds"]
                + time.monotonic() - self.started}


def _train_scheduled_daac(data, config, monitor, world, roots, seed, resume):
    c, name = config["daac"], f"daac/seed_{seed}"
    count = c.get("episodes_per_epoch", 128)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Each decision epoch must collect positive complete units")
    folder = monitor.weights / "daac" / f"seed_{seed}"
    checkpoint = folder / "last.pt"
    if resume and not checkpoint.is_file():
        raise ValueError("DECISION_RESUME_CHECKPOINT_MISSING")
    if not resume and checkpoint.exists():
        raise ValueError("DECISION_OUTPUT_EXISTS_USE_RESUME_OR_NEW_RUN")
    saved = load_checkpoint(checkpoint) if resume else None
    initialization = saved["metadata"].get("initialization") if saved else None
    initial = None
    if c.get("init_from"):
        source = project_path(c["init_from"])
        initial = load_checkpoint(source)
        if (initial.get("metadata", {}).get("stage") != "daac"
                or initial["metadata"].get("data_identity") != data.identity
                or initial["metadata"].get("biological_identity") != world.biological_identity
                or not has_policy_update("daac", initial.get("counters", {}))):
            raise ValueError("ACTOR_INITIALIZATION_CHECKPOINT_INCOMPATIBLE")
        initialization = {"source_sha256": file_hash(source), "source_path": str(source),
                          "components": ["actor"]}
    settings = {"learner": asdict(daac_settings(config)),
        "stage_config": {key: value for key, value in c.items()
                         if key not in {"init_from", "max_seconds", "resume_budget_seconds"}},
        "base_seed": config.get("seed", 20260923), "policy_seed": seed,
        "scoring_repeats": config.get("scoring_repeats", 32),
        "validation_candidates": config.get("validation_candidates", 8 if config.get("synthetic") else 256),
        "validation_repeats": config.get("decision_validation_repeats", config.get("scoring_repeats", 32)),
        "diagnostic_repeats": config.get("diagnostic_repeats", 2 if config.get("synthetic") else 64),
        "feature_layout": "frozen-summary-private-history-v1", "initialization": initialization}
    identity = config_hash({"stage": "daac", "data": data.identity, "versions": world.adapter.versions,
                            "network": asdict(world.registry.spec), "settings": settings})
    if saved is not None and saved["identity"] != identity:
        raise ValueError("CHECKPOINT_IDENTITY_MISMATCH: data/model/reward/protocol changed")
    seed_base = stream_seed(config.get("seed", 20260923), "daac", seed)
    learner = DAACTrainer(world.registry, world.adapter.versions, daac_settings(config),
        config["device"], seed_base, actor_schedule=c["actor_schedule"],
        legal_mask=world.adapter.protocol.legal_mask(world.registry, ()))
    metadata = {"stage": "daac", "data_identity": data.identity,
        "biological_identity": world.biological_identity, "synthetic": config.get("synthetic", False),
        "config": config, "policy_seed": seed, "network_spec": asdict(world.registry.spec),
        "settings": settings, "initialization": initialization, "train_role": "T", "selection_role": "V",
        "task_id": world.adapter.task_id, "versions": world.adapter.versions, "evidence": world.adapter.evidence}
    epoch, cursor, best = 0, _new_epoch(), -float("inf")
    runtime_state = saved["extra"]["decision_runtime"] if saved else None
    if saved is not None and not isinstance(runtime_state, dict):
        raise ValueError("DECISION_RUNTIME_STATE_REQUIRED")
    if runtime_state is not None and saved is not None and (folder / "runtime.json").is_file():
        report = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
        expected = {"identity": identity, "epoch": saved["epoch"],
            "counters": saved["counters"], "cursor_phase": saved["extra"]["cursor"]["phase"],
            "cursor_unit": saved["extra"]["cursor"]["unit"],
            **{key: runtime_state[key] for key in ("status", "reason")}}
        if (all(report.get(key) == value for key, value in expected.items())
                and report.get("elapsed_seconds", -1) >= runtime_state["elapsed_seconds"]):
            # The matching sidecar includes the I/O cost of writing this checkpoint itself.
            runtime_state = {key: report.get(key, value) for key, value in runtime_state.items()}
    runtime = _DecisionRuntime(runtime_state)
    if saved is not None:
        learner.load_state_dict(saved["learner"])
        epoch, cursor, best = saved["epoch"], saved["extra"]["cursor"], saved["extra"]["best"]
        restore_rng(saved["rng"])
    elif initial is not None:
        learner.initialize_actor(initial["learner"])
    schedule = learner.actor_schedule
    assert schedule is not None

    def save(filename="last.pt"):
        started = time.monotonic()
        _boundary_save(folder, "daac", learner, None, epoch, cursor, best, identity, metadata,
                       filename, runtime=runtime)
        runtime.values["checkpoint_seconds"] += time.monotonic() - started
        atomic_json(folder / "runtime.json", {"identity": identity,
            "epoch": epoch, "counters": {key: getattr(learner, key) for key in
                ("total_transitions", "critic_step", "actor_step", "episodes")},
            "cursor_phase": cursor["phase"], "cursor_unit": cursor["unit"],
            "actor_step": learner.actor_step, "actor_schedule": schedule.state_dict(),
            "active_actor_parameters": learner.active_actor_parameters, **runtime.state_dict()})

    def pause(reason):
        runtime.stop(reason)
        save()
        monitor.say(f"{name}: {reason}; resume checkpoint={checkpoint}")
        return str(checkpoint)

    if runtime.values["status"] == "complete":
        return str(checkpoint)
    if saved is None:
        save()
    while epoch < config["epochs"]:
        if cursor["phase"] == "collect":
            indices = np.random.default_rng(stream_seed(seed_base, epoch, "roots")).integers(len(roots), size=count)
            bar = monitor.progress(total=count, initial=cursor["unit"], desc=f"{name} epoch {epoch+1}")
            try:
                for unit in range(cursor["unit"], count):
                    if schedule.complete:
                        break
                    phase = "warmup" if learner.total_transitions < learner.config.warmup_transitions else learner.actor_stage
                    started = time.monotonic()
                    _collect_daac(learner, world, roots[int(indices[unit])],
                        stream_seed(seed_base, epoch, unit, "training-unit"), cursor, monitor, name,
                        config.get("scoring_repeats", 32))
                    runtime.values["stage_seconds"][phase] += time.monotonic() - started
                    runtime.values["environment_units"] += 1
                    runtime.values["training_feedback_calls"] += 1
                    runtime.values["training_mc_draws"] += config.get("scoring_repeats", 32)
                    cursor["unit"] = unit + 1
                    bar.update()
                    save()  # Every completed episode, including a stage boundary, is durable.
            finally:
                bar.close()
            cursor["phase"] = "validate"
            save()
        started = time.monotonic()
        validation = validate_decision("daac", learner, world, config, monitor, seed, epoch)
        runtime.values["validation_seconds"] += time.monotonic() - started
        trained = learner.actor_step > 0 and learner.critic_step > 0
        improved = trained and validation["val/score"] > best
        if improved:
            best = validation["val/score"]
        metrics = {**_epoch_metrics("daac", cursor, learner, world.registry), **validation}
        epoch += 1
        cursor = _new_epoch()
        if schedule.complete:
            runtime.stop("actor_schedule_complete")
        elif epoch >= config["epochs"]:
            runtime.stop("epoch_limit")
        save()
        if improved:
            save("best.pt")
        monitor.record(name, epoch, metrics)
        if schedule.complete:
            return str(checkpoint)
    return pause("epoch_limit")
