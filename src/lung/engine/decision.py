"""Fresh DA-AC / DLPA training with model exports and independent V scores."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lung.rl import (DAACConfig, DAACTrainer, DLPAConfig, DLPAPlanner, ModelRollout,
                     ProposalDistiller, generate_fixed_sequences,
                     reevaluate_fixed_sequences, stream_seed)
from .checkpoint import atomic_save, load_checkpoint
from .config import StageUnavailable, config_hash
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
    return {"base": [], "shaped": [],
            "length": [], "search": [], "kl": [], "updates": [], "actions": {}}


def has_policy_update(stage, counters):
    """Warmup or Critic-only updates are not completed policy learning."""
    if stage == "daac":
        return counters.get("actor_step", 0) > 0 and counters.get("critic_step", 0) > 0
    return counters.get("distill_step", 0) > 0


def _save_model(folder, stage, learner, planner, epoch, best, identity, metadata,
                filename, completion=None):
    model = learner.actor if stage == "daac" else learner.proposal
    counters = ({key: getattr(learner, key) for key in
                 ("total_transitions", "critic_step", "actor_step", "episodes")}
                if stage == "daac" else {"distill_step": learner.distill_step})
    if planner is not None:
        counters = {**counters, "root_count": planner.root_count, "plan_iteration": planner.plan_iteration}
    extra = {"best": best}
    if stage == "daac" and learner.actor_schedule is not None:
        extra.update(actor_schedule=learner.actor_schedule.state_dict(),
                     actor_stage_events=learner.actor_stage_events)
    if completion is not None:
        extra["training_completion"] = completion
    atomic_save(folder / filename, {"schema_version": 1, "model": model.state_dict(),
        "learner": learner.state_dict(), "epoch": epoch, "identity": identity,
        "metadata": metadata, "counters": counters, "extra": extra})


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
    scores = []
    with torch.random.fork_rng():
        for root in monitor.progress(roots, desc=f"{stage} independent V evaluation"):
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
            scores.append(audit.means[winner])
    return {"val/score": float(np.mean(scores))}


def train_decision(stage, data, config, monitor, pipeline):
    if stage not in {"daac", "dlpa"}:
        raise ValueError("Expected a DA-AC or DLPA stage")
    world = BiologicalWorld(data, config, pipeline)
    world.adapter.validate_ready()
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
        paths.append(_train_seed(stage, data, config, monitor, world, roots, policy_seed))
    result = {**pipeline, stage: paths}
    if stage == "daac" and config[stage].get("actor_schedule"):
        result["daac_best"] = [str(Path(path).with_name("best.pt")) for path in paths
                               if Path(path).with_name("best.pt").is_file()]
    return result


def _train_seed(stage, data, config, monitor, world, roots, seed):
    c, name = config[stage], f"{stage}/seed_{seed}"
    folder = monitor.weights / stage / f"seed_{seed}"
    count = c.get("episodes_per_epoch", 128) if stage == "daac" else c.get("roots_per_epoch", 64)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Each decision epoch must collect positive complete units")
    seed_base = stream_seed(config.get("seed", 20260923), stage, seed)
    if stage == "daac":
        learner = DAACTrainer(world.registry, world.adapter.versions, daac_settings(config),
            config["device"], seed_base, actor_schedule=c.get("actor_schedule"),
            legal_mask=world.adapter.protocol.legal_mask(world.registry, ()))
        planner = None
    else:
        learner = ProposalDistiller(world.registry, world.adapter.versions, config["device"], seed_base,
                                     c.get("lr", 1e-4), c.get("grad_clip", 1.))
        planner = DLPAPlanner(world.adapter, learner.proposal, dlpa_settings(config))
    schedule = learner.actor_schedule if stage == "daac" else None
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
    best = -float("inf")
    control = TrainingControl({**config, "training_schedule": {"mode": "full"}}, config["epochs"])
    for epoch in range(config["epochs"]):
        cursor = _new_epoch()
        if stage == "dlpa":
            learner.begin_epoch()
        indices = np.random.default_rng(stream_seed(seed_base, epoch, "roots")).integers(len(roots), size=count)
        for unit in monitor.progress(range(count), desc=f"{name} epoch {epoch+1}"):
            if schedule is not None and schedule.complete:
                break
            root = roots[int(indices[unit])]
            draw_seed = stream_seed(seed_base, epoch, unit, "training-unit")
            if stage == "daac":
                _collect_daac(learner, world, root, draw_seed, cursor, monitor, name,
                              config.get("scoring_repeats", 32))
            else:
                result = planner.plan(root, draw_seed, progress=lambda metrics:
                    monitor.record(name, metrics["plan_iteration"], metrics, axis="plan_iteration"))
                learner.add(result)
                cursor["search"].append(max(result.search_scores.values()))
        if stage == "dlpa":
            for _ in monitor.progress(range(c.get("distill_updates", 200)), desc=f"{name} joint distillation"):
                metrics = learner.update(c.get("batch_size", 256))
                cursor["kl"].append(metrics["distill_kl"])
                monitor.record(name, learner.distill_step, metrics, axis="distill_step")
        updates = learner.critic_step if stage == "daac" else learner.distill_step
        control.advance(epoch + 1, updates)
        validation = (validate_decision(stage, learner, world, config, monitor, seed, epoch)
                      if schedule is not None or control.validation_due() or best == -float("inf") or control.stopped else {})
        metrics = {**_epoch_metrics(stage, cursor, learner, world.registry), **validation}
        trained = has_policy_update(stage, {"actor_step": learner.actor_step, "critic_step": learner.critic_step} if stage == "daac"
                                     else {"distill_step": learner.distill_step})
        if trained and "val/score" in validation:
            if validation["val/score"] > best:
                best = validation["val/score"]
                _save_model(folder, stage, learner, planner, epoch + 1, best, identity, metadata, "best.pt")
            if schedule is None:
                control.observe(-validation["val/score"])
        monitor.record(name, epoch + 1, metrics)
        if schedule is not None and schedule.complete or schedule is None and control.stopped:
            break
    completion = control.completion()
    if schedule is not None:
        completion = {**completion, "status": "complete", "reason":
                      "actor_schedule_complete" if schedule.complete else "epoch_limit"}
    _save_model(folder, stage, learner, planner, epoch + 1, best, identity, metadata, "final.pt", completion)
    best_path = folder / "best.pt"
    if best == -float("inf") or not best_path.exists():
        raise StageUnavailable(f"TRAINED_DECISION_REQUIRED: {name}")
    selected = load_checkpoint(best_path)
    selected["extra"] = {**selected.get("extra", {}), "training_completion": completion}
    atomic_save(best_path, selected)
    final = folder / "final.pt" if schedule is not None else best_path
    if not has_policy_update(stage, load_checkpoint(final)["counters"]):
        raise StageUnavailable(f"TRAINED_POLICY_UPDATE_REQUIRED: {name}")
    return str(final)
