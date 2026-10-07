"""Frozen-reference value targets, T-only fitting and V-only checkpoint selection."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from lung.rl import PotentialNetwork, stream_seed
from .checkpoint import atomic_save, load_checkpoint, restore_rng, save_training, seed_all
from .config import StageUnavailable, config_hash
from .training_control import TrainingControl


def _prepare_world(data, config, monitor, pipeline):
    from .diagnostics import prepare_frozen_environment
    return prepare_frozen_environment(data, config, monitor, pipeline)


def uniform_completion(adapter, prefix, generator):
    """Uniform over COMPLETE legal actions, including STOP only when registered."""
    result = tuple(prefix)
    while not adapter.protocol.position(adapter.registry, result)[2]:
        mask = adapter.protocol.legal_mask(adapter.registry, result)
        legal = torch.where(mask)[0]
        if not len(legal):
            raise StageUnavailable("UNSUPPORTED_PROTOCOL: reference continuation cannot reach T*")
        index = int(legal[int(torch.randint(len(legal), (1,), generator=generator))])
        result = (*result, adapter.registry.candidates[index].action_id)
    return result


def reference_prefix_rows(world, role, count, continuations, seed, monitor):
    """Every target averages independent fixed-plan population outcomes at the same prefix."""
    if role not in {"T", "V"}:
        raise ValueError("Phi target generation may only use T and V")
    roots = world.roots(role)
    if not roots or count < 1 or continuations < 1:
        raise StageUnavailable(f"POTENTIAL_SUPERVISION_UNAVAILABLE: {role}")
    rows = []
    for index in monitor.progress(range(count), desc=f"Phi {role} reference prefixes"):
        generator = torch.Generator().manual_seed(stream_seed(seed, role, index, "reference-policy"))
        root = roots[int(torch.randint(len(roots), (1,), generator=generator))]
        complete = uniform_completion(world.adapter, (), generator)
        # A length-L complete episode has L nonabsorbing decision prefixes 0..L-1.
        prefix = complete[:int(torch.randint(len(complete), (1,), generator=generator))]
        planning_seed = stream_seed(seed, role, index, "prefix-planning")
        state = world.adapter.observe(root, prefix, planning_seed)
        score_requests, completion_ids = [], []
        for repeat in range(continuations):
            continuation_seed = stream_seed(seed, role, index, repeat, "reference-continuation")
            continuation_rng = torch.Generator().manual_seed(continuation_seed)
            sequence = uniform_completion(world.adapter, prefix, continuation_rng)
            score_seed = stream_seed(seed, role, index, repeat, "reference-hidden-score")
            score_requests.append((sequence, (score_seed,)))
            completion_ids.append(sequence)
        # Group only complete identical plans. Preserve each independent seed and
        # restore the original continuation order before computing its mean.
        grouped = {}
        for sequence, seeds in score_requests:
            grouped.setdefault(sequence, []).extend(seeds)
        requests = [(sequence, tuple(seeds)) for sequence, seeds in grouped.items()]
        batch_score = getattr(world.adapter, "score_many", None)
        scored = (batch_score(root, requests) if batch_score else
                  [world.adapter.score(root, sequence, seeds) for sequence, seeds in requests])
        by_request = {(sequence, seed): float(score[index])
                      for (sequence, seeds), score in zip(requests, scored)
                      for index, seed in enumerate(seeds)}
        target_scores = [by_request[(sequence, seeds[0])] for sequence, seeds in score_requests]
        rows.append({"state": state, "target": float(np.mean(target_scores)),
                     "root_id": root.root_id, "role": role, "prefix": prefix,
                     "planning_seed": planning_seed, "generation_seed": seed,
                     "generation_index": index, "continuations": continuations,
                     "sampled_sequences": tuple(completion_ids),
                     "target_scope": "uniform_complete_action_model_reference_value"})
    return rows


def _cache_rows(world, role, count, config, monitor, identity):
    path = monitor.experiment / "cache" / f"potential-{role}-{identity}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity or saved.get("role") != role:
            raise ValueError("POTENTIAL_CACHE_IDENTITY_MISMATCH")
        rows = saved["rows"]
    else:
        rows = reference_prefix_rows(world, role, count, config.get("potential_continuations", 64),
                                     config.get("seed", 20260923), monitor)
        atomic_save(path, {"identity": identity, "role": role,
                           "versions": world.adapter.versions, "rows": rows,
                           "synthetic": config.get("synthetic", False)})
    if len(rows) != count or any(row["role"] != role or row["state"].role != role for row in rows):
        raise ValueError("POTENTIAL_CACHE_ROLE_OR_SIZE_MISMATCH")
    return rows


@torch.no_grad()
def evaluate_potential(model, rows, batch_size, device):
    if not rows or any(row["state"].role != "V" for row in rows):
        raise ValueError("Phi checkpoint selection requires V-only prefix rows")
    total = 0.
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        values = model([row["state"] for row in batch])
        target = torch.tensor([row["target"] for row in batch], device=device)
        total += float((values - target).square().sum())
    return total / len(rows)


def train_potential(data, config, monitor, pipeline, resume=False):
    # The new Phi must not include an older Phi file in its target/environment identity.
    dependencies = {key: value for key, value in pipeline.items() if key not in {"potential", "daac", "dlpa"}}
    world, prepared = _prepare_world(data, config, monitor, dependencies)
    world.adapter.validate_ready(require_potential=False)
    settings = {"train_prefixes": config.get("potential_train_prefixes", 10000),
                "val_prefixes": config.get("potential_val_prefixes", 2000),
                "continuations": config.get("potential_continuations", 64),
                "batch_size": config.get("potential_batch", 128),
                "lr": config.get("potential_lr", 1e-4), "seed": config.get("seed", 20260923),
                "early_stopping": config.get("early_stopping", False),
                "training_control": config.get("training_control", {})}
    identity = config_hash({"stage": "potential", "data": data.identity,
                            "biological_identity": world.biological_identity,
                            "versions": world.adapter.versions, "settings": settings,
                            "spec": asdict(world.registry.spec)})
    train = _cache_rows(world, "T", settings["train_prefixes"], config, monitor, identity)
    selection = _cache_rows(world, "V", settings["val_prefixes"], config, monitor, identity)
    if any(row["state"].role != "T" for row in train):
        raise ValueError("Phi fitting cannot read V/G/C/E targets")
    seed_all(settings["seed"])
    device = torch.device(config["device"])
    model = PotentialNetwork(world.registry.spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    folder = monitor.weights / "potential"
    epoch0, step, best = 0, 0, float("inf")
    saved_control = None
    metadata = {"stage": "potential", "data_identity": data.identity,
                "biological_identity": world.biological_identity,
                "synthetic": config.get("synthetic", False), "config": config,
                "train_role": "T", "selection_role": "V", "settings": settings,
                "network_spec": asdict(world.registry.spec), "versions": world.adapter.versions,
                "target_scope": "frozen_uniform_complete_action_model_reference_value"}
    if resume and (folder / "last.pt").exists():
        saved = load_checkpoint(folder / "last.pt", identity)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch0, step, best = saved["epoch"], saved["counters"]["optimizer_step"], saved["extra"]["best"]
        saved_control = saved["extra"].get("training_control")
        restore_rng(saved["rng"])
    else:
        save_training(folder / "last.pt", model, optimizer, epoch=0, identity=identity,
                      metadata=metadata, counters={"optimizer_step": 0}, extra={"best": best})
    controller = TrainingControl({**config, "training_schedule": {"mode": "full"}},
                                 config["epochs"], saved_control)
    if saved_control is None and epoch0:
        controller.advance(epoch0, step)
    for epoch in range(epoch0, config["epochs"]):
        if controller.stopped:
            break
        model.train()
        order = np.random.default_rng(settings["seed"] + epoch).permutation(len(train))
        total = 0.
        for start in monitor.progress(range(0, len(train), settings["batch_size"]), desc=f"Phi epoch {epoch+1}"):
            batch = [train[int(i)] for i in order[start:start + settings["batch_size"]]]
            targets = torch.tensor([row["target"] for row in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model([row["state"] for row in batch]) - targets).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("NONFINITE_POTENTIAL_LOSS")
            loss.backward()
            optimizer.step()
            step += 1
            total += float(loss.detach()) * len(batch)
            monitor.record("potential", step, {"mse": float(loss.detach())}, axis="optimizer_step")
        model.eval()
        controller.advance(epoch + 1, step)
        improved = False
        values = {"train/mse": total / len(train)}
        if controller.validation_due() or controller.stopped:
            validation = evaluate_potential(model, selection, settings["batch_size"], device)
            controller.observe(validation)
            improved = validation < best
            best = min(best, validation)
            values["val/mse"] = validation
        arguments = dict(epoch=epoch+1, identity=identity, metadata=metadata,
                         counters={"optimizer_step": step}, extra={"best": best,
                         "training_control": controller.state_dict()})
        save_training(folder / "last.pt", model, optimizer, **arguments)
        if improved:
            save_training(folder / "best.pt", model, optimizer, **arguments)
        if (epoch+1) % config.get("save_period", 10) == 0:
            save_training(folder / f"epoch_{epoch+1}.pt", model, optimizer, **arguments)
        monitor.record("potential", epoch+1, values)
    final = folder / "best.pt"
    if not final.exists():
        raise StageUnavailable("TRAINED_POTENTIAL_REQUIRED")
    return {**prepared, "potential": str(final)}
