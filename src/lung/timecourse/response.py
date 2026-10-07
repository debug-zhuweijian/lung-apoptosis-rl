"""Time response with legal internal validation and complete-update recovery."""
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lung.models import ModelConfig, build_response_model, rbf_mmd
from lung.engine.config import config_hash, file_hash, merge, read_yaml
from lung.engine.checkpoint import seed_all, save_training, load_checkpoint, restore_rng, atomic_json
from lung.engine.response import train_condition, tensor, condition_kwargs, projected, response_rngs, learning_rate
from .protocol import settings
from lung.engine.training_control import TrainingControl, checkpoint_completion
from lung.engine.response_training import generation_noise, train_conditions, condition_batch_mode, prefetch_enabled
from lung.engine.response_prefetch import ResponsePrefetch
from lung.engine.checkpoint import rng_state


def response_settings(config):
    base = read_yaml(Path(__file__).parents[3] / "configs/response.yaml")
    return merge(merge(base, config.get("response", {})), settings(config).get("response", {}))


@torch.no_grad()
def validation_loss(model, data, cfg, device):
    state = rng_state()
    was_training = model.training
    values = []
    try:
        model.eval()
        for key in data.condition_ids("V"):
            rng = np.random.default_rng(cfg["seed"])
            batch = data.sample_response(key, cfg.get("validation_population", 32), rng, role="V")
            reference, target = tensor(batch["control"], device), tensor(batch["target"], device)
            generator = torch.Generator(device=device).manual_seed(cfg["seed"])
            generated = model.sample(reference, model.encode_condition(**condition_kwargs(batch, len(reference), device)),
                noise=generation_noise(model, reference, generator), steps=cfg["inference_ode_steps"], checkpoint=False)
            values.append(float(rbf_mmd(generated, target, bandwidth=data.bandwidth, unbiased=False)))
        if not values:
            raise ValueError("NO_LEGAL_TIME_VALIDATION_POPULATIONS")
        return float(np.mean(values))
    finally:
        model.train(was_training)
        restore_rng(state)


def selected_checkpoint(path, identity=None):
    """A selected early checkpoint is usable only with its completed-fit receipt."""
    path = Path(path).resolve()
    saved = load_checkpoint(path, identity)
    if path.name == "last.pt":
        return saved, checkpoint_completion(saved, saved["metadata"]["target_epochs"])
    receipt = json.loads((path.parent / "completion.json").read_text(encoding="utf-8"))
    if (receipt["identity"] != saved["identity"] or receipt["selected"] != str(path)
            or receipt["selected_hash"] != file_hash(path)
            or receipt["last_hash"] != file_hash(receipt["last"])):
        raise ValueError("TIME_SELECTION_COMPLETION_BINDING_MISMATCH")
    last = load_checkpoint(receipt["last"], saved["identity"])
    completion = checkpoint_completion(last, saved["metadata"]["target_epochs"])
    state = saved["extra"]["training_control"]
    if (receipt["completion"] != completion or last["extra"].get("best_selection") != {
            "path": str(path), "hash": receipt["selected_hash"]}
            or state["stage"] != "full" or state["full_updates"] < 1 or state["full_validations"] < 1
            or saved["counters"]["optimizer_step"] > completion["actual_updates"]):
        raise ValueError("TIME_SELECTION_NOT_FROM_COMPLETED_FULL_FIT")
    return saved, completion


def train_response(data, config, output, resume=False, progress=None, initial_checkpoints=None):
    cfg = response_settings(config)
    control_config = {**config, "training_schedule": cfg.get("training_schedule", {})}
    early = config.get("early_stopping", False)
    early_enabled = bool(early.get("enabled", True)) if isinstance(early, dict) else bool(early)
    use_validation = early_enabled or cfg.get("training_schedule", {}).get("mode", "full") != "full"
    if use_validation and not 0 < cfg.get("validation_fraction", .2) < 1:
        raise ValueError("INDEPENDENT_TIME_VALIDATION_FRACTION_REQUIRED")
    data = data.training_view(known24_only=getattr(data, "known24_only", False),
                              validation_fraction=cfg.get("validation_fraction", .2) if use_validation else 0.)
    device = torch.device(config.get("device", "cuda:0"))
    epochs = int(settings(config).get("epochs", config.get("epochs", 1000)))
    if epochs < 1 or cfg["members"] < 1:
        raise ValueError("POSITIVE_EPOCHS_AND_MEMBERS_REQUIRED")
    shape = ModelConfig(**cfg["model"])
    if shape.n_genes != data.n_genes or (not data.synthetic and data.n_genes != 32461):
        raise ValueError("RESPONSE_GENE_AXIS_MISMATCH")
    package = Path(__file__).parents[1]
    code_paths = [Path(__file__), package / "engine/response.py", package / "engine/checkpoint.py", package / "engine/response_training.py", package / "engine/response_prefetch.py", package / "engine/inference_noise.py", package / "engine/training_control.py", package / "timecourse/data.py",
                  *sorted((package / "models").glob("*.py"))]
    code = {str(p.relative_to(package)): file_hash(p) for p in code_paths}
    identity = config_hash({"data": data.view_identity, "config": cfg, "epochs": epochs, "code": code, "control": control_config.get("training_control", {}), "early": config.get("early_stopping", False), "initial": {str(p): file_hash(p) for p in initial_checkpoints or []}})
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists():
        import json
        previous = json.loads((output / "identity.json").read_text(encoding="utf-8"))
        if previous["identity"] != identity or not resume:
            raise ValueError("TIME_RESPONSE_RESUME_IDENTITY_MISMATCH")
    result_path = output / "result.json"
    if resume and result_path.exists():
        import json
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if completed.get("identity") != identity or completed.get("status") != "complete":
            raise ValueError("TIME_RESPONSE_RESULT_IDENTITY_MISMATCH")
        for path, digest in completed["hashes"].items():
            if file_hash(path) != digest:
                raise ValueError("TIME_RESPONSE_COMPLETED_ARTIFACT_CHANGED")
        for path in completed["checkpoints"]:
            selected_checkpoint(path, identity)
        return {**completed, "artifacts": [*completed["artifacts"], str(result_path)]}
    atomic_json(output / "identity.json", {"identity": identity, "fit_drugs": sorted(data.fit_drugs),
        "coordinate_identity": data.coordinate_identity, "synthetic": data.synthetic,
        "selection": "internal_validation_best" if use_validation else "fixed_final_epoch", "epochs": epochs})
    paths, completions, completion_artifacts = [], [], []
    if initial_checkpoints is not None and len(initial_checkpoints) != cfg["members"]:
        raise ValueError("ONE_PRETRAIN_CHECKPOINT_PER_MEMBER_REQUIRED")
    for member in range(cfg["members"]):
        folder = output / f"member_{member+1}"
        folder.mkdir(exist_ok=True)
        last = folder / "last.pt"
        if last.exists() and not resume:
            raise FileExistsError("Use resume or a new response directory")
        seed_all(cfg["seed"] + member)
        model = build_response_model(shape, tensor(data.neighbors, device, torch.long)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        metadata = {"stage": "time_response", "member": member, "model_config": asdict(shape), "response_config": cfg,
            "data_identity": data.identity, "coordinate_identity": data.coordinate_identity,
            "fit_drugs": sorted(data.fit_drugs), "synthetic": data.synthetic,
            "selection": "internal_validation_best" if use_validation else "fixed_final_epoch", "target_epochs": epochs, "code": code, "known24_only": getattr(data, "known24_only", False), "validation_scope": "within_legal_fit_population_holdout_not_outer_oof" if use_validation else "disabled", "validation_fraction": getattr(data, "validation_fraction", 0.)}
        control = TrainingControl(control_config, epochs)
        epoch0, step, pending, best_selection = 0, 0, None, None
        if last.exists():
            saved = load_checkpoint(last, identity)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"])
            epoch0, step = saved["epoch"], saved["counters"]["optimizer_step"]
            pending = saved["extra"].get("update_resume")
            best_selection = saved["extra"].get("best_selection")
            if best_selection is not None and file_hash(best_selection["path"]) != best_selection["hash"]:
                raise ValueError("TIME_BEST_RESUME_HASH_MISMATCH")
            control = TrainingControl(control_config, epochs, saved["extra"].get("training_control"))
            if not 0 <= epoch0 <= epochs or (pending is not None and epoch0 == epochs):
                raise ValueError("TIME_RESPONSE_RESUME_EPOCH_MISMATCH")
        else:
            if initial_checkpoints is not None:
                initial, _ = selected_checkpoint(initial_checkpoints[member])
                if not initial["metadata"].get("known24_only") or initial["metadata"]["coordinate_identity"] != data.coordinate_identity or initial["metadata"]["data_identity"] != data.identity or initial["metadata"].get("member") != member or initial["metadata"].get("validation_fraction", 0.) != getattr(data, "validation_fraction", 0.):
                    raise ValueError("ILLEGAL_PRETRAIN_INITIALIZATION")
                model.load_state_dict(initial["model"])
            save_training(last, model, optimizer, epoch=0, identity=identity, metadata=metadata,
                          counters={"optimizer_step": 0})
        keys = data.condition_ids("T")
        if not keys:
            raise ValueError("EMPTY_TIME_TRAINING_POOL")
        updates = math.ceil(len(keys) / cfg["conditions_per_update"])
        event_dir = folder / "events"
        if config.get("log_root"):
            anchor = Path(config.get("log_anchor", output)).resolve()
            event_dir = Path(config["log_root"]) / output.relative_to(anchor) / folder.name / "events"
        writer = SummaryWriter(str(event_dir), purge_step=epoch0 + 1)
        history_path = folder / "metrics.csv"
        # Update checkpoints precede telemetry; only epoch commits retain telemetry.
        history = []
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as stream:
                history = [row for row in csv.DictReader(stream) if int(row["epoch"]) <= epoch0]
        def observe_validation(checkpoint_epoch):
            nonlocal best_selection
            stage_before = control.stage
            improved = control.observe(validation_loss(model, data, cfg, device))
            if improved and stage_before == "full" and control.state["full_updates"] > 0:
                selected = folder / f"best-update-{step}.pt"
                save_training(selected, model, optimizer, epoch=checkpoint_epoch, identity=identity,
                    metadata=metadata, counters={"optimizer_step": step},
                    extra={"training_control": control.state_dict()})
                best_selection = {"path": str(selected), "hash": file_hash(selected)}
                return True
            return False

        prefetch = None
        try:
            for epoch in range(epoch0, epochs):
                if control.stopped:
                    break
                model.train()
                rng, mmd_rng, generator = response_rngs(cfg["seed"] + 100000 * member + epoch, device)
                keys = list(data.iter_response_epoch("T", cfg["seed"] + 100000 * member + epoch))
                loss = 0.
                next_condition = 0
                lr = optimizer.param_groups[0]["lr"]
                if pending is not None:
                    next_condition = pending["next_condition_index"]
                    if (pending["schema_version"] != 1 or pending["epoch"] != epoch
                            or pending["keys"] != keys or not 0 < next_condition <= len(keys)
                            or (next_condition != len(keys) and next_condition % cfg["conditions_per_update"])
                            or pending["loss_count"] != next_condition
                            or step != epoch * updates + math.ceil(next_condition / cfg["conditions_per_update"])):
                        raise ValueError("TIME_RESPONSE_UPDATE_RESUME_MISMATCH")
                    loss = pending["loss"]
                    rng.bit_generator.state = pending["numpy_rng"]
                    mmd_rng.bit_generator.state = pending["mmd_numpy_rng"]
                    generator.set_state(pending["mmd_torch_rng"].cpu())
                    pending = None
                prefetch = ResponsePrefetch(data, cfg, enabled=prefetch_enabled(
                    model, {**cfg, "_training_stage": control.stage}))
                for start in range(next_condition, len(keys), cfg["conditions_per_update"]):
                    group = keys[start:start+cfg["conditions_per_update"]]
                    group_batches = prefetch.consume(group, rng, mmd_rng)
                    prefetch.queue(keys[start+len(group):start+len(group)+cfg["conditions_per_update"]], rng, mmd_rng)
                    optimizer.zero_grad(set_to_none=True)
                    lr = learning_rate(step, epochs * updates, min(cfg["warmup_epochs"], epochs) * updates,
                                       cfg["lr"], cfg["min_lr"])
                    for parameters in optimizer.param_groups:
                        parameters["lr"] = lr
                    active_cfg = {**cfg, "_training_stage": control.stage}
                    batch_width, _ = condition_batch_mode(model, active_cfg)
                    for offset in range(0, len(group), batch_width):
                        values = train_conditions(model, data, group[offset:offset+batch_width], rng,
                            active_cfg, device, len(group), progress=progress,
                            mmd_rng=mmd_rng, mmd_generator=generator,
                            batches=group_batches[offset:offset+batch_width] if group_batches is not None else None)
                        loss += sum(row["loss"] for row in values) / len(keys)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    control.advance(epoch, step)
                    improved = False
                    if use_validation and control.validation_due():
                        prefetch.drain()
                        improved = observe_validation(epoch)
                    if control.checkpoint_due(force=improved):
                        control.checkpoint_saved()
                        save_training(last, model, optimizer, epoch=epoch, identity=identity,
                        metadata=metadata, counters={"optimizer_step": step}, extra={"training_control": control.state_dict(), "best_selection": best_selection, "update_resume": {
                            "schema_version": 1, "epoch": epoch, "keys": keys,
                            "next_condition_index": start + len(group), "loss": loss,
                            "loss_count": start + len(group), "numpy_rng": rng.bit_generator.state,
                            "mmd_numpy_rng": mmd_rng.bit_generator.state,
                            "mmd_torch_rng": generator.get_state()}})
                    if control.stopped:
                        break
                prefetch.close()
                prefetch = None
                finished_epoch = epoch + int(start + len(group) == len(keys)) if next_condition < len(keys) else epoch + 1
                previous_stage = control.stage
                control.advance(finished_epoch, step)
                if use_validation and previous_stage == control.stage and (control.validation_due() or control.stopped):
                    observe_validation(finished_epoch)
                history.append({"epoch": finished_epoch, "optimizer_step": step, "loss": loss, "lr": lr})
                temporary = history_path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8") as stream:
                    csvwriter = csv.DictWriter(stream, fieldnames=["epoch", "optimizer_step", "loss", "lr"])
                    csvwriter.writeheader()
                    csvwriter.writerows(history)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, history_path)
                writer.add_scalar("train/loss", loss, epoch+1)
                writer.add_scalar("train/lr", lr, epoch+1)
                writer.flush()
                save_training(last, model, optimizer, epoch=finished_epoch, identity=identity,
                              metadata=metadata, counters={"optimizer_step": step}, extra={"loss": loss, "training_control": control.state_dict(), "best_selection": best_selection})
                if progress:
                    progress(f"time_response_member_{member+1}", finished_epoch, epochs)
        finally:
            if prefetch is not None:
                prefetch.close()
            writer.close()
        completion = checkpoint_completion(load_checkpoint(last, identity), epochs)
        completions.append(completion)
        if use_validation:
            if best_selection is None:
                raise ValueError("FULL_TIME_VALIDATION_SELECTION_REQUIRED")
            receipt_path = folder / "completion.json"
            atomic_json(receipt_path, {"identity": identity, "selected": best_selection["path"],
                "selected_hash": best_selection["hash"], "last": str(last), "last_hash": file_hash(last),
                "completion": completion})
            selected_checkpoint(best_selection["path"], identity)
            paths.append(best_selection["path"])
            completion_artifacts.extend([str(last), str(receipt_path)])
        else:
            paths.append(str(last))
        del optimizer, model
    result = {"status": "complete", "checkpoints": paths, "identity": identity,
              "artifacts": [str(output / "identity.json"), *completion_artifacts, *[str(p) for p in output.glob("member_*/metrics.csv")]],
              "synthetic": data.synthetic, "completion": completions}
    result["hashes"] = {path: file_hash(path) for path in paths + result["artifacts"]}
    atomic_json(output / "result.json", result)
    result["artifacts"].append(str(output / "result.json"))
    return result


def load_frozen_response(path, data, device):
    saved = load_checkpoint(path)
    metadata = saved["metadata"]
    if metadata.get("coordinate_identity") != data.coordinate_identity or metadata.get("data_identity") != data.identity:
        raise ValueError("TIME_MODEL_DATA_IDENTITY_MISMATCH")
    if metadata.get("stage") != "time_response":
        raise ValueError("FINAL_TIME_RESPONSE_REQUIRED")
    try:
        selected_checkpoint(path)
    except (ValueError, FileNotFoundError) as error:
        raise ValueError("FINAL_TIME_RESPONSE_REQUIRED") from error
    # Initialization is overwritten by the checkpoint and must not consume training RNG.
    with torch.random.fork_rng(devices=[]):
        model = build_response_model(ModelConfig(**metadata["model_config"]), tensor(data.neighbors, device, torch.long)).to(device)
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False), metadata["response_config"]


@torch.no_grad()
def predict_response(paths, data, condition_id, n, seed, device, reference=None, models=None, as_tensor=False):
    if not paths or n < 1:
        raise ValueError("MODELS_AND_POSITIVE_POPULATION_REQUIRED")
    device = torch.device(device)
    result = []
    if reference is not None:
        reference = torch.as_tensor(reference, dtype=torch.float32, device=device)
        if reference.shape != (n, data.n_genes) or not torch.isfinite(reference).all():
            raise ValueError("FIXED_REFERENCE_SHAPE_OR_VALUE_MISMATCH")
    offset = 0
    # Mixture of ensemble populations, rather than averaging independent cells.
    for member, path in enumerate(paths):
        count = n // len(paths) + int(member < n % len(paths))
        if not count:
            continue
        model, cfg = models[member] if models is not None else load_frozen_response(path, data, device)
        batch = (data.sample_controls(condition_id, count, np.random.default_rng(seed + member))
                 if reference is None else data.action_features(condition_id))
        start = tensor(batch["control"], device) if reference is None else reference[offset:offset+count]
        offset += count
        generator = torch.Generator(device=device).manual_seed(seed + member)
        noise = generation_noise(model, start, generator)
        micro = cfg.get("inference_microbatch", 2)
        raw = torch.cat([model.sample(start[i:i+micro], model.encode_condition(
            **condition_kwargs(batch, len(start[i:i+micro]), device)), noise=noise[i:i+micro],
            steps=cfg["inference_ode_steps"], checkpoint=False) for i in range(0, count, micro)])
        result.append(projected(raw, data))
    joined = torch.cat(result)
    return joined if as_tensor else joined.cpu().numpy()
