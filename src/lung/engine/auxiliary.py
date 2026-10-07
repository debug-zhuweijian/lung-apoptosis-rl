"""Fit-only auxiliary stages with frozen biological representations and provenance."""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from lung.models import (PopulationPool, StateClassifier, TemperatureCalibrator,
                         BackwardModel, CorrectionModel, ErrorEstimator)
from lung.models.state_objectives import aggregate_nll, state_loss
from .checkpoint import (atomic_save, save_training, load_checkpoint, seed_all,
                         restore_rng)
from .config import config_hash, file_hash, project_path, StageUnavailable, state_objective
from .response import load_response, tensor


def calibration_audit_binding(data, config, pipeline):
    """Actual frozen inputs to a pre-C numerical audit, recomputed from disk."""
    keys = ("response", "pool", "state", "backward", "correction", "correction_gate", "uncertainty", "endpoint")
    weights = {}
    for key in keys:
        if pipeline.get(key):
            paths = pipeline[key] if isinstance(pipeline[key], (list, tuple)) else [pipeline[key]]
            weights[key] = [file_hash(path) for path in paths]
    protocol = config.get("protocol")
    protocol_hash = file_hash(project_path(protocol)) if protocol else config_hash(
        {"name": "single-exposure-24h", "horizon_hours": 24, "stop_tail": None})
    return {"data_identity": data.identity, "weights": weights, "protocol_hash": protocol_hash,
            "task": config.get("task"), "history_support": config.get("history_support", "single_exposure_only"),
            "population": config.get("population", 512),
            "inference_microbatch": config["response"].get("inference_microbatch", 1),
            "inference_ode_steps": pipeline.get("inference_ode_steps", config["response"]["inference_ode_steps"])}


def require_population_audit(data, config, pipeline):
    path = pipeline.get("numerical_audit")
    if not path:
        raise StageUnavailable("NUMERICAL_AUDIT_REQUIRED_BEFORE_C_CALIBRATION")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("passed") is not True or (not config.get("synthetic", False) and report.get("synthetic_test_only", False)):
        raise StageUnavailable("NUMERICAL_AUDIT_REQUIRED_BEFORE_C_CALIBRATION")
    identity = pipeline.get("precalibration_identity")
    if not identity or report.get("biological_identity") != identity:
        raise StageUnavailable("NUMERICAL_AUDIT_IDENTITY_MISMATCH")
    if report.get("calibration_binding") != calibration_audit_binding(data, config, pipeline):
        raise StageUnavailable("NUMERICAL_AUDIT_DEPENDENCY_MISMATCH")


def dependency_record(pipeline, keys, files=()):
    """Store training-time content identities, not only mutable file locations."""
    values = {}
    for key in keys:
        if not pipeline.get(key):
            raise StageUnavailable(f"AUX_DEPENDENCY_REQUIRED: {key}")
        paths = pipeline[key] if isinstance(pipeline[key], (list, tuple)) else [pipeline[key]]
        values[key] = [file_hash(path) for path in paths]
    return {"pipeline": values,
            "files": [{"path": str(Path(path).resolve()), "sha256": file_hash(path)} for path in files]}


def verify_auxiliary_dependencies(path, pipeline):
    dependencies = load_checkpoint(path)["metadata"].get("dependencies")
    if not dependencies:
        raise StageUnavailable("AUX_DEPENDENCIES_UNRECORDED: rebuild this auxiliary checkpoint")
    for key, expected in dependencies["pipeline"].items():
        if not pipeline.get(key):
            raise StageUnavailable(f"AUX_DEPENDENCY_MISSING: {key}")
        current = pipeline[key] if isinstance(pipeline[key], (list, tuple)) else [pipeline[key]]
        if [file_hash(value) for value in current] != expected:
            raise StageUnavailable(f"AUX_DEPENDENCY_MISMATCH: {key}")
    for entry in dependencies["files"]:
        if not Path(entry["path"]).is_file() or file_hash(entry["path"]) != entry["sha256"]:
            raise StageUnavailable("AUX_FEATURE_ASSET_MISMATCH")
    return True


def pipeline_identity(stage, data, config, pipeline):
    name = stage.rsplit("/", 1)[-1]
    upstream = {"pool": ("response",), "state": ("response",),
                "backward": ("response", "pool"), "residual_predictions": ("response", "pool"),
                "correction": ("response", "pool", "backward"),
                "uncertainty": ("response", "pool", "backward", "correction")}[name]
    paths = {key: [value] if isinstance(value, str) else value
             for key, value in pipeline.items() if key in upstream}
    hashes = {key: [file_hash(path) for path in values] for key, values in paths.items()}
    settings = {key: value for key, value in config.items()
                if key in {"seed", "population", "state_fold", "state_batch", "aux_batch", "aux_lr", "calibration_steps", "calibration_lr", "early_stopping", "training_control"}}
    settings["inference_ode_steps"] = config["response"]["inference_ode_steps"]
    if name in {"pool", "state"}:
        settings["inference_microbatch"] = config["response"].get("inference_microbatch", 1)
    if name == "state":
        settings.update(state_objective(config))
        settings["execution"] = "state-batched-v2"
    return config_hash({"stage": stage, "data": data.identity, "dependencies": hashes, "settings": settings})


def fit_epochs(stage, model, rows, loss_function, data, config, monitor, identity,
               metadata, validation=None, resume=False, batch_size=None, batch_loss=None):
    """Loss is returned per independent unit. No G/C/E checkpoint selection."""
    if not rows:
        raise StageUnavailable(f"SUPERVISION_UNAVAILABLE: {stage} has no permitted fitting units")
    directory = monitor.weights / stage
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("aux_lr", 1e-4))
    batch_size = batch_size or config.get("aux_batch", 16)
    start, step, best = 0, 0, float("inf")
    from .training_control import TrainingControl
    controller_config = {**config, "training_schedule": {"mode": "full"}}
    if validation is None:
        controller_config["early_stopping"] = False
    saved_control = None
    metadata = {**metadata, "stage": stage.split("/")[-1], "data_identity": data.identity,
                "synthetic": config.get("synthetic", False)}
    if resume and (directory / "last.pt").exists():
        saved = load_checkpoint(directory / "last.pt", identity)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, step, best = saved["epoch"], saved["counters"]["optimizer_step"], saved["extra"]["best"]
        saved_control = saved["extra"].get("training_control")
        restore_rng(saved["rng"])
    controller = TrainingControl(controller_config, config["epochs"], saved_control)
    if saved_control is None and start:
        controller.advance(start, step)
    for epoch in range(start, config["epochs"]):
        if controller.stopped:
            break
        model.train()
        order = np.random.default_rng(config.get("seed", 20260923) + epoch).permutation(len(rows))
        total = 0.
        for offset in monitor.progress(range(0, len(order), batch_size), desc=f"{stage} epoch {epoch+1}"):
            selected = order[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if batch_loss is not None:
                losses = batch_loss(model, [rows[int(index)] for index in selected])
                if losses.shape != (len(selected),) or not torch.isfinite(losses).all():
                    raise FloatingPointError(f"NONFINITE_AUX_LOSS: {stage}")
                losses.mean().backward()
                total += float(losses.detach().sum()) / len(rows)
            else:
                for index in selected:
                    loss = loss_function(model, rows[int(index)])
                    if loss.ndim or not torch.isfinite(loss):
                        raise FloatingPointError(f"NONFINITE_AUX_LOSS: {stage}")
                    (loss / len(selected)).backward()
                    total += float(loss.detach()) / len(rows)
            optimizer.step()
            step += 1
        controller.advance(epoch + 1, step)
        values = {"train/loss": total}
        improved = False
        if validation is not None and (controller.validation_due() or controller.stopped):
            model.eval()
            with torch.no_grad():
                values.update(validation(model))
            score = values["val/loss"]
            controller.observe(score)
            improved = score < best
            best = min(best, score)
        arguments = dict(epoch=epoch + 1, identity=identity, metadata=metadata,
                         counters={"optimizer_step": step}, extra={"best": best,
                         "training_control": controller.state_dict(),
                         "selection_available": validation is not None})
        save_training(directory / "last.pt", model, optimizer, **arguments)
        if improved:
            save_training(directory / "best.pt", model, optimizer, **arguments)
        if (epoch + 1) % config.get("save_period", 10) == 0:
            save_training(directory / f"epoch_{epoch+1}.pt", model, optimizer, **arguments)
        monitor.record(stage, epoch + 1, values)
    final = directory / ("best.pt" if validation is not None else "last.pt")
    if not final.exists():
        raise StageUnavailable(f"TRAINED_AUX_REQUIRED: {final}")
    model.load_state_dict(load_checkpoint(final, identity)["model"])
    model.eval()
    return str(final)


@torch.no_grad()
def encode_vectors(reference, expression, device, *, batch_size=1, output_device: str | torch.device = "cpu"):
    if not len(expression):
        raise StageUnavailable("EMPTY_EXPRESSION_GROUP")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Expression encoding batch_size must be a positive integer.")
    vectors = []
    for start in range(0, len(expression), batch_size):
        batch = expression[start:start + batch_size]
        values = batch.to(device=device, dtype=torch.float32) if isinstance(batch, torch.Tensor) else tensor(batch, device)
        vectors.append(reference.encode_cells(values)[1].to(output_device))
    return torch.cat(vectors)


def feature_cache(path, identity, build):
    path = Path(path)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("AUX_FEATURE_CACHE_IDENTITY_MISMATCH")
        return saved["rows"]
    rows = build()
    atomic_save(path, {"identity": identity, "rows": rows})
    return rows


def train_pool(data, config, monitor, pipeline, resume=False):
    device = torch.device(config["device"])
    reference = load_response(pipeline["response"][0], data, device)
    width = reference.config.d_model
    identity = pipeline_identity("pool", data, config, {"response": pipeline["response"]})
    seed_all(config.get("seed", 20260923))
    model = PopulationPool(width).to(device)

    def records(role):
        def build():
            rng = np.random.default_rng(config.get("seed", 20260923))
            rows = []
            for key in monitor.progress(data.condition_ids(role), desc=f"cache pool {role}"):
                batch = data.sample_response(key, config.get("population", 512), rng, role=role)
                z = encode_vectors(reference, batch["target"], device,
                                   batch_size=config["response"].get("inference_microbatch", 1))
                rows.append({"statistics": model.statistics(z), "condition_id": key})
            return rows
        return feature_cache(monitor.experiment / "cache" / f"pool-{role}-{identity}.pt", identity, build)

    train, selection = records("T"), records("V")
    if not selection:
        raise StageUnavailable("POOL_SELECTION_UNAVAILABLE: V is empty")

    def loss(network, row):
        statistics = row["statistics"].to(device)
        return F.mse_loss(network.reconstruction(network.encoder(statistics)), statistics)

    def validate(network):
        return {"val/loss": float(torch.stack([loss(network, row) for row in selection]).mean())}

    path = fit_epochs("pool", model, train, loss, data, config, monitor, identity,
                      {"width": width, "train_role": "T", "selection_role": "V",
                       "reference_response": pipeline["response"][0],
                       "reference_response_hash": file_hash(pipeline["response"][0]),
                       "dependencies": dependency_record(pipeline, ("response",))}, validate, resume)
    return {**pipeline, "pool": path}


def state_metrics(logits, labels):
    """Unweighted probability/recognition metrics for the original sorting labels."""
    prediction, probabilities = logits.argmax(-1), logits.softmax(-1)
    confusion = torch.zeros(3, 3, device=labels.device, dtype=torch.float32)
    confusion.index_put_((labels, prediction), torch.ones_like(labels, dtype=torch.float32), accumulate=True)
    recall = confusion.diagonal() / confusion.sum(1).clamp_min(1)
    precision = confusion.diagonal() / confusion.sum(0).clamp_min(1)
    f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-12)
    nll = float(F.cross_entropy(logits, labels))
    positive = (labels > 0).to(probabilities.dtype)
    score = probabilities[:, 1:].sum(-1)
    predicted_positive = score >= .5
    positive_mask = labels > 0
    tp = (predicted_positive & positive_mask).sum().float()
    tn = (~predicted_positive & ~positive_mask).sum().float()
    fp = (predicted_positive & ~positive_mask).sum().float()
    fn = (~predicted_positive & positive_mask).sum().float()
    # Ten fixed equal-width bins. The final bin includes probability 1 exactly.
    bins = (score * 10).long().clamp(max=9)
    ece = score.new_zeros(())
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.float().mean() * (score[mask].mean() - positive[mask].mean()).abs()
    result = {"val/loss": nll, "val/nll": nll, "val/macro_f1": float(f1.mean()),
              "val/accuracy": float((prediction == labels).float().mean()),
              "val/balanced_accuracy": float(recall.mean()),
              "val/brier": float((probabilities - F.one_hot(labels, 3)).square().sum(1).mean()),
              "val/aggregate_nll": float(aggregate_nll(logits, labels).mean()),
              "val/aggregate_brier": float((score - positive).square().mean()),
              "val/aggregate_ece_10": float(ece),
              "val/aggregate_mean_score": float(score.mean()),
              "val/aggregate_label_fraction": float(positive.mean()),
              "val/aggregate_mean_bias": float((score - positive).mean()),
              "val/aggregate_accuracy": float((predicted_positive == positive_mask).float().mean()),
              "val/aggregate_recall": float(tp / (tp + fn).clamp_min(1)),
              "val/aggregate_specificity": float(tn / (tn + fp).clamp_min(1)),
              "val/aggregate_precision": float(tp / (tp + fp).clamp_min(1)),
              "val/aggregate_f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1))}
    result.update({f"val/recall_{i}": float(recall[i]) for i in range(3)})
    result.update({f"val/confusion_{i}_{j}": float(confusion[i, j]) for i in range(3) for j in range(3)})
    return result


def state_batch_loss(network, rows, device, objective, weights):
    """One loss per original unit, using fixed fit-set weights, including tails."""
    vectors = torch.stack([row[0] for row in rows]).to(device)
    labels = torch.stack([row[1] for row in rows]).to(device)
    return state_loss(network(vectors), labels, objective=objective["state_loss"],
                      aggregate_lambda=objective["state_aggregate_lambda"], weights=weights)


def train_state(data, config, monitor, pipeline, resume=False):
    objective = state_objective(config)
    device = torch.device(config["device"])
    reference = load_response(pipeline["response"][0], data, device)
    width, fold = reference.config.d_model, config.get("state_fold", "1_to_8")
    identity = pipeline_identity("state", data, config, {"response": pipeline["response"]})
    seed_all(config.get("seed", 20260923))
    model = StateClassifier(width).to(device)

    def records(role):
        role_identity = config_hash({"state_identity": identity, "role": role})
        def build():
            batch = data.state_arrays(fold, role)
            z = encode_vectors(reference, batch["x"], device,
                               batch_size=config["response"].get("inference_microbatch", 1))
            return {"vectors": z, "labels": torch.as_tensor(batch["labels"], dtype=torch.long)}
        return feature_cache(monitor.experiment / "cache" / f"state-{role}-{identity}.pt", role_identity, build)

    fitting, selection = records("fit"), records("selection")
    weights = StateClassifier.class_weights(fitting["labels"]).to(device)
    # Apply a fixed global class weighting, independent of microbatch composition.
    weights = weights / weights[fitting["labels"].to(device)].mean()
    rows = list(zip(fitting["vectors"], fitting["labels"]))

    def loss(network, row):
        vector, label = row[0].to(device), row[1].to(device)
        return state_loss(network(vector).reshape(1, 3), label.reshape(1),
                          objective=objective["state_loss"],
                          aggregate_lambda=objective["state_aggregate_lambda"], weights=weights)[0]

    def validate(network):
        return state_metrics(network(selection["vectors"].to(device)), selection["labels"].to(device))

    metadata = {"width": width, "state_fold": fold, "train_role": "fit", "selection_role": "selection",
                **objective, "selection_metric": "unweighted_three_class_nll",
                "label_semantics": "original_TIDYI_sorting_labels_not_apoptosis_outcomes",
                "aggregate_classes": [1, 2], "class_weights": weights.cpu().tolist() if objective["state_loss"] == "weighted_ce" else None,
                "class_order": list(StateClassifier.LABELS), "reference_response": pipeline["response"][0],
                "reference_response_hash": file_hash(pipeline["response"][0]),
                "dependencies": dependency_record(pipeline, ("response",))}
    best_path = fit_epochs("state", model, rows, loss, data, config, monitor, identity, metadata,
                           validate, resume, config.get("state_batch", 64),
                           batch_loss=lambda network, selected: state_batch_loss(
                               network, selected, device, objective, weights))
    calibration = records("calibration")
    logits = model(calibration["vectors"].to(device)).detach()
    labels = calibration["labels"].to(device)
    calibrator = TemperatureCalibrator().to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=config.get("calibration_lr", 1e-2))
    for _ in monitor.progress(range(config.get("calibration_steps", 2000)), desc="state temperature"):
        optimizer.zero_grad(set_to_none=True)
        error = F.cross_entropy(calibrator(logits), labels)
        error.backward()
        optimizer.step()
    model.set_temperature(calibrator.temperature)
    saved = load_checkpoint(best_path)
    monitor.record("state_calibration", saved["epoch"],
                   {key.replace("val/", "calibration/"): value for key, value in
                    state_metrics(logits / model.temperature, labels).items()})
    calibrated_path = monitor.weights / "state" / "calibrated.pt"
    save_training(calibrated_path, model, None, epoch=saved["epoch"], identity=identity,
                  metadata={**saved["metadata"], "calibration_role": "calibration", "uncalibrated_best": best_path,
                            "calibration_objective": "unweighted_three_class_nll"},
                  counters=saved["counters"], extra={**saved.get("extra", {}), "temperature": float(model.temperature),
                  "calibration_lr": config.get("calibration_lr", 1e-2)})
    return {**pipeline, "state": str(calibrated_path)}


def load_auxiliary(path, data, device):
    saved = load_checkpoint(path)
    metadata = saved["metadata"]
    if metadata.get("data_identity") != data.identity or saved["epoch"] < 1:
        raise ValueError("AUX_DATA_OR_TRAINING_IDENTITY_MISMATCH")
    stage, width = metadata["stage"], metadata["width"]
    if stage == "endpoint":
        from .endpoints import load_endpoint
        return load_endpoint(path, data, device)
    constructors = {"pool": lambda: PopulationPool(width), "state": lambda: StateClassifier(width),
                    "backward": lambda: BackwardModel(metadata["n_genes"], width),
                    "correction": lambda: CorrectionModel(metadata["n_genes"], width),
                    "uncertainty": lambda: ErrorEstimator(width)}
    if stage not in constructors:
        raise ValueError(f"Unknown auxiliary checkpoint stage: {stage}")
    model = constructors[stage]()
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False)


def train_auxiliary(stage, data, config, monitor, pipeline, resume=False):
    if not pipeline.get("response"):
        raise StageUnavailable("TRAINED_RESPONSE_REQUIRED")
    if stage == "pool":
        return train_pool(data, config, monitor, pipeline, resume)
    if stage == "state":
        return train_state(data, config, monitor, pipeline, resume)
    if stage in {"backward", "correction", "uncertainty"}:
        from .residuals import train_residual
        return train_residual(stage, data, config, monitor, pipeline, resume)
    if stage == "endpoint":
        from .endpoints import train_endpoint
        return train_endpoint(data, config, monitor, pipeline, resume)
    raise ValueError(f"Unsupported auxiliary stage: {stage}")
