"""Independent original-sorting C fitting and same-P three-class sidecars.

The default entry reads metadata only. Explicit fitting keeps the established
weighted CE and calibration roles; scoring never interprets class 1+2 as an
Annexin endpoint, population death rate, or decision reward.
"""
from pathlib import Path
import json
import math

from filelock import FileLock
import numpy as np
import pandas as pd
import torch

from lung.models import StateClassifier
from lung.data.common import dataset_config
from . import full_protocol_response as producer
from . import full_protocol_response_learning as learning
from . import auxiliary
from .checkpoint import atomic_json, atomic_save, load_checkpoint
from .config import ROOT, config_hash, file_hash, merge, state_objective
from .data import EngineData
from .representation import load_representation, require_finite_state, require_trained
from .representation_cache import encoder_identity
from .training_control import checkpoint_completion

SCHEMA = "raw-sorting-state-C-v1"
SIDECAR_SCHEMA = "raw-sorting-state-sidecar-v1"
LABELS = list(StateClassifier.LABELS)
GATE_FOLLOWUP = ROOT / "experiments/20261011-data-alignment-round22/root_state_gate_followup.json"
FLAGS = {"source_C_transfer_validated": False, "cross_source_C_calibration_verified": False,
    "Annexin_interpretation": False, "scientific_qualified": False,
    "functional_Y_read": False, "reward_generated": False}
REPLAY_POLICY = "exact_float64_P_moments_replay"


def require(ok, reason):
    if not ok:
        raise ValueError("RAW_STATE_C_" + reason)


def state_config(config):
    return merge(learning.joint_ae_config(config), {"stage": "state", "task": config["task"]})


def settings(config):
    require(state_objective(config) == {"state_loss": "weighted_ce", "state_aggregate_lambda": 0.},
            "ORIGINAL_WEIGHTED_CE_REQUIRED")
    require(config.get("state_fold", "1_to_8") == "1_to_8", "READY_ORIGINAL_FOLD_REQUIRED")
    value = {k: config[k] for k in ("seed", "device", "epochs", "aux_lr", "calibration_steps", "calibration_lr")}
    value.update({"state_fold": "1_to_8", "state_batch": config.get("state_batch", 64),
        "inference_microbatch": config["response"].get("inference_microbatch", 1),
        "state_loss": "weighted_ce", "state_aggregate_lambda": 0.,
        "early_stopping": config.get("early_stopping", False),
        "training_control": config.get("training_control", {})})
    require(type(value["seed"]) is int and value["seed"] >= 0, "SEED")
    require(all(type(value[k]) is int and value[k] > 0 for k in
        ("epochs", "state_batch", "inference_microbatch", "calibration_steps")), "INTEGER_SETTINGS")
    require(all(type(value[k]) in (int, float) and math.isfinite(value[k]) and value[k] > 0
        for k in ("aux_lr", "calibration_lr")), "OPTIMIZER_SETTINGS")
    return value


def verify_ae_origin(ae_run, data_identity):
    """JSON/bytes only: a completed independent matching AE, never old runs."""
    ae_run = Path(ae_run).resolve()
    require(ae_run.is_relative_to(ROOT / "experiments"), "AE_ORIGIN_PROJECT_BOUNDARY")
    result = json.loads((ae_run / "AE_result.json").read_text(encoding="utf-8"))
    learning.verify_completed_ae_result(result, ae_run, data_identity)
    manifest = json.loads((ae_run / "run_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == learning.SCHEMA + ":joint-AE"
        and manifest.get("data_identity") == data_identity
        and manifest.get("config", {}).get("synthetic") is False
        and manifest["config"].get("data") == learning.JOINT_CONFIG, "AE_ORIGIN_IDENTITY")
    return {"run": str(ae_run), "result_sha256": file_hash(ae_run / "AE_result.json"),
        "manifest_sha256": file_hash(ae_run / "run_manifest.json"),
        "weights": result["weights"], "checkpoint_sha256": result["checkpoint_sha256"],
        "final_checkpoint": result["final_checkpoint"], "final_checkpoint_sha256": result["final_checkpoint_sha256"]}


def preflight(config, *, ae_run=None):
    """Check current bound assets and split metadata; no RNA values/PT."""
    config = state_config(config)
    data_config = dataset_config(ROOT, ROOT / config["data"])
    assets = ROOT / data_config["assets"]
    manifest_path = assets / "manifest.json"
    require(file_hash(manifest_path) == producer.inputs.PINS[manifest_path.relative_to(ROOT).as_posix()],
            "STATE_DATA_MANIFEST_CHANGED")
    manifest = producer.inputs.read(manifest_path)
    require(manifest["status"] == "complete" and manifest["config"] == data_config, "STATE_DATA_CONFIG_CHANGED")
    require(file_hash(ROOT / data_config["bundle"] / "manifest.json") == manifest["source_manifest_sha256"],
            "STATE_SOURCE_MANIFEST_CHANGED")
    for name in ("genes.parquet", "normalization.npz", "tidyi_splits.parquet"):
        require(file_hash(assets / name) == manifest["files"][name]["sha256"], "STATE_ASSET_CHANGED")
    data_identity = config_hash({"signature": manifest["signature"], "files": manifest["files"]})
    require(manifest["metadata"]["tidyi_folds"].get("1_to_8", {}).get("status") == "ready",
            "STATE_SPLIT_INSUFFICIENT")
    frame = pd.read_parquet(assets / "tidyi_splits.parquet")
    frame = frame.loc[frame["fold"].eq("1_to_8")].copy()
    require(frame["cell_id"].notna().all() and frame["cell_id"].is_unique, "ROLE_OVERLAP")
    require(frame["state_class"].eq(frame["state_label_original"].map(dict(zip(LABELS, range(3))))).all()
        and frame["label_scope"].eq("original_sorted_state_not_natural_apoptosis_rate").all(),
        "ORIGINAL_LABEL_SEMANTICS_CHANGED")
    groups = {r: frame.loc[frame["role"].eq(r)].copy()
              for r in ("fit", "selection", "calibration")}
    require(all(len(q) and set(q["state_class"]) == {0, 1, 2}
        and q["cell_id"].is_unique for q in groups.values()), "ORIGINAL_ROLES_OR_CLASSES")
    role_plan = {r: {"cells": len(q), "class_counts": [int(q["state_class"].eq(i).sum()) for i in range(3)],
        "cell_ids_sha256": config_hash(q["cell_id"].tolist())} for r, q in groups.items()}
    origin, missing = None, "matching_completed_joint_AE_missing"
    if ae_run and (Path(ae_run) / "AE_result.json").is_file():
        origin, missing = verify_ae_origin(ae_run, data_identity), None
    return {"schema": SCHEMA + ":preflight", "data_identity": data_identity,
        "gene_axis_sha256": manifest["files"]["genes.parquet"]["sha256"],
        "normalization_sha256": manifest["files"]["normalization.npz"]["sha256"],
        "settings": settings(config),
        "class_order": LABELS, "roles": role_plan,
        "raw_label_semantics": "original_sorted_state_not_natural_apoptosis_rate",
        "AE_origin": origin, "remaining_dependency": missing, "fit_inputs_ready": origin is not None,
        "RNA_matrix_values_reread": 0, "PT_loads": 0, "model_forwards": 0, "optimizer_steps": 0,
        "final_RNA_read": False, "class_probabilities_generated": 0, **FLAGS}


class StateMonitor:
    def __init__(self, output):
        self.experiment, self.weights = Path(output), Path(output) / "weights"

    def progress(self, values, **kwargs):
        return values

    def record(self, stage, epoch, values):
        with (self.experiment / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "epoch": epoch, **values}) + "\n")


def validate_state_declaration(path, *, allow_synthetic=False):
    """Origin/completion/integrity only; not leakage or transfer certification."""
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "experiments"), "STATE_DECLARATION_PROJECT_BOUNDARY")
    body = json.loads(path.read_text(encoding="utf-8"))
    require(body.get("schema") == SCHEMA + ":completed" and body.get("state_identity")
        == config_hash({k: v for k, v in body.items() if k != "state_identity"}), "STATE_DECLARATION_CHANGED")
    require(body.get("synthetic") is False or allow_synthetic, "SYNTHETIC_STATE_HAS_NO_PRODUCTION_QUALIFICATION")
    require(body.get("class_order") == LABELS and all(body.get(k) is v for k, v in FLAGS.items())
        and body.get("training_completion", {}).get("status") == "complete"
        and body.get("calibration_completion", {}).get("status") == "complete"
        and type(body["calibration_completion"].get("actual_steps")) is int
        and body["calibration_completion"]["actual_steps"] == body["calibration_completion"].get("target_steps"),
        "STATE_SCOPE_OR_COMPLETION")
    run = path.parent
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    require(body.get("manifest_sha256") == file_hash(run / "run_manifest.json")
        and manifest.get("schema") == SCHEMA + ":run" and manifest.get("synthetic") is body["synthetic"]
        and manifest.get("data_identity") == body.get("data_identity")
        and manifest.get("AE_origin") == body.get("AE_origin"), "STATE_RUN_IDENTITY_CHANGED")
    for key, name in (("serving", "state_weights.pt"), ("calibrated", "weights/state/calibrated.pt"),
        ("best", "weights/state/best.pt"), ("final", "weights/state/final.pt")):
        entry = body["assets"][key]
        require(Path(entry["path"]).resolve() == run / name and file_hash(run / name) == entry["sha256"],
                "STATE_ASSET_CHANGED")
    cfg, completion, calibration = manifest["config"], body["training_completion"], body["calibration_completion"]
    settings(cfg)
    require(body.get("state_fold") == cfg.get("state_fold", "1_to_8") == "1_to_8"
        and type(body.get("width")) is int and body["width"] > 0
        and body["representation_identity"].get("d_model") == body["width"]
        and type(completion.get("actual_epochs")) is int and 0 < completion["actual_epochs"] <= cfg["epochs"]
        and type(completion.get("actual_updates")) is int and completion["actual_updates"] > 0
        and type(completion.get("target_epochs")) is int and completion.get("target_epochs") == cfg["epochs"]
        and completion.get("reason") in ("max_epochs", "validation_plateau")
        and (completion["reason"] != "max_epochs" or completion["actual_epochs"] == cfg["epochs"])
        and type(body.get("selected_epoch")) is int and 0 < body["selected_epoch"] <= completion["actual_epochs"]
        and type(body.get("selected_optimizer_step")) is int
        and 0 < body["selected_optimizer_step"] <= completion["actual_updates"], "STATE_MANIFEST_FOLD_OR_COUNTERS")
    require(type(calibration.get("target_steps")) is int and calibration["target_steps"] == cfg["calibration_steps"]
        and calibration.get("role") == "calibration" and calibration.get("objective") == "unweighted_three_class_nll"
        and type(calibration.get("lr")) in (int, float) and calibration["lr"] == cfg["calibration_lr"],
        "STATE_CALIBRATION_DEPENDENCIES_CHANGED")
    if not body["synthetic"]:
        require(verify_ae_origin(body["AE_origin"]["run"], body["data_identity"]) == body["AE_origin"],
                "STATE_AE_ORIGIN_CHANGED")
    return body


def fit_state(data, config, ae_run, output, *, synthetic=False, monitor=None):
    """Future explicit C fit using original train_state and only the frozen AE."""
    settings(config)
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / "experiments"), "FIT_OUTPUT_PROJECT_BOUNDARY")
    origin = verify_ae_origin(ae_run, data.identity)
    manifest = {"schema": SCHEMA + ":run", "data_identity": data.identity, "config": config,
        "AE_origin": origin, "synthetic": synthetic}
    with FileLock(str(output) + ".lock", timeout=0):
        require(not output.exists(), "NEW_STATE_NAMESPACE_REQUIRED")
        output.mkdir(parents=True)
        atomic_json(output / "run_manifest.json", manifest)
        pipeline = auxiliary.train_state(data, config, monitor or StateMonitor(output),
            {"representation": origin["weights"]})
        directory = output / "weights/state"
        calibrated, final, best = [load_checkpoint(directory / n) for n in
            ("calibrated.pt", "final.pt", "best.pt")]
        completion = checkpoint_completion(final, config["epochs"])
        meta, calibration = calibrated["metadata"], calibrated.get("extra", {}).get("calibration_completion", {})
        require(meta.get("stage") == "state" and meta.get("data_identity") == data.identity
            and meta.get("class_order") == LABELS and meta.get("train_role") == "fit"
            and meta.get("selection_role") == "selection" and meta.get("calibration_role") == "calibration"
            and meta.get("reference_representation_hash") == origin["checkpoint_sha256"]
            and meta.get("state_loss") == "weighted_ce" and meta.get("state_aggregate_lambda") == 0.
            and Path(pipeline["state"]).resolve() == directory / "calibrated.pt"
            and Path(meta["uncalibrated_best"]).resolve() == directory / "best.pt", "STATE_TRAINING_CONTRACT")
        require(calibration.get("status") == "complete" and type(calibration.get("actual_steps")) is int
            and calibration.get("actual_steps") == config["calibration_steps"]
            and calibration.get("target_steps") == config["calibration_steps"],
            "STATE_CALIBRATION_NOT_COMPLETE")
        require_trained(best, "RAW_STATE_C_NO_UPDATES")
        require_finite_state(calibrated["model"])
        value = calibrated["model"].get("temperature")
        require(isinstance(value, torch.Tensor) and value.numel() == 1 and float(value) > 1e-3,
                "STATE_TEMPERATURE_INVALID")
        snapshot = {"schema": SCHEMA + ":safe-serving", "model": calibrated["model"],
            "metadata": meta, "identity": calibrated["identity"], "epoch": calibrated["epoch"],
            "counters": calibrated["counters"]}
        atomic_save(output / "state_weights.pt", snapshot)
        body = {"schema": SCHEMA + ":completed", "synthetic": synthetic, "data_identity": data.identity,
            "AE_origin": origin, "representation_identity": meta["representation_identity"],
            "width": meta["width"], "class_order": LABELS, "state_fold": meta["state_fold"],
            "selected_epoch": calibrated["epoch"], "selected_optimizer_step": calibrated["counters"]["optimizer_step"],
            "training_completion": completion, "calibration_completion": calibration,
            "training_identity": calibrated["identity"], "manifest_sha256": file_hash(output / "run_manifest.json"),
            "assets": {k: {"path": str(p), "sha256": file_hash(p)} for k, p in
                (("serving", output / "state_weights.pt"), ("calibrated", directory / "calibrated.pt"),
                 ("best", directory / "best.pt"), ("final", directory / "final.pt"))}, **FLAGS}
        body["state_identity"] = config_hash(body)
        atomic_json(output / "state_declaration.json", body)
        return validate_state_declaration(output / "state_declaration.json", allow_synthetic=synthetic)


def state_asset(path, declaration):
    return {"declaration_path": str(Path(path).resolve()), "declaration_sha256": file_hash(path),
        "serving_path": declaration["assets"]["serving"]["path"],
        "serving_sha256": declaration["assets"]["serving"]["sha256"],
        "state_identity": declaration["state_identity"],
        "reference_AE_origin_sha256": declaration["AE_origin"]["checkpoint_sha256"],
        "representation_identity": declaration["representation_identity"]}


def load_state_for_prediction(path, prediction, model):
    declaration = validate_state_declaration(path)
    require(declaration["data_identity"] == prediction["bindings"]["data_identity"], "P_C_DATA_IDENTITY_MISMATCH")
    p_meta = prediction["model_metadata"]["representation"]
    exported = torch.load(producer.ROOT / p_meta["checkpoint"], map_location="cpu", weights_only=True)
    require(exported.get("metadata", {}).get("source_checkpoint_sha256")
        == declaration["AE_origin"]["checkpoint_sha256"], "P_AE_ORIGIN_MISMATCH")
    strip = lambda q: {k: v for k, v in q.items() if k != "checkpoint_sha256"}
    current = encoder_identity(model.shared, model.shared.config)
    require(strip(current) == strip(declaration["representation_identity"]), "P_C_REPRESENTATION_MISMATCH")
    return load_classifier(declaration, next(model.parameters()).device), declaration


def load_classifier(declaration, device):
    """Check the actual safe C stage/roles/objective/counters before construct."""
    saved = torch.load(declaration["assets"]["serving"]["path"], map_location="cpu", weights_only=True)
    meta = saved.get("metadata", {})
    require(saved.get("schema") == SCHEMA + ":safe-serving" and saved.get("identity") == declaration["training_identity"]
        and type(saved.get("epoch")) is int and saved["epoch"] == declaration["selected_epoch"]
        and type(saved.get("counters", {}).get("optimizer_step")) is int
        and saved["counters"]["optimizer_step"] == declaration["selected_optimizer_step"], "C_SERVING_COUNTERS_MISMATCH")
    require(meta.get("stage") == "state" and meta.get("width") == declaration["width"]
        and meta.get("synthetic") is declaration["synthetic"] and meta.get("data_identity") == declaration["data_identity"]
        and meta.get("representation_identity") == declaration["representation_identity"]
        and meta.get("reference_representation_hash") == declaration["AE_origin"]["checkpoint_sha256"]
        and Path(meta.get("reference_representation", "")).resolve() == Path(declaration["AE_origin"]["weights"]).resolve()
        and meta.get("dependencies") == {"pipeline": {"representation": [declaration["AE_origin"]["checkpoint_sha256"]]}, "files": []}
        and meta.get("class_order") == LABELS and meta.get("state_fold") == declaration["state_fold"] == "1_to_8"
        and meta.get("input_coordinate") == "h" and meta.get("train_role") == "fit"
        and meta.get("selection_role") == "selection" and meta.get("calibration_role") == "calibration"
        and meta.get("state_loss") == "weighted_ce" and type(meta.get("state_aggregate_lambda")) in (float, int)
        and meta["state_aggregate_lambda"] == 0.
        and meta.get("selection_metric") == meta.get("calibration_objective") == "unweighted_three_class_nll"
        and meta.get("label_semantics") == "original_TIDYI_sorting_labels_not_apoptosis_outcomes"
        and Path(meta.get("uncalibrated_best", "")).resolve() == Path(declaration["assets"]["best"]["path"]).resolve(),
        "C_SERVING_CONTRACT_MISMATCH")
    require_trained(saved, "RAW_STATE_C_NO_UPDATES")
    require_finite_state(saved["model"])
    classifier = StateClassifier(declaration["width"])
    classifier.load_state_dict(saved["model"], strict=True)
    require(float(classifier.temperature) > 1e-3, "C_TEMPERATURE_INVALID")
    return classifier.to(device).eval().requires_grad_(False)


@torch.no_grad()
def replay_probabilities(model, classifier, reader, candidate, x, context, prediction, *, microbatch, device):
    """Exactly replay P, then restore its physical outlet to the same AE input."""
    require(type(microbatch) is int and microbatch > 0, "SCORING_MICROBATCH")
    generator = torch.Generator(device=device).manual_seed(prediction["seed"])
    noise = torch.randn((len(x), model.full.member.config.latent_dim), device=device, generator=generator)
    h_reference = producer.prepare_reference_h(model.shared, x, model.reference_contract["variant"],
        device=device, microbatch=microbatch)
    expression, probabilities = [], []
    mean = torch.as_tensor(context["mean"], device=device, dtype=torch.float32)
    scale = torch.as_tensor(context["scale"], device=device, dtype=torch.float32)
    require(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all(), "C_INPUT_NORMALIZATION")
    for start in range(0, len(x), microbatch):
        end = min(start + microbatch, len(x))
        kwargs, physical = producer.numeric_batch(reader, candidate, end-start, device)
        result = model.predict_h(h_reference[start:end], physical, kwargs, noise=noise[start:end], steps=prediction["steps"])
        require(torch.isfinite(result.y).all() and (result.y >= 0).all(), "PHYSICAL_P_OUTLET")
        # D/P y is common library10000 log1p. C was fitted on standardized E x.
        normalized = (result.y - mean) / scale
        _, h = model.shared.encode_h(normalized)
        probability = classifier.probabilities(h)
        require(probability.shape == (end-start, 3) and torch.isfinite(probability).all()
            and (probability >= 0).all() and (probability <= 1).all(), "C_PROBABILITIES_INVALID")
        expression.append(result.y.cpu().numpy())
        probabilities.append(probability.cpu().numpy())
    y = np.concatenate(expression).astype(np.float64)
    require(np.array_equal(y.mean(0), np.asarray(prediction["expression_mean"]))
        and np.array_equal(y.var(0), np.asarray(prediction["expression_variance"])), "P_REPLAY_MOMENTS_DIFFER")
    return np.concatenate(probabilities).astype(np.float64)


def score_prediction(prediction_path, declaration_path, output, *, microbatch=64, device="cpu"):
    """Future explicitly authorized model scoring; preparation never calls it."""
    prediction = producer.read_prediction(prediction_path)
    declaration = validate_state_declaration(declaration_path)
    require(declaration["data_identity"] == prediction["bindings"]["data_identity"], "P_C_DATA_IDENTITY_MISMATCH")
    prediction_sha, declaration_sha = file_hash(prediction_path), file_hash(declaration_path)
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / "experiments"), "SCORING_OUTPUT_PROJECT_BOUNDARY")
    with FileLock(str(output) + ".lock", timeout=0):
        require(not output.exists(), "NEW_SIDECAR_OUTPUT_REQUIRED")
        output.mkdir(parents=True)
        try:
            reader, _, context, bindings = producer.bound_inputs()
            model = producer.load_model(prediction["model_metadata"], bindings, context, device)
            classifier, loaded = load_state_for_prediction(declaration_path, prediction, model)
            require(loaded == declaration, "C_DECLARATION_CHANGED")
            x, reference = producer.read_proxy_reference(context)
            require(reference == prediction["reference"], "P_REFERENCE_REPLAY_CHANGED")
            probabilities = replay_probabilities(model, classifier, reader, prediction["candidate_id"], x,
                context, prediction, microbatch=microbatch, device=device)
            require(file_hash(prediction_path) == prediction_sha and file_hash(declaration_path) == declaration_sha
                and producer.read_prediction(prediction_path) == prediction
                and validate_state_declaration(declaration_path) == declaration, "SCORING_ASSET_CHANGED")
            body = {"schema": SIDECAR_SCHEMA, "candidate_id": prediction["candidate_id"],
                "prediction": {"path": str(Path(prediction_path).resolve()), "sha256": prediction_sha,
                    "prediction_identity": prediction["prediction_identity"]},
                "state_asset": state_asset(declaration_path, declaration), "class_order": LABELS,
                "P_AE_bridge": {"export_checkpoint_sha256": prediction["model_metadata"]["representation"]["checkpoint_sha256"],
                    "source_checkpoint_sha256": declaration["AE_origin"]["checkpoint_sha256"],
                    "representation_identity": {k: v for k, v in declaration["representation_identity"].items()
                        if k != "checkpoint_sha256"}},
                "probability_mean": probabilities.mean(0).tolist(),
                "probability_variance": probabilities.var(0).tolist(), "sample_count": len(probabilities),
                "predict_domain": producer.OUTLET,
                "sampling": {"seed": prediction["seed"], "steps": prediction["steps"],
                    "microbatch": microbatch, "device": device, "policy": REPLAY_POLICY},
                "P_replay_moments_exact": True, "C_input_coordinate": "same_AE_standardized_common_log1p_to_h",
                "probability_summary": "equal_reference_cell_weighted_original_three_class_probabilities",
                "raw_label_semantics": "original_TIDYI_sorting_labels_not_apoptosis_outcomes", **FLAGS}
            body["sidecar_identity"] = config_hash(body)
            atomic_json(output / "state_sidecar.json", body)
            return read_state_sidecar(output / "state_sidecar.json")
        except BaseException as error:
            atomic_json(output / "failure.json", {"schema": SIDECAR_SCHEMA + ":failure",
                "candidate_id": prediction["candidate_id"], "error_type": type(error).__name__, "error": str(error),
                "partial_preserved": True, **FLAGS})
            raise


def read_state_sidecar(path):
    """Read-only C outlet integrity/AE/P binding; no PT or model operations."""
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "experiments"), "SIDECAR_PROJECT_BOUNDARY")
    body = json.loads(path.read_text(encoding="utf-8"))
    require(body.get("schema") == SIDECAR_SCHEMA and body.get("sidecar_identity")
        == config_hash({k: v for k, v in body.items() if k != "sidecar_identity"}), "SIDECAR_CONTENT_CHANGED")
    p = body["prediction"]
    require(file_hash(p["path"]) == p["sha256"], "SIDECAR_P_BYTES_CHANGED")
    prediction = producer.read_prediction(p["path"])
    require(p["prediction_identity"] == prediction["prediction_identity"]
        and body.get("candidate_id") == prediction["candidate_id"], "SIDECAR_P_IDENTITY_CHANGED")
    asset = body["state_asset"]
    declaration = validate_state_declaration(asset["declaration_path"])
    require(asset == state_asset(asset["declaration_path"], declaration)
        and declaration["data_identity"] == prediction["bindings"]["data_identity"], "SIDECAR_C_IDENTITY_CHANGED")
    require(body.get("P_AE_bridge") == {
        "export_checkpoint_sha256": prediction["model_metadata"]["representation"]["checkpoint_sha256"],
        "source_checkpoint_sha256": declaration["AE_origin"]["checkpoint_sha256"],
        "representation_identity": {k: v for k, v in declaration["representation_identity"].items()
            if k != "checkpoint_sha256"}}, "SIDECAR_P_C_AE_BRIDGE_CHANGED")
    require(body.get("class_order") == LABELS and body.get("predict_domain") == producer.OUTLET
        and body.get("P_replay_moments_exact") is True
        and body.get("C_input_coordinate") == "same_AE_standardized_common_log1p_to_h"
        and body.get("probability_summary") == "equal_reference_cell_weighted_original_three_class_probabilities"
        and body.get("raw_label_semantics") == "original_TIDYI_sorting_labels_not_apoptosis_outcomes"
        and all(body.get(k) is v for k, v in FLAGS.items()), "SIDECAR_CLASS_OR_SCIENCE_CHANGED")
    sampling = body.get("sampling", {})
    require(sampling.get("seed") == prediction["seed"] and type(sampling.get("seed")) is int
        and sampling.get("steps") == prediction["steps"] and type(sampling.get("steps")) is int
        and type(sampling.get("microbatch")) is int and sampling["microbatch"] > 0
        and isinstance(sampling.get("device"), str) and sampling["device"]
        and sampling.get("policy") == REPLAY_POLICY and type(body.get("sample_count")) is int
        and body["sample_count"] == prediction["reference"]["shape"][0], "SIDECAR_SAMPLING_CHANGED")
    means, variances = [np.asarray(body.get(k)) for k in ("probability_mean", "probability_variance")]
    require(means.shape == variances.shape == (3,) and means.dtype.kind == variances.dtype.kind == "f"
        and np.isfinite(means).all() and np.isfinite(variances).all() and (means >= 0).all()
        and (means <= 1).all() and abs(float(means.sum())-1) <= 1e-6
        and (variances >= 0).all() and (variances <= .25+1e-8).all(), "SIDECAR_PROBABILITY_MOMENTS")
    return body


def gate_sensitivity_plan(frame):
    """Frozen metadata-only sensitivity; never a primary relabel/exclusion."""
    followup = json.loads(GATE_FOLLOWUP.read_text(encoding="utf-8"))
    require(followup.get("flagged_original_negative_cells") == 58
        and followup.get("no_label_error_proven") is True and followup.get("primary_labels_or_split_changed") is False,
        "ORIGINAL_GATE_FOLLOWUP_REQUIRED")
    for path, expected in followup["source_bindings"].items():
        require(Path(path).resolve().is_relative_to(ROOT) and file_hash(path) == expected, "GATE_SOURCE_CHANGED")
    ids_path = Path(followup["frozen_ID_file"]).resolve()
    require(ids_path.is_relative_to(ROOT / "experiments")
        and file_hash(ids_path) == followup["frozen_ID_sha256"], "FROZEN_GATE_IDS_CHANGED")
    audit = pd.read_csv(ids_path)
    ids = audit["cell_id"].tolist()
    require(len(ids) == len(set(ids)) == 58 and set(ids) <= set(frame["cell_id"])
        and frame.loc[frame["cell_id"].isin(ids), "state_class"].eq(0).all()
        and audit["label_changed"].eq(False).all() and audit["excluded_from_primary_analysis"].eq(False).all(),
        "GATE_IDS_ARE_FROZEN_ORIGINAL_NEGATIVE_METADATA")
    return {"followup_path": str(GATE_FOLLOWUP), "followup_sha256": file_hash(GATE_FOLLOWUP),
        "ID_file": str(ids_path), "ID_file_sha256": file_hash(ids_path), "flagged_original_cell_ids": ids,
        "flagged_count": 58, "primary_sample_count": len(frame), "sensitivity_sample_count": len(frame)-58,
        "rule": "frozen_export_fields_do_not_directly_confirm_negative_gate_not_proven_wrong",
        "primary_labels_or_split_changed": False, "model_outputs_used_to_define_subset": False,
        "source_channel_gate_derivation_verified": False, "plate1_gate_fields_recovered": False}


def final_preflight(config, *, declaration_path=None, device="cpu"):
    """Freeze E role/sample/label/source/C metadata, without reading final RNA."""
    report = preflight(config)
    require(isinstance(device, str) and bool(device), "FINAL_DEVICE")
    frame = pd.read_parquet(ROOT / producer.inputs.JOINT / "tidyi_splits.parquet",
        filters=[("fold", "==", "1_to_8"), ("role", "==", "final_test")])
    require(len(frame) == 368 and frame["cell_id"].is_unique and set(frame["state_class"]) == {0, 1, 2},
            "FROZEN_FINAL_ROLE_METADATA")
    declaration = validate_state_declaration(declaration_path) if declaration_path else None
    if declaration:
        require(declaration["data_identity"] == report["data_identity"], "FINAL_C_DATA_IDENTITY")
        manifest = json.loads((Path(declaration_path).parent / "run_manifest.json").read_text(encoding="utf-8"))
        require(manifest["config"] == state_config(config), "FINAL_FROZEN_C_CONFIG_CHANGED")
    body = {"schema": SCHEMA + ":final-plan", "role": "final_test", "state_fold": "1_to_8",
        "data_identity": report["data_identity"], "gene_axis_sha256": report["gene_axis_sha256"],
        "normalization_sha256": report["normalization_sha256"], "class_order": LABELS,
        "split_path": str(ROOT / producer.inputs.JOINT / "tidyi_splits.parquet"),
        "split_sha256": file_hash(ROOT / producer.inputs.JOINT / "tidyi_splits.parquet"),
        "cell_ids": frame["cell_id"].tolist(), "class_ids": frame["state_class"].astype(int).tolist(),
        "sample_count": len(frame), "class_counts": [int(frame["state_class"].eq(i).sum()) for i in range(3)],
        "state_asset": state_asset(declaration_path, declaration) if declaration else None,
        "model_status": "frozen_matching_C_ready" if declaration else "completed_matching_C_pending",
        "device": device, "dtype": "float32", "settings": report["settings"],
        "raw_label_semantics": report["raw_label_semantics"],
        "model_selection": False, "statistic_refitting": False, "calibration_refitting": False,
        "final_RNA_values_read": False, "PT_loads": 0, "model_forwards": 0, "optimizer_steps": 0,
        "source_channel_gate_derivation_verified": False, "gate_metadata_sensitivity": gate_sensitivity_plan(frame),
        "gate_sensitivity_status": "all_368_original_labels_primary_frozen_58_export_metadata_sensitivity_secondary",
        **FLAGS}
    body["final_plan_identity"] = config_hash(body)
    return body


def evaluate_final(config, declaration_path, final_plan_path, output, *, device="cpu"):
    """Future explicit frozen-C evaluation; only true final_test is opened."""
    require(Path(final_plan_path).resolve().is_relative_to(ROOT / "experiments"), "FINAL_PLAN_PROJECT_BOUNDARY")
    plan = json.loads(Path(final_plan_path).read_text(encoding="utf-8"))
    require(plan == final_preflight(config, declaration_path=declaration_path, device=device)
        and plan["state_asset"] is not None, "FINAL_PLAN_OR_FROZEN_ASSET_CHANGED")
    declaration = validate_state_declaration(declaration_path)
    before = {k: file_hash(v["path"]) for k, v in declaration["assets"].items()}
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / "experiments"), "FINAL_OUTPUT_PROJECT_BOUNDARY")
    with FileLock(str(output) + ".lock", timeout=0):
        require(not output.exists(), "NEW_FINAL_OUTPUT_REQUIRED")
        output.mkdir(parents=True)
        atomic_json(output / "frozen_final_plan.json", plan)
        try:
            data = EngineData(state_config(config), roles=("E",), allow_test=True)
            require(data.identity == plan["data_identity"] and data.reader.allow_test is True
                and data.reader.allowed_roles == frozenset(("final_test",)), "EXPLICIT_FINAL_ONLY_READER_REQUIRED")
            reference = load_representation(declaration["AE_origin"]["weights"], data, device)
            require(reference.identity == declaration["representation_identity"], "FINAL_C_AE_REPRESENTATION_CHANGED")
            classifier = load_classifier(declaration, device)
            rows = data.state_arrays("1_to_8", "final_test")
            require(np.array_equal(rows["labels"], np.asarray(plan["class_ids"], dtype=np.int64))
                and rows["x"].shape == (plan["sample_count"], data.n_genes), "FINAL_LABEL_ORDER_OR_AXIS_CHANGED")
            with torch.no_grad():
                h = auxiliary.encode_vectors(reference, rows["x"], torch.device(device),
                    batch_size=plan["settings"]["inference_microbatch"], output_device=device)
                logits = classifier(h) / classifier.temperature
                metrics = auxiliary.state_metrics(logits, torch.as_tensor(rows["labels"], device=device, dtype=torch.long))
                probabilities = logits.softmax(-1).cpu().numpy()
            wanted = ("nll", "brier", "macro_f1", "accuracy", "balanced_accuracy",
                *(f"recall_{i}" for i in range(3)), *(f"confusion_{i}_{j}" for i in range(3) for j in range(3)))
            metrics = {"final/" + k: metrics["val/" + k] for k in wanted}
            sensitivity = None
            if plan.get("gate_metadata_sensitivity"):
                gate = plan["gate_metadata_sensitivity"]
                excluded = set(gate["flagged_original_cell_ids"])
                keep = torch.tensor([sid not in excluded for sid in plan["cell_ids"]], device=device)
                require(int(keep.sum()) == gate["sensitivity_sample_count"], "FINAL_FROZEN_GATE_SUBSET_CHANGED")
                with torch.no_grad():
                    sub = auxiliary.state_metrics(logits[keep], torch.as_tensor(rows["labels"], device=device, dtype=torch.long)[keep])
                sensitivity = {"sample_count": int(keep.sum()), "frozen_gate_ID_file_sha256": gate["ID_file_sha256"],
                    "metrics": {"sensitivity/" + k: sub["val/" + k] for k in wanted},
                    "primary_labels_or_split_changed": False, "wrong_label_inferred": False, "subset_used_for_model_selection": False}
            require(all(math.isfinite(v) for v in metrics.values()) and probabilities.shape == (plan["sample_count"], 3)
                and np.isfinite(probabilities).all(), "FINAL_THREE_CLASS_METRICS_NONFINITE")
            require(final_preflight(config, declaration_path=declaration_path, device=device) == plan
                and validate_state_declaration(declaration_path) == declaration
                and all(file_hash(declaration["assets"][k]["path"]) == v for k, v in before.items()),
                "FINAL_FROZEN_INPUT_CHANGED_DURING_EVALUATION")
            np.save(output / "original_three_class_probabilities.npy", probabilities, allow_pickle=False)
            body = {"schema": SCHEMA + ":final-result", "final_plan_identity": plan["final_plan_identity"],
                "state_asset": plan["state_asset"], "class_order": LABELS, "role": "final_test",
                "sample_count": plan["sample_count"], "cell_ids": plan["cell_ids"], "class_ids": plan["class_ids"],
                "metrics": metrics, "probabilities_sha256": file_hash(output / "original_three_class_probabilities.npy"),
                "frozen_C_assets_sha256": before, "final_RNA_values_read": True, "model_selection": False,
                "statistic_refitting": False, "calibration_refitting": False, "optimizer_steps": 0,
                "source_channel_gate_derivation_verified": False, "gate_sensitivity_status": plan["gate_sensitivity_status"],
                "prespecified_gate_metadata_sensitivity": sensitivity,
                **FLAGS}
            atomic_json(output / "original_three_class_final_metrics.json", body)
            return body
        except BaseException as error:
            atomic_json(output / "failure.json", {"schema": SCHEMA + ":final-failure", "error_type": type(error).__name__,
                "error": str(error), "partial_preserved": True, "optimizer_steps": 0, **FLAGS})
            raise
