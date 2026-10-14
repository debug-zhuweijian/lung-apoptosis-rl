"""Measured Annexin supervision with explicit observed/predicted feature contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lung.models import ModelConfig, EndpointHead, HistoryEncoder, MonotoneEndpointCalibrator
from .auxiliary import (fit_epochs, require_population_audit, dependency_record,
                        calibration_audit_binding, verify_auxiliary_dependencies)
from .checkpoint import atomic_save, load_checkpoint, save_training, seed_all
from .config import StageUnavailable, config_hash, file_hash, project_path
from .endpoint_contract import (validate_manifest_v2, load_units_v2,
                                dependency_files_v2)


def frozen_feature_identity(pipeline):
    keys = ("response", "pool", "backward", "correction", "correction_gate")
    values = {}
    for key in keys:
        if key in pipeline:
            paths = pipeline[key] if isinstance(pipeline[key], list) else [pipeline[key]]
            values[key] = [file_hash(path) for path in paths]
    return config_hash(values)


def read_manifest(data, config, pipeline):
    value = config.get("endpoint_manifest")
    if not value:
        raise StageUnavailable("ENDPOINT_UNAVAILABLE: no verified measured-Annexin supervision manifest; PROXY substitution forbidden")
    path = project_path(value)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") == 2:
        validate_manifest_v2(path, manifest, data.identity, pipeline)
        return path, manifest
    required = (manifest.get("schema_version") == 1,
                manifest.get("label_type") == "Annexin_positive_fraction",
                bool(manifest.get("denominator")), bool(manifest.get("sampling_gate")),
                bool(manifest.get("protocol_identity")),
                manifest.get("data_identity") == data.identity,
                manifest.get("frozen_feature_identity") == frozen_feature_identity(pipeline))
    if not all(required):
        raise StageUnavailable("ENDPOINT_MANIFEST_UNVERIFIED: label/denominator/protocol/data/frozen-feature identity mismatch")
    return path, manifest


def load_units(path, manifest, role, shape):
    """Role files keep C/E labels unopened during network fitting and G selection."""
    if manifest.get("schema_version") == 2:
        return load_units_v2(path, manifest, role, shape)
    entry = manifest.get("partitions", {}).get(role)
    if not entry:
        raise StageUnavailable(f"ENDPOINT_UNAVAILABLE: matched {role} partition missing")
    source = (path.parent / entry["path"]).resolve()
    if file_hash(source) != entry.get("sha256"):
        raise ValueError("ENDPOINT_PARTITION_HASH_MISMATCH")
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise StageUnavailable(f"ENDPOINT_UNAVAILABLE: no matched {role} units")
    keys = set()
    expected_kind = "observed" if role in {"T", "V"} else "predicted"
    for row in rows:
        if (row.get("role") != role or row.get("feature_kind") != expected_kind or
                row.get("pairing_verified") is not True or not row.get("pairing_evidence") or
                not row.get("expression_sample_id") or not row.get("endpoint_sample_id") or
                not row.get("experiment_id") or not row.get("group_id")):
            raise StageUnavailable("ENDPOINT_PAIRING_UNVERIFIED")
        key = (row["experiment_id"], row["endpoint_sample_id"])
        if key in keys:
            raise ValueError("DUPLICATE_ENDPOINT_UNIT: population labels must not be copied onto cells")
        keys.add(key)
        y, elapsed, horizon = row["label_fraction"], row["elapsed"], row["horizon"]
        if not np.isfinite(y) or not 0 <= y <= 1 or not 0 <= elapsed <= horizon or horizon <= 0:
            raise ValueError("Invalid endpoint fraction or common endpoint time")
        if len(row["population"]) != shape.d_model or len(row["bio"]) != 27 or len(row["context"]) != 3 * shape.context_dim:
            raise ValueError("ENDPOINT_FEATURE_SHAPE_MISMATCH")
        if any(len(segment) != shape.history_input_dim for segment in row["history"]):
            raise ValueError("ENDPOINT_HISTORY_SHAPE_MISMATCH")
    return rows


class MatchedEndpointModel(nn.Module):
    """Own history GRU trains jointly with G_A; frozen expression features do not."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.history = HistoryEncoder(shape)
        self.head = EndpointHead(shape.state_dim, shape.d_model)
        self.calibrator = MonotoneEndpointCalibrator().requires_grad_(False)
        self.calibrated = False
        self.endpoint_schema_version = 1
        self.endpoint_protocol_ids = ()
        self.endpoint_definition = None
        self.feature_generation = None

    def raw(self, row):
        if self.endpoint_schema_version == 2:
            if row.get("protocol_id") not in self.endpoint_protocol_ids:
                raise StageUnavailable("ENDPOINT_PROTOCOL_OUTSIDE_TRAINED_SCOPE")
            if any(row.get(key) != value for key, value in self.endpoint_definition.items()):
                raise StageUnavailable("ENDPOINT_OBSERVATION_DEFINITION_MISMATCH")
        device = next(self.parameters()).device
        def value(key):
            return torch.as_tensor(row[key], device=device, dtype=torch.float32)
        history = value("history").reshape(1, -1, self.shape.history_input_dim)
        hidden = self.history(history)[0]
        elapsed, horizon = float(row["elapsed"]), float(row["horizon"])
        times = hidden.new_tensor([elapsed / horizon, (horizon - elapsed) / horizon])
        features = torch.cat((value("population"), value("bio"), hidden, value("context"), times))
        if not torch.isfinite(features).all():
            raise ValueError("ENDPOINT_FEATURE_NONFINITE")
        return self.head(features)

    def forward(self, row, history=None, lengths=None, *, endpoint_protocol_id=None):
        if isinstance(row, dict):
            probability = self.raw(row)
        else:
            if self.endpoint_schema_version == 2 and endpoint_protocol_id not in self.endpoint_protocol_ids:
                raise StageUnavailable("ENDPOINT_PROTOCOL_OUTSIDE_TRAINED_SCOPE")
            if row.ndim != 2 or row.shape[-1] != self.shape.state_dim or history is None:
                raise ValueError("Endpoint needs features [batch,state_dim] and raw ordered history.")
            hidden = self.history(history, lengths)
            slot = self.shape.d_model + 27
            # Never reuse the policy's mutable history representation for G_A.
            features = torch.cat((row[:, :slot], hidden, row[:, slot + self.shape.d_model:]), -1)
            probability = self.head(features)
        return self.calibrator(probability) if self.calibrated else probability


def train_endpoint(data, config, monitor, pipeline, resume=False):
    path, manifest = read_manifest(data, config, pipeline)
    shape = ModelConfig(**load_checkpoint(pipeline["response"][0])["metadata"]["model_config"])
    if manifest["schema_version"] == 2:
        return _train_predicted_endpoint(data, config, monitor, pipeline, path, manifest, shape, resume)
    units = {role: load_units(path, manifest, role, shape) for role in ("T", "V", "R", "G")}
    groups = {role: {row["group_id"] for row in rows} for role, rows in units.items()}
    for index, role in enumerate(groups):
        for other in list(groups)[index + 1:]:
            if groups[role] & groups[other]:
                raise ValueError("ENDPOINT_GROUP_LEAKAGE")
    identity = config_hash({"manifest": file_hash(path), "features": frozen_feature_identity(pipeline)})
    seed_all(config.get("seed", 20260923))
    model = MatchedEndpointModel(shape).to(config["device"])

    def loss(network, row):
        return (network.raw(row) - float(row["label_fraction"])).square()

    def validation(network):
        return {"val/loss": float(torch.stack([loss(network, row) for row in units["V"]]).mean())}

    metadata = {"width": shape.d_model, "model_config": vars(shape), "train_role": "T",
                "selection_role": "V", "label_type": manifest["label_type"],
                "denominator": manifest["denominator"], "sampling_gate": manifest["sampling_gate"],
                "manifest_path": str(path), "manifest_hash": file_hash(path),
                "frozen_feature_identity": frozen_feature_identity(pipeline), "calibrated": False,
                "dependencies": dependency_record(pipeline, [key for key in
                    ("response", "pool", "backward", "correction", "correction_gate") if pipeline.get(key)],
                    [path] + [path.parent / manifest["partitions"][role]["path"] for role in ("T", "V", "R", "G")]),
                "development_groups": sorted(set().union(*groups.values()))}
    observed_path = fit_epochs("endpoint/observed", model, units["T"], loss, data, config,
                              monitor, identity + "-T", metadata, validation, resume)
    predicted = copy.deepcopy(model)
    predicted_path = fit_epochs("endpoint/predicted", predicted, units["R"], loss, data, config,
                                monitor, identity + "-R", {**metadata, "train_role": "R", "selection_role": None},
                                resume=resume)
    with torch.no_grad():
        scores = [float(torch.stack([loss(network, row) for row in units["G"]]).mean()) for network in (model, predicted)]
    winner = 1 if scores[1] < scores[0] else 0
    saved = load_checkpoint((observed_path, predicted_path)[winner])
    selected = monitor.weights / "endpoint" / "selected.pt"
    saved["metadata"] = {**saved["metadata"], "stage": "endpoint", "selected_by": "G",
                         "selected_version": ("T", "R")[winner], "calibration_status": "AWAITING_NUMERICAL_AUDIT"}
    saved["extra"] = {**saved["extra"], "G_scores": {"T_version": scores[0], "R_version": scores[1]}}
    atomic_save(selected, saved)
    return {**pipeline, "endpoint": str(selected)}


def _train_predicted_endpoint(data, config, monitor, pipeline, path, manifest, shape, resume):
    """Fit measured Y directly on frozen predicted features; select using V only."""
    units = {role: load_units(path, manifest, role, shape) for role in ("T", "V")}
    feature_identity = manifest["frozen_feature_identity"]
    identity = config_hash({"manifest": file_hash(path), "features": feature_identity})
    seed_all(config.get("seed", 20260923))
    model = MatchedEndpointModel(shape).to(config["device"])
    model.endpoint_schema_version = 2
    model.endpoint_protocol_ids = tuple(sorted(manifest["protocols"]))
    protocol = next(iter(manifest["protocols"].values()))
    model.endpoint_definition = {key: protocol[key] for key in ("label_type", "denominator", "sampling_gate")}
    model.feature_generation = manifest["feature_generation"]

    def loss(network, row):
        return (network.raw(row) - float(row["label_fraction"])).square()

    def validation(network):
        return {"val/loss": float(torch.stack([loss(network, row) for row in units["V"]]).mean())}

    metadata = {"width": shape.d_model, "model_config": vars(shape), "train_role": "T",
                "selection_role": "V", "endpoint_schema_version": 2,
                "endpoint_protocol_ids": list(model.endpoint_protocol_ids),
                "endpoint_definition": model.endpoint_definition,
                "feature_generation": manifest["feature_generation"],
                "supervision_kind": manifest["supervision_kind"],
                "feature_kind": "predicted", "label_kind": "measured",
                "label_type": manifest["label_type"], "protocols": manifest["protocols"],
                "split_policy": manifest["split_policy"], "manifest_path": str(path),
                "manifest_hash": file_hash(path), "frozen_feature_identity": feature_identity,
                "calibrated": False,
                "dependencies": dependency_record(pipeline,
                    manifest["feature_dependencies"]["pipeline"], dependency_files_v2(path, manifest)),
                "development_groups": sorted({row["group_id"] for rows in units.values() for row in rows})}
    trained = fit_epochs("endpoint/predicted_measured", model, units["T"], loss, data, config,
                         monitor, identity, metadata, validation, resume)
    saved = load_checkpoint(trained)
    selected = monitor.weights / "endpoint" / "selected.pt"
    saved["metadata"] = {**saved["metadata"], "stage": "endpoint", "selected_by": "V",
                         "selected_version": "schema2_T_predicted_measured",
                         "calibration_status": "AWAITING_NUMERICAL_AUDIT"}
    atomic_save(selected, saved)
    return {**pipeline, "endpoint": str(selected)}


def load_endpoint(path, data, device):
    saved = load_checkpoint(path)
    if saved["metadata"]["data_identity"] != data.identity or saved["epoch"] < 1:
        raise ValueError("ENDPOINT_DATA_OR_TRAINING_IDENTITY_MISMATCH")
    model = MatchedEndpointModel(ModelConfig(**saved["metadata"]["model_config"]))
    model.load_state_dict(saved["model"])
    model.calibrated = saved["metadata"].get("calibrated", False)
    model.endpoint_schema_version = saved["metadata"].get("endpoint_schema_version", 1)
    if model.endpoint_schema_version == 2:
        model.endpoint_protocol_ids = tuple(saved["metadata"]["endpoint_protocol_ids"])
        model.endpoint_definition = saved["metadata"]["endpoint_definition"]
        model.feature_generation = saved["metadata"]["feature_generation"]
        model.feature_pipeline_stages = tuple(sorted(saved["metadata"]["dependencies"]["pipeline"]))
    return model.to(device).eval().requires_grad_(False)


def finalize_calibration(data, config, monitor, pipeline):
    if not pipeline.get("endpoint"):
        raise StageUnavailable("ENDPOINT_UNAVAILABLE")
    verify_auxiliary_dependencies(pipeline["endpoint"], pipeline)
    path, manifest = read_manifest(data, config, pipeline)
    model = load_endpoint(pipeline["endpoint"], data, config["device"])
    saved = load_checkpoint(pipeline["endpoint"])
    binding = calibration_audit_binding(data, config, pipeline)
    binding["weights"].pop("endpoint", None)
    if model.calibrated:
        original = saved["metadata"].get("calibration_base", {})
        if (saved["metadata"].get("calibration_binding") != binding or
                not original.get("path") or file_hash(original["path"]) != original.get("sha256")):
            raise StageUnavailable("CALIBRATION_VERSION_CHANGED: start a new experiment")
        return pipeline
    require_population_audit(data, config, pipeline)
    units = load_units(path, manifest, "C", model.shape)
    if set(saved["metadata"]["development_groups"]) & {row["group_id"] for row in units}:
        raise StageUnavailable("ENDPOINT_CALIBRATION_GROUP_LEAKAGE")
    with torch.no_grad():
        probabilities = torch.stack([model.raw(row) for row in units])
    target = probabilities.new_tensor([row["label_fraction"] for row in units])
    model.calibrator.requires_grad_(True)
    optimizer = torch.optim.Adam(model.calibrator.parameters(), lr=config.get("calibration_lr", 1e-2))
    for _ in monitor.progress(range(config.get("calibration_steps", 2000)), desc="Annexin C calibration"):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.calibrator(probabilities) - target).square().mean()
        loss.backward()
        optimizer.step()
    output = monitor.weights / "endpoint" / "calibrated.pt"
    dependencies = copy.deepcopy(saved["metadata"]["dependencies"])
    c_path = path.parent / manifest["partitions"]["C"]["path"]
    dependencies["files"].append({"path": str(c_path.resolve()), "sha256": file_hash(c_path)})
    save_training(output, model, None, epoch=saved["epoch"], identity=saved["identity"],
                  metadata={**saved["metadata"], "calibrated": True, "calibration_role": "C",
                            "calibration_status": "COMPLETE", "calibration_binding": binding,
                            "calibration_base": {"path": str(Path(pipeline["endpoint"]).resolve()),
                                                 "sha256": file_hash(pipeline["endpoint"])},
                            "dependencies": dependencies}, counters=saved["counters"],
                  extra={**saved["extra"], "calibration_groups": sorted({row["group_id"] for row in units}),
                         "calibration_lr": config.get("calibration_lr", 1e-2)})
    return {**pipeline, "endpoint": str(output)}
