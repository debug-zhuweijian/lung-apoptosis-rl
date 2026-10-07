"""Tiny synthetic auxiliary-stage integration, never biological performance evidence."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models import ModelConfig, ResponseModel
from lung.engine.auxiliary import (train_auxiliary, load_auxiliary, verify_auxiliary_dependencies,
                                   pipeline_identity, calibration_audit_binding, require_population_audit)
from lung.engine.checkpoint import save_training, load_checkpoint
from lung.engine.config import StageUnavailable, file_hash
from lung.engine.response import condition_kwargs, tensor
from lung.engine.synthetic import SyntheticData
from lung.engine.residuals import finalize_calibration as finalize_uncertainty
from lung.engine.endpoints import (frozen_feature_identity, MatchedEndpointModel,
                                   finalize_calibration as finalize_endpoint, load_endpoint)


class Monitor:
    def __init__(self, directory):
        self.weights, self.experiment = directory / "weights", directory / "experiment"
        self.weights.mkdir(parents=True)
        self.experiment.mkdir(parents=True)
        self.metrics = []

    def progress(self, iterable=None, **kwargs):
        return iterable

    def say(self, text):
        pass

    def record(self, stage, epoch, metrics):
        self.metrics.append((stage, epoch, metrics))


class TrackedData(SyntheticData):
    def __init__(self, config, enough_groups=False):
        super().__init__(config)
        self.read_roles = []
        self.enough_groups = enough_groups

    def condition_ids(self, role):
        count = 5 if role == "R" and self.enough_groups else 10 if role == "C" else 2
        return [f"{role}-{index}" for index in range(count)]

    def sample_response(self, condition_id, n, rng, role):
        self.read_roles.append(role)
        return super().sample_response(condition_id, n, rng, role)

    def state_arrays(self, fold, role):
        self.read_roles.append("state-" + role)
        return super().state_arrays(fold, role)


def audited_pipeline(data, config, monitor, pipeline):
    path = monitor.experiment / "synthetic-population-audit.json"
    identity = "SYNTHETIC-ENGINEERING-AUDIT"
    path.write_text(json.dumps({"passed": True, "biological_identity": identity,
                               "synthetic_test_only": True,
                               "calibration_binding": calibration_audit_binding(data, config, pipeline)}), encoding="utf-8")
    return {**pipeline, "numerical_audit": str(path), "precalibration_identity": identity}


@pytest.fixture
def setup(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    shape = ModelConfig(n_genes=8, d_model=8, n_heads=2, encoder_layers=1,
                        pad_layers=1, query_chunk=4, context_dim=2)
    config = {"device": "cpu", "epochs": 1, "seed": 20260923, "population": 2,
              "aux_batch": 2, "state_batch": 6, "aux_lr": 1e-3, "save_period": 1,
              "calibration_steps": 2, "state_fold": "1_to_8", "synthetic": True,
              "response": {"model": vars(shape), "inference_ode_steps": 1}}
    data = TrackedData(config)
    monitor = Monitor(tmp_path)
    model = ResponseModel(shape, torch.as_tensor(data.neighbors))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = data.sample_response("T-0", 2, np.random.default_rng(11), "T")
    condition = model.encode_condition(**condition_kwargs(sample, 2, "cpu"))
    loss = model.cfm_loss(tensor(sample["control"], "cpu"), tensor(sample["target"], "cpu"), condition)
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "response.pt"
    save_training(checkpoint, model, optimizer, epoch=1, identity="synthetic-test",
                  counters={"optimizer_step": 1},
                  metadata={"stage": "response", "data_identity": data.identity,
                            "model_config": vars(shape), "synthetic": True})
    data.read_roles.clear()
    yield data, config, monitor, {"response": [str(checkpoint)]}
    torch.set_num_threads(threads)


def test_pool_state_stages_resume_and_no_final_test_read(setup):
    data, config, monitor, pipeline = setup
    pipeline = train_auxiliary("pool", data, config, monitor, pipeline)
    pipeline = train_auxiliary("state", data, config, monitor, pipeline)
    assert "E" not in data.read_roles and "state-test" not in data.read_roles
    assert set(data.read_roles) == {"T", "V", "state-fit", "state-selection", "state-calibration"}
    assert Path(pipeline["pool"]).name == "best.pt"
    assert Path(pipeline["state"]).name == "calibrated.pt"
    assert (monitor.weights / "state" / "best.pt").exists()
    state = load_auxiliary(pipeline["state"], data, "cpu")
    assert state.temperature > 0 and all(not p.requires_grad for p in state.parameters())
    assert torch.isfinite(state.probabilities(torch.randn(3, 8))).all()
    assert load_checkpoint(pipeline["state"])["extra"]["calibration_lr"] == .01
    assert verify_auxiliary_dependencies(pipeline["state"], pipeline)
    changed = monitor.experiment / "changed-response.pt"
    saved = load_checkpoint(pipeline["response"][0])
    saved["model"]["cell_encoder.gene_embedding.weight"] += .01
    torch.save(saved, changed)
    with pytest.raises(StageUnavailable, match="DEPENDENCY_MISMATCH"):
        verify_auxiliary_dependencies(pipeline["state"], {**pipeline, "response": [str(changed)]})
    assert pipeline_identity("pool", data, config, pipeline) == pipeline_identity("pool", data, config, {"response": pipeline["response"]})
    pipeline = train_auxiliary("pool", data, {**config, "epochs": 2}, monitor, pipeline, resume=True)
    last = load_checkpoint(monitor.weights / "pool" / "last.pt")
    assert last["epoch"] == 2 and last["counters"]["optimizer_step"] == 2


def test_real_residual_fits_gate_and_insufficient_oof_refusal(setup):
    data, config, monitor, pipeline = setup
    for stage in ("pool", "backward", "correction"):
        pipeline = train_auxiliary(stage, data, config, monitor, pipeline)
    gate = json.loads(Path(pipeline["correction_gate"]).read_text())
    assert not any(domain["enabled"] for domain in gate["domains"].values())
    assert all(domain["reason"] == "FEWER_THAN_20_GROUPS" for domain in gate["domains"].values())
    assert "C" not in data.read_roles and "E" not in data.read_roles
    with pytest.raises(StageUnavailable, match="five R"):
        train_auxiliary("uncertainty", data, config, monitor, pipeline)


def test_uncertainty_strict_oof_and_delayed_calibration(setup):
    data, config, monitor, pipeline = setup
    data.enough_groups = True
    for stage in ("pool", "backward", "correction", "uncertainty"):
        pipeline = train_auxiliary(stage, data, config, monitor, pipeline)
    saved = load_checkpoint(pipeline["uncertainty"])
    assert set(saved["metadata"]["folds"].values()) == set(range(5))
    cached = next((monitor.experiment / "cache").glob("uncertainty-oof-*.pt"))
    records = torch.load(cached, weights_only=False)["rows"]
    assert len(records) == 10
    for row in records:
        assert row["group_id"] not in row["fit_groups"]
    assert "C" not in data.read_roles and "E" not in data.read_roles
    with pytest.raises(StageUnavailable, match="NUMERICAL_AUDIT"):
        finalize_uncertainty(data, config, monitor, pipeline)
    pipeline = finalize_uncertainty(data, config, monitor, audited_pipeline(data, config, monitor, pipeline))
    calibration = json.loads(Path(pipeline["uncertainty_calibration"]).read_text())
    assert calibration["role"] == "C" and len(calibration["groups"]) == 10
    assert calibration["coverage_guarantee"] is False and "E" not in data.read_roles
    count = len(data.read_roles)
    again = finalize_uncertainty(data, config, monitor, pipeline)
    assert again["uncertainty_calibration"] == pipeline["uncertainty_calibration"]
    assert len(data.read_roles) == count


def write_endpoint_manifest(directory, data, pipeline, shape):
    partitions = {}
    for role in ("T", "V", "R", "G", "C"):
        rows = []
        for i in range(3):
            rows.append({"role": role, "feature_kind": "observed" if role in {"T", "V"} else "predicted",
                         "pairing_verified": True, "pairing_evidence": "synthetic test fixture only",
                         "expression_sample_id": f"{role}-expr-{i}", "endpoint_sample_id": f"{role}-end-{i}",
                         "experiment_id": f"{role}-experiment-{i}", "group_id": f"{role}-group-{i}",
                         "population": [i / 10] * shape.d_model, "bio": [0.] * 27,
                         "context": [0.] * (3 * shape.context_dim),
                         "history": [[.1] * shape.history_input_dim],
                         "elapsed": 24., "horizon": 24., "label_fraction": .2 + .2 * i})
        path = directory / f"endpoint-{role}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        partitions[role] = {"path": path.name, "sha256": file_hash(path)}
    # An unreadable final-test path proves network training does not load it.
    partitions["E"] = {"path": "E-MUST-NOT-BE-READ.json", "sha256": "not-used"}
    manifest = {"schema_version": 1, "label_type": "Annexin_positive_fraction", "denominator": "synthetic",
                "sampling_gate": "synthetic", "protocol_identity": "SYNTHETIC", "data_identity": data.identity,
                "frozen_feature_identity": frozen_feature_identity(pipeline), "partitions": partitions}
    path = directory / "endpoint-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_endpoint_unavailable_then_verified_manifest_and_private_history(setup):
    data, config, monitor, pipeline = setup
    with pytest.raises(StageUnavailable, match="ENDPOINT_UNAVAILABLE"):
        train_auxiliary("endpoint", data, config, monitor, pipeline)
    shape = ModelConfig(**config["response"]["model"])
    path = write_endpoint_manifest(monitor.experiment, data, pipeline, shape)
    cfg = {**config, "endpoint_manifest": str(path)}
    pipeline = train_auxiliary("endpoint", data, cfg, monitor, pipeline)
    selected = load_checkpoint(pipeline["endpoint"])
    assert selected["metadata"]["selected_by"] == "G" and not selected["metadata"]["calibrated"]
    with pytest.raises(StageUnavailable, match="NUMERICAL_AUDIT"):
        finalize_endpoint(data, cfg, monitor, pipeline)
    pipeline = finalize_endpoint(data, cfg, monitor, audited_pipeline(data, cfg, monitor, pipeline))
    assert load_endpoint(pipeline["endpoint"], data, "cpu").calibrated
    assert finalize_endpoint(data, cfg, monitor, pipeline) == pipeline
    deployed = load_auxiliary(pipeline["endpoint"], data, "cpu")
    assert deployed(torch.zeros(2, shape.state_dim), torch.zeros(2, 1, shape.history_input_dim),
                    torch.tensor([1, 1])).shape == (2,)
    standalone = MatchedEndpointModel(shape)
    row = json.loads((monitor.experiment / "endpoint-T.json").read_text())[0]
    standalone.raw(row).backward()
    assert standalone.history.gru.weight_ih_l0.grad.abs().sum() > 0


def test_population_audit_rejects_old_weights_even_with_reused_identity(setup):
    data, config, monitor, pipeline = setup
    pipeline = audited_pipeline(data, config, monitor, pipeline)
    require_population_audit(data, config, pipeline)
    changed = monitor.experiment / "response-different-weights.pt"
    state = load_checkpoint(pipeline["response"][0])
    state["model"]["cell_encoder.gene_embedding.weight"] += 1.
    torch.save(state, changed)
    with pytest.raises(StageUnavailable, match="AUDIT_DEPENDENCY_MISMATCH"):
        require_population_audit(data, config, {**pipeline, "response": [str(changed)]})
    with pytest.raises(StageUnavailable, match="AUDIT_IDENTITY_MISMATCH"):
        require_population_audit(data, config, {**pipeline, "precalibration_identity": "another-world"})
    with pytest.raises(StageUnavailable, match="AUDIT_DEPENDENCY_MISMATCH"):
        require_population_audit(data, {**config, "population": 1024}, pipeline)
    with pytest.raises(StageUnavailable, match="NUMERICAL_AUDIT_REQUIRED"):
        require_population_audit(data, {**config, "synthetic": False}, pipeline)
    with pytest.raises(StageUnavailable, match="NUMERICAL_AUDIT_REQUIRED"):
        require_population_audit(data, {**config, "synthetic": False}, {"population_audit_passed": True})
