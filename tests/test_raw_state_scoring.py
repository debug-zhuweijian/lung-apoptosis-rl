"""Original C toy fitting/export and sidecar integrity, never real ML/PT."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from lung.engine import raw_state_scoring as module
from lung.engine.checkpoint import load_checkpoint, save_training, seed_all
from lung.engine.config import load_config
from lung.models import ModelConfig, StateClassifier
from lung.models.latent_staged import SharedAutoencoder
from conftest import assert_same

pytestmark = pytest.mark.usefixtures("cpu_only")


class ToyData:
    identity, n_genes, synthetic = "synthetic-C-only", 4, True
    neighbors = np.arange(4)[:, None]

    def __init__(self):
        self.roles = []

    def state_arrays(self, fold, role):
        assert fold == "1_to_8" and role in ("fit", "selection", "calibration")
        self.roles.append(role)
        offset = {"fit": 0., "selection": .1, "calibration": .2}[role]
        return {"x": np.arange(24, dtype=np.float32).reshape(6, 4)/10 + offset,
                "labels": np.asarray([0, 1, 2, 0, 1, 2])}


def config():
    return {"task": "PROXY-A549-V1", "synthetic": True, "seed": 8321, "device": "cpu", "epochs": 2,
        "state_fold": "1_to_8", "state_batch": 3, "aux_batch": 2, "aux_lr": .01,
        "calibration_steps": 3, "calibration_lr": .01, "early_stopping": False,
        "state_loss": "weighted_ce",
        "state_aggregate_lambda": 0., "response": {"inference_microbatch": 2, "inference_ode_steps": 1}}


def fitting_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "experiments").mkdir()

    seed_all(8123)
    data = ToyData()
    shape = ModelConfig(n_genes=4, d_model=8, n_heads=2, context_dim=2, latent_dim=3,
        backbone="compact_full_gene", checkpoint_blocks=False)
    shared = SharedAutoencoder(shape, torch.arange(4)[:, None])
    ae = tmp_path / "experiments/toy-AE.pt"
    save_training(ae, shared, epoch=1, identity="syntheticAE",
        metadata={"stage": "latent_ae", "data_identity": data.identity, "model_config": asdict(shape),
                  "synthetic_fixture_only": True}, counters={"optimizer_step": 1})
    origin = {"run": str(ae.parent), "weights": str(ae), "checkpoint_sha256": module.file_hash(ae),
        "synthetic_fixture_only": True}
    monkeypatch.setattr(module, "verify_ae_origin", lambda *a: deepcopy(origin))
    return data, ae.parent


def test_state_config_keeps_original_objective_and_user_config():
    actual = load_config("configs/modular_research.yaml")
    before = deepcopy(actual)
    value = module.state_config(actual)
    assert actual == before
    assert value["data"] == module.learning.JOINT_CONFIG
    assert value["reward"] == actual["reward"]
    assert module.settings(value)["state_loss"] == "weighted_ce"
    assert value["state_aggregate_lambda"] == 0.


def test_toy_original_C_fit_only_uses_three_named_roles_and_freezes_AE(tmp_path, monkeypatch):
    data, ae_run = fitting_fixture(tmp_path, monkeypatch)
    ae_sha = module.file_hash(ae_run / "toy-AE.pt")
    body = module.fit_state(data, config(), ae_run, tmp_path / "experiments/C", synthetic=True)
    assert data.roles == ["fit", "selection", "calibration"]
    assert module.file_hash(ae_run / "toy-AE.pt") == ae_sha
    assert body["class_order"] == ["PI-APC-", "PI-APC+", "PI+APC+"]
    assert body["training_completion"]["actual_updates"] == 4
    assert body["calibration_completion"]["actual_steps"] == 3
    assert all(body[k] is False for k in module.FLAGS)
    saved = torch.load(body["assets"]["serving"]["path"], weights_only=True)
    assert saved["metadata"]["dependencies"]["pipeline"] == {"representation": [ae_sha]}
    assert saved["metadata"]["input_coordinate"] == "h"
    assert "response" not in saved["metadata"]["dependencies"]["pipeline"]
    with pytest.raises(ValueError, match="SYNTHETIC_STATE_HAS_NO_PRODUCTION"):
        module.validate_state_declaration(tmp_path / "experiments/C/state_declaration.json")


@pytest.mark.parametrize("change", [None, "source_origin", "tensor_identity", "data_identity"])
def test_actual_toy_safe_C_loader_checks_AE_origin_and_same_tensors(tmp_path, monkeypatch, change):
    data, ae_run = fitting_fixture(tmp_path, monkeypatch)
    output = tmp_path / "experiments/C"
    body = module.fit_state(data, config(), ae_run, output, synthetic=True)
    path = output / "state_declaration.json"
    original_validator = module.validate_state_declaration
    monkeypatch.setattr(module, "validate_state_declaration", lambda p: original_validator(p, allow_synthetic=True))
    saved = load_checkpoint(ae_run / "toy-AE.pt")
    shared = SharedAutoencoder(ModelConfig(**saved["metadata"]["model_config"]), torch.arange(4)[:, None])
    shared.load_state_dict(saved["model"])
    if change == "tensor_identity":
        with torch.no_grad():
            shared.decoder[0].weight[0, 0] += .1
    model = torch.nn.Module()
    model.add_module("shared", shared)
    export = tmp_path / "experiments/export.pt"
    source_hash = "wrong-origin" if change == "source_origin" else body["AE_origin"]["checkpoint_sha256"]
    torch.save({"metadata": {"source_checkpoint_sha256": source_hash}, "model": saved["model"],
                "synthetic_fixture_only": True}, export)
    prediction = {"bindings": {"data_identity": "wrong-data" if change == "data_identity" else data.identity},
        "model_metadata": {"representation": {"checkpoint": "experiments/export.pt"}}}
    monkeypatch.setattr(module.producer, "ROOT", tmp_path)
    if change:
        monkeypatch.setattr(module, "StateClassifier", lambda *a: pytest.fail("Reject before C construction"))
        with pytest.raises(ValueError, match="MISMATCH"):
            module.load_state_for_prediction(path, prediction, model)
    else:
        classifier, declared = module.load_state_for_prediction(path, prediction, model)
        assert declared == body and not classifier.training
        assert all(not p.requires_grad for p in classifier.parameters())
        assert_same(classifier.state_dict(), torch.load(body["assets"]["serving"]["path"], weights_only=True)["model"])


class ToyShared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_genes=4, d_model=4, latent_dim=3)
        self.seen = []

    def encode_h(self, x):
        self.seen.append(x.clone())
        return x[:, None], x


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = ToyShared()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))
        self.full = SimpleNamespace(member=SimpleNamespace(config=SimpleNamespace(latent_dim=3)))
        self.reference_contract = module.producer.reference_contract("observed")

    def forward(self, x, physical, numeric, *, noise, steps):
        # Synthetic demonstration only: same random population and outlet.
        raw = x + noise[:, :1]
        counts = raw.softmax(-1) * 10000
        return SimpleNamespace(y=counts.log1p())

    predict_h = forward


def sidecar_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "experiments"
    directory.mkdir()
    model, x = ToyModel(), np.arange(20, dtype=np.float32).reshape(5, 4)/13
    seed_all(333)
    classifier = StateClassifier(4).eval()
    classifier.set_temperature(1.3)
    monkeypatch.setattr(classifier, "population_proxy", lambda *a: pytest.fail("Never merge labels"))
    context = {"mean": np.full(4, .7, dtype=np.float32), "scale": np.full(4, 2., dtype=np.float32)}
    noise = torch.randn((5, 3), generator=torch.Generator().manual_seed(64))
    y = model(torch.from_numpy(x), None, None, noise=noise, steps=2).y.detach().numpy().astype(np.float64)
    reference = {"shape": [5, 4], "reference_role": "fit", "synthetic_fixture_only": True}
    prediction = {"candidate_id": "toy-full-P", "prediction_identity": "toy-P-ID", "seed": 64, "steps": 2,
        "expression_mean": y.mean(0).tolist(), "expression_variance": y.var(0).tolist(), "reference": reference,
        "bindings": {"data_identity": "toy"}, "model_metadata": {"representation": {"checkpoint_sha256": "toy-exportSHA"}}}
    pp, dp, sp = [directory / n for n in ("P.json", "C-declaration.json", "synthetic-C-weights.pt")]
    module.atomic_json(pp, prediction)
    sp.write_text("synthetic fixture, not a real C checkpoint", encoding="utf-8")
    declaration = {"data_identity": "toy", "state_identity": "toy-C-ID", "AE_origin": {"checkpoint_sha256": "toy-originalSHA"},
        "representation_identity": {"weights": "toy-tensor-ID", "n_genes": 4, "d_model": 4, "latent_dim": 3},
        "assets": {"serving": {"path": str(sp), "sha256": module.file_hash(sp)}}}
    module.atomic_json(dp, declaration)
    monkeypatch.setattr(module.producer, "read_prediction", lambda p: deepcopy(json.loads(Path(p).read_text())))
    monkeypatch.setattr(module.producer, "bound_inputs", lambda: (None, None, context, {"toy": True}))
    monkeypatch.setattr(module.producer, "numeric_batch", lambda *a: ({}, None))
    monkeypatch.setattr(module.producer, "read_proxy_reference", lambda c: (x.copy(), deepcopy(reference)))
    monkeypatch.setattr(module.producer, "load_model", lambda *a: model)
    monkeypatch.setattr(module, "validate_state_declaration", lambda p: deepcopy(declaration))
    monkeypatch.setattr(module, "load_state_for_prediction", lambda *a: (classifier, deepcopy(declaration)))
    return model, classifier, context, prediction, x, pp, dp


def test_exact_P_replay_restores_C_input_coordinates_and_keeps_three_classes(tmp_path, monkeypatch):
    model, classifier, context, prediction, x, pp, dp = sidecar_fixture(tmp_path, monkeypatch)
    body = module.score_prediction(pp, dp, tmp_path / "experiments/score", microbatch=2)
    assert body["class_order"] == module.LABELS and len(body["probability_mean"]) == 3
    assert body["sample_count"] == 5 and body["P_replay_moments_exact"] is True
    assert all(body[k] is False for k in module.FLAGS)
    expected_y = model(torch.from_numpy(x), None, None,
        noise=torch.randn((5, 3), generator=torch.Generator().manual_seed(64)), steps=2).y
    actual_normalized = torch.cat(model.shared.seen[-3:])
    assert torch.equal(actual_normalized, (expected_y-.7)/2.)
    expected_probabilities = classifier.probabilities(actual_normalized).detach().numpy().astype(np.float64)
    assert np.array_equal(body["probability_mean"], expected_probabilities.mean(0))
    assert np.array_equal(body["probability_variance"], expected_probabilities.var(0))


def test_different_P_moments_preserves_failed_namespace_and_refuses_reuse(tmp_path, monkeypatch):
    _, _, _, prediction, _, pp, dp = sidecar_fixture(tmp_path, monkeypatch)
    prediction["expression_mean"][0] += .01
    module.atomic_json(pp, prediction)
    output = tmp_path / "experiments/fail"
    with pytest.raises(ValueError, match="P_REPLAY_MOMENTS_DIFFER"):
        module.score_prediction(pp, dp, output)
    assert (output / "failure.json").is_file() and not (output / "state_sidecar.json").exists()
    with pytest.raises(ValueError, match="NEW_SIDECAR_OUTPUT_REQUIRED"):
        module.score_prediction(pp, dp, output)


@pytest.mark.parametrize("change", ["classes", "boolean_mean", "wrong_n", "P_clock", "C_origin", "science", "prob_sum"])
def test_reader_rejects_scope_sampling_AE_and_probability_changes(tmp_path, monkeypatch, change):
    _, _, _, _, _, pp, dp = sidecar_fixture(tmp_path, monkeypatch)
    output = tmp_path / "experiments/score"
    body = module.score_prediction(pp, dp, output)
    if change == "classes":
        body["class_order"] = list(reversed(module.LABELS))
    elif change == "boolean_mean":
        body["probability_mean"] = [True, False, False]
    elif change == "wrong_n":
        body["sample_count"] = 4
    elif change == "P_clock":
        body["sampling"]["steps"] += 1
    elif change == "C_origin":
        body["P_AE_bridge"]["source_checkpoint_sha256"] = "wrong"
    elif change == "science":
        body["source_C_transfer_validated"] = True
    else:
        body["probability_mean"] = [.8, .8, .8]
    body["sidecar_identity"] = module.config_hash({k: v for k, v in body.items() if k != "sidecar_identity"})
    module.atomic_json(output / "state_sidecar.json", body)
    with pytest.raises(ValueError):
        module.read_state_sidecar(output / "state_sidecar.json")


def test_sidecar_reader_is_zero_PT_model_and_bytes_bound(tmp_path, monkeypatch):
    _, _, _, _, _, pp, dp = sidecar_fixture(tmp_path, monkeypatch)
    path = tmp_path / "experiments/score/state_sidecar.json"
    body = module.score_prediction(pp, dp, path.parent)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No PT in reader"))
    monkeypatch.setattr(module, "StateClassifier", lambda *a: pytest.fail("No model in reader"))
    assert module.read_state_sidecar(path) == body
    pp.write_text("changed actual P bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="SIDECAR_P_BYTES_CHANGED"):
        module.read_state_sidecar(path)


def test_final_metadata_preflight_does_not_open_final_RNA_or_models(monkeypatch):
    forbid = lambda *a, **k: pytest.fail("No final RNA/PT/model during metadata preflight")
    monkeypatch.setattr(torch, "load", forbid)
    monkeypatch.setattr(module, "EngineData", forbid)
    monkeypatch.setattr(module, "load_representation", forbid)
    monkeypatch.setattr(module.auxiliary, "train_state", forbid)
    monkeypatch.setattr(module.producer, "bound_inputs", forbid)
    plan = module.final_preflight(load_config("configs/modular_research.yaml"))
    assert plan["role"] == "final_test" and plan["sample_count"] == 368
    assert plan["class_counts"] == [167, 93, 108]
    assert plan["gate_metadata_sensitivity"]["flagged_count"] == 58
    assert plan["gate_metadata_sensitivity"]["sensitivity_sample_count"] == 310
    assert plan["gate_metadata_sensitivity"]["primary_labels_or_split_changed"] is False
    assert plan["state_asset"] is None
    assert all(plan[k] is False for k in module.FLAGS)
    assert plan["PT_loads"] == plan["model_forwards"] == plan["optimizer_steps"] == 0
    assert plan["final_RNA_values_read"] is False


@pytest.mark.parametrize("change,reason", [(None, None), ("bytes", "STATE_ASSET_CHANGED"),
    ("overlap", "ROLE_OVERLAP"), ("label", "ORIGINAL_LABEL_SEMANTICS_CHANGED")])
def test_preflight_uses_current_split_identity_and_labels(tmp_path, monkeypatch, change, reason):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assets = tmp_path / module.producer.inputs.JOINT
    assets.mkdir(parents=True)
    rows = [{"fold": "1_to_8", "role": role, "cell_id": f"{role}-{i}", "state_class": i,
        "state_label_original": label, "label_scope": "original_sorted_state_not_natural_apoptosis_rate"}
        for role in ("fit", "selection", "calibration", "final_test") for i, label in enumerate(module.LABELS)]
    if change == "overlap":
        rows[-1]["cell_id"] = rows[0]["cell_id"]
    elif change == "label":
        rows[0]["state_class"] = 1
    path = assets / "tidyi_splits.parquet"
    pd.DataFrame(rows).to_parquet(path)
    source = tmp_path / "source/manifest.json"
    source.parent.mkdir()
    source.write_text("{}", encoding="utf-8")
    for name in ("genes.parquet", "normalization.npz"):
        (assets / name).write_bytes(b"fixture-only-asset-bytes")
    data_config = {"assets": module.producer.inputs.JOINT, "bundle": "source"}
    monkeypatch.setattr(module, "dataset_config", lambda *_: data_config)
    (assets / "manifest.json").write_text(json.dumps({
        "signature": "fixture", "status": "complete", "config": data_config,
        "source_manifest_sha256": module.file_hash(source),
        "metadata": {"tidyi_folds": {"1_to_8": {"status": "ready"}}},
        "files": {name: {"sha256": "changed" if change == "bytes" and name == path.name
            else module.file_hash(assets / name)} for name in ("genes.parquet", "normalization.npz", path.name)}}),
        encoding="utf-8")
    monkeypatch.setattr(module.producer.inputs, "PINS", {
        (assets / "manifest.json").relative_to(tmp_path).as_posix(): module.file_hash(assets / "manifest.json")})

    if reason:
        with pytest.raises(ValueError, match=reason):
            module.preflight(config())
    else:
        report = module.preflight(config())
        assert report["roles"]["fit"]["cells"] == 3
        assert report["RNA_matrix_values_reread"] == report["PT_loads"] == 0
        assert "reused_real_readback" not in report


def toy_final_fixture(tmp_path, monkeypatch):
    data, ae_run = fitting_fixture(tmp_path, monkeypatch)
    c_run = tmp_path / "experiments/C"
    declaration = module.fit_state(data, config(), ae_run, c_run, synthetic=True)
    dp = c_run / "state_declaration.json"
    validator = module.validate_state_declaration
    monkeypatch.setattr(module, "validate_state_declaration", lambda p: validator(p, allow_synthetic=True))
    values = data.state_arrays("1_to_8", "selection")
    ids = ["synthetic-final-" + str(i) for i in range(6)]
    plan = {"role": "final_test", "state_fold": "1_to_8", "data_identity": data.identity,
        "state_asset": module.state_asset(dp, declaration), "sample_count": 6, "cell_ids": ids,
        "class_ids": values["labels"].tolist(), "class_order": module.LABELS,
        "settings": {"inference_microbatch": 2}, "gate_sensitivity_status": "synthetic_not_gate_evidence",
        "synthetic_fixture_only": True}
    plan["final_plan_identity"] = module.config_hash(plan)
    pp = tmp_path / "experiments/toy-final-plan.json"
    module.atomic_json(pp, plan)
    monkeypatch.setattr(module, "final_preflight", lambda *a, **k: deepcopy(plan))
    final_roles = []
    class FinalData:
        identity, n_genes = data.identity, data.n_genes
        neighbors = data.neighbors
        reader = SimpleNamespace(allow_test=True, allowed_roles=frozenset(("final_test",)))
        def state_arrays(self, fold, role):
            assert fold == "1_to_8" and role == "final_test"
            final_roles.append(role)
            return deepcopy(values)
    def open_final(cfg, *, roles, allow_test):
        assert roles == ("E",) and allow_test is True
        return FinalData()
    monkeypatch.setattr(module, "EngineData", open_final)
    return dp, pp, plan, declaration, final_roles


def test_actual_toy_frozen_C_final_uses_only_E_role_and_three_class_metrics(tmp_path, monkeypatch):
    dp, pp, plan, declaration, roles = toy_final_fixture(tmp_path, monkeypatch)
    before = {k: module.file_hash(v["path"]) for k, v in declaration["assets"].items()}
    forbid = lambda *a, **k: pytest.fail("Final evaluation cannot fit or construct optimizer")
    monkeypatch.setattr(module.auxiliary, "train_state", forbid)
    monkeypatch.setattr(torch.optim, "Adam", forbid)
    result = module.evaluate_final(config(), dp, pp, tmp_path / "experiments/final")
    assert roles == ["final_test"] and result["role"] == "final_test"
    assert result["sample_count"] == 6 and result["class_order"] == module.LABELS
    assert result["optimizer_steps"] == 0 and all(result[k] is False for k in module.FLAGS)
    assert result["model_selection"] is result["calibration_refitting"] is result["statistic_refitting"] is False
    assert set(result["metrics"]) == {"final/nll", "final/brier", "final/macro_f1", "final/accuracy", "final/balanced_accuracy",
        *(f"final/recall_{i}" for i in range(3)), *(f"final/confusion_{i}_{j}" for i in range(3) for j in range(3))}
    assert all("aggregate" not in k for k in result["metrics"])
    assert before == result["frozen_C_assets_sha256"] == {k: module.file_hash(v["path"]) for k, v in declaration["assets"].items()}


@pytest.mark.parametrize("change", ["role", "state_asset", "cell_ids"])
def test_final_plan_changes_reject_before_PT_or_E_data(tmp_path, monkeypatch, change):
    dp, pp, plan, _, _ = toy_final_fixture(tmp_path, monkeypatch)
    wrong = deepcopy(plan)
    if change == "role":
        wrong["role"] = "selection"
    elif change == "state_asset":
        wrong["state_asset"]["serving_sha256"] = "different-state"
    else:
        wrong["cell_ids"] = list(reversed(wrong["cell_ids"]))
    module.atomic_json(pp, wrong)
    forbid = lambda *a, **k: pytest.fail("Final drift must reject before PT or E data")
    monkeypatch.setattr(torch, "load", forbid)
    monkeypatch.setattr(module, "EngineData", forbid)
    with pytest.raises(ValueError, match="FINAL_PLAN_OR_FROZEN_ASSET_CHANGED"):
        module.evaluate_final(config(), dp, pp, tmp_path / "experiments/final")


def test_final_label_order_mismatch_preserves_failure_without_metrics(tmp_path, monkeypatch):
    dp, pp, _, _, _ = toy_final_fixture(tmp_path, monkeypatch)
    factory = module.EngineData
    def changed(*a, **k):
        data = factory(*a, **k)
        rows = data.state_arrays("1_to_8", "final_test")
        rows["labels"] = np.roll(rows["labels"], 1)
        data.state_arrays = lambda *a, **k: rows
        return data
    monkeypatch.setattr(module, "EngineData", changed)
    output = tmp_path / "experiments/final"
    with pytest.raises(ValueError, match="FINAL_LABEL_ORDER_OR_AXIS_CHANGED"):
        module.evaluate_final(config(), dp, pp, output)
    assert (output / "failure.json").is_file()
    assert not (output / "original_three_class_final_metrics.json").exists()


def test_frozen_metadata_gate_sensitivity_keeps_full_primary_and_model(tmp_path, monkeypatch):
    dp, pp, plan, declaration, _ = toy_final_fixture(tmp_path, monkeypatch)
    plan["gate_metadata_sensitivity"] = {"flagged_original_cell_ids": [plan["cell_ids"][0]],
        "sensitivity_sample_count": 5, "ID_file_sha256": "synthetic-frozen-gate-ID-only"}
    module.atomic_json(pp, plan)
    before = {k: module.file_hash(v["path"]) for k, v in declaration["assets"].items()}
    result = module.evaluate_final(config(), dp, pp, tmp_path / "experiments/final")
    assert result["sample_count"] == 6 and result["class_ids"] == [0, 1, 2, 0, 1, 2]
    assert result["prespecified_gate_metadata_sensitivity"]["sample_count"] == 5
    assert result["prespecified_gate_metadata_sensitivity"]["primary_labels_or_split_changed"] is False
    assert result["prespecified_gate_metadata_sensitivity"]["wrong_label_inferred"] is False
    assert result["prespecified_gate_metadata_sensitivity"]["subset_used_for_model_selection"] is False
    assert all("aggregate" not in k for k in result["prespecified_gate_metadata_sensitivity"]["metrics"])
    assert before == {k: module.file_hash(v["path"]) for k, v in declaration["assets"].items()}
