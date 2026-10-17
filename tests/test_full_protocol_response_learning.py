"""Only synthetic optimization/export; real RNA/weights never reach a model."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from lung.engine import full_protocol_response_learning as module
from lung.engine.config import load_config
from lung.engine.checkpoint import load_checkpoint, seed_all
from lung.models import ModelConfig
from lung.models.latent_staged import SharedAutoencoder


class ToyData:
    synthetic = True
    identity = "synthetic-not-joint"
    protocol_identity = "synthetic-single-nominal-event"
    n_genes = 4
    mu, scale, bandwidth = np.ones(4), np.ones(4), 1.0

    def __init__(self):
        self.data = self
        self.requests = {key: {"dose_uM": float(i+1), "actual_protocol": {"duration_h": 24.}}
            for i, key in enumerate(("fitA", "fitB", "selectionA"))}

    def action_features(self, key):
        return {"fingerprint": np.ones(2048), "dose": self.requests[key]["dose_uM"],
            "duration": 24., "source_id": 0, "background_id": 0, "protocol_id": 0,
            "is_vehicle": False}

    batch = module.NativeProtocolData.batch

    def condition_ids(self, role):
        assert role in ("T", "V")
        return ("fitA", "fitB") if role == "T" else ("selectionA",)

    def iter_response_epoch(self, role, seed):
        return np.random.default_rng(seed).permutation(self.condition_ids(role)).tolist()

    def sample_response(self, key, n, rng, role):
        assert key in self.condition_ids(role)
        return {"control": rng.uniform(.5, 2., (n, 4)).astype(np.float32),
            "target": rng.uniform(.5, 2., (n, 4)).astype(np.float32)}


def shared():
    seed_all(997)
    shape = ModelConfig(n_genes=4, d_model=8, n_heads=2, context_dim=2, latent_dim=3,
        n_sources=1, n_protocols=1, backbone="compact_full_gene", checkpoint_blocks=False)
    return SharedAutoencoder(shape, torch.arange(4)[:, None])


def settings(refine=False):
    return {"seed": 998, "device": "cpu", "epochs": 2, "lr": .001, "weight_decay": .0001,
        "grad_clip": 1., "conditions_per_update": 1, "ot_samples": 2, "microbatch": 1,
        "mmd_samples": 2, "train_ode_steps": 1, "inference_ode_steps": 1,
        "early_stopping": {"enabled": False}, "coupling_space": "latent",
        "numeric_mode": "fp32_explicit_independent_version", "refine": {"enabled": refine,
            "switch_epoch": 1, "weight": 1.}}


def run(tmp_path, name="run", *, refine=False):
    ae = tmp_path / "synthetic_AE.pt"
    if not ae.exists():
        torch.save({"synthetic_fixture_only": True}, ae)
    return module.fit_response(ToyData(), shared(), settings(refine), tmp_path / name,
        bindings={"synthetic": True}, representation_path=ae, synthetic=True)


def assert_state(left, right):
    assert set(left) == set(right)
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            assert module.producer.equal_weight(a, b), key
        elif isinstance(a, dict):
            assert_state(a, b)
        elif isinstance(a, list):
            for x, y in zip(a, b):
                assert_state(x, y) if isinstance(x, dict) else None
                if not isinstance(x, dict):
                    assert x == y
        else:
            assert a == b


def test_nominal_native_event_keeps_unknown_exposure_recovery_history():
    kw, physical = ToyData().batch("fitA", 2, "cpu")
    assert kw["duration"].tolist() == [24., 24.]
    assert physical.endpoint_hours.tolist() == [24., 24.]
    assert physical.known[:, :, 0].all() and not physical.known[:, :, 1:].any()
    assert torch.isnan(physical.values[:, :, 1:]).all()
    assert physical.lengths.tolist() == [1, 1]
    assert not module.COVERAGE["event_list_verified_exhaustive"]
    assert module.COVERAGE["Viel_pulse_washout_recovery_protocol_OOD"]
    assert not module.COVERAGE["washout_RNA_supervision"]


def test_independent_joint_config_does_not_mutate_active_configuration():
    config = load_config("configs/modular_research.yaml")
    before = deepcopy(config)
    candidate = module.joint_config(config)
    assert config == before
    assert candidate["data"] == module.JOINT_CONFIG
    assert candidate["response"]["model"]["n_protocols"] == 2
    assert candidate["response"]["model"]["n_sources"] == 2
    assert candidate["reward"] == config["reward"]
    # Old B1 E has no protocol embedding; explicit export changes only context
    # capacity in the new P shape, never a numerical AE parameter.
    source = module._model_config(candidate, module.latent_settings(candidate), 32461)
    assert source.n_protocols == 1


def test_synthetic_CFM_optimizes_new_namespace_and_freezes_AE_decoder(tmp_path):
    result = run(tmp_path)
    saved = load_checkpoint(tmp_path / "run/final.pt")
    original = shared().state_dict()
    assert result["optimizer_steps"] == 4
    assert "optimizer" not in saved and "rng" not in saved
    assert saved["metadata"]["training_protocol_coverage"] == module.COVERAGE
    assert saved["metadata"]["stage"] == module.producer.STAGE
    assert any(k.startswith("full_protocol.history.") for k in saved["model"])
    assert_state(original, saved["shared"])
    assert all(module.producer.equal_weight(v, original[k.replace("member.decoder.", "decoder.")])
        for k, v in saved["model"].items() if k.startswith("member.decoder."))
    serving = torch.load(tmp_path / "run/response_weights.pt", weights_only=True)
    assert serving["metadata"]["synthetic"] is True
    assert result["selected_optimizer_step"] == serving["counters"]["optimizer_step"]
    assert not result["production_registered"] and not result["scientific_qualified"]


def test_completed_new_AE_wrapper_checks_best_and_final(tmp_path, monkeypatch):
    config = load_config("configs/modular_research.yaml")
    output = tmp_path / "experiments/new-AE"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "EngineData", lambda *a, **k: SimpleNamespace(identity="synthetic-wrapper-only"))

    def fake_B1(cfg, monitor, data):
        folder = monitor.weights / "latent_ae"
        folder.mkdir(parents=True)
        common = {"schema_version": 1, "model": {"weight": torch.ones(1)}, "identity": "synthetic-wrapper-only",
            "epoch": 2, "counters": {"optimizer_step": 1}, "extra": {"best": 1.,
                "training_completion": {"status": "complete"}}}
        torch.save({**common, "epoch": 1}, folder / "best.pt")
        torch.save(common, folder / "final.pt")
        return {"weights": str(folder / "best.pt"), "best": 1., "identity": "synthetic-wrapper-only"}
    monkeypatch.setattr(module, "run_latent_ae", fake_B1)
    result = module.fit_matching_ae(config, output)
    assert result["checkpoint_sha256"] != result["final_checkpoint_sha256"]
    module.verify_completed_ae_result(result, output, "synthetic-wrapper-only")
    with (output / "weights/latent_ae/final.pt").open("ab") as stream:
        stream.write(b"synthetic completed-final drift")
    with pytest.raises(ValueError, match="COMPLETED_AE_CHECKPOINT_CHANGED"):
        module.verify_completed_ae_result(result, output, "synthetic-wrapper-only")


def test_synthetic_weights_cannot_export_as_real_matching_AE(tmp_path, monkeypatch):
    data = ToyData()
    monkeypatch.setattr(module, "load_checkpoint", lambda *a, **k: pytest.fail("Reject synthetic before PT"))
    with pytest.raises(ValueError, match="REAL_JOINT_AE_EXPORT_REQUIRED"):
        module.export_matching_ae(tmp_path / "weights/latent_ae/best.pt", data, tmp_path,
            {"data_identity": data.identity}, {})


def test_synthetic_B1_checkpoint_format_export_keeps_every_tensor_and_safe_snapshot(tmp_path, monkeypatch):
    # A 4-gene checkpoint-format fixture under a mocked root; this is not a
    # project data identity or execution registration for production.
    config = load_config("configs/modular_research.yaml")
    config["research"]["latent"].update({"d_model": 8, "latent_dim": 3})
    data = ToyData()
    data.synthetic = False  # exercise the B1-format export branch, mocked root only
    data.context_sizes = {"n_sources": 2, "n_backgrounds": 1, "n_protocols": 2}
    data.neighbors = np.arange(4)[:, None]
    monkeypatch.setattr(module, "ROOT", tmp_path)

    source_shape = module._model_config(config, module.latent_settings(config), 4)
    model = SharedAutoencoder(source_shape, torch.arange(4)[:, None])
    folder = tmp_path / "experiments/fixture/weights/latent_ae"
    folder.mkdir(parents=True)
    origin = folder.parents[1]
    ae = folder / "best.pt"
    torch.save({"schema_version": 1, "identity": "synthetic-format-fixture", "model": model.state_dict(),
        "metadata": {"stage": "latent_ae", "model_config": asdict(source_shape), "data_identity": data.identity},
        "epoch": 1, "counters": {"optimizer_step": 1}, "extra": {"training_completion": {"status": "complete"}}}, ae)
    (origin / "run_manifest.json").write_text(json.dumps({"schema": module.SCHEMA + ":joint-AE",
        "config": module.joint_config(config)}), encoding="utf-8")
    last = folder / "final.pt"
    last.write_bytes(ae.read_bytes())
    declaration = {"schema": module.AE_RESULT_SCHEMA, "stage": "latent_ae", "weights": str(ae),
        "checkpoint_sha256": module.producer.inputs.sha(ae), "final_checkpoint": str(last),
        "final_checkpoint_sha256": module.producer.inputs.sha(last),
        "identity": "synthetic-format-fixture", "data_identity": data.identity,
        "training_completion": {"status": "complete"}}
    (origin / "AE_result.json").write_text(json.dumps(declaration), encoding="utf-8")
    output = tmp_path / "experiments/fixture-export"
    output.mkdir()
    bound = {"data_identity": data.identity, "synthetic_format_test_only": True}
    result, exported, _ = module.export_matching_ae(ae, data, output, bound, config)
    assert result.config.n_protocols == 2 and source_shape.n_protocols == 1
    assert_state(model.state_dict(), result.state_dict())
    value = torch.load(exported, weights_only=True)
    assert_state(model.state_dict(), value["model"])
    assert value["metadata"]["source_model_config"] == asdict(source_shape)
    digest = module.producer.inputs.sha(exported)
    module.export_matching_ae(ae, data, output, bound, config)
    assert module.producer.inputs.sha(exported) == digest
    # A fabricated completed declaration cannot change actual strict counters.
    saved = load_checkpoint(ae)
    saved["counters"]["optimizer_step"] = True
    torch.save(saved, ae)
    declaration["checkpoint_sha256"] = module.producer.inputs.sha(ae)
    (origin / "AE_result.json").write_text(json.dumps(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match="MATCHING_COMPLETED_AE_REQUIRED"):
        module.export_matching_ae(ae, data, output, bound, config)


@pytest.mark.parametrize("variant", ("observed", "zero_h", "mean_h"))
def test_three_independent_reference_train_selection_and_B4_inputs_share_the_same_variant(tmp_path, monkeypatch, variant):
    cfg = settings(refine=True)
    cfg["reference_variant"] = variant
    original = module.producer.apply_reference_variant
    calls = []
    def track(h, value):
        result = original(h, value)
        calls.append((value, result.detach().clone()))
        return result
    monkeypatch.setattr(module.producer, "apply_reference_variant", track)
    ae = tmp_path / "toyAE.pt"
    torch.save({"synthetic_only": True}, ae)
    result = module.fit_response(ToyData(), shared(), cfg, tmp_path / variant,
        bindings={"synthetic_only": True}, representation_path=ae, synthetic=True)
    saved = torch.load(tmp_path / variant / "response_weights.pt", weights_only=True)
    assert result["reference_contract"] == saved["metadata"]["reference_contract"] == module.producer.reference_contract(variant)
    assert len(calls) == 8 and all(v == variant for v, _ in calls)  # 4 CFM, 2 B4, 2 selection
    if variant == "zero_h":
        assert all((h == 0).all() for _, h in calls)
    elif variant == "mean_h":
        assert all((h == h[0]).all() for _, h in calls)


def test_P_variant_settings_do_not_change_shared_AE_or_C_configs():
    config = load_config("configs/modular_research.yaml")
    ae = module.joint_ae_config(config)
    from lung.engine.raw_state_scoring import state_config
    C = state_config(config)
    for variant in module.producer.REFERENCE_VARIANTS:
        candidate = deepcopy(config)
        candidate["research"]["full_protocol_response"] = {"reference_variant": variant}
        assert module.joint_ae_config(candidate) == ae
        assert state_config(candidate) == C
        assert module.settings_from_config(candidate)["reference_variant"] == variant
        assert candidate["reward"] == config["reward"]
