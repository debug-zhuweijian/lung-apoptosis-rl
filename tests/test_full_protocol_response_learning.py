"""Only synthetic optimization/restore; real RNA/weights never reach a model."""
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


def run(tmp_path, name="run", *, refine=False, resume=False, hook=None):
    ae = tmp_path / "synthetic_AE.pt"
    if not ae.exists():
        torch.save({"synthetic_fixture_only": True}, ae)
    return module.fit_response(ToyData(), shared(), settings(refine), tmp_path / name,
        bindings={"synthetic": True}, representation_path=ae, synthetic=True, resume=resume, after_update=hook)


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
    saved = load_checkpoint(tmp_path / "run/last.pt")
    original = shared().state_dict()
    assert result["optimizer_steps"] == 4
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


@pytest.mark.parametrize("refine,cut", [(False, 1), (True, 2), (True, 3)])
def test_synthetic_interrupted_exact_restore_including_B4_switch(tmp_path, refine, cut):
    run(tmp_path, "complete", refine=refine)
    def stop(step):
        if step == cut:
            raise KeyboardInterrupt("synthetic durable update interrupt")
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "resumed", refine=refine, hook=stop)
    result = run(tmp_path, "resumed", refine=refine, resume=True)
    left, right = [load_checkpoint(tmp_path / name / "last.pt") for name in ("complete", "resumed")]
    assert result["optimizer_steps"] == 4
    assert_state(left["model"], right["model"])
    assert_state(left["optimizer"], right["optimizer"])
    assert_state(left["shared"], right["shared"])
    assert left["extra"] == right["extra"]
    if refine:
        assert len(right["optimizer"]["param_groups"]) == 2
        original = shared().state_dict()
        assert any(not module.producer.equal_weight(v, original[k.replace("member.decoder.", "decoder.")])
            for k, v in right["model"].items() if k.startswith("member.decoder."))


def test_resume_dependency_drift_refuses_before_PT_or_new_model(tmp_path, monkeypatch):
    run(tmp_path)
    monkeypatch.setattr(module, "execution_dependencies", lambda: {"changed": True})
    monkeypatch.setattr(module, "LatentMember", lambda *a, **k: pytest.fail("Must fail before model construction"))
    monkeypatch.setattr(module, "load_checkpoint", lambda *a, **k: pytest.fail("Must fail before PT load"))
    with pytest.raises(ValueError, match="PRE_PT_RESUME_IDENTITY_CHANGED"):
        run(tmp_path, resume=True)


def test_completed_early_stop_resume_returns_selected_asset_without_new_fit(tmp_path, monkeypatch):
    data, cfg = ToyData(), settings()
    cfg.update({"epochs": 4, "early_stopping": {"enabled": True, "patience": 1,
        "min_delta": 0., "min_updates": 0}})
    monkeypatch.setattr(module, "_validation", lambda *a: 1.)
    ae = tmp_path / "synthetic_AE.pt"
    torch.save({"synthetic_fixture_only": True}, ae)
    output = tmp_path / "stopped"
    kwargs = {"bindings": {"synthetic": True}, "representation_path": ae, "synthetic": True}
    result = module.fit_response(data, shared(), cfg, output, **kwargs)
    before = {n: module.producer.inputs.sha(output / n) for n in ("last.pt", "response_weights.pt")}
    assert result["optimizer_steps"] == 4
    assert load_checkpoint(output / "last.pt")["epoch"] == 2
    monkeypatch.setattr(module, "LatentMember", lambda *a, **k: pytest.fail("Completed resume must not construct P"))
    monkeypatch.setattr(module.torch.optim, "AdamW", lambda *a, **k: pytest.fail("Completed resume must not construct optimizer"))
    resumed = module.fit_response(data, shared(), cfg, output, **kwargs, resume=True)
    assert resumed == result
    assert all(module.producer.inputs.sha(output / n) == h for n, h in before.items())
    with (output / "response_weights.pt").open("ab") as stream:
        stream.write(b"synthetic selected-file drift")
    with pytest.raises(ValueError, match="COMPLETED_RESPONSE_PROOF_PENDING"):
        module.fit_response(data, shared(), cfg, output, **kwargs, resume=True)


def test_completed_new_AE_wrapper_checks_best_and_last_without_old_loop_or_PT(tmp_path, monkeypatch):
    config = load_config("configs/modular_research.yaml")
    output = tmp_path / "experiments/new-AE"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "EngineData", lambda *a, **k: SimpleNamespace(identity="synthetic-wrapper-only"))
    monkeypatch.setattr(module, "execution_dependencies", lambda: {"synthetic_mock_only": True})
    def fake_B1(cfg, monitor, data):
        folder = monitor.weights / "latent_ae"
        folder.mkdir(parents=True)
        common = {"schema_version": 1, "model": {"weight": torch.ones(1)}, "identity": "synthetic-wrapper-only",
            "epoch": 2, "counters": {"optimizer_step": 1}, "extra": {"best": 1.,
                "early_stop": {"best": 1., "bad": 1, "updates": 2, "steps": 1},
                "training_completion": {"status": "complete"}}}
        torch.save({**common, "epoch": 1}, folder / "best.pt")
        torch.save(common, folder / "last.pt")
        return {"weights": str(folder / "best.pt"), "best": 1., "identity": "synthetic-wrapper-only"}
    monkeypatch.setattr(module, "run_latent_ae", fake_B1)
    result = module.fit_matching_ae(config, output)
    assert result["checkpoint_sha256"] != result["last_checkpoint_sha256"]
    original_load = module.load_checkpoint
    monkeypatch.setattr(module, "run_latent_ae", lambda *a, **k: pytest.fail("Completed AE must not reenter old loop"))
    monkeypatch.setattr(module, "load_checkpoint", lambda *a, **k: pytest.fail("Completed AE receipt should not load PT"))
    assert module.fit_matching_ae(config, output, resume=True) == result
    with (output / "weights/latent_ae/last.pt").open("ab") as stream:
        stream.write(b"synthetic completed-last drift")
    with pytest.raises(ValueError, match="COMPLETED_AE_CHECKPOINT_CHANGED"):
        module.fit_matching_ae(config, output, resume=True)
    (output / "AE_result.json").unlink()
    monkeypatch.setattr(module, "load_checkpoint", original_load)
    with pytest.raises(ValueError, match="COMPLETED_AE_RECEIPT_MISSING_NO_RETRAIN"):
        module.fit_matching_ae(config, output, resume=True)


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
    monkeypatch.setattr(module, "execution_dependencies", lambda: {"synthetic_mock_only": True})
    source_shape = module._model_config(config, module.latent_settings(config), 4)
    model = SharedAutoencoder(source_shape, torch.arange(4)[:, None])
    folder = tmp_path / "experiments/fixture/weights/latent_ae"
    folder.mkdir(parents=True)
    origin = folder.parents[1]
    ae = folder / "best.pt"
    torch.save({"schema_version": 1, "identity": "synthetic-format-fixture", "model": model.state_dict(),
        "metadata": {"stage": "latent_ae", "model_config": asdict(source_shape), "data_identity": data.identity},
        "epoch": 1, "counters": {"optimizer_step": 1}, "extra": {"budget_exhausted": False}}, ae)
    (origin / "run_manifest.json").write_text(json.dumps({"schema": module.SCHEMA + ":joint-AE",
        "config": module.joint_config(config), "execution_dependencies": module.execution_dependencies()}), encoding="utf-8")
    last = folder / "last.pt"
    last.write_bytes(ae.read_bytes())
    declaration = {"schema": module.AE_RESULT_SCHEMA, "stage": "latent_ae", "weights": str(ae),
        "checkpoint_sha256": module.producer.inputs.sha(ae), "last_checkpoint": str(last),
        "last_checkpoint_sha256": module.producer.inputs.sha(last),
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


@pytest.mark.parametrize("change", ["role", "mask", "settings"])
def test_changed_learning_contract_cannot_resume(tmp_path, change):
    run(tmp_path)
    data, cfg = ToyData(), settings()
    if change == "role":
        data.protocol_identity = "changed-selection-assignment"
    elif change == "mask":
        data.protocol_identity = "falsely-known-exposure"
    else:
        cfg["microbatch"] = 2
    with pytest.raises(ValueError, match="PRE_PT_RESUME_IDENTITY_CHANGED"):
        module.fit_response(data, shared(), cfg, tmp_path / "run", bindings={"synthetic": True},
            representation_path=tmp_path / "synthetic_AE.pt", synthetic=True, resume=True)


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


def test_reference_variant_resume_drift_rejects_before_PT_and_synthetic_exact_restore(tmp_path, monkeypatch):
    cfg = settings()
    cfg["reference_variant"] = "mean_h"
    ae = tmp_path / "toyAE.pt"
    torch.save({"synthetic_only": True}, ae)
    def run_one(name, resume=False, after_update=None):
        return module.fit_response(ToyData(), shared(), cfg, tmp_path / name,
            bindings={"synthetic_only": True}, representation_path=ae, synthetic=True,
            resume=resume, after_update=after_update)
    run_one("full")
    def stop(step):
        if step == 1:
            raise KeyboardInterrupt("synthetic interruption")
    with pytest.raises(KeyboardInterrupt):
        run_one("resumed", after_update=stop)
    run_one("resumed", resume=True)
    a, b = [load_checkpoint(tmp_path / p / "last.pt") for p in ("full", "resumed")]
    for key in ("model", "optimizer", "shared", "metadata", "extra"):
        assert_state(a[key], b[key])
    cfg["reference_variant"] = "zero_h"
    monkeypatch.setattr(module, "load_checkpoint", lambda *a, **k: pytest.fail("variant drift must reject before PT"))
    with pytest.raises(ValueError, match="PRE_PT_RESUME_IDENTITY_CHANGED"):
        run_one("resumed", resume=True)


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
