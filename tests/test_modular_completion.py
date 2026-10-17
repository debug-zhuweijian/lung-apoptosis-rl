"""Tiny model outputs retain completion and calibrated inference metadata."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import auxiliary
from lung.engine.checkpoint import atomic_save, load_checkpoint
from lung.engine.config import StageUnavailable, load_config, merge
from lung.engine.latent_pipeline import run_latent_ae, run_latent_cache, run_latent_cfm
from lung.engine.training_control import checkpoint_completion
from synthetic_data import smoke_config
from test_modular_world_v1 import Monitor, components, cpu_threads
from test_response_research_runtime import TinyData


class LatentMonitor(Monitor):
    def say(self, message):
        pass


def tiny_config():
    return merge(smoke_config(load_config()), {
        "progress": False, "early_stopping": False,
        "response": {"members": 1, "conditions_per_update": 1, "ot_samples": 2,
                     "mmd_samples": 2, "inference_microbatch": 2, "inference_ode_steps": 2},
        "research": {"precision": {"mode": "fp32"},
                     "latent": {"d_model": 8, "latent_dim": 4, "ae": {"batch": 2}}}})


@pytest.mark.parametrize("stage", ["pool", "state"])
def test_completed_auxiliary_weights_are_portable(components, tmp_path, stage):
    data, config, monitor, pipeline, _ = components
    complete = auxiliary.train_auxiliary(stage, data, config, monitor, pipeline)
    raw = load_checkpoint(complete[stage])
    assert raw["counters"]["optimizer_step"] > 0
    selected = Path(complete[stage])
    record = checkpoint_completion(raw, config["epochs"])
    assert record["status"] == "complete"
    assert record["actual_updates"] >= raw["counters"]["optimizer_step"]
    imported = tmp_path / f"portable-{stage}" / "trained.pt"
    imported.parent.mkdir()
    shutil.copy2(selected, imported)
    assert checkpoint_completion(load_checkpoint(imported), config["epochs"]) == record
    auxiliary.load_auxiliary(imported, data, config["device"])


@pytest.mark.parametrize("early_stop,actual_epochs", [(False, 3), (True, 2)])
def test_actual_latent_completion_keeps_early_best_weights(tmp_path, monkeypatch, early_stop, actual_epochs):
    from lung.engine import latent_pipeline
    config = {**tiny_config(), "workflow": "modular_stage", "epochs": 3,
              "early_stopping": {"enabled": early_stop, "patience": 1, "min_updates": 0}}
    data, monitor = TinyData(), LatentMonitor(tmp_path)
    selected_models = {}
    original_save = latent_pipeline.atomic_save

    def capture_best(path, payload):
        if Path(path).name == "best.pt":
            selected_models[payload["metadata"]["stage"]] = deepcopy(payload["model"])
        original_save(path, payload)

    monkeypatch.setattr(latent_pipeline, "atomic_save", capture_best)
    ae_losses, cfm_losses = iter([1., 2., 3.]), iter([1., 2., 3.])
    monkeypatch.setattr(latent_pipeline, "_ae_validation", lambda *a, **kw: {"loss": next(ae_losses)})
    monkeypatch.setattr(latent_pipeline, "_cfm_validation", lambda *a, **kw: {"loss": next(cfm_losses)})
    ae = run_latent_ae(config, monitor, data)
    run_latent_cache(config, monitor, data)
    response = run_latent_cfm(config, monitor, data)
    for stage, result in (("latent_ae", ae), ("latent_cfm", response)):
        selected = load_checkpoint(result["weights"])
        final = load_checkpoint(Path(result["weights"]).with_name("final.pt"))
        completion = checkpoint_completion(selected, 3)
        assert Path(result["weights"]).name == "best.pt"
        assert selected["epoch"] == 1 and selected["counters"]["optimizer_step"] == 2
        assert completion["actual_epochs"] == final["epoch"] == actual_epochs
        assert completion["actual_updates"] == 2 * actual_epochs
        assert completion["reason"] == ("validation_plateau" if early_stop else "max_epochs")
        for name, value in selected_models[stage].items():
            observed = selected["model"][name]
            if value.is_sparse:
                value, observed = value.to_dense(), observed.to_dense()
            torch.testing.assert_close(value, observed, rtol=0, atol=0)
        assert any(not torch.equal(value, final["model"][name])
                   for name, value in selected["model"].items() if not value.is_sparse)
    assert not list(monitor.weights.rglob("completed-*.pt"))
    assert not list(monitor.weights.rglob("selected-*.pt"))



def test_uncalibrated_state_is_rejected_despite_completed_classifier_training(components):
    data, config, monitor, pipeline, _ = components
    calibrated = auxiliary.train_auxiliary("state", data, config, monitor, pipeline)
    path = monitor.weights / "state" / "best.pt"
    assert checkpoint_completion(load_checkpoint(path), config["epochs"])["status"] == "complete"
    assert "calibration_role" not in load_checkpoint(path)["metadata"]
    auxiliary.load_auxiliary(calibrated["state"], data, config["device"])
    with pytest.raises(StageUnavailable, match="STATE_CALIBRATION_INCOMPLETE"):
        auxiliary.load_auxiliary(path, data, config["device"])


@pytest.mark.parametrize("change", ["nan_metadata", "inf_metadata", "nan_buffer", "inf_buffer", "zero", "mismatch"])
def test_invalid_or_inconsistent_calibration_temperature_is_rejected(components, tmp_path, change):
    data, config, monitor, pipeline, _ = components
    calibrated = auxiliary.train_auxiliary("state", data, config, monitor, pipeline)
    saved = deepcopy(load_checkpoint(calibrated["state"]))
    if change.endswith("metadata"):
        saved["extra"]["temperature"] = float("nan" if change.startswith("nan") else "inf")
    elif change.endswith("buffer"):
        saved["model"]["temperature"] = torch.tensor(float("nan" if change.startswith("nan") else "inf"))
    elif change == "zero":
        saved["extra"]["temperature"] = 0.
        saved["model"]["temperature"] = torch.tensor(0.)
    else:
        saved["extra"]["temperature"] = float(saved["model"]["temperature"]) + .25
    path = tmp_path / f"invalid-{change}" / "state.pt"
    atomic_save(path, saved)
    with pytest.raises((ValueError, StageUnavailable), match="STATE_CALIBRATION_INCOMPLETE|NONFINITE_MODEL_WEIGHTS"):
        auxiliary.load_auxiliary(path, data, config["device"])
