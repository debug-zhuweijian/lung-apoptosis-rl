"""Real tiny CPU training proves completion and calibrated reuse boundaries."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import auxiliary
from lung.engine.checkpoint import atomic_save, load_checkpoint
from lung.engine.config import StageUnavailable
from lung.engine.latent_pipeline import run_latent_ae, run_latent_cache, run_latent_cfm
from lung.engine.modular_budget import StagePaused
from lung.engine.modular_completion import completion_record, seal_completed
from lung.engine.modular_research import verify_artifact
from test_modular_research_v1 import tiny_config
from test_modular_world_v1 import Monitor, components, cpu_threads
from test_response_research_runtime import Budget, TinyData


class LatentMonitor(Monitor):
    def say(self, message):
        pass


class ManualInterruptBudget(Budget):
    """A second update is interrupted after the first durable update existed."""

    def check(self, step, epoch):
        if step >= 2:
            raise KeyboardInterrupt("Synthetic manual stop during a latent training epoch")
        return False


@pytest.mark.parametrize("interruption", ["budget", "manual_epoch", "manual_update"])
def test_real_fit_epochs_intermediate_best_and_tail_cannot_be_reused(components, monkeypatch, interruption):
    data, original_config, monitor, pipeline, _ = components
    config = {**original_config, "epochs": 3, "early_stopping": False, "aux_batch": 1,
              "training_control": {"checkpoint_every_updates": 1, "checkpoint_every_seconds": 0}}
    monitor.stop_check = (lambda step, epoch: step >= 3 and epoch >= 1) if interruption == "budget" else (
        lambda step, epoch: False)
    if interruption == "manual_epoch":
        def interrupt_record(stage, epoch, metrics):
            if stage == "pool" and epoch == 1:
                raise KeyboardInterrupt("Manual stop after an intermediate selected epoch")
        monkeypatch.setattr(monitor, "record", interrupt_record)
    elif interruption == "manual_update":
        original_save = auxiliary.save_training

        def interrupt_after_update(path, *args, **kwargs):
            original_save(path, *args, **kwargs)
            if Path(path).name == "last.pt" and kwargs["epoch"] >= 1 and kwargs.get("extra", {}).get("update_resume"):
                raise KeyboardInterrupt("Manual stop after a durable incomplete update")

        monkeypatch.setattr(auxiliary, "save_training", interrupt_after_update)
    with pytest.raises(StagePaused if interruption == "budget" else KeyboardInterrupt):
        auxiliary.train_auxiliary("pool", data, config, monitor, pipeline)
    folder = monitor.weights / "pool"
    last = load_checkpoint(folder / "last.pt")
    best = load_checkpoint(folder / "best.pt")
    assert last["epoch"] >= 1 and best["epoch"] >= 1
    assert last["extra"]["training_control"]["status"] == "running"
    if interruption in {"budget", "manual_update"}:
        assert last["extra"]["update_resume"]["epoch"] == 1
        assert last["counters"]["optimizer_step"] == 3
    if interruption == "budget":
        assert last["extra"]["pause_reason"] == "budget_exhausted"
    for path in (folder / "last.pt", folder / "best.pt"):
        with pytest.raises(StageUnavailable, match="MODULE_TRAINING_INCOMPLETE"):
            completion_record(path, config)
        with pytest.raises(StageUnavailable, match="MODULE_TRAINING_INCOMPLETE"):
            seal_completed(path, config)
        with pytest.raises(StageUnavailable, match="MODULE_TRAINING_INCOMPLETE"):
            verify_artifact("pool", {**pipeline, "pool": str(path)}, data, config)


@pytest.mark.parametrize("stage", ["pool", "state"])
def test_completed_auxiliary_is_portable_without_mutable_last_checkpoint(components, tmp_path, stage):
    data, config, monitor, pipeline, _ = components
    complete = auxiliary.train_auxiliary(stage, data, config, monitor, pipeline)
    raw = load_checkpoint(complete[stage])
    assert raw["counters"]["optimizer_step"] > 0
    sealed = Path(seal_completed(complete[stage], config))
    record = completion_record(sealed, config)
    assert record["status"] == "complete"
    assert record["selected_updates"] == raw["counters"]["optimizer_step"]
    assert record["actual_updates"] >= record["selected_updates"]
    imported = tmp_path / f"portable-{stage}" / "trained.pt"
    imported.parent.mkdir()
    shutil.copy2(sealed, imported)
    assert not (imported.parent / "last.pt").exists()
    assert completion_record(imported, config) == record
    assert seal_completed(imported, config) == str(imported)
    verify_artifact(stage, {**pipeline, stage: str(imported)}, data, config)


@pytest.mark.parametrize("stage", ["ae", "response"])
def test_manual_latent_update_checkpoint_with_resume_cursor_is_not_complete(tmp_path, stage):
    config = {**tiny_config(), "workflow": "modular_stage"}
    data, monitor = TinyData(), LatentMonitor(tmp_path)
    if stage == "ae":
        operation = run_latent_ae
        pipeline = {}
        folder = monitor.weights / "latent_ae"
    else:
        ae = run_latent_ae(config, monitor, data, Budget())
        pipeline = {"representation": ae["weights"]}
        run_latent_cache(config, monitor, data, Budget())
        operation = run_latent_cfm
        folder = monitor.weights / "latent_cfm" / "member_1"
    with pytest.raises(KeyboardInterrupt):
        operation(config, monitor, data, ManualInterruptBudget())
    path = folder / "last.pt"
    saved = load_checkpoint(path)
    assert saved["counters"]["optimizer_step"] == 1
    assert saved["extra"]["resume"] and not saved["extra"]["budget_exhausted"]
    pipeline = {**pipeline, ("representation" if stage == "ae" else "response"):
                str(path) if stage == "ae" else [str(path)]}
    with pytest.raises(StageUnavailable, match="MODULE_TRAINING_INCOMPLETE"):
        completion_record(path, config)
    with pytest.raises(StageUnavailable, match="MODULE_TRAINING_INCOMPLETE"):
        verify_artifact(stage, pipeline, data, config)


def test_actual_latent_completion_records_allow_portable_ae_and_response(tmp_path):
    config = {**tiny_config(), "workflow": "modular_stage"}
    data, monitor = TinyData(), LatentMonitor(tmp_path)
    ae = run_latent_ae(config, monitor, data, Budget())
    ae_saved = load_checkpoint(ae["weights"])
    completion = ae_saved["extra"]["training_completion"]
    assert completion["status"] == "complete" and completion["actual_epochs"] == ae_saved["epoch"] == 1
    assert completion["actual_updates"] == ae_saved["counters"]["optimizer_step"] == 2
    portable = tmp_path / "portable-latent"
    portable.mkdir()
    representation = portable / "representation.pt"
    shutil.copy2(seal_completed(ae["weights"], config), representation)
    config["research"]["latent"]["ae_checkpoint"] = str(representation)
    run_latent_cache(config, monitor, data, Budget())
    response = run_latent_cfm(config, monitor, data, Budget())
    response_saved = load_checkpoint(response["weights"])
    assert response_saved["extra"]["training_completion"]["status"] == "complete"
    assert response_saved["extra"]["training_completion"]["actual_updates"] == response_saved["counters"]["optimizer_step"] == 2
    response_path = portable / "response.pt"
    shutil.copy2(seal_completed(response["weights"], config), response_path)
    assert not (portable / "last.pt").exists()
    pipeline = {"representation": str(representation), "response": [str(response_path)]}
    for stage in ("ae", "response"):
        verify_artifact(stage, pipeline, data, config)


def test_uncalibrated_state_is_rejected_despite_completed_classifier_training(components):
    data, config, monitor, pipeline, _ = components
    calibrated = auxiliary.train_auxiliary("state", data, config, monitor, pipeline)
    path = monitor.weights / "state" / "best.pt"
    assert completion_record(path, config)["status"] == "complete"
    assert "calibration_role" not in load_checkpoint(path)["metadata"]
    verify_artifact("state", calibrated, data, config)
    with pytest.raises(StageUnavailable, match="STATE_CALIBRATION_INCOMPLETE"):
        verify_artifact("state", {**pipeline, "state": str(path)}, data, config)


@pytest.mark.parametrize("change", ["nan_metadata", "inf_metadata", "nan_buffer", "inf_buffer", "zero", "mismatch"])
def test_invalid_or_inconsistent_calibration_temperature_is_rejected(components, tmp_path, change):
    data, config, monitor, pipeline, _ = components
    calibrated = auxiliary.train_auxiliary("state", data, config, monitor, pipeline)
    sealed = seal_completed(calibrated["state"], config)
    saved = deepcopy(load_checkpoint(sealed))
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
    assert not (path.parent / "last.pt").exists()
    with pytest.raises((ValueError, StageUnavailable), match="STATE_CALIBRATION_INCOMPLETE|NONFINITE_MODEL_WEIGHTS"):
        verify_artifact("state", {**pipeline, "state": str(path)}, data, config)
