"""CPU equivalence for resumable auxiliary features and state temperature fitting."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import auxiliary
from lung.engine.checkpoint import atomic_save, load_checkpoint
from lung.engine.modular_budget import StagePaused
from test_modular_aux_budget import assert_same


class Monitor:
    def __init__(self, root):
        self.weights, self.experiment = root / "weights", root / "experiment"
        self.weights.mkdir(parents=True)
        self.experiment.mkdir(parents=True)
        self.phase = ""
        self.records = []

    def progress(self, values, **kwargs):
        self.phase = kwargs.get("desc", "")
        return values

    def record(self, *args, **kwargs):
        self.records.append((args, kwargs))


class BudgetMonitor(Monitor):
    def __init__(self, root, predicate=None):
        super().__init__(root)
        self.predicate = predicate or (lambda step, epoch: False)

    def stop_check(self, step, epoch):
        return "test_budget" if self.predicate(step, epoch) else None


class Data:
    identity = "AUX-PREPARATION-CPU"

    def __init__(self):
        self.response_reads = []
        self.state_reads = []

    def condition_ids(self, role):
        return [f"{role}-{index}" for index in range(3 if role == "T" else 2)]

    def sample_response(self, key, count, rng, role):
        assert key in self.condition_ids(role)
        values = rng.normal(size=(count, 4)).astype(np.float32)
        self.response_reads.append((role, key, values.copy()))
        return {"target": values}

    def state_arrays(self, fold, role):
        assert fold == "1_to_8" and role in {"fit", "selection", "calibration"}
        self.state_reads.append(role)
        return {"x": np.random.default_rng({"fit": 1, "selection": 2, "calibration": 3}[role]).normal(
                    size=(6, 4)).astype(np.float32), "labels": np.tile(np.arange(3), 2)}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reference = SimpleNamespace(config=SimpleNamespace(d_model=4), encode_cells=lambda x: (None, x))
    monkeypatch.setattr(auxiliary, "auxiliary_representation", lambda *args: (reference, {"fixture": True}))
    asset = tmp_path / "representation.pt"
    atomic_save(asset, {"fixture": "unchanged"})
    pipeline = {"representation": str(asset)}
    config = {"device": "cpu", "seed": 31, "epochs": 2, "synthetic": True,
              "population": 3, "aux_batch": 2, "aux_lr": .001, "state_batch": 3,
              "calibration_steps": 5, "calibration_lr": .01,
              "response": {"inference_microbatch": 2, "inference_ode_steps": 1}}
    yield config, pipeline
    torch.set_num_threads(threads)


def test_pool_prefix_resume_preserves_sampling_order_and_numpy_generator(inputs, tmp_path):
    config, pipeline = inputs
    full_data, resumed_data = Data(), Data()
    full = BudgetMonitor(tmp_path / "full")
    auxiliary.train_pool(full_data, config, full, pipeline)
    paused = BudgetMonitor(tmp_path / "resumed")
    paused.predicate = lambda step, epoch: paused.phase == "cache pool T" and step >= 2
    with pytest.raises(StagePaused) as caught:
        auxiliary.train_pool(resumed_data, config, paused, pipeline)
    assert caught.value.stage == "pool/cache/T"
    assert caught.value.cursor == 2
    assert len(resumed_data.response_reads) == 2
    assert len(list((paused.experiment / "cache").glob("*.parts/row-*.pt"))) == 2
    paused.predicate = lambda *args: False
    auxiliary.train_pool(resumed_data, config, paused, pipeline, resume=True)
    assert_same(full_data.response_reads, resumed_data.response_reads)
    assert_same(load_checkpoint(full.weights / "pool/last.pt"),
                load_checkpoint(paused.weights / "pool/last.pt"))


@pytest.mark.parametrize("changed", ["role", "order", "rng"])
def test_pool_partial_cache_checks_role_order_and_rng(inputs, tmp_path, changed):
    config, pipeline = inputs
    monitor = BudgetMonitor(tmp_path / "run")
    monitor.predicate = lambda step, epoch: monitor.phase == "cache pool T" and step >= 1
    with pytest.raises(StagePaused):
        auxiliary.train_pool(Data(), config, monitor, pipeline)
    path = next((monitor.experiment / "cache").glob("*.parts/row-00000000.pt"))
    saved = torch.load(path, weights_only=False)
    if changed == "role":
        saved["role"] = "V"
    elif changed == "order":
        saved["row"]["condition_id"] = "T-other"
    else:
        saved["sample_rng"] = {"bit_generator": "wrong"}
    atomic_save(path, saved)
    monitor.predicate = lambda *args: False
    data = Data()
    with pytest.raises(ValueError, match="POOL_PREPARATION_"):
        auxiliary.train_pool(data, config, monitor, pipeline, resume=True)
    assert data.response_reads == []


@pytest.mark.parametrize("role", ["fit", "selection", "calibration"])
def test_state_feature_role_pause_reuses_saved_features(inputs, tmp_path, role):
    config, pipeline = inputs
    monitor = BudgetMonitor(tmp_path / "run")
    monitor.predicate = lambda *args: bool(list((monitor.experiment / "cache").glob(f"state-{role}-*.pt")))
    data = Data()
    with pytest.raises(StagePaused) as caught:
        auxiliary.train_state(data, config, monitor, pipeline)
    assert caught.value.stage == f"state/cache/{role}"
    assert Path(caught.value.checkpoint_path).is_file()
    assert not (monitor.weights / "state/calibrated.pt").exists()
    monitor.predicate = lambda *args: False
    auxiliary.train_state(data, config, monitor, pipeline, resume=True)
    assert data.state_reads == ["fit", "selection", "calibration"]
    assert load_checkpoint(monitor.weights / "state/last.pt")["counters"]["optimizer_step"] == 4


@pytest.mark.parametrize("stop_step", [2, 5])
def test_temperature_pause_resume_preserves_adam_and_fitted_temperature(inputs, tmp_path, stop_step):
    config, pipeline = inputs
    full, paused = BudgetMonitor(tmp_path / "full"), BudgetMonitor(tmp_path / "resumed")
    expected = auxiliary.train_state(Data(), config, full, pipeline)
    paused.predicate = lambda step, epoch: paused.phase == "state temperature" and step >= stop_step
    data = Data()
    with pytest.raises(StagePaused) as caught:
        auxiliary.train_state(data, config, paused, pipeline)
    assert caught.value.stage == "state/temperature"
    assert caught.value.cursor == stop_step
    assert not (paused.weights / "state/calibrated.pt").exists()
    progress = load_checkpoint(caught.value.checkpoint_path)
    assert progress["extra"]["calibration_state"]["status"] != "complete"
    assert progress["extra"]["calibration_state"]["cursor"] == stop_step
    paused.predicate = lambda *args: False
    result = auxiliary.train_state(data, config, paused, pipeline, resume=True)
    assert data.state_reads == ["fit", "selection", "calibration"]
    for name in ("model", "rng", "counters", "epoch", "extra"):
        assert_same(load_checkpoint(expected["state"])[name], load_checkpoint(result["state"])[name])
    completed = load_checkpoint(result["state"])["extra"]["calibration_state"]
    assert completed["status"] == "complete"
    assert completed["cursor"] == completed["target_steps"] == config["calibration_steps"]
    for name in ("model", "optimizer", "rng", "counters", "extra"):
        assert_same(load_checkpoint(full.weights / "state/temperature-last.pt")[name],
                    load_checkpoint(paused.weights / "state/temperature-last.pt")[name])


def test_temperature_resume_rejects_changed_uncalibrated_target(inputs, tmp_path):
    config, pipeline = inputs
    monitor = BudgetMonitor(tmp_path / "run")
    monitor.predicate = lambda step, epoch: monitor.phase == "state temperature" and step >= 2
    with pytest.raises(StagePaused):
        auxiliary.train_state(Data(), config, monitor, pipeline)
    path = monitor.weights / "state/best.pt"
    saved = load_checkpoint(path)
    key = next(name for name in saved["model"] if name.endswith("weight"))
    saved["model"][key] = saved["model"][key] + .1
    atomic_save(path, saved)
    monitor.predicate = lambda *args: False
    with pytest.raises(ValueError, match="CHECKPOINT_IDENTITY_MISMATCH"):
        auxiliary.train_state(Data(), config, monitor, pipeline, resume=True)
    assert not (monitor.weights / "state/calibrated.pt").exists()


@pytest.mark.parametrize("stage", ["pool", "state"])
def test_initial_auxiliary_budget_skips_preparation(inputs, tmp_path, stage):
    config, pipeline = inputs
    data = Data()
    monitor = BudgetMonitor(tmp_path / "run", lambda *args: True)
    with pytest.raises(StagePaused):
        getattr(auxiliary, f"train_{stage}")(data, config, monitor, pipeline)
    assert data.response_reads == data.state_reads == []
    assert not list(monitor.weights.rglob("*.pt"))
    assert not list(monitor.experiment.rglob("*.pt"))


@pytest.mark.parametrize("stage", ["pool", "state"])
def test_modular_preparation_preserves_legacy_training_result(inputs, tmp_path, stage):
    config, pipeline = inputs
    legacy, modular = Monitor(tmp_path / "legacy"), BudgetMonitor(tmp_path / "modular")
    old = getattr(auxiliary, f"train_{stage}")(Data(), config, legacy, pipeline)
    new = getattr(auxiliary, f"train_{stage}")(Data(), config, modular, pipeline)
    for key in ("model", "optimizer", "rng", "counters", "epoch", "identity"):
        assert_same(load_checkpoint(old[stage])[key], load_checkpoint(new[stage])[key])
