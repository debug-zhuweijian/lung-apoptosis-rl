"""Bounded CPU checks for resumable auxiliary-stage budget stops."""

import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine.auxiliary import fit_epochs
from lung.engine.checkpoint import load_checkpoint, seed_all
from lung.engine.modular_budget import StagePaused
from lung.engine.training_control import checkpoint_completion


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(threads)


class Monitor:
    def __init__(self, directory, limit=None):
        self.weights = directory
        self.limit = limit
        self.records = []

    def progress(self, values, **kwargs):
        return values

    def record(self, *values):
        self.records.append(values)

    def stop_check(self, step, epoch):
        return "test_budget" if self.limit is not None and step >= self.limit else None


def fixture_inputs():
    seed_all(71)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.25),
                                torch.nn.Linear(4, 1))
    rows = [torch.tensor([float(i), .1 * i]) for i in range(5)]
    config = {"epochs": 2, "seed": 9, "aux_batch": 2, "aux_lr": .01,
              "training_control": {"checkpoint_every_updates": 1}, "synthetic": True}
    return model, rows, config


def scalar_loss(model, row):
    noise = random.random() + float(np.random.rand()) + torch.rand(())
    return (model(row).squeeze() - noise).square()


def batch_loss(model, rows):
    noise = random.random() + float(np.random.rand()) + torch.rand(len(rows))
    return (model(torch.stack(rows)).squeeze(-1) - noise).square()


def validation(model):
    noise = random.random() + float(np.random.rand()) + torch.rand(())
    return {"val/loss": float((model(torch.ones(1, 2)).squeeze() - noise).square())}


def run(model, rows, config, monitor, **kwargs):
    return fit_epochs("state", model, rows, scalar_loss, SimpleNamespace(identity="data"),
                      config, monitor, "fixture", {}, validation=validation, **kwargs)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_same(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for one, other in zip(left, right):
            assert_same(one, other)
    else:
        assert left == right


@pytest.mark.parametrize("stop_step", [1, 3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_pause_resume_matches_continuous_optimizer_rng_and_control(tmp_path, stop_step, batched):
    initial, rows, config = fixture_inputs()
    options = {"batch_loss": batch_loss} if batched else {}
    full = copy.deepcopy(initial)
    full_monitor = Monitor(tmp_path / "full")
    seed_all(89)
    run(full, rows, config, full_monitor, **options)

    paused = copy.deepcopy(initial)
    paused_monitor = Monitor(tmp_path / "resumed", stop_step)
    seed_all(89)
    with pytest.raises(StagePaused) as caught:
        run(paused, rows, config, paused_monitor, **options)
    saved = load_checkpoint(paused_monitor.weights / "state/last.pt")
    assert caught.value.reason == "test_budget"
    assert caught.value.step == stop_step
    assert caught.value.checkpoint_path == str(paused_monitor.weights / "state/last.pt")
    assert saved["counters"]["optimizer_step"] == stop_step
    assert saved["epoch"] == (stop_step - 1) // 3
    assert saved["extra"]["update_resume"]["cursor"] == min(((stop_step - 1) % 3 + 1) * 2, 5)
    assert saved["extra"]["training_control"]["status"] != "complete"
    with pytest.raises(ValueError, match="TRAINING_NOT_COMPLETE"):
        checkpoint_completion(saved, config["epochs"])

    # Model construction and unrelated random draws cannot alter resumed training.
    resumed = copy.deepcopy(initial)
    seed_all(321)
    resumed_monitor = Monitor(paused_monitor.weights)
    run(resumed, rows, config, resumed_monitor, resume=True, **options)
    final = load_checkpoint(resumed_monitor.weights / "state/last.pt")
    assert_same(final, load_checkpoint(full_monitor.weights / "state/last.pt"))
    assert_same(resumed.state_dict(), full.state_dict())
    assert paused_monitor.records + resumed_monitor.records == full_monitor.records
    assert "update_resume" not in final["extra"]
    assert checkpoint_completion(final, config["epochs"])["actual_updates"] == 6


def test_initial_budget_exhaustion_does_not_train_or_save_random_weights(tmp_path):
    model, rows, config = fixture_inputs()
    initial = copy.deepcopy(model.state_dict())
    monitor = Monitor(tmp_path, 0)
    with pytest.raises(StagePaused) as caught:
        run(model, rows, config, monitor)
    assert caught.value.step == 0
    assert caught.value.checkpoint_path is None
    assert_same(model.state_dict(), initial)
    assert not list(tmp_path.rglob("*.pt"))
    assert monitor.records == []


@pytest.mark.parametrize("changed", ["rows", "batch_size"])
def test_partial_resume_rejects_changed_batch_geometry(tmp_path, changed):
    model, rows, config = fixture_inputs()
    with pytest.raises(StagePaused):
        run(model, rows, config, Monitor(tmp_path, 1))
    if changed == "rows":
        rows = rows[:-1]
    else:
        config = {**config, "aux_batch": 3}
    with pytest.raises(ValueError, match="AUX_UPDATE_RESUME_MISMATCH"):
        run(model, rows, config, Monitor(tmp_path), resume=True)


def test_update_checkpoint_survives_failure_before_next_update(tmp_path):
    model, rows, config = fixture_inputs()
    calls = 0

    def failing_batch(network, selected):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected_batch_failure")
        return batch_loss(network, selected)

    with pytest.raises(RuntimeError, match="injected_batch_failure"):
        run(model, rows, config, Monitor(tmp_path), batch_loss=failing_batch)
    saved = load_checkpoint(tmp_path / "state/last.pt")
    assert saved["epoch"] == 0
    assert saved["counters"]["optimizer_step"] == 1
    assert saved["extra"]["update_resume"]["cursor"] == 2
    with pytest.raises(ValueError, match="TRAINING_NOT_COMPLETE"):
        checkpoint_completion(saved, config["epochs"])
