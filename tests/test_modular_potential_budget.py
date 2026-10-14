"""CPU coverage for persistent reference prefixes and potential training resume."""

from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import potential
from lung.engine.checkpoint import atomic_save, load_checkpoint
from lung.engine.modular_budget import StagePaused
from lung.engine.training_control import checkpoint_completion
from test_decision_engine_v1 import Monitor, synthetic_world
from conftest import assert_same


pytestmark = pytest.mark.usefixtures("cpu_only")


class BudgetMonitor(Monitor):
    def __init__(self, root, limit=None, training_only=False):
        super().__init__(root)
        self.limit = limit
        self.training_only = training_only
        self.in_training = False

    def progress(self, iterable=None, **kwargs):
        self.in_training = kwargs.get("desc", "").startswith("potential epoch")
        return iterable

    def stop_check(self, step, epoch):
        if self.training_only and not self.in_training:
            return False
        return "test_budget" if self.limit is not None and step >= self.limit else None


def assert_rows_equal(left, right):
    assert len(left) == len(right)
    for expected, actual in zip(left, right):
        assert_same({**expected, "state": asdict(expected["state"])},
                    {**actual, "state": asdict(actual["state"])})


def test_prefix_pause_reuses_committed_rows_and_exact_reference_stream(synthetic_world, tmp_path):
    world, config, _, calls = synthetic_world
    expected = potential._cache_rows(world, "T", 5, config, Monitor(tmp_path / "full"), "cache")
    calls.clear()
    monitor = BudgetMonitor(tmp_path / "resumed", limit=2)
    with pytest.raises(StagePaused) as caught:
        potential._cache_rows(world, "T", 5, config, monitor, "cache")
    parts = monitor.experiment / "cache/potential-T-cache.parts"
    assert caught.value.stage == "potential/cache/T"
    assert caught.value.cursor == 2
    assert Path(caught.value.checkpoint_path) == parts / "manifest.pt"
    assert len(list(parts.glob("row-*.pt"))) == 2
    assert not (monitor.experiment / "cache/potential-T-cache.pt").exists()
    first_seeds = [call[3] for call in calls if call[0] == "observe"]
    assert first_seeds == [row["planning_seed"] for row in expected[:2]]

    calls.clear()
    monitor.limit = None
    actual = potential._cache_rows(world, "T", 5, config, monitor, "cache")
    assert_rows_equal(expected, actual)
    assert [call[3] for call in calls if call[0] == "observe"] == [
        row["planning_seed"] for row in expected[2:]]
    calls.clear()
    assert_rows_equal(actual, potential._cache_rows(world, "T", 5, config, monitor, "cache"))
    assert calls == []


def test_zero_generation_budget_writes_no_targets_or_weights(synthetic_world, tmp_path):
    world, config, data, calls = synthetic_world
    monitor = BudgetMonitor(tmp_path, limit=0)
    with pytest.raises(StagePaused):
        potential.train_potential(data, config, monitor, {})
    assert calls == []
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("changed", ["identity", "count", "row_role", "row_index"])
def test_partial_prefix_cache_rejects_incompatible_content(synthetic_world, tmp_path, changed):
    world, config, _, calls = synthetic_world
    monitor = BudgetMonitor(tmp_path, limit=1)
    with pytest.raises(StagePaused):
        potential._cache_rows(world, "T", 4, config, monitor, "cache")
    parts = monitor.experiment / "cache/potential-T-cache.parts"
    path = parts / ("row-00000000.pt" if changed.startswith("row_") else "manifest.pt")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if changed == "identity":
        saved["identity"] = "other"
    elif changed == "count":
        saved["count"] += 1
    elif changed == "row_role":
        saved["row"]["role"] = "V"
    else:
        saved["row"]["generation_index"] += 1
    atomic_save(path, saved)
    calls.clear()
    monitor.limit = None
    with pytest.raises(ValueError, match="POTENTIAL_CACHE_"):
        potential._cache_rows(world, "T", 4, config, monitor, "cache")
    assert calls == []


@pytest.mark.parametrize("stop_step", [1, 2, 3])
def test_potential_update_resume_matches_continuous_and_keeps_roles(synthetic_world, tmp_path, stop_step):
    _, config, data, calls = synthetic_world
    config = {**config, "epochs": 2}
    full = BudgetMonitor(tmp_path / "full")
    potential.train_potential(data, config, full, {})
    paused = BudgetMonitor(tmp_path / "resumed", stop_step, training_only=True)
    with pytest.raises(StagePaused) as caught:
        potential.train_potential(data, config, paused, {})
    assert caught.value.stage == "potential"
    checkpoint = load_checkpoint(paused.weights / "potential/last.pt")
    assert checkpoint["counters"]["optimizer_step"] == stop_step
    with pytest.raises(ValueError, match="TRAINING_NOT_COMPLETE"):
        checkpoint_completion(checkpoint, 2)
    assert {call[1] for call in calls} == {"T", "V"}
    calls.clear()
    resumed = BudgetMonitor(tmp_path / "resumed")
    result = potential.train_potential(data, config, resumed, {}, resume=True)
    assert calls == []
    final = load_checkpoint(resumed.weights / "potential/last.pt")
    assert_same(final, load_checkpoint(full.weights / "potential/last.pt"))
    assert final["metadata"]["train_role"] == "T"
    assert final["metadata"]["selection_role"] == "V"
    assert checkpoint_completion(final, 2)["actual_updates"] == 4
    assert Path(result["potential"]).is_file()


def test_failed_prefix_save_preserves_earlier_committed_rows(synthetic_world, tmp_path, monkeypatch):
    world, config, _, calls = synthetic_world
    monitor = BudgetMonitor(tmp_path)
    original_save = potential.atomic_save

    def fail_second_row(path, value):
        if Path(path).name == "row-00000001.pt":
            raise OSError("injected_save_failure")
        return original_save(path, value)

    monkeypatch.setattr(potential, "atomic_save", fail_second_row)
    with pytest.raises(OSError, match="injected_save_failure"):
        potential._cache_rows(world, "T", 4, config, monitor, "cache")
    first_seed = next(call[3] for call in calls if call[0] == "observe")
    monkeypatch.setattr(potential, "atomic_save", original_save)
    calls.clear()
    rows = potential._cache_rows(world, "T", 4, config, monitor, "cache")
    assert len(rows) == 4
    assert first_seed not in [call[3] for call in calls if call[0] == "observe"]


def test_modular_potential_updates_match_legacy_mse_objective(synthetic_world, tmp_path):
    _, config, data, _ = synthetic_world
    config = {**config, "epochs": 2}
    legacy, modular = Monitor(tmp_path / "legacy"), BudgetMonitor(tmp_path / "modular")
    potential.train_potential(data, config, legacy, {})
    potential.train_potential(data, config, modular, {})
    old = load_checkpoint(legacy.weights / "potential/last.pt")
    new = load_checkpoint(modular.weights / "potential/last.pt")
    for key in ("model", "optimizer", "rng", "counters", "epoch", "identity", "metadata"):
        assert_same(old[key], new[key])
    assert old["extra"]["best"] == new["extra"]["best"]
