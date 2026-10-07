"""Decision-stage orchestration against an explicitly synthetic callback environment."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import decision, potential
from lung.engine.checkpoint import load_checkpoint
from lung.rl import (Candidate, CandidateRegistry, DecisionState, NetworkSpec, PotentialNetwork,
                     ProtocolSpec, RootContext, SimulationAdapter, VersionStamp)


class Monitor:
    def __init__(self, root, fail=None):
        self.weights = root / "weights"
        self.experiment = root / "experiment"
        self.weights.mkdir(parents=True, exist_ok=True)
        self.experiment.mkdir(parents=True, exist_ok=True)
        self.records, self.messages, self.fail = [], [], fail

    def progress(self, iterable=None, **kwargs):
        return tqdm(iterable, disable=True, **kwargs)

    def say(self, message):
        self.messages.append(message)

    def record(self, stage, step, metrics, axis="epoch"):
        self.records.append((stage, step, dict(metrics), axis))
        if self.fail == (axis, step):
            raise KeyboardInterrupt("Synthetic interruption at an explicitly tested boundary")


@pytest.fixture
def synthetic_world(monkeypatch):
    torch.set_num_threads(1)
    spec = NetworkSpec(state_dim=16, drug_dim=4, dose_dim=2, time_dim=2,
                       history_dim=4, history_start=5, protocol_dim=2,
                       encoder_hidden=(12, 8), scorer_hidden=(10, 6), critic_hidden=(12, 8))
    candidates = [Candidate("low", "drug", "low", "1", 1, torch.tensor([1., 0, 0, 0, 1, 0, 1, 0])),
                  Candidate("high", "drug", "high", "1", 1, torch.tensor([1., 0, 0, 0, 0, 1, 1, 0])),
                  Candidate("STOP", "STOP", "", "", 0, torch.zeros(8), True)]
    registry = CandidateRegistry(candidates, spec)
    protocol = ProtocolSpec(2, "active", frozenset({"active"}),
                            {("active", "low"): "active", ("active", "high"): "active"},
                            {("active", 0): "tail-2", ("active", 1): "tail-1"}, "test-protocol")
    versions = VersionStamp("test-biology", protocol.version, "test-proxy", registry.version)
    calls = []

    def observe(root, prefix, seed):
        elapsed = len(prefix)
        features = torch.arange(16).float()/100
        features[0] = elapsed/2
        history = torch.stack([torch.cat((registry[aid].embedding,
                                         torch.tensor([i/2, .1, .2]))) for i, aid in enumerate(prefix)]) \
            if prefix else torch.empty(0, spec.history_input_dim)
        calls.append(("observe", root.role, prefix, seed))
        return DecisionState(features, history, prefix, elapsed, root.root_id, root.role)

    def score(root, sequence, seeds):
        calls.append(("score", root.role, sequence, seeds))
        return torch.tensor([.2 + .2*sequence.count("high") for _ in seeds])

    adapter = SimulationAdapter(registry, protocol, versions, observe, score,
        lambda state: .2+.1*state.elapsed_ticks, "PROXY-A549-V1", True, True, 3,
        {"synthetic_test": True, "transfer_status": "synthetic"})
    world = SimpleNamespace(registry=registry, adapter=adapter, biological_identity="test-biology",
                            roots=lambda role: [RootContext(f"{role}-root-{i}", role) for i in range(2)])
    monkeypatch.setattr(decision, "BiologicalWorld", lambda *args, **kwargs: world)
    monkeypatch.setattr(potential, "_prepare_world", lambda data, config, monitor, pipeline: (world, dict(pipeline)))
    config = {"task": "PROXY-A549-V1", "device": "cpu", "epochs": 1, "seed": 31,
              "synthetic": True, "policy_seeds": [0], "population": 3,
              "scoring_repeats": 2, "save_period": 1, "validation_candidates": 4,
              "decision_validation_repeats": 2, "diagnostic_repeats": 2,
              "potential_train_prefixes": 4, "potential_val_prefixes": 2,
              "potential_continuations": 2, "potential_batch": 2,
              "daac": {"episodes_per_epoch": 4, "warmup_transitions": 2, "batch_size": 2,
                       "replay_capacity": 32, "validation_roots": 2, "checkpoint_units": 1},
              "dlpa": {"roots_per_epoch": 2, "distill_updates": 2, "batch_size": 2,
                       "iterations": 2, "candidates": 4, "elites": 2, "validation_roots": 2}}
    data = SimpleNamespace(identity="SYNTHETIC-TEST", synthetic=True)
    return world, config, data, calls


def test_potential_fixed_reference_cache_split_isolation_and_resume(synthetic_world, monkeypatch, tmp_path):
    world, config, data, calls = synthetic_world
    forward_roles = []

    class RecordingPotential(PotentialNetwork):
        def forward(self, states):
            forward_roles.append((self.training, tuple(s.role for s in states)))
            return super().forward(states)

    monkeypatch.setattr(potential, "PotentialNetwork", RecordingPotential)
    monitor = Monitor(tmp_path)
    pipeline = potential.train_potential(data, config, monitor, {})
    saved = load_checkpoint(pipeline["potential"])
    assert saved["epoch"] == 1 and saved["metadata"]["biological_identity"] == world.biological_identity
    assert saved["metadata"]["synthetic"] is True
    assert all(all(role == ("T" if fitting else "V") for role in roles) for fitting, roles in forward_roles)
    assert {call[1] for call in calls} == {"T", "V"}
    cache_files = list((monitor.experiment / "cache").glob("potential-*.pt"))
    assert len(cache_files) == 2
    rows = [row for path in cache_files for row in torch.load(path, weights_only=False)["rows"]]
    assert len(rows) == 6 and all(row["continuations"] == 2 for row in rows)
    calls_before = len(calls)
    extended = {**config, "epochs": 2}
    potential.train_potential(data, extended, monitor, pipeline, resume=True)
    assert len(calls) == calls_before  # Frozen target cache is reused; V never becomes fit.
    last = load_checkpoint(monitor.weights / "potential/last.pt")
    assert last["epoch"] == 2 and last["counters"]["optimizer_step"] == 4


@pytest.mark.parametrize("stage", ["daac", "dlpa"])
def test_decision_epochs_real_counters_curves_and_resume(synthetic_world, tmp_path, stage):
    _, config, data, _ = synthetic_world
    monitor = Monitor(tmp_path)
    pipeline = decision.train_decision(stage, data, config, monitor, {})
    assert isinstance(pipeline[stage], list) and len(pipeline[stage]) == 1
    best = load_checkpoint(pipeline[stage][0])
    assert best["metadata"]["stage"] == stage and best["metadata"]["train_role"] == "T"
    assert best["metadata"]["selection_role"] == "V" and best["epoch"] == 1
    assert "learner" in best and best["extra"]["cursor"]["unit"] == 0
    checkpoint = monitor.weights / stage / "seed_0/last.pt"
    old = load_checkpoint(checkpoint)
    if stage == "daac":
        assert old["counters"]["episodes"] == 4
        assert old["counters"]["critic_step"] == old["counters"]["total_transitions"] - 2
    else:
        assert old["counters"]["root_count"] == 2
        assert old["counters"]["plan_iteration"] == 4
        assert old["counters"]["distill_step"] == 2
    epoch_metrics = [row[2] for row in monitor.records if row[3] == "epoch"]
    assert len(epoch_metrics) == 1 and "val/score" in epoch_metrics[0]
    decision.train_decision(stage, data, {**config, "epochs": 2}, monitor, pipeline, resume=True)
    new = load_checkpoint(checkpoint)
    assert new["epoch"] == 2
    assert new["counters"]["episodes" if stage == "daac" else "root_count"] == (8 if stage == "daac" else 4)
    assert (checkpoint.parent / "epoch_2.pt").exists()
    with pytest.raises(ValueError, match="CHECKPOINT_IDENTITY_MISMATCH"):
        decision.train_decision(stage, data, {**config, "seed": 999}, monitor, pipeline, resume=True)


@pytest.mark.parametrize("stage,fail", [("daac", ("env_step", 3)), ("dlpa", ("plan_iteration", 3))])
def test_interruption_resumes_committed_complete_units_exactly(synthetic_world, tmp_path, stage, fail):
    _, config, data, _ = synthetic_world
    uninterrupted = Monitor(tmp_path / "uninterrupted")
    decision.train_decision(stage, data, config, uninterrupted, {})
    interrupted = Monitor(tmp_path / "resumed", fail=fail)
    with pytest.raises(KeyboardInterrupt):
        decision.train_decision(stage, data, config, interrupted, {})
    committed = load_checkpoint(interrupted.weights / stage / "seed_0/last.pt")
    assert committed["epoch"] == 0 and committed["extra"]["cursor"]["unit"] < (
        config[stage]["episodes_per_epoch"] if stage == "daac" else config[stage]["roots_per_epoch"])
    interrupted.fail = None
    decision.train_decision(stage, data, config, interrupted, {}, resume=True)
    complete1 = load_checkpoint(uninterrupted.weights / stage / "seed_0/last.pt")
    complete2 = load_checkpoint(interrupted.weights / stage / "seed_0/last.pt")
    assert complete1["counters"] == complete2["counters"]
    assert all(torch.equal(value, complete2["model"][key]) for key, value in complete1["model"].items())


def test_five_seed_default_and_no_test_role_training(synthetic_world, monkeypatch, tmp_path):
    world, config, data, _ = synthetic_world
    seen = []
    monkeypatch.setattr(decision, "_train_seed", lambda stage, data, cfg, monitor, w, roots, seed, resume:
                        seen.append((seed, tuple(root.role for root in roots))) or f"seed-{seed}.pt")
    formal = {key: value for key, value in config.items() if key != "policy_seeds"}
    formal["synthetic"] = False
    result = decision.train_decision("daac", data, formal, Monitor(tmp_path), {})
    assert [seed for seed, _ in seen] == [0, 1, 2, 3, 4]
    assert all(set(roles) == {"T"} for _, roles in seen)
    assert len(result["daac"]) == 5
