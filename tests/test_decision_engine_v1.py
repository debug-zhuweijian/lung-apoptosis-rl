"""Decision-stage orchestration against an explicitly synthetic callback environment."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import decision
from lung.engine.checkpoint import load_checkpoint
from lung.rl import (Candidate, CandidateRegistry, DecisionState, NetworkSpec,
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
    config = {"task": "PROXY-A549-V1", "device": "cpu", "epochs": 1, "seed": 31,
              "synthetic": True, "policy_seeds": [0], "population": 3,
              "scoring_repeats": 2, "validation_candidates": 4,
              "decision_validation_repeats": 2, "diagnostic_repeats": 2,
              "daac": {"episodes_per_epoch": 4, "warmup_transitions": 2, "batch_size": 2,
                       "replay_capacity": 32, "validation_roots": 2, "checkpoint_units": 1},
              "dlpa": {"roots_per_epoch": 2, "distill_updates": 2, "batch_size": 2,
                       "iterations": 2, "candidates": 4, "elites": 2, "validation_roots": 2}}
    data = SimpleNamespace(identity="SYNTHETIC-TEST", synthetic=True)
    return world, config, data, calls


@pytest.mark.parametrize("stage", ["daac", "dlpa"])
def test_decision_epochs_real_counters_curves_and_fresh_restart(synthetic_world, tmp_path, stage):
    _, config, data, _ = synthetic_world
    monitor = Monitor(tmp_path)
    pipeline = decision.train_decision(stage, data, config, monitor, {})
    assert isinstance(pipeline[stage], list) and len(pipeline[stage]) == 1
    best = load_checkpoint(pipeline[stage][0])
    assert best["metadata"]["stage"] == stage and best["metadata"]["train_role"] == "T"
    assert best["metadata"]["selection_role"] == "V" and best["epoch"] == 1
    assert "learner" in best and "cursor" not in best["extra"]
    assert best["extra"]["training_completion"]["status"] == "complete"
    checkpoint = monitor.weights / stage / "seed_0/final.pt"
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
    decision.train_decision(stage, data, {**config, "epochs": 2}, monitor, pipeline)
    new = load_checkpoint(checkpoint)
    assert new["epoch"] == 2
    assert new["counters"]["episodes" if stage == "daac" else "root_count"] == (8 if stage == "daac" else 4)
    assert not (checkpoint.parent / "epoch_2.pt").exists()
    decision.train_decision(stage, data, config, monitor, pipeline)
    fresh = load_checkpoint(checkpoint)
    assert fresh["counters"] == old["counters"]
    for name, value in old["model"].items():
        torch.testing.assert_close(value, fresh["model"][name], atol=0, rtol=0)


def test_five_seed_default_and_no_test_role_training(synthetic_world, monkeypatch, tmp_path):
    world, config, data, _ = synthetic_world
    seen = []
    monkeypatch.setattr(decision, "_train_seed", lambda stage, data, cfg, monitor, w, roots, seed:
                        seen.append((seed, tuple(root.role for root in roots))) or f"seed-{seed}.pt")
    formal = {key: value for key, value in config.items() if key != "policy_seeds"}
    formal["synthetic"] = False
    result = decision.train_decision("daac", data, formal, Monitor(tmp_path), {})
    assert [seed for seed, _ in seen] == [0, 1, 2, 3, 4]
    assert all(set(roles) == {"T"} for _, roles in seen)
    assert len(result["daac"]) == 5
