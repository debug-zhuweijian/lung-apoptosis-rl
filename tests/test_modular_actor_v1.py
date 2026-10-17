"""Synthetic checks of one Actor, stage ownership and completed training."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import decision
from lung.engine.checkpoint import load_checkpoint
from lung.rl import DAACConfig, DAACTrainer, CandidateRegistry
from test_decision_engine_v1 import Monitor, synthetic_world
from test_rl_v1 import complete_episode, system


def learner_for(system, schedule=None, seed=4):
    registry, adapter, root, _ = system
    trainer = DAACTrainer(registry, adapter.versions,
        DAACConfig(batch_size=2, capacity=32, warmup_transitions=2, actor_delay=1),
        seed=seed, actor_schedule=schedule)
    complete_episode(trainer, adapter, root, 3, ("a_low_1", "STOP"))
    return trainer


@pytest.mark.parametrize("stage,active,frozen", [
    ("drug", ("encoder.", "history.", "drug.", "stop_token"), ("dose.", "time.")),
    ("dose", ("encoder.", "history.", "dose."), ("drug.", "time.", "stop_token")),
    ("joint", ("encoder.", "history.", "drug.", "dose.", "time."), ()),
])
def test_stage_changes_only_active_actor_and_updates_both_critics(system, stage, active, frozen):
    trainer = learner_for(system)
    trainer.update()  # Materialize Adam momentum before freezing a previously active head.
    optimizer = trainer.actor_optimizer
    trainer.set_actor_stage(stage)
    before = deepcopy(trainer.actor.state_dict())
    critic_before = [deepcopy(q.state_dict()) for q in trainer.critics]
    trainer.update()
    assert trainer.actor_optimizer is optimizer
    for name, value in trainer.actor.named_parameters():
        if name.startswith(frozen):
            assert torch.equal(value, before[name]) and value.grad is None
    for prefix in active:
        assert any(not torch.equal(value, before[name]) for name, value in trainer.actor.named_parameters()
                   if name.startswith(prefix))
    for q, old in zip(trainer.critics, critic_before):
        assert any(not torch.equal(value, old[name]) for name, value in q.named_parameters())


@pytest.mark.parametrize("stage,head", [("drug", "dose"), ("drug", "time"), ("dose", "drug")])
def test_frozen_head_keeps_gradient_to_private_features(system, stage, head):
    registry, adapter, root, _ = system
    trainer = learner_for(system)
    trainer.set_actor_stage(stage)
    inputs = []

    def capture(module, args, output):
        args[0].retain_grad()
        inputs.append(args[0])

    hook = getattr(trainer.actor, head).register_forward_hook(capture)
    state = adapter.observe(root, ("a_low_1",), 123)
    p = trainer.actor([state], torch.ones(1, len(registry), dtype=torch.bool))
    (p * torch.arange(len(registry)).float()).sum().backward()
    hook.remove()
    latent = registry.spec.encoder_hidden[-1]
    assert any(x.grad is not None and x.grad[..., :latent].abs().sum() > 0 for x in inputs)
    assert all(p.grad is None for p in getattr(trainer.actor, head).parameters())
    assert state.features.grad is None and state.history.grad is None


def test_schedule_counts_actual_actor_updates_and_redistributes_missing_branches(system):
    registry, adapter, _, _ = system
    candidates = [a for a in registry.candidates if a.dose_id != "high" and a.time_id != "2"]
    reduced = CandidateRegistry(candidates, registry.spec)
    trainer = DAACTrainer(reduced, replace(adapter.versions, candidates=reduced.version),
                          actor_schedule={"drug": 2, "dose": 3, "joint": 4})
    assert trainer.actor_schedule.quotas == {"drug": 2, "dose": 0, "joint": 7}
    assert not trainer.actor_schedule.choices["dose"] and not trainer.actor_schedule.choices["time"]
    assert all(not p.requires_grad for p in trainer.actor.time.parameters())
    delayed = learner_for(system, {"drug": 1, "dose": 1, "joint": 1})
    delayed.config = replace(delayed.config, actor_delay=2)
    first = delayed.update()
    assert first["critic_step"] == 1 and delayed.actor_step == 0 and delayed.actor_stage == "drug"
    second = delayed.update()
    assert all(isinstance(value, (int, float)) for value in second.values())
    assert delayed.actor_step == 1 and delayed.actor_stage == "dose"


def scheduled_config(config):
    return {**config, "epochs": 20, "daac": {**config["daac"], "actor_delay": 1,
        "actor_schedule": {"drug": 1, "dose": 1, "joint": 2}}}


def test_engine_stage_quota_complete_and_model_export(synthetic_world, tmp_path):
    _, config, data, _ = synthetic_world
    config = scheduled_config(config)
    monitor = Monitor(tmp_path)
    pipeline = decision.train_decision("daac", data, config, monitor, {})
    final = load_checkpoint(monitor.weights / "daac/seed_0/final.pt")
    assert final["counters"]["actor_step"] == 4
    assert final["extra"]["training_completion"]["reason"] == "actor_schedule_complete"
    assert final["extra"]["actor_schedule"]["completed"] == {"drug": 1, "dose": 1, "joint": 2}
    assert load_checkpoint(pipeline["daac"][0])["counters"]["actor_step"] > 0
    assert not {"optimizer", "replay", "rng", "planner"} & set(final)
    assert not {"actor_optimizer", "critic_optimizer", "replay", "global_rng"} & set(final["learner"])
