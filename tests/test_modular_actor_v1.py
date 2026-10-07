"""Bounded synthetic checks of one Actor, stage ownership and exact continuation."""
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


@pytest.mark.parametrize("boundary", [1, 2])
def test_stage_boundary_resume_matches_next_update_and_optimizer(system, boundary):
    schedule = {"drug": 1, "dose": 1, "joint": 2}
    trainer = learner_for(system, schedule)
    for _ in range(boundary):
        trainer.update()
    saved = deepcopy(trainer.state_dict())
    restored = learner_for(system, schedule, seed=999)
    restored.load_state_dict(saved)
    assert trainer.actor_stage == restored.actor_stage == ("dose" if boundary == 1 else "joint")
    left, right = trainer.update(), restored.update()
    assert left == right
    assert trainer.actor_step == restored.actor_step == boundary + 1
    for component in ("actor", "critics", "targets"):
        assert all(torch.equal(value, getattr(restored, component).state_dict()[name])
                   for name, value in getattr(trainer, component).state_dict().items())
    assert trainer.actor_schedule.state_dict() == restored.actor_schedule.state_dict()


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


def test_actor_initialization_imports_weights_only_and_rejects_identity_changes(system):
    source = learner_for(system, {"drug": 1, "dose": 1, "joint": 1})
    source.update()
    registry, adapter, _, _ = system
    config = replace(source.config, actor_lr=.003)
    fresh = DAACTrainer(registry, adapter.versions, config, seed=9,
                        actor_schedule={"drug": 2, "dose": 2, "joint": 2})
    old_critics = deepcopy(fresh.critics.state_dict())
    old_rng = fresh.generator.get_state().clone()
    fresh.initialize_actor(deepcopy(source.state_dict()))
    assert len(fresh.replay) == fresh.actor_step == fresh.critic_step == fresh.total_transitions == 0
    assert not fresh.actor_optimizer.state and not fresh.critic_optimizer.state
    assert torch.equal(old_rng, fresh.generator.get_state())
    assert all(torch.equal(value, old_critics[name]) for name, value in fresh.critics.state_dict().items())
    assert all(torch.equal(value, source.actor.state_dict()[name]) for name, value in fresh.actor.state_dict().items())
    bad = deepcopy(source.state_dict())
    bad["versions"] = replace(adapter.versions, environment="different-world")
    with pytest.raises(ValueError, match="ACTOR_INITIALIZATION"):
        fresh.initialize_actor(bad)
    bad = deepcopy(source.state_dict())
    bad["feature_layout"] = "different-feature-order"
    with pytest.raises(ValueError, match="ACTOR_INITIALIZATION"):
        fresh.initialize_actor(bad)
    bad = deepcopy(source.state_dict())
    bad["actor"]["action_features"] = bad["actor"]["action_features"].flip(0)
    with pytest.raises(ValueError, match="ACTOR_INITIALIZATION"):
        fresh.initialize_actor(bad)


def scheduled_config(config):
    return {**config, "epochs": 20, "daac": {**config["daac"], "actor_delay": 1,
        "actor_schedule": {"drug": 1, "dose": 1, "joint": 2}, "max_seconds": 30.}}


def test_engine_stage_quota_complete_and_strict_resume(synthetic_world, tmp_path):
    _, config, data, _ = synthetic_world
    config = scheduled_config(config)
    monitor = Monitor(tmp_path)
    pipeline = decision.train_decision("daac", data, config, monitor, {})
    last = load_checkpoint(monitor.weights / "daac/seed_0/last.pt")
    assert last["counters"]["actor_step"] == 4
    assert last["extra"]["decision_runtime"]["status"] == "complete"
    runtime = last["extra"]["decision_runtime"]
    assert runtime["validation_seconds"] > 0 and runtime["checkpoint_seconds"] > 0
    assert sum(runtime["stage_seconds"].values()) < runtime["elapsed_seconds"]
    assert runtime["training_mc_draws"] == runtime["training_feedback_calls"] * config["scoring_repeats"]
    assert last["learner"]["actor_schedule"]["completed"] == {"drug": 1, "dose": 1, "joint": 2}
    assert load_checkpoint(pipeline["daac"][0])["counters"]["actor_step"] > 0
    with pytest.raises(ValueError, match="INIT_FROM_RESUME_MUTUALLY_EXCLUSIVE"):
        decision.train_decision("daac", data, {**config, "daac": {**config["daac"],
            "init_from": pipeline["daac"][0]}}, monitor, {}, resume=True)


def test_engine_budget_exhaustion_can_resume_without_restarting_schedule(synthetic_world, tmp_path):
    _, config, data, _ = synthetic_world
    config = scheduled_config(config)
    config["daac"]["max_seconds"] = 1e-9
    monitor = Monitor(tmp_path)
    pipeline = decision.train_decision("daac", data, config, monitor, {})
    last = load_checkpoint(pipeline["daac"][0])
    paused = last["extra"]["decision_runtime"]
    assert paused["status"] == "paused" and paused["reason"] == "budget_exhausted"
    assert last["counters"]["actor_step"] == 0
    result = decision.train_decision("daac", data, config, monitor, pipeline, resume=True)
    assert load_checkpoint(result["daac"][0])["counters"]["actor_step"] == 0
    extended = {**config, "daac": {**config["daac"], "max_seconds": 30. + 1e-9,
                                "resume_budget_seconds": 30.}}
    decision.train_decision("daac", data, extended, monitor, pipeline, resume=True)
    resumed = load_checkpoint(monitor.weights / "daac/seed_0/last.pt")
    assert resumed["counters"]["actor_step"] == 4
    assert resumed["extra"]["decision_runtime"]["status"] == "complete"
    assert resumed["extra"]["decision_runtime"]["elapsed_seconds"] >= paused["elapsed_seconds"]
    assert resumed["extra"]["decision_runtime"]["budget_seconds"] == pytest.approx(30. + 1e-9)


@pytest.mark.parametrize("boundary", [1, 2])
def test_engine_stage_boundary_interruption_matches_continuous_run(synthetic_world, monkeypatch, tmp_path, boundary):
    world, config, data, _ = synthetic_world
    protocol = replace(world.adapter.protocol, horizon_ticks=1, stop_tails={("active", 0): "tail-1"})
    world.adapter = replace(world.adapter, protocol=protocol)
    config = scheduled_config(config)
    full = Monitor(tmp_path / "full")
    decision.train_decision("daac", data, config, full, {})
    interrupted = Monitor(tmp_path / "interrupted")
    original = decision._boundary_save

    def interrupt_after_durable_boundary(*args, **kwargs):
        original(*args, **kwargs)
        if args[2].actor_step == boundary:
            raise KeyboardInterrupt("Test interruption after a committed Actor boundary")

    monkeypatch.setattr(decision, "_boundary_save", interrupt_after_durable_boundary)
    with pytest.raises(KeyboardInterrupt):
        decision.train_decision("daac", data, config, interrupted, {})
    committed = load_checkpoint(interrupted.weights / "daac/seed_0/last.pt")
    assert committed["counters"]["actor_step"] == boundary
    monkeypatch.setattr(decision, "_boundary_save", original)
    decision.train_decision("daac", data, config, interrupted, {}, resume=True)
    continuous = load_checkpoint(full.weights / "daac/seed_0/last.pt")
    resumed = load_checkpoint(interrupted.weights / "daac/seed_0/last.pt")
    assert continuous["counters"] == resumed["counters"]
    assert continuous["learner"]["actor_stage_events"] == resumed["learner"]["actor_stage_events"]
    for part in ("actor", "critics", "targets"):
        assert all(torch.equal(value, resumed["learner"][part][key])
                   for key, value in continuous["learner"][part].items())


def test_engine_init_from_starts_fresh_and_then_resumes_its_own_experiment(synthetic_world, tmp_path):
    _, config, data, _ = synthetic_world
    config = scheduled_config(config)
    source_monitor = Monitor(tmp_path / "source")
    source_path = decision.train_decision("daac", data, config, source_monitor, {})["daac"][0]
    source = load_checkpoint(source_path)
    new_monitor = Monitor(tmp_path / "new")
    new_config = {**config, "daac": {**config["daac"], "max_seconds": 1e-9, "init_from": source_path}}
    initialized_path = decision.train_decision("daac", data, new_config, new_monitor, {})["daac"][0]
    initialized = load_checkpoint(initialized_path)
    assert initialized["counters"]["actor_step"] == 0
    assert initialized["learner"]["replay"]["records"] == []
    assert not initialized["learner"]["actor_optimizer"]["state"]
    assert initialized["metadata"]["initialization"]["components"] == ["actor"]
    assert all(torch.equal(value, source["model"][name]) for name, value in initialized["model"].items())
    continued = {**new_config, "daac": {key: value for key, value in new_config["daac"].items() if key != "init_from"}}
    continued["daac"]["resume_budget_seconds"] = 30.
    decision.train_decision("daac", data, continued, new_monitor, {}, resume=True)
    complete = load_checkpoint(new_monitor.weights / "daac/seed_0/last.pt")
    assert complete["identity"] == initialized["identity"]
    assert complete["counters"]["actor_step"] == 4


def test_resume_rejects_candidate_buffers_and_changed_schedule(system):
    original = learner_for(system, {"drug": 1, "dose": 1, "joint": 1})
    original.update()
    changed = learner_for(system, {"drug": 1, "dose": 1, "joint": 2})
    with pytest.raises(ValueError, match="SCHEDULE_RESUME_MISMATCH"):
        changed.load_state_dict(deepcopy(original.state_dict()))
    saved = deepcopy(original.state_dict())
    saved["actor"]["action_features"] = saved["actor"]["action_features"].flip(0)
    same = learner_for(system, {"drug": 1, "dose": 1, "joint": 1})
    with pytest.raises(ValueError, match="ACTOR_RESUME_CANDIDATE_ENCODING_MISMATCH"):
        same.load_state_dict(saved)


def test_budget_pause_after_policy_update_preserves_exact_continuation(synthetic_world, monkeypatch, tmp_path):
    world, config, data, _ = synthetic_world
    world.adapter = replace(world.adapter, protocol=replace(world.adapter.protocol,
        horizon_ticks=1, stop_tails={("active", 0): "tail-1"}))
    config = scheduled_config(config)
    full = Monitor(tmp_path / "full")
    decision.train_decision("daac", data, config, full, {})
    interrupted = Monitor(tmp_path / "paused")
    original = decision._DecisionRuntime.exhausted

    def controlled_budget(runtime):
        if runtime.values["environment_units"] >= 3:
            runtime.budget.total = runtime.budget.summary()["consumed_seconds"]
            runtime.budget.exhausted = True
            return True
        return original(runtime)

    monkeypatch.setattr(decision._DecisionRuntime, "exhausted", controlled_budget)
    path = decision.train_decision("daac", data, config, interrupted, {})["daac"][0]
    paused = load_checkpoint(path)
    assert paused["counters"]["actor_step"] == 1
    assert paused["learner"]["actor_stage"] == "dose"
    assert paused["extra"]["decision_runtime"]["status"] == "paused"
    assert paused["extra"]["decision_runtime"]["budget_exhausted"]
    monkeypatch.setattr(decision._DecisionRuntime, "exhausted", original)
    extended = {**config, "daac": {**config["daac"], "resume_budget_seconds": 30.}}
    decision.train_decision("daac", data, extended, interrupted, {}, resume=True)
    continuous = load_checkpoint(full.weights / "daac/seed_0/last.pt")
    resumed = load_checkpoint(interrupted.weights / "daac/seed_0/last.pt")
    assert continuous["counters"] == resumed["counters"]
    for part in ("actor", "critics", "targets"):
        assert all(torch.equal(value, resumed["learner"][part][key])
                   for key, value in continuous["learner"][part].items())
