"""Small synthetic arithmetic/protocol tests, never biological efficacy experiments."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.rl import (
    AffineCritic, Candidate, CandidateRegistry, ConditionalActor, ConditionalTable,
    DAACConfig, DAACTrainer, DLPAConfig, DLPAPlanner, DecisionState, ModelRollout,
    NetworkSpec, ProposalDistiller, ProtocolSpec, RLUnavailable, RootContext,
    SimulationAdapter, VersionStamp, audit_winners, critic_mc_diagnostic,
    exact_time_ticks, generate_fixed_sequences, joint_kl, min_expected_target,
    precision_repeats, prefix_advantages, reevaluate_fixed_sequences,
)


@pytest.fixture
def system():
    torch.set_num_threads(1)
    spec = NetworkSpec(state_dim=16, drug_dim=4, dose_dim=2, time_dim=2,
                       history_dim=4, history_start=5, protocol_dim=2,
                       encoder_hidden=(12, 8), scorer_hidden=(10, 6), critic_hidden=(12, 8))
    candidates = []
    for drug, dose, time in (("a", "low", 1), ("a", "low", 2),
                             ("a", "high", 1), ("a", "high", 2), ("b", "low", 1)):
        embedding = torch.tensor(([1., 0., 0., 0.] if drug == "a" else [0., 1., 0., 0.])
                                 + ([1., 0.] if dose == "low" else [0., 1.])
                                 + [float(time), 1.])
        candidates.append(Candidate(f"{drug}_{dose}_{time}", drug, dose, str(time), time, embedding))
    candidates.append(Candidate("STOP", "STOP", "", "", 0, torch.zeros(8), True))
    registry = CandidateRegistry(candidates, spec)
    protocol = ProtocolSpec(2, "active", frozenset({"active"}),
                            {("active", a.action_id): "active" for a in candidates if not a.is_stop},
                            {("active", 0): "washout_2", ("active", 1): "washout_1"}, "protocol-v1")
    versions = VersionStamp("test-world-v1", protocol.version, "test-endpoint-v1", registry.version)
    calls = {"observe": [], "score": [], "phi": []}

    def observe(root, prefix, seed):
        calls["observe"].append((prefix, seed))
        _, elapsed, _ = protocol.position(registry, prefix)
        features = torch.arange(spec.state_dim).float() / 50
        features[0] = elapsed / 2
        # Deliberately contaminate the old slot to prove private networks replace it.
        features[spec.history_start:spec.history_start + spec.history_dim] = 999
        history, ticks = [], 0
        for aid in prefix:
            a = registry[aid]
            history.append(torch.cat((a.embedding, torch.tensor([ticks / 2., .3, .4]))))
            ticks += a.duration_ticks
        raw = torch.stack(history) if history else torch.empty(0, spec.history_input_dim)
        return DecisionState(features, raw, prefix, elapsed, root.root_id, root.role)

    def score(root, sequence, seeds):
        calls["score"].append((sequence, seeds))
        value = .1 + .3 * sum("high" in a for a in sequence) + .1 * sum(a.startswith("b") for a in sequence)
        return torch.full((len(seeds),), min(.95, value))

    def potential(state):
        calls["phi"].append(state.prefix)
        return .2 + .1 * state.elapsed_ticks

    adapter = SimulationAdapter(registry, protocol, versions, observe, score, potential,
                                "TEST-SYNTHETIC", True, True, 3, {"synthetic_test": True})
    return registry, adapter, RootContext("root1", "T"), calls


def complete_episode(trainer, adapter, root, seed, force=None):
    env = ModelRollout(adapter, root, seed, score_repeats=4)
    records, metrics = [], []
    while not env.done:
        index, probabilities = trainer.select(env.state, env.legal_mask)
        if force is not None:
            index = adapter.registry.by_id[force[len(records)]]
        record = env.step(index, probabilities, trainer.actor_step)
        records.append(record)
        metrics.append(trainer.observe(record))
    return records, metrics


def test_exact_integer_time_and_missing_protocol(system):
    assert exact_time_ticks(["0.5", "1.25", "24"]) == (4, (2, 5, 96))
    with pytest.raises(ValueError):
        exact_time_ticks([.5])
    _, adapter, _, _ = system
    with pytest.raises(RLUnavailable, match="PROTOCOL_INCOMPLETE"):
        replace(adapter, protocol=None).validate_ready()
    with pytest.raises(RLUnavailable, match="ENDPOINT_UNAVAILABLE"):
        replace(adapter, endpoint_available=False).validate_ready()
    with pytest.raises(RLUnavailable, match="POPULATION_RESOLUTION_UNRESOLVED"):
        replace(adapter, population_resolution_passed=False).validate_ready()
    with pytest.raises(RLUnavailable, match="POTENTIAL_UNAVAILABLE"):
        replace(adapter, potential_fn=None).validate_ready(require_potential=True)


def test_protocol_masks_noncompletable_suffix_and_missing_stop(system):
    registry, adapter, _, _ = system
    protocol = replace(adapter.protocol, stop_tails={},
                       transitions={("active", "a_low_1"): "dead", ("active", "a_low_2"): "active"})
    mask = protocol.legal_mask(registry, ())
    assert torch.where(mask)[0].tolist() == [registry.by_id["a_low_2"]]
    with pytest.raises(RLUnavailable):
        protocol.validate_complete(registry, ("STOP",))
    protocol.validate_complete(registry, ("a_low_2",))


def test_masked_joint_distribution_all_roles_and_private_history_gradients(system):
    registry, adapter, root, _ = system
    actor = ConditionalActor(registry)
    state = adapter.observe(root, ("a_low_1",), 12)
    masks = torch.tensor([[True, True, True, True, True, False]])
    p = actor([state], masks)
    assert float(p.sum().detach()) == pytest.approx(1.) and p[0, -1] == 0
    loss = (p * torch.tensor([[0., .8, .3, 1., -.2, 0.]])).sum()
    loss.backward()
    for component in (actor.drug, actor.dose, actor.time, actor.history.gru):
        assert sum(float(x.grad.abs().sum()) for x in component.parameters() if x.grad is not None) > 0
    assert state.features.grad is None and state.history.grad is None
    clean = replace(state, features=state.features.clone().index_fill(0, torch.arange(5, 9), -99))
    assert torch.allclose(actor([state], masks), actor([clean], masks))
    with pytest.raises(RLUnavailable, match="UNSUPPORTED_PROTOCOL"):
        actor([state], torch.zeros_like(masks))


@pytest.mark.parametrize("sequence", [("a_low_1", "STOP"), ("a_high_2",), ("STOP",)])
def test_reward_telescope_single_settlement_and_hidden_scores(system, sequence):
    registry, adapter, root, calls = system
    env = ModelRollout(adapter, root, 3, score_repeats=4)
    initial_phi, rewards = env.phi, []
    for aid in sequence:
        p = env.legal_mask.float() / env.legal_mask.sum()
        record = env.step(registry.by_id[aid], p)
        rewards.append(record.reward)
        if not env.done:
            assert not calls["score"]
    assert sum(rewards) == pytest.approx(record.base_reward - initial_phi)
    assert len(calls["score"]) == 1
    assert len(calls["phi"]) == len(sequence)
    assert record.next_state is None and record.phi_next == 0
    with pytest.raises(ValueError):
        env.step(0, p)


def test_affine_expectation_and_conditional_credit(system):
    registry, adapter, root, _ = system
    critic = AffineCritic(registry)
    state = adapter.observe(root, (), 1)
    p, q = torch.softmax(torch.randn(1, len(registry)), -1), torch.softmax(torch.randn(1, len(registry)), -1)
    lhs = critic.distribution_value([state], .3 * p + .7 * q)
    rhs = .3 * critic.distribution_value([state], p) + .7 * critic.distribution_value([state], q)
    assert torch.allclose(lhs, rhs, atol=1e-6)
    values = critic([state])
    advantages = prefix_advantages(registry, p, values)
    assert torch.allclose(sum(advantages), values - (p * values).sum(-1, keepdim=True), atol=1e-6)
    assert advantages[1][0, -1] == advantages[2][0, -1] == 0
    target = min_expected_target(torch.tensor([[.5, .5]]), torch.tensor([[0., 10.]]), torch.tensor([[10., 0.]]))
    assert target.item() == 5  # min per action first would incorrectly be zero.


@pytest.mark.parametrize("critic_type", ["affine", "icl_control"])
def test_daac_warmup_delayed_updates_targets_and_resume(system, tmp_path, critic_type):
    registry, adapter, root, _ = system
    config = DAACConfig(batch_size=2, capacity=32, warmup_transitions=2, critic_type=critic_type)
    trainer = DAACTrainer(registry, adapter.versions, config, seed=4)
    old_actor = deepcopy(trainer.actor.state_dict())
    records, metrics = complete_episode(trainer, adapter, root, 3, ("a_low_1", "STOP"))
    assert trainer.critic_step == trainer.actor_step == 0
    assert metrics[-1]["warmup"] == 1 and trainer.epsilon == .2
    old_target = deepcopy(trainer.targets.state_dict())
    records2, _ = complete_episode(trainer, adapter, root, 5, ("a_high_1", "STOP"))
    assert trainer.critic_step == 2 and trainer.actor_step == 1
    assert any(not torch.equal(v, old_actor[k]) for k, v in trainer.actor.state_dict().items())
    for name, value in trainer.targets.named_parameters():
        expected = old_target[name] * .995 + trainer.critics.state_dict()[name] * .005
        assert torch.allclose(value, expected, atol=1e-7) and not value.requires_grad
    assert trainer._targets([records[-1]]).item() == pytest.approx(records[-1].reward)
    checkpoint = tmp_path / "rl.pt"
    torch.save(trainer.state_dict(), checkpoint)
    restored = DAACTrainer(registry, adapter.versions, config, seed=99)
    restored.load_state_dict(torch.load(checkpoint, weights_only=False))
    state = adapter.observe(root, (), 10)
    mask = adapter.protocol.legal_mask(registry, ())
    assert trainer.select(state, mask)[0] == restored.select(state, mask)[0]
    first, second = trainer.update(), restored.update()
    assert first["critic_loss_1"] == pytest.approx(second["critic_loss_1"], abs=1e-8)
    for left, right in zip(trainer.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(left, right)
    with pytest.raises(ValueError, match="version mismatch"):
        trainer.observe(replace(records2[0], versions=replace(adapter.versions, reward="new-reward")))


def test_prefix_conditional_mle_not_joint_arithmetic_average(system):
    registry, _, _, _ = system
    table = ConditionalTable(registry, torch.ones(6, dtype=torch.bool),
                             torch.tensor([.1, .2, .1, .1, .2, .3]))
    table.update(torch.tensor([1., 0, 0, 0, 0, 0]), .8)
    p = table.probabilities
    assert p[:4].sum().item() == pytest.approx(.9)
    assert (p[:2].sum() / p[:4].sum()).item() == pytest.approx(.92)
    assert (p[0] / p[:2].sum()).item() == pytest.approx(.2 / 3 + .8)
    assert p[-1].item() == pytest.approx(.06)


def test_dlpa_deduplicated_scoring_joint_distillation_and_resume(system):
    registry, adapter, root, calls = system
    distiller = ProposalDistiller(registry, adapter.versions, seed=2, lr=.01)
    config = DLPAConfig(rounds=2, candidates_per_round=80, elite_count=8, score_repeats=4)
    planner = DLPAPlanner(adapter, distiller.proposal, config)
    result = planner.plan(root, seed=3)
    assert len(calls["score"]) == sum(result.metadata["round_unique_counts"])
    assert all(n < 80 for n in result.metadata["round_unique_counts"])
    assert all(len(seeds) == len(set(seeds)) == 4 for _, seeds in calls["score"])
    distiller.add(result)
    nodes = result.teachers
    states, masks = [n.state for n in nodes], torch.stack([n.mask for n in nodes])
    teacher = torch.stack([n.probabilities for n in nodes])
    before = joint_kl(teacher, distiller.proposal(states, masks)).mean().item()
    for _ in range(20):
        distiller.update(batch_size=16)
    after = joint_kl(teacher, distiller.proposal(states, masks)).mean().item()
    assert after < before
    with pytest.raises(ValueError, match="T-root"):
        distiller.add(replace(result, role="E"))
    d2 = ProposalDistiller(registry, adapter.versions, seed=99, lr=.01)
    d2.load_state_dict(deepcopy(distiller.state_dict()))
    p2 = DLPAPlanner(adapter, d2.proposal, config)
    p2.load_state_dict(planner.state_dict())
    next1, next2 = planner.plan(root, 7), p2.plan(root, 7)
    assert next1.sequences == next2.sequences
    assert all(torch.equal(a.probabilities, b.probabilities) for a, b in zip(next1.teachers, next2.teachers))
    assert distiller.update(8)["distill_kl"] == pytest.approx(d2.update(8)["distill_kl"])


def test_fixed_reevaluation_never_calls_actor_or_observe_and_keeps_provenance(system):
    registry, adapter, root, calls = system
    actor = ConditionalActor(registry)
    plans = generate_fixed_sequences(actor, adapter, root, count=20, seed=8)
    assert calls["score"] == []
    def forbidden(*args):
        raise AssertionError("Scoring attempted to observe future state or reselect an action")
    evaluation_adapter = replace(adapter, observe_fn=forbidden)
    result = reevaluate_fixed_sequences(evaluation_adapter, root, plans, seed=8, repeats=8, block_size=3)
    assert result.metadata["actor_calls"] == 0 and not result.metadata["fixed_pool_bound_met"]
    assert precision_repeats(len(plans)) > 1024
    own = {"daac": [plans[0]], "dlpa": [plans[-1]]}
    audit = audit_winners(evaluation_adapter, root, result, own, seed=8, repeats=4)
    assert audit["winners"] == {"daac": plans[0], "dlpa": plans[-1]}
    assert not audit["reselected_after_audit"]
    reevaluation_seeds = {s for _, seeds in calls["score"][:-2] for s in seeds}
    audit_seeds = {s for _, seeds in calls["score"][-2:] for s in seeds}
    assert not reevaluation_seeds & audit_seeds
    with pytest.raises(ValueError, match="same root"):
        audit_winners(evaluation_adapter, replace(root, root_id="different-root"), result, own, seed=8, repeats=4)


def test_frozen_mc_diagnostic_compares_shaped_values(system):
    registry, adapter, root, _ = system
    actor = ConditionalActor(registry)
    initial = deepcopy(actor.state_dict())
    result = critic_mc_diagnostic(actor, adapter, root, "a_high_2", .2, repeats=4)
    assert result["shaped_target"] == pytest.approx(result["base_target"] - .2)
    assert all(torch.equal(v, initial[k]) for k, v in actor.state_dict().items())
