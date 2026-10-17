"""State ablations and native decisions have observable, separate evaluation semantics."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine.config import load_config
from synthetic_data import SyntheticData, smoke_config
from lung.engine.decision_evaluation import load_policy_models, native_sequence
from lung.engine.response import exact_mmd_backward
from lung.models import ModelConfig, ResponseModel, rbf_mmd
from lung.rl import DAACConfig, DAACTrainer
from test_models_v1 import tiny_config, neighbors, conditions
from test_rl_v1 import system


@pytest.mark.parametrize("mode", ["observed", "condition_only", "population_mean"])
def test_ablation_full_population_gradient_is_microbatch_invariant(mode):
    torch.set_num_threads(1)
    torch.manual_seed(76)
    shape = replace(tiny_config(), reference_mode=mode)
    native = ResponseModel(shape, neighbors())
    replay = ResponseModel(shape, neighbors())
    replay.load_state_dict(native.state_dict())
    x, y, noise = torch.randn(5, 7), torch.randn(4, 7), torch.randn(5, 7)
    base = conditions(1)
    def encode(model, n):
        return model.encode_condition(**{k: v.expand(n, *v.shape[1:]) for k, v in base.items()})
    expected = rbf_mmd(native.sample(x, encode(native, 5), noise=noise, steps=2), y, .7)
    expected.backward()
    actual = exact_mmd_backward(replay, x, y, lambda n: encode(replay, n),
                               bandwidth=.7, steps=2, noise=noise, microbatch=2)
    assert actual == pytest.approx(float(expected.detach()), abs=1e-6)
    for name, parameter in native.named_parameters():
        other = dict(replay.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=5e-4, atol=3e-6, msg=name)


def test_ablation_changes_only_control_context_and_never_uses_targets():
    x = torch.arange(21).reshape(3, 7).float()
    mean_model = ResponseModel(replace(tiny_config(), reference_mode="population_mean"), neighbors())
    zero_model = ResponseModel(replace(tiny_config(), reference_mode="condition_only"), neighbors())
    torch.testing.assert_close(mean_model.condition_reference(x), x.mean(0).expand_as(x))
    assert not zero_model.condition_reference(x).any()
    assert torch.equal(x, torch.arange(21).reshape(3, 7).float())


def test_native_choice_cannot_be_rescued_by_world_reranking(system):
    registry, adapter, root, calls = system
    class Actor:
        def __call__(self, states, masks):
            probabilities = torch.zeros((len(states), len(registry)))
            probabilities[:, registry.by_id["a_low_2"]] = .99
            probabilities[:, registry.by_id["a_high_2"]] = .01
            return probabilities
    class Critic:
        def __call__(self, states):
            q = torch.zeros((len(states), len(registry)))
            q[:, registry.by_id["a_high_2"]] = 1.
            return q
    actor_plan, actor_trace = native_sequence(Actor(), adapter, root, 7)
    q_plan, q_trace = native_sequence(Actor(), adapter, root, 7, critics=[Critic(), Critic()])
    assert actor_plan == ("a_low_2",) and q_plan == ("a_high_2",)
    assert not calls["score"]
    assert actor_trace[0]["selection"] == "actor_greedy"
    assert q_trace[0]["ranking"][0]["q1"] == 1.


@pytest.mark.parametrize("actor_type", ["conditional", "joint_control"])
def test_frozen_loader_uses_checkpoint_architecture_without_rng_or_optimizer_updates(system, actor_type):
    registry, adapter, root, _ = system
    trainer = DAACTrainer(registry, adapter.versions,
        DAACConfig(capacity=8, batch_size=2, warmup_transitions=2, actor_type=actor_type))
    snapshot = {"model": trainer.actor.state_dict(), "learner": trainer.state_dict()}
    before = torch.get_rng_state().clone()
    actor, critics, kind = load_policy_models(snapshot, registry, "cpu", "daac")
    assert torch.equal(before, torch.get_rng_state())
    assert type(actor) is type(trainer.actor)
    assert kind == "affine" and len(critics) == 2
    assert not any(p.requires_grad for p in actor.parameters())


@pytest.mark.parametrize("mode", ["observed", "population_mean"])
def test_E_population_metric_does_not_claim_iid_for_shared_mean(tmp_path, mode):
    from lung.engine.evaluate import evaluate_expression
    cfg = smoke_config(load_config())
    data = SyntheticData(cfg)
    shape = ModelConfig(**cfg["response"]["model"], reference_mode=mode)
    models = [ResponseModel(shape, torch.tensor(data.neighbors)) for _ in range(2)]
    report = evaluate_expression(models, data, cfg, tmp_path)
    import pandas as pd
    rows = pd.read_csv(tmp_path / "conditions.csv")
    assert np.isfinite(rows.population_mmd_v).all()
    assert report["mmd_estimator"] == "empirical_population_V_not_unbiased_population_estimate"
    assert rows.shared_control_mmd_u.isna().all() if mode == "population_mean" else rows.shared_control_mmd_u.notna().all()
