"""Tiny synthetic / mathematical checks; never biological performance evidence."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import auxiliary
from lung.engine.checkpoint import load_checkpoint
from lung.engine.config import load_config, state_objective, validate_config, StageUnavailable
from lung.models import StateClassifier
from lung.models.state_objectives import aggregate_nll, state_loss


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_lambda_zero_matches_ce_and_gradients_exactly():
    logits = torch.tensor([[1., -2., 3.], [0., 10000., -10000.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 0])
    ce = state_loss(logits, labels, objective="ce")
    combined = state_loss(logits, labels, objective="ce_aggregate", aggregate_lambda=0.)
    assert torch.equal(ce, combined)
    assert torch.equal(torch.autograd.grad(ce.sum(), logits, retain_graph=True)[0],
                       torch.autograd.grad(combined.sum(), logits)[0])


def test_aggregate_matches_probability_bce_and_gradient():
    logits = torch.tensor([[1., -2., 3.], [-1., 2., 0.], [.5, .2, -.4]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    direct = F.binary_cross_entropy(logits.softmax(-1)[:, 1:].sum(-1), (labels > 0).double(), reduction="none")
    stable = aggregate_nll(logits, labels)
    torch.testing.assert_close(stable, direct)
    torch.testing.assert_close(torch.autograd.grad(stable.sum(), logits, retain_graph=True)[0],
                               torch.autograd.grad(direct.sum(), logits)[0])


@pytest.mark.parametrize("scale", [1e4, 1e20])
def test_extreme_logits_finite_losses_and_gradients(scale):
    logits = torch.tensor([[scale, -scale, -scale], [-scale, scale, scale],
                           [scale, scale, scale]], requires_grad=True)
    labels = torch.tensor([1, 0, 2])
    losses = state_loss(logits, labels, objective="ce_aggregate", aggregate_lambda=1.)
    assert torch.isfinite(losses).all()
    losses.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_expected_excess_loss_is_two_kls():
    q, p = torch.tensor([.2, .3, .5], dtype=torch.float64), torch.tensor([.6, .1, .3], dtype=torch.float64)
    labels = torch.arange(3)
    loss = lambda probs: state_loss(probs.log().repeat(3, 1), labels, objective="ce_aggregate", aggregate_lambda=1.)
    observed = (q * (loss(p) - loss(q))).sum()
    q_binary, p_binary = torch.stack((q[0], q[1:].sum())), torch.stack((p[0], p[1:].sum()))
    expected = (q * (q.log() - p.log())).sum() + (q_binary * (q_binary.log() - p_binary.log())).sum()
    torch.testing.assert_close(observed, expected)
    assert observed > 0


def test_weighted_ce_preserves_global_fit_weights():
    labels = torch.tensor([0, 0, 0, 1, 2])
    logits = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 5
    weights = StateClassifier.class_weights(labels)
    weights = weights / weights[labels].mean()
    actual = state_loss(logits, labels, weights=weights)
    expected = torch.stack([F.cross_entropy(row[None], label[None]) * weights[label]
                            for row, label in zip(logits, labels)])
    torch.testing.assert_close(actual, expected)
    assert state_objective({}) == {"state_loss": "weighted_ce", "state_aggregate_lambda": 0.}


def test_probability_metrics_match_direct_definitions():
    probabilities = torch.tensor([[.8, .1, .1], [.2, .6, .2], [.4, .5, .1]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 2])
    report = auxiliary.state_metrics(probabilities.log(), labels)
    expected_brier = ((probabilities - F.one_hot(labels, 3)) ** 2).sum(1).mean()
    positive = probabilities[:, 1:].sum(1)
    assert report["val/nll"] == pytest.approx(float(-probabilities[torch.arange(3), labels].log().mean()))
    assert report["val/brier"] == pytest.approx(float(expected_brier))
    assert report["val/aggregate_nll"] == pytest.approx(float(-torch.tensor([.8, .8, .6]).log().mean()))
    assert report["val/aggregate_brier"] == pytest.approx(float(((positive - (labels > 0).double()) ** 2).mean()))
    assert report["val/aggregate_accuracy"] == 1.
    assert report["val/aggregate_ece_10"] == pytest.approx((.2 + .2 + .4) / 3)
    assert report["val/recall_2"] == 0.
    assert all(np.isfinite(value) for value in report.values())


@pytest.mark.parametrize("objective, weight", [("bad", 0.), ([], 0.), ("ce", 1.), ("weighted_ce", 1.),
                                               ("ce_aggregate", -1.), ("ce_aggregate", float("nan")),
                                               ("ce_aggregate", float("inf")), ("ce_aggregate", True),
                                               ("ce_aggregate", "1")])
def test_invalid_objective_config_rejected(objective, weight):
    config = load_config()
    with pytest.raises(ValueError, match="state_"):
        validate_config({**config, "state_loss": objective, "state_aggregate_lambda": weight})


@pytest.mark.parametrize("name, objective, weight", [("weighted", "weighted_ce", 0.), ("ce", "ce", 0.), ("aggregate", "ce_aggregate", 1.)])
def test_pilot_config_contract(name, objective, weight):
    config = load_config(f"configs/pilot_state_{name}.yaml")
    assert config["stage"] == "state" and config["epochs"] == 1000
    assert config["progress"] and config["dashboard"] and not config["auto_prepare"]
    assert config["response"]["model"]["n_genes"] == 32461
    assert config["pipeline"] is None
    assert state_objective(config) == {"state_loss": objective, "state_aggregate_lambda": weight}
    with pytest.raises(StageUnavailable, match="TRAINED_REPRESENTATION_OR_RESPONSE_REQUIRED"):
        auxiliary.train_auxiliary("state", None, config, None, {})


class TinyStateData:
    identity = "synthetic-state-only"

    def __init__(self):
        self.roles = []

    def state_arrays(self, fold, role):
        assert fold == "1_to_8" and role in {"fit", "selection", "calibration"}
        self.roles.append(role)
        rng = np.random.default_rng({"fit": 1, "selection": 2, "calibration": 3}[role])
        return {"x": rng.normal(size=(6, 4)), "labels": np.array([0, 1, 2, 0, 1, 2])}


class TinyMonitor:
    def __init__(self, path):
        self.weights, self.experiment, self.values = path / "weights", path / "experiment", []

    def progress(self, items, **kwargs):
        return items

    def record(self, stage, epoch, metrics):
        self.values.append((stage, epoch, metrics))


@pytest.fixture
def tiny_state(tmp_path, monkeypatch):
    reference = SimpleNamespace(config=SimpleNamespace(d_model=4), encode_cells=lambda x: (None, x))
    monkeypatch.setattr(auxiliary, "load_response", lambda *args: reference)
    upstream = tmp_path / "frozen-response-placeholder.bin"
    upstream.write_bytes(b"synthetic-frozen-representation-fixture")
    config = {**load_config("configs/pilot_state_aggregate.yaml"), "device": "cpu", "epochs": 1,
              "synthetic": True, "state_batch": 3, "calibration_steps": 2}
    return TinyStateData(), config, TinyMonitor(tmp_path), {"response": [str(upstream)]}


@pytest.mark.parametrize("objective, weight", [("weighted_ce", 0.), ("ce", 0.), ("ce_aggregate", 1.)])
def test_synthetic_state_fit_roles_metadata_and_resume(tiny_state, objective, weight):
    data, base, monitor, pipeline = tiny_state
    config = {**base, "state_loss": objective, "state_aggregate_lambda": weight}
    completed = auxiliary.train_state(data, config, monitor, pipeline)
    saved = load_checkpoint(completed["state"])
    assert data.roles == ["fit", "selection", "calibration"]
    assert saved["metadata"]["state_loss"] == objective
    assert saved["metadata"]["state_aggregate_lambda"] == weight
    assert saved["metadata"]["selection_metric"] == "unweighted_three_class_nll"
    assert saved["metadata"]["calibration_role"] == "calibration"
    assert saved["metadata"]["aggregate_classes"] == [1, 2]
    assert saved["extra"]["temperature"] > 0
    assert auxiliary.verify_auxiliary_dependencies(completed["state"], completed)
    assert any("val/aggregate_nll" in row[2] for row in monitor.values)
    assert monitor.values[-1][0] == "state_calibration"
    auxiliary.train_state(data, config, monitor, pipeline, resume=True)
    assert data.roles == ["fit", "selection", "calibration"]  # cache reuse
    changed = {**config, "state_loss": "ce_aggregate", "state_aggregate_lambda": 2.}
    with pytest.raises(ValueError, match="CHECKPOINT_IDENTITY_MISMATCH"):
        auxiliary.train_state(data, changed, monitor, pipeline, resume=True)


def test_state_identity_binds_objective_lambda_upstream_and_fold(tiny_state):
    data, config, _, pipeline = tiny_state
    identity = auxiliary.pipeline_identity("state", data, config, pipeline)
    for replacement in ({"state_aggregate_lambda": 0.}, {"state_loss": "ce", "state_aggregate_lambda": 0.},
                        {"state_loss": "weighted_ce", "state_aggregate_lambda": 0.}, {"state_fold": "8_to_1"}):
        assert identity != auxiliary.pipeline_identity("state", data, {**config, **replacement}, pipeline)
    Path(pipeline["response"][0]).write_bytes(b"different-upstream")
    assert identity != auxiliary.pipeline_identity("state", data, config, pipeline)


def test_role_cache_swap_is_rejected(tiny_state):
    data, config, monitor, pipeline = tiny_state
    auxiliary.train_state(data, config, monitor, pipeline)
    fitting = next((monitor.experiment / "cache").glob("state-fit-*.pt"))
    calibration = next((monitor.experiment / "cache").glob("state-calibration-*.pt"))
    fitting.write_bytes(calibration.read_bytes())
    with pytest.raises(ValueError, match="AUX_FEATURE_CACHE_IDENTITY_MISMATCH"):
        auxiliary.train_state(data, config, monitor, pipeline, resume=True)
