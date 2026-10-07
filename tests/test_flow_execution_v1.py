"""Trusted internally generated times preserve the checked flow and all gradients."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models import ModelConfig, ResponseModel, heun_integrate


@pytest.fixture(autouse=True)
def single_thread_and_seed():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(924)
    yield
    torch.set_num_threads(before)


def model_and_condition_inputs(checkpoint=True):
    config = ModelConfig(n_genes=7, d_model=8, n_heads=2, encoder_layers=1,
                         pad_layers=2, query_chunk=3, checkpoint_blocks=checkpoint,
                         checkpoint_attention=checkpoint)
    genes = torch.arange(config.n_genes)
    neighbors = torch.stack((genes, (genes + 1) % config.n_genes), dim=-1)
    model = ResponseModel(config, neighbors)
    with torch.no_grad():
        for block in model.velocity_model.blocks:
            for modulation in block.modulations:
                modulation[-1].weight.normal_(std=.01)
                modulation[-1].bias.fill_(.07)
    inputs = dict(fingerprint=torch.randn(2, 2048), dose=torch.tensor([.5, 1.]),
                  duration=torch.full((2,), 24.), background=torch.zeros(2, dtype=torch.long),
                  source=torch.tensor([0, 1]), protocol=torch.zeros(2, dtype=torch.long),
                  elapsed=torch.zeros(2), horizon=torch.full((2,), 24.))
    return model, inputs


def assert_all_gradients_equal(first, second):
    others = dict(second.named_parameters())
    for name, parameter in first.named_parameters():
        other = others[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)
    assert first.cell_encoder.blocks[0].attention.q1.weight.grad.abs().sum() > 0
    assert first.velocity_model.time_encoder.network[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_heun_trusted_times_equal_checked_outputs_and_all_gradients(checkpoint):
    fast, condition_inputs = model_and_condition_inputs(checkpoint)
    checked = deepcopy(fast)
    reference = torch.randn(2, 7)
    noise = torch.randn_like(reference)
    actual = fast.sample(reference, fast.encode_condition(**condition_inputs), noise,
                         steps=3, checkpoint=checkpoint, cache_reference=False)
    reference_tokens, _ = checked.encode_cells(reference)
    condition = checked.encode_condition(**condition_inputs)
    expected = heun_integrate(
        lambda state, tau: checked.velocity(state, tau, reference_tokens, condition),
        noise, steps=3, checkpoint=checkpoint)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert_all_gradients_equal(fast, checked)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_sampled_cfm_time_equals_explicit_checked_time_and_all_gradients(checkpoint):
    fast, condition_inputs = model_and_condition_inputs(checkpoint)
    checked = deepcopy(fast)
    reference, target, noise = (torch.randn(2, 7) for _ in range(3))
    torch.manual_seed(711)
    actual = fast.cfm_loss(reference, target, fast.encode_condition(**condition_inputs), noise)
    torch.manual_seed(711)
    tau = torch.rand(2)
    expected = checked.cfm_loss(reference, target, checked.encode_condition(**condition_inputs), noise, tau)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    assert_all_gradients_equal(fast, checked)


@pytest.mark.parametrize("bad_value", [-.01, 1.01, float("nan"), float("inf")])
@pytest.mark.parametrize("entry", ["encoder", "velocity", "forward", "cfm"])
def test_public_paths_still_reject_invalid_external_times(bad_value, entry):
    model, condition_inputs = model_and_condition_inputs(False)
    expression = torch.randn(2, 7)
    condition = model.encode_condition(**condition_inputs)
    tau = torch.tensor([.5, bad_value])
    with pytest.raises(ValueError, match="Generative time tau"):
        if entry == "encoder":
            model.velocity_model.time_encoder(tau)
        elif entry == "velocity":
            tokens, _ = model.encode_cells(expression)
            model.velocity(expression, tau, tokens, condition)
        elif entry == "forward":
            model(expression, tau, expression, condition)
        else:
            model.cfm_loss(expression, expression, condition, tau=tau)


def test_internal_paths_skip_repeated_public_time_validation(monkeypatch):
    model, condition_inputs = model_and_condition_inputs(False)
    expression = torch.randn(2, 7)
    condition = model.encode_condition(**condition_inputs)
    calls = []
    encoder = model.velocity_model.time_encoder
    checked_forward = encoder.forward

    def count_validation(tau):
        calls.append(tau)
        return checked_forward(tau)

    monkeypatch.setattr(encoder, "forward", count_validation)
    model.sample(expression, condition, steps=3, checkpoint=False)
    model.cfm_loss(expression, expression, condition)
    assert len(calls) == 0
    model.cfm_loss(expression, expression, condition, tau=torch.tensor([0., 1.]))
    assert len(calls) == 1


def test_public_time_shape_and_condition_shape_checks_remain():
    model, condition_inputs = model_and_condition_inputs(False)
    expression = torch.randn(2, 7)
    condition = model.encode_condition(**condition_inputs)
    with pytest.raises(ValueError, match="Generative time tau"):
        model.velocity_model.time_encoder(torch.ones(2, 1))
    with pytest.raises(ValueError, match="generative-time shape mismatch"):
        model.cfm_loss(expression, expression, condition, tau=torch.ones(2, 1))
    with pytest.raises(ValueError, match="Condition batch"):
        model.sample(expression, condition[:1], steps=1, checkpoint=False)
    with pytest.raises(ValueError, match="Condition batch"):
        model.cfm_loss(expression, expression, condition[:1])
