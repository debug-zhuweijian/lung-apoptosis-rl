"""CPU synthetic contracts; no real-data predictive-accuracy claim."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.models import build_response_model
from lung.engine.response_training import mmd_backward, resolve_mmd_mode, one_step_loss
from test_models_v1 import tiny_config, neighbors, conditions




@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(291)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('backbone', ['compact_full_gene', 'gene_token', 'latent_flow'])
def test_full_32_step_loss_all_gradients_and_adam_update_equal(backbone):
    direct = build_response_model(replace(tiny_config(False), backbone=backbone, latent_dim=3), neighbors())
    replay = copy.deepcopy(direct)
    x, y = torch.randn(3, 7), torch.randn(3, 7)
    epsilon = direct.sample_noise(x)
    tau = torch.tensor([.1, .5, .9])
    base = conditions(1)
    def encode(model, n):
        return model.encode_condition(**{k: v.expand(n, *v.shape[1:]) for k, v in base.items()})
    results = []
    for model, mode in [(direct, 'direct'), (replay, 'replay')]:
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        cfm = model.cfm_loss(x, y, encode(model, len(x)), noise=epsilon, tau=tau)
        cfm.backward()
        mmd = mmd_backward(model, x, y, lambda n: encode(model, n), mode=mode,
                           noise=epsilon, bandwidth=.7, steps=32, microbatch=2)
        gradients = {k: p.grad.clone() if p.grad is not None else None for k, p in model.named_parameters()}
        optim.step()
        results.append((float(cfm.detach()) + mmd, gradients))
    assert results[0][0] == pytest.approx(results[1][0], abs=2e-6)
    for name, left in results[0][1].items():
        right = results[1][1][name]
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, rtol=5e-4, atol=3e-6, msg=name)
    for (name, left), right in zip(direct.named_parameters(), replay.parameters()):
        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-6, msg=name)
    assert resolve_mmd_mode(direct) == ('replay' if backbone == 'gene_token' else 'direct')


@pytest.mark.parametrize('backbone', ['compact_full_gene', 'gene_token'])
def test_one_step_uses_noisy_endpoint_not_target(backbone):
    model = build_response_model(replace(tiny_config(False), backbone=backbone, latent_dim=3), neighbors())
    x, y = torch.randn(2, 7), torch.randn(2, 7)
    noise, tau = model.sample_noise(x), torch.tensor([.2, .7])
    objective = y - (1-model.config.sigma_min)*noise
    with patch.object(model, 'velocity', return_value=objective):
        loss = model.one_step_loss(x, y, model.encode_condition(**conditions(2)),
                                  bandwidth=.7, noise=noise, tau=tau)
    assert float(loss.detach()) < 1e-7


@pytest.mark.parametrize('backbone', ['compact_full_gene', 'gene_token'])
@pytest.mark.parametrize('microbatch', [1, 2, 3])
def test_one_step_full_population_mmd_and_all_gradients_ignore_partition(backbone, microbatch):
    from lung.models import rbf_mmd
    model = build_response_model(replace(tiny_config(False), backbone=backbone), neighbors())
    other = copy.deepcopy(model)
    x, y, noise = torch.randn(5, 7), torch.randn(5, 7), torch.randn(5, 7)
    tau, cond = torch.linspace(.1, .9, 5), conditions(5)
    predicted, endpoint = model.one_step_endpoints(x, y, model.encode_condition(**cond), noise=noise, tau=tau)
    expected = rbf_mmd(predicted, endpoint, bandwidth=.7, unbiased=False)
    incorrect_micro_average = torch.stack([rbf_mmd(predicted[i:i+1], endpoint[i:i+1],
        bandwidth=.7, unbiased=False) for i in range(5)]).mean()
    assert not torch.isclose(expected, incorrect_micro_average, rtol=1e-3, atol=1e-5)
    expected.backward()
    actual = one_step_loss(other, x, y, other.encode_condition(**cond), bandwidth=.7,
                          noise=noise, tau=tau, microbatch=microbatch)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for (name, left), right in zip(model.named_parameters(), other.parameters()):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=5e-4, atol=2e-6, msg=name)


def test_latent_one_step_rejected():
    model = build_response_model(replace(tiny_config(False), backbone='latent_flow', latent_dim=3), neighbors())
    x, y = torch.randn(2, 7), torch.randn(2, 7)
    with pytest.raises(ValueError, match='LATENT_ONE_STEP_UNSUPPORTED'):
        one_step_loss(model, x, y, model.encode_condition(**conditions(2)), bandwidth=1.)


def test_one_step_independent_32_cell_draw_does_not_advance_cfm_rng():
    from lung.engine.config import load_config
    from synthetic_data import smoke_config, SyntheticData
    from lung.engine.response import train_condition, response_rngs
    config = smoke_config(load_config())
    cfg = {**config['response'], 'coupling': 'random_product_control', 'mmd_samples': 32}
    data = SyntheticData(config)
    from lung.models import ModelConfig
    model = build_response_model(ModelConfig(**cfg['model']), torch.as_tensor(data.neighbors))
    outcomes = []
    for stage in ['cfm', 'one_step']:
        torch.manual_seed(203)
        rng, extra_rng, generator = response_rngs(203, 'cpu')
        with patch.object(data, 'sample_response', wraps=data.sample_response) as sample:
            result = train_condition(copy.deepcopy(model), data, 'T-0', rng,
                {**cfg, '_training_stage': stage}, 'cpu', 1,
                mmd_rng=extra_rng, mmd_generator=generator)
        outcomes.append((result, torch.get_rng_state(), rng.bit_generator.state))
        if stage == 'one_step':
            assert sample.call_count == 2
            assert sample.call_args.args[1] == 32 and sample.call_args.args[2] is extra_rng
            assert sample.call_args.kwargs['role'] == 'T'
            assert result['one_step_auxiliary'] > 0
    assert outcomes[0][0]['cfm_loss'] == outcomes[1][0]['cfm_loss']
    assert torch.equal(outcomes[0][1], outcomes[1][1])
    assert outcomes[0][2] == outcomes[1][2]


def test_latent_all_gene_reconstruction_generation_decode_once_and_reload():
    model = build_response_model(replace(tiny_config(False), backbone='latent_flow', latent_dim=3), neighbors())
    x, y = torch.randn(2, 7), torch.randn(2, 7)
    condition = model.encode_condition(**conditions(2))
    losses = model.loss_components(x, y, condition)
    (losses['latent_cfm_loss'] + losses['reconstruction_loss']).backward()
    assert torch.all(model.cell_encoder.expression_projection.weight.grad.abs().sum(0) > 0)
    assert torch.all(model.decoder[-1].weight.grad.abs().sum(1) > 0)
    noise = model.sample_noise(x)
    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decode:
        generated = model.sample(x, condition, noise=noise, steps=32, checkpoint=False)
    assert decode.call_count == 1
    assert generated.shape == x.shape and torch.isfinite(generated).all()
    restored = build_response_model(model.config, neighbors())
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(generated, restored.sample(x, condition, noise=noise, steps=32, checkpoint=False))
    with pytest.raises(ValueError, match='latent_dim'):
        model.sample(x, condition, noise=torch.randn_like(x))








@pytest.mark.parametrize('count,micro,mmd_micro', [(8,128,32), (3,31,7)])
def test_two_condition_batch_full_32_steps_all_gradients_update_and_rng(count, micro, mmd_micro):
    from lung.engine.config import load_config
    from synthetic_data import smoke_config, SyntheticData
    from lung.engine.response import response_rngs
    from lung.engine.response_training import train_conditions
    from lung.models import ModelConfig
    config = smoke_config(load_config())
    cfg = {**config['response'], 'coupling': 'random_product_control',
           'cfm_samples': 128, 'mmd_samples': 32, 'microbatch': micro,
           'mmd_microbatch': mmd_micro, 'train_ode_steps': 32}
    data = SyntheticData(config)
    initial = build_response_model(ModelConfig(**cfg['model']), torch.as_tensor(data.neighbors))
    keys = ['T-0', 'T-1'] * (count//2) + (['T-0'] if count%2 else [])
    results = []
    for width in [1, 2]:
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.manual_seed(217)
        rng, extra_rng, generator = response_rngs(217, 'cpu')
        rows = []
        for start in range(0, len(keys), width):
            rows.extend(train_conditions(model, data, keys[start:start+width], rng,
                {**cfg, 'condition_microbatch': width}, 'cpu', len(keys),
                mmd_rng=extra_rng, mmd_generator=generator))
        gradients = {name: p.grad.clone() if p.grad is not None else None for name,p in model.named_parameters()}
        optimizer.step()
        results.append((model, rows, gradients, torch.get_rng_state(), generator.get_state(),
                        rng.bit_generator.state, extra_rng.bit_generator.state))
    left, right = results
    assert left[5:] == right[5:]
    assert torch.equal(left[3], right[3]) and torch.equal(left[4], right[4])
    for a,b in zip(left[1], right[1]):
        for key in a:
            assert a[key] is b[key] if a[key] is None else a[key] == pytest.approx(b[key], abs=3e-6)
    for name, grad in left[2].items():
        other = right[2][name]
        assert (grad is None) == (other is None)
        if grad is not None:
            torch.testing.assert_close(grad, other, rtol=5e-4, atol=3e-6, msg=name)
    for (name,a), b in zip(left[0].named_parameters(), right[0].parameters()):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6, msg=name)


@pytest.mark.parametrize('backbone,reference,stage,mode,reason', [
    ('gene_token','observed','full','auto','serial_backbone'),
    ('latent_flow','observed','full','auto','serial_backbone'),
    ('compact_full_gene','population_mean','full','auto','serial_reference_mode'),
    ('compact_full_gene','observed','one_step','auto','serial_training_stage'),
    ('compact_full_gene','observed','full','replay','serial_backward_mode')])
def test_condition_batching_explicit_serial_fallback(backbone, reference, stage, mode, reason):
    from lung.engine.response_training import condition_batch_mode
    model = build_response_model(replace(tiny_config(False), backbone=backbone, reference_mode=reference), neighbors())
    assert condition_batch_mode(model, {'condition_microbatch':2, '_training_stage':stage,
                                      'mmd_backward':mode}) == (1, reason)


def test_cfm_stage_and_condition_only_grouping_enabled():
    """W2 contract change: CFM warm-up and condition_only now batch (task 2026-09-29)."""
    from lung.engine.response_training import condition_batch_mode
    model = build_response_model(replace(tiny_config(False), backbone='compact_full_gene'), neighbors())
    assert condition_batch_mode(model, {'condition_microbatch': 2, '_training_stage': 'cfm',
                                      'mmd_backward': 'auto'}) == (2, 'grouped_compact_observed_cfm_direct')
    other = build_response_model(replace(tiny_config(False), backbone='compact_full_gene',
                                         reference_mode='condition_only'), neighbors())
    assert condition_batch_mode(other, {'condition_microbatch': 3, '_training_stage': 'full',
                                       'mmd_backward': 'auto'}) == (3, 'grouped_compact_condition_only_full_direct')
