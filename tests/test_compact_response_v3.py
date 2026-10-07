"""CPU synthetic contracts for the compact full-gene candidate, not efficacy tests."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models import ModelConfig, ResponseModel, CompactResponseModel, build_response_model, rbf_mmd


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(291)
    yield
    torch.set_num_threads(threads)


def config(**kwargs):
    return replace(ModelConfig(n_genes=7, d_model=8, n_heads=2, pad_layers=6,
                               backbone="compact_full_gene"), **kwargs)


def graph(genes=7):
    own = torch.arange(genes)
    return torch.stack((own, (own + 1) % genes), -1)


def conditions(batch=2):
    return dict(fingerprint=torch.randn(batch, 2048), dose=torch.ones(batch),
                duration=torch.full((batch,), 24.), background=torch.zeros(batch, dtype=torch.long),
                source=torch.zeros(batch, dtype=torch.long), protocol=torch.zeros(batch, dtype=torch.long),
                elapsed=torch.zeros(batch), horizon=torch.full((batch,), 24.))


def test_cfm_formula_and_all_trainable_paths():
    model = build_response_model(config(), graph())
    x = torch.randn(2, 7, requires_grad=True)
    y, noise = torch.randn_like(x), torch.randn_like(x)
    cond = model.encode_condition(**conditions())
    cond.retain_grad()
    tau = torch.tensor([.2, .8])
    state = (1 - .999 * tau[:, None]) * noise + tau[:, None] * y
    expected = (model(state, tau, x, cond) - (y - .999 * noise)).square().mean()
    actual = model.cfm_loss(x, y, cond, noise, tau)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert (x.grad.abs().sum(0) > 0).all()
    assert cond.grad.abs().sum() > 0
    for name in ("cell_encoder.expression_projection.weight", "cell_encoder.graph_projection.weight",
                 "condition_encoder.fusion.0.weight", "velocity_model.decoder.1.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0, name
    for bad in (torch.tensor([-1., .2]), torch.tensor([.2, float('nan')]), torch.ones(3)):
        with pytest.raises(ValueError):
            model.cfm_loss(x, y, cond, noise, bad)


def test_full_32461_gene_axis_and_sparse_graph_use():
    model = build_response_model(config(n_genes=32461, checkpoint_blocks=False), graph(32461))
    x = torch.randn(1, 32461, requires_grad=True)
    encoder = model.cell_encoder
    torch.testing.assert_close(encoder.graph_mean(x), x[:, graph(32461)].mean(-1))
    assert encoder.graph.layout == torch.sparse_coo
    assert encoder.graph._nnz() == 2 * 32461
    tokens, vector = model.encode_cells(x)
    assert tokens.shape == (1, 1, 8) and vector.shape == (1, 8)
    vector.square().sum().backward()
    assert (x.grad.abs() > 0).all()
    with torch.no_grad():
        before = vector.clone()
        encoder.graph.values().zero_()
        assert not torch.allclose(model.encode_cells(x)[1], before)
    result = model(x, torch.tensor([.5]), x, torch.randn(1, 8))
    assert result.shape == (1, 32461) and result.dtype == torch.float32


@pytest.mark.parametrize("blocks", [False, True])
@pytest.mark.parametrize("ode", [False, True])
def test_complete_32_steps_checkpoint_and_reference_once(blocks, ode):
    model = build_response_model(config(checkpoint_blocks=blocks), graph())
    x, noise = torch.randn(2, 7, requires_grad=True), torch.randn(2, 7, requires_grad=True)
    c = torch.randn(2, 8, requires_grad=True)
    progress = []
    with patch.object(model, 'encode_cells', wraps=model.encode_cells) as encode:
        endpoint = model.sample(x, c, noise, steps=32, checkpoint=ode,
                                progress=lambda step, total: progress.append((step, total)))
        assert encode.call_count == 1
        endpoint.square().mean().backward()
        assert encode.call_count == 1
    assert progress == [(i, 32) for i in range(1, 33)]
    for value in (x, noise, c):
        assert value.grad.isfinite().all() and value.grad.abs().sum() > 0
    assert model.velocity_model.state_projection.weight.grad.abs().sum() > 0


def test_checkpoint_numerical_and_gradient_equivalence():
    a = build_response_model(config(checkpoint_blocks=False), graph())
    b = build_response_model(config(checkpoint_blocks=True), graph())
    b.load_state_dict(a.state_dict())
    x, n, c = torch.randn(2, 7), torch.randn(2, 7), torch.randn(2, 8)
    ya, yb = a.sample(x, c, n, checkpoint=False), b.sample(x, c, n, checkpoint=True)
    torch.testing.assert_close(ya, yb)
    ya.square().mean().backward()
    yb.square().mean().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad)


def test_32_step_two_pass_mmd_matches_direct_gradients():
    a = build_response_model(config(), graph())
    b = build_response_model(config(), graph())
    b.load_state_dict(a.state_dict())
    x, n, y = torch.randn(3, 7), torch.randn(3, 7), torch.randn(4, 7)
    args = conditions(3)
    direct = rbf_mmd(a.sample(x, a.encode_condition(**args), n, steps=32), y, .7)
    direct.backward()
    with torch.no_grad():
        outputs = torch.cat([b.sample(x[i:i+1], b.encode_condition(**{k: v[i:i+1] for k, v in args.items()}),
                                     n[i:i+1], steps=32) for i in range(3)])
    leaf = outputs.requires_grad_()
    loss = rbf_mmd(leaf, y, .7)
    gradient, = torch.autograd.grad(loss, leaf)
    for i in range(3):
        output = b.sample(x[i:i+1], b.encode_condition(**{k: v[i:i+1] for k, v in args.items()}),
                          n[i:i+1], steps=32)
        output.backward(gradient[i:i+1])
    torch.testing.assert_close(direct, loss)
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=3e-6, rtol=3e-4, msg=name)


@pytest.mark.parametrize("backbone", ["gene_token", "compact_full_gene"])
def test_factory_legacy_default_and_state_roundtrip(backbone, tmp_path):
    cfg = config(backbone=backbone, pad_layers=1)
    a = build_response_model(cfg, graph())
    assert isinstance(a, CompactResponseModel if backbone == 'compact_full_gene' else ResponseModel)
    path = tmp_path / 'weights.pt'
    torch.save(dict(config=asdict(cfg), model=a.state_dict()), path)
    loaded = torch.load(path, weights_only=True)
    b = build_response_model(ModelConfig(**loaded['config']), graph())
    b.load_state_dict(loaded['model'])
    x, n, c = torch.randn(2, 7), torch.randn(2, 7), torch.randn(2, 8)
    with torch.no_grad():
        torch.testing.assert_close(a.sample(x, c, n, steps=2), b.sample(x, c, n, steps=2))
    old = asdict(cfg)
    del old['backbone']
    assert ModelConfig(**old).backbone == 'gene_token'
    with pytest.raises(ValueError):
        config(backbone='unknown')


@pytest.mark.parametrize("mode", ['observed', 'population_mean', 'condition_only'])
def test_reference_ablation(mode):
    model = build_response_model(config(reference_mode=mode), graph())
    x = torch.randn(3, 7)
    expected = x if mode == 'observed' else x.mean(0).expand_as(x) if mode == 'population_mean' else torch.zeros_like(x)
    torch.testing.assert_close(model.condition_reference(x), expected)
    torch.testing.assert_close(model.condition_reference(expected), expected)
