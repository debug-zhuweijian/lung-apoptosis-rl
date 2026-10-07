"""Numerical and gradient contracts, using explicitly synthetic tiny panels."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models import (ModelConfig, ResponseModel, PopulationPool, StateClassifier,
                         EndpointHead, PotentialHead, BackwardModel, CorrectionModel,
                         ErrorEstimator, HistoryEncoder, entropic_ot, sample_ot_pairs,
                         physical_projection, rbf_mmd, rbf_kernel, heun_integrate,
                         biological_features, average_gene_ranks, split_conformal_quantile)
from lung.models.sparse import SparseAttention, validate_neighbors


@pytest.fixture(autouse=True)
def single_thread_and_seed():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(919)
    yield
    torch.set_num_threads(before)


def tiny_config(checkpoint=True):
    return ModelConfig(n_genes=7, d_model=8, n_heads=2, encoder_layers=1, pad_layers=1,
                       query_chunk=3, checkpoint_blocks=checkpoint,
                       checkpoint_attention=checkpoint, n_sources=2)


def neighbors(genes=7):
    own = torch.arange(genes)
    return torch.stack((own, (own + 1) % genes, (own + 2) % genes), dim=-1)


def conditions(batch=2):
    return dict(fingerprint=torch.randn(batch, 2048), dose=torch.full((batch,), .5),
                duration=torch.full((batch,), 24.0), background=torch.zeros(batch, dtype=torch.long),
                source=torch.zeros(batch, dtype=torch.long), protocol=torch.zeros(batch, dtype=torch.long),
                elapsed=torch.zeros(batch), horizon=torch.full((batch,), 24.0))


def test_defaults_match_architecture_dimensions():
    cfg = ModelConfig()
    assert (cfg.n_genes, cfg.d_model, cfg.n_heads, cfg.encoder_layers, cfg.pad_layers) == (32461, 256, 4, 2, 6)
    assert (cfg.condition_input_dim, cfg.history_input_dim, cfg.state_dim) == (674, 353, 637)
    with pytest.raises(ValueError):
        ModelConfig(d_model=255)


@pytest.mark.parametrize("differential", [False, True])
def test_sparse_attention_equals_masked_dense_reference(differential):
    layer = SparseAttention(8, 2, 3, differential, True)
    x = torch.randn(2, 7, 8, requires_grad=True)
    y = torch.randn(2, 7, 8, requires_grad=True)
    graph = neighbors()
    q = layer.q1(x).reshape(2, 7, 2, 4).transpose(1, 2)
    k = layer.k1(y).reshape(2, 7, 2, 4).transpose(1, 2)
    v = layer.value(y).reshape(2, 7, 2, 4).transpose(1, 2)
    mask = torch.zeros(7, 7, dtype=torch.bool).scatter(1, graph, True)
    scores = (q @ k.transpose(-1, -2) / 2).masked_fill(~mask, -torch.inf)
    weights = scores.softmax(-1)
    if differential:
        q2 = layer.q2(x).reshape(2, 7, 2, 4).transpose(1, 2)
        k2 = layer.k2(y).reshape(2, 7, 2, 4).transpose(1, 2)
        w2 = (q2 @ k2.transpose(-1, -2) / 2).masked_fill(~mask, -torch.inf).softmax(-1)
        weights = weights - 2 * layer.lambda_logit.sigmoid() * w2
    expected = layer.output((weights @ v).transpose(1, 2).reshape(2, 7, 8))
    actual = layer(x, y, graph)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert x.grad.isfinite().all() and y.grad.isfinite().all()
    bad = graph.clone()
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError, match="Repeated"):
        validate_neighbors(bad, 7)


def test_cfm_target_and_condition_errors():
    model = ResponseModel(tiny_config(), neighbors())
    x, y, noise = (torch.randn(2, 7) for _ in range(3))
    args = conditions()
    cond = model.encode_condition(**args)
    tau = torch.tensor([.2, .8])
    z = (1 - .999 * tau[:, None]) * noise + tau[:, None] * y
    actual = model.cfm_loss(x, y, cond, noise, tau)
    expected = (model(z, tau, x, cond) - (y - .999 * noise)).square().mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert model.velocity_model.decoder[-1].weight.grad.abs().sum() > 0
    assert model.condition_encoder.drug.network[0][0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="UNSUPPORTED_CONTEXT"):
        model.encode_condition(**{**args, "source": torch.tensor([0, 99])})
    with pytest.raises(ValueError, match="INVALID_DOSE"):
        model.encode_condition(**{**args, "dose": torch.tensor([float("nan"), .5])})
    valid = model.encode_condition(**{**args, "dose": torch.tensor([float("nan"), .5]),
                                     "vehicle": torch.tensor([True, False])})
    assert valid.isfinite().all()


def test_complete_heun_trajectory_and_derivative():
    coefficient = torch.tensor(.3, requires_grad=True)
    initial = torch.ones(2, 3, requires_grad=True)
    result = heun_integrate(lambda z, tau: coefficient * z, initial, 32, True)
    increment = 1 + coefficient / 32 + .5 * (coefficient / 32).square()
    expected = initial * increment.pow(32)
    torch.testing.assert_close(result, expected)
    gradient = torch.autograd.grad(result.sum(), (coefficient, initial))
    exact_gradient = torch.autograd.grad(expected.sum(), (coefficient, initial))
    for actual, wanted in zip(gradient, exact_gradient):
        torch.testing.assert_close(actual, wanted)
    with pytest.raises(FloatingPointError, match="NONFINITE_ODE"):
        heun_integrate(lambda z, tau: z * float("nan"), initial.detach(), 2)


def test_full_flow_mmd_backward_and_two_pass_gradients():
    """Exact endpoint-chain-rule microbatching equals a retained full ODE graph."""
    first = ResponseModel(tiny_config(), neighbors())
    second = ResponseModel(tiny_config(), neighbors())
    # Exercise nonzero cross-attention and encoder gradients after initial gates.
    with torch.no_grad():
        for module in first.velocity_model.blocks[0].modulations:
            module[-1].bias.fill_(.03)
    second.load_state_dict(first.state_dict())
    x, noise, observed = torch.randn(3, 7), torch.randn(3, 7), torch.randn(4, 7)
    args = conditions(3)
    generated = first.sample(x, first.encode_condition(**args), noise, steps=2)
    native_loss = rbf_mmd(generated, observed, bandwidth=.7)
    native_loss.backward()
    with torch.no_grad():
        endpoints = torch.cat([second.sample(x[i:i + 1], second.encode_condition(**{k: v[i:i + 1] for k, v in args.items()}),
                                              noise[i:i + 1], steps=2) for i in range(3)])
    leaf = endpoints.detach().requires_grad_()
    micro_loss = rbf_mmd(leaf, observed, bandwidth=.7)
    gradient, = torch.autograd.grad(micro_loss, leaf)
    for index in range(3):
        sample = second.sample(x[index:index + 1],
                               second.encode_condition(**{k: v[index:index + 1] for k, v in args.items()}),
                               noise[index:index + 1], steps=2)
        sample.backward(gradient[index:index + 1])
    torch.testing.assert_close(native_loss, micro_loss)
    parameters = dict(second.named_parameters())
    for name, parameter in first.named_parameters():
        other = parameters[name]
        if parameter.grad is not None:
            assert other.grad is not None, name
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-4, msg=name)
    assert first.cell_encoder.blocks[0].attention.q1.weight.grad.abs().sum() > 0


def test_mmd_three_terms_v_and_u_are_distinct():
    x = torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
    y = torch.tensor([[1., 0.], [3., 2.], [0., 0.]])
    xx, yy, xy = rbf_kernel(x, x, 1.), rbf_kernel(y, y, 1.), rbf_kernel(x, y, 1.)
    expected = xx.mean() + yy.mean() - 2 * xy.mean()
    torch.testing.assert_close(rbf_mmd(x, y, 1.), expected)
    expected_u = (xx.sum() - xx.trace()) / 2 + (yy.sum() - yy.trace()) / 6 - 2 * xy.mean()
    torch.testing.assert_close(rbf_mmd(x, y, 1., unbiased=True), expected_u)
    rbf_mmd(x, y, 1.).backward()
    assert x.grad.isfinite().all()


def test_ot_marginals_detached_and_failure_explicit():
    x, y = torch.randn(12, 7, requires_grad=True), torch.randn(10, 7)
    coupling, info = entropic_ot(x, y, return_info=True)
    assert coupling.dtype == torch.float64 and not coupling.requires_grad
    assert info["row_l1"] <= 1e-6 and info["column_l1"] <= 1e-6
    i, j = sample_ot_pairs(coupling, 40)
    assert i.shape == j.shape == (40,) and i.max() < 12 and j.max() < 10
    uniform = entropic_ot(torch.zeros(3, 7), torch.zeros(2, 7))
    torch.testing.assert_close(uniform, torch.full((3, 2), 1 / 6, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="OT_NUMERICAL_FAILURE"):
        entropic_ot(x, y, max_iter=1, tolerance=1e-18)


def test_projection_mass_invalid_rows_and_gradient():
    raw = torch.randn(2, 7, requires_grad=True)
    projected = physical_projection(raw, torch.ones(7) * 2, torch.ones(7))
    assert projected.valid.all()
    torch.testing.assert_close(projected.y.expm1().sum(1), torch.ones(2) * 10000)
    projected.x.square().mean().backward()
    assert raw.grad.isfinite().all()
    invalid = physical_projection(torch.ones(1, 7) * -10, torch.zeros(7), torch.ones(7))
    assert not invalid.valid.any() and invalid.x.isnan().all()


def test_history_pool_and_diagnostic_heads():
    cfg = tiny_config()
    history = HistoryEncoder(cfg)
    assert torch.equal(history(torch.empty(2, 0, cfg.history_input_dim)), torch.zeros(2, 8))
    sequence = torch.randn(2, 3, cfg.history_input_dim)
    encoded = history(sequence, torch.tensor([0, 2]))
    assert torch.equal(encoded[0], torch.zeros(8))
    encoded.sum().backward()
    assert history.gru.weight_ih_l0.grad.abs().sum() > 0
    pool = PopulationPool(8)
    cells = torch.randn(2, 5, 8)
    summary = pool(cells)
    pool.reconstruction_loss(cells).backward()
    assert pool.encoder[0].weight.grad.abs().sum() > 0
    cond, e_b, uncertainty = torch.randn(2, 8), torch.ones(2), torch.ones(2) * .2
    backward, correction, estimator = BackwardModel(7, 8), CorrectionModel(7, 8), ErrorEstimator(8)
    assert backward(summary, cond).shape == (2, 7)
    assert backward.diagnostic(summary, cond, None) is None
    raw = torch.randn(2, 4, 7)
    assert correction(raw, summary, cond, gate=False) is raw
    adjusted = correction(raw, summary.detach(), cond, e_b, uncertainty, training_candidate=True)
    torch.testing.assert_close(adjusted, raw)
    (adjusted + 1).square().mean().backward()
    assert correction.network[-1].weight.grad.abs().sum() > 0
    assert (estimator(summary, cond, e_b, uncertainty, torch.tensor([0, 1])) > 0).all()
    features = torch.randn(2, cfg.state_dim)
    assert EndpointHead(cfg.state_dim, 8)(features).shape == (2,)
    assert PotentialHead(cfg.state_dim, 8)(features, torch.tensor([True, False]))[0] == 0


def test_state_class_order_proxy_and_program_masks():
    classifier = StateClassifier(8)
    z = torch.randn(5, 8)
    torch.testing.assert_close(classifier.population_proxy(z), classifier.probabilities(z)[:, 1:].sum(-1).mean())
    with pytest.raises(ValueError, match="missing"):
        classifier.class_weights(torch.tensor([0, 1, 1]))
    torch.testing.assert_close(classifier.class_weights(torch.tensor([0, 1, 2])), torch.ones(3))
    x = torch.tensor([[3., 3., 1., 0., 0., 0., 2.]])
    torch.testing.assert_close(average_gene_ranks(x), torch.tensor([[1.5, 1.5, 4., 6., 6., 6., 3.]]))
    features, scores = biological_features(x, [list(range(7)), None, [0]], [7, 0, 100])
    assert features.shape == (27,) and scores.shape == (1, 3)
    torch.testing.assert_close(features[21:24], torch.tensor([0., 1., 1.]))
    torch.testing.assert_close(features[24:], torch.ones(3))
    assert split_conformal_quantile(torch.ones(3)).isinf()
    assert split_conformal_quantile(torch.arange(10.).float()) == 9
