"""CPU equivalence against original gene-major sparse/PAD operations."""
from copy import deepcopy
from pathlib import Path
import math
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.models import ModelConfig
from lung.models.sparse import SparseAttention, PADBlock


@pytest.fixture(autouse=True)
def cpu_seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(927)
    yield
    torch.set_num_threads(previous)


def graph():
    own = torch.arange(7)
    return torch.stack((own, (own + 3) % 7, (own + 6) % 7), dim=1)


def original_attention(module, query, keys, neighbors):
    batch, genes, _ = query.shape
    shape = (batch, genes, module.heads, module.head_dim)
    q1, k1, value = (layer(x).reshape(shape) for layer, x in
                     ((module.q1, query), (module.k1, keys), (module.value, keys)))
    q2 = module.q2(query).reshape(shape) if module.differential else None
    k2 = module.k2(keys).reshape(shape) if module.differential else None
    outputs = []
    for start in range(0, genes, module.chunk):
        end = min(start + module.chunk, genes)
        indices = neighbors[start:end]
        def gather(x):
            return x[:, indices.reshape(-1)].reshape(batch, end-start, indices.shape[1], module.heads, module.head_dim)
        score = torch.einsum('bchd,bckhd->bchk', q1[:, start:end], gather(k1)) / math.sqrt(module.head_dim)
        weights = score.softmax(-1)
        if module.differential:
            score2 = torch.einsum('bchd,bckhd->bchk', q2[:, start:end], gather(k2)) / math.sqrt(module.head_dim)
            weights = weights - 2 * module.lambda_logit.sigmoid() * score2.softmax(-1)
        outputs.append(torch.einsum('bchk,bckhd->bchd', weights, gather(value)).reshape(batch, end-start, module.width))
    return module.output(torch.cat(outputs, dim=1))


def compare(module, reference, inputs, actual, expected):
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert list(module.state_dict()) == list(reference.state_dict())
    for name, parameter in module.named_parameters():
        other = dict(reference.named_parameters())[name]
        assert parameter.grad is not None, name
        assert other.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-11, rtol=1e-10, msg=name)
    for left, right in inputs:
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize('differential', [False, True])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('projected', [False, True])
def test_attention_original_outputs_and_all_gradients(differential, checkpoint, projected):
    module = SparseAttention(12, 3, chunk=3, differential=differential, checkpoint_chunks=checkpoint).double()
    reference = deepcopy(module)
    q, k = [torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    qr, kr = [x.detach().clone().requires_grad_() for x in (q, k)]
    keys = module.project_keys(k)
    assert keys[0].shape == (2, 3, 7, 4) and keys[0].is_contiguous()
    actual = module.forward_projected(q, keys, graph()) if projected else module(q, k, graph())
    expected = original_attention(reference, qr, kr, graph())
    compare(module, reference, [(q, qr), (k, kr)], actual, expected)


def original_pad(module, tokens, reference, condition, time, neighbors):
    expanded = condition[:, None].expand(-1, tokens.shape[1], -1)
    tokens = tokens + module.injection(torch.cat((module.injection_norm(tokens), expanded), -1))
    conditioning = torch.cat((condition, time), -1)
    for index in range(3):
        alpha, beta, gate = module.modulations[index](conditioning).chunk(3, -1)
        value = (1 + alpha[:, None]) * module.norms[index](tokens) + beta[:, None]
        if index == 0:
            update = original_attention(module.self_attention, value, value, neighbors)
        elif index == 1:
            update = original_attention(module.cross_attention, value, module.cross_norm(reference), neighbors)
        else:
            update = module.ffn(value)
        tokens = tokens + gate[:, None].tanh() * update
    return tokens


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('cached', [False, True])
def test_pad_original_and_reused_reference_all_gradients(checkpoint, cached):
    config = ModelConfig(n_genes=7, d_model=12, n_heads=3, query_chunk=3, checkpoint_attention=checkpoint)
    module = PADBlock(config).double()
    with torch.no_grad():
        for modulation in module.modulations:
            modulation[-1].weight.normal_(std=.08)
            modulation[-1].bias.normal_(std=.08)
    reference = deepcopy(module)
    shapes = [(2, 7, 12), (2, 7, 12), (2, 12), (2, 12)]
    xs = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in shapes]
    rs = [x.detach().clone().requires_grad_() for x in xs]
    prepared = module.prepare_reference(xs[1]) if cached else None
    actual = sum(module(xs[0] * scale, *xs[1:], graph(), reference_keys=prepared) for scale in (1., .7))
    expected = sum(original_pad(reference, rs[0] * scale, *rs[1:], graph()) for scale in (1., .7))
    compare(module, reference, list(zip(xs, rs)), actual, expected)


def test_injection_split_linear_equivalence():
    layer = torch.nn.Linear(24, 24).double()
    reference = deepcopy(layer)
    tokens = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    condition = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
    tr, cr = [x.detach().clone().requires_grad_() for x in (tokens, condition)]
    split = F.gelu(F.linear(tokens, layer.weight[:, :12]) + F.linear(condition, layer.weight[:, 12:], layer.bias)[:, None])
    old = F.gelu(reference(torch.cat((tr, cr[:, None].expand(-1, 7, -1)), -1)))
    compare(layer, reference, [(tokens, tr), (condition, cr)], split, old)
