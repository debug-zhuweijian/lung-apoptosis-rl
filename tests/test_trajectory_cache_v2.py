"""Trajectory-local projection reuse preserves complete ODE values and derivatives."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models import ModelConfig, ResponseModel


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model(checkpoint):
    torch.manual_seed(928)
    config = ModelConfig(n_genes=7, d_model=8, n_heads=2, encoder_layers=1,
                         pad_layers=2, query_chunk=3, checkpoint_blocks=checkpoint,
                         checkpoint_attention=checkpoint)
    genes = torch.arange(7)
    neighbors = torch.stack([genes, (genes + 3) % 7, (genes + 5) % 7], dim=1)
    result = ResponseModel(config, neighbors).double()
    with torch.no_grad():
        for block in result.velocity_model.blocks:
            for modulation in block.modulations:
                modulation[-1].weight.normal_(0., .02)
                modulation[-1].bias.normal_(0., .03)
    return result


@pytest.mark.parametrize("checkpoint", [False, True])
def test_full_32_steps_cached_reference_matches_outputs_and_all_gradients(checkpoint):
    cached = model(checkpoint)
    reference_model = deepcopy(cached)
    x = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    c = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    c_reference = c.detach().clone().requires_grad_()
    noise = torch.randn_like(x)
    out = cached.sample(x, c, noise=noise, steps=32, checkpoint=checkpoint, cache_reference=True)
    expected = reference_model.sample(x_reference, c_reference, noise=noise,
                                      steps=32, checkpoint=checkpoint, cache_reference=False)
    torch.testing.assert_close(out, expected, rtol=2e-10, atol=2e-11)
    out.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(c.grad, c_reference.grad, rtol=2e-9, atol=2e-10)
    other = dict(reference_model.named_parameters())
    for name, p in cached.named_parameters():
        q = other[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-9, atol=2e-10, msg=name)
    assert cached.cell_encoder.gene_embedding.weight.grad.abs().sum() > 0
    for block in cached.velocity_model.blocks:
        assert block.cross_attention.k1.weight.grad.abs().sum() > 0


def test_reference_projection_is_once_per_sample_and_not_reused_across_calls():
    net = model(False)
    counts = [0, 0]
    handles = []
    for index, block in enumerate(net.velocity_model.blocks):
        def count(_module, _args, _output, index=index):
            counts[index] += 1
        handles.append(block.cross_norm.register_forward_hook(count))
    x, c = torch.randn(2, 7, dtype=torch.float64), torch.randn(2, 8, dtype=torch.float64)
    with torch.no_grad():
        net.sample(x, c, steps=32)
        assert counts == [1, 1]
        net.sample(x + 1, c, steps=32)
        assert counts == [2, 2]
        net.sample(x, c, steps=2, cache_reference=False)
        assert counts == [6, 6]
    for handle in handles:
        handle.remove()
    assert not any("cache" in name for name in net.state_dict())
