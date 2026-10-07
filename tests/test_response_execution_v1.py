"""Batching and UI throttling preserve the response training objective."""

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.models import ResponseModel, rbf_mmd
from lung.engine.response import exact_mmd_backward, TrainingDetailProgress
from test_models_v1 import tiny_config, neighbors, conditions


@pytest.mark.parametrize('microbatch', [1, 2, 3, 5])
def test_batched_full_ode_mmd_gradient_equals_native(microbatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(675)
        native = ResponseModel(tiny_config(), neighbors())
        replay = ResponseModel(replace(tiny_config(), query_chunk=7), neighbors())
        with torch.no_grad():
            for module in native.velocity_model.blocks[0].modulations:
                module[-1].bias.fill_(.03)
        replay.load_state_dict(native.state_dict())
        x, y, noise = torch.randn(5, 7), torch.randn(4, 7), torch.randn(5, 7)
        base = conditions(1)
        def encode(model, n):
            return model.encode_condition(**{k: v.expand(n, *v.shape[1:]) for k, v in base.items()})
        generated = native.sample(x, encode(native, 5), noise=noise, steps=3)
        expected = rbf_mmd(generated, y, bandwidth=.7)
        (expected * .125).backward()
        actual = exact_mmd_backward(replay, x, y, lambda n: encode(replay, n),
            bandwidth=.7, steps=3, scale=.125, noise=noise, microbatch=microbatch)
        assert actual == pytest.approx(float(expected.detach()), abs=1e-6)
        candidates = dict(replay.named_parameters())
        for name, parameter in native.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, candidates[name].grad,
                                           rtol=3e-4, atol=2e-6, msg=name)
        assert native.cell_encoder.gene_embedding.weight.grad.abs().sum() > 0
    finally:
        torch.set_num_threads(old)


def test_training_subprogress_throttles_but_phase_and_end_remain_visible():
    bar = Mock()
    with patch('lung.engine.response.time.monotonic', side_effect=[0., .1, .2, .6, .7, .8]):
        progress = TrainingDetailProgress(bar)
        progress('CFM', 0, 128)
        progress('CFM', 1, 128)
        progress('CFM', 2, 128)
        progress('CFM', 40, 128)
        progress('CFM', 128, 128)
        progress('MMD', 0, 32)
    assert bar.set_postfix_str.call_count == 4
    assert bar.set_postfix_str.call_args.args[0] == 'MMD 0/32'
