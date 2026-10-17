"""Small synthetic regressions only; no real training or biological claims."""

import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.checkpoint import seed_all
from lung.engine.config import load_config
from lung.engine.response import response_rngs, train_condition
from synthetic_data import SyntheticData, smoke_config
from lung.models import ResponseModel


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_training():
    config = smoke_config(load_config())
    config["response"]["members"] = 1
    data = SyntheticData(config)
    from lung.models import ModelConfig
    shape = ModelConfig(**config["response"]["model"])
    seed_all(13)
    model = ResponseModel(shape, torch.as_tensor(data.neighbors))
    return config, data, model


def test_zero_weight_bypasses_second_sampling_ode_and_mmd():
    config, data, model = small_training()
    cfg = {**config["response"], "mmd_weight": 0.}
    cfg.pop("mmd_samples")
    cfg.pop("train_ode_steps")
    with patch.object(data, "sample_response", wraps=data.sample_response) as sample:
        with patch("lung.engine.response.exact_mmd_backward", side_effect=AssertionError("must bypass")):
            result = train_condition(model, data, "T-0", np.random.default_rng(13), cfg, "cpu", 1)
    assert sample.call_count == 1
    assert result["mmd_loss"] is None and result["weighted_mmd_loss"] is None
    assert result["mmd_computed"] == 0 and result["loss"] == result["cfm_loss"]


def test_shared_condition_cell_ot_tau_epsilon_streams_match_with_or_without_mmd():
    config, data, initial = small_training()
    records = []
    for weight in (0., 1.):
        model = copy.deepcopy(initial)
        seed_all(37)
        cfm_rng, extra_rng, extra_torch = response_rngs(37, "cpu")
        cells, cfm_inputs = [], []
        original_sample, original_loss = data.sample_response, model.cfm_loss

        def sample(key, n, rng, role):
            value = original_sample(key, n, rng, role)
            if rng is cfm_rng:
                cells.append((key, value["control"].copy(), value["target"].copy()))
            return value

        def loss(reference, target, condition):
            epsilon = torch.randn_like(target)
            tau = torch.rand(len(target), device=target.device)
            cfm_inputs.append(tuple(value.detach().clone() for value in (reference, target, epsilon, tau)))
            return original_loss(reference, target, condition, noise=epsilon, tau=tau)

        keys = list(data.iter_response_epoch("T", 37)) * 2
        with patch.object(data, "sample_response", side_effect=sample), patch.object(model, "cfm_loss", side_effect=loss):
            metrics = [train_condition(model, data, key, cfm_rng,
                       {**config["response"], "mmd_weight": weight}, "cpu", 1,
                       mmd_rng=extra_rng, mmd_generator=extra_torch) for key in keys]
        records.append((keys, cells, cfm_inputs, cfm_rng.bit_generator.state,
                        torch.get_rng_state().clone(), metrics, extra_torch.get_state().clone()))
    a, b = records
    assert a[0] == b[0] and a[3] == b[3]
    for left, right in zip(a[1], b[1]):
        assert left[0] == right[0]
        for x, y in zip(left[1:], right[1:]):
            np.testing.assert_array_equal(x, y)
    for left, right in zip(a[2], b[2]):
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(a[4], b[4], rtol=0, atol=0)
    assert not torch.equal(a[6], b[6])
    assert [m["cfm_loss"] for m in a[5]] == [m["cfm_loss"] for m in b[5]]


def test_extra_branch_weight_and_stream_requirements():
    config, data, model = small_training()
    shared, extra, torch_extra = response_rngs(14, "cpu")
    cfg = {**config["response"], "mmd_weight": .25}
    with pytest.raises(ValueError, match="separate"):
        train_condition(model, data, "T-0", shared, cfg, "cpu", 2)
    with patch("lung.engine.response.exact_mmd_backward", return_value=4.) as mmd:
        result = train_condition(model, data, "T-0", shared, cfg, "cpu", 2,
                                 mmd_rng=extra, mmd_generator=torch_extra)
    assert mmd.call_args.kwargs["scale"] == .125
    assert mmd.call_args.kwargs["noise"].shape == (cfg["mmd_samples"], data.n_genes)
    assert result["weighted_mmd_loss"] == 1.
    assert result["loss"] == result["cfm_loss"] + 1.
