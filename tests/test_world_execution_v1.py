"""Frozen-world batching preserves per-cell streams and keeps tensors on device."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import auxiliary, world
from lung.engine.response import condition_kwargs
from synthetic_data import SyntheticData
from lung.models import (ModelConfig, ResponseModel, PopulationPool, StateClassifier,
                         BackwardModel, CorrectionModel)
from lung.rl import stream_seed


@pytest.fixture(autouse=True)
def cpu_thread_and_seed():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(926)
    yield
    torch.set_num_threads(before)


def config_and_data(batch_size=1):
    shape = ModelConfig(n_genes=7, d_model=8, n_heads=2, encoder_layers=1,
                        pad_layers=1, query_chunk=3, context_dim=2,
                        checkpoint_blocks=False, checkpoint_attention=False)
    config = {"device": "cpu", "task": "PROXY-A549-V1", "population": 5,
              "synthetic": True, "history_support": "single_exposure_only",
              "response": {"model": vars(shape), "inference_ode_steps": 2,
                           "inference_microbatch": batch_size}}
    return shape, config, SyntheticData(config)


def response_model(shape, data):
    model = ResponseModel(shape, torch.as_tensor(data.neighbors)).eval()
    with torch.no_grad():
        for block in model.velocity_model.blocks:
            for modulation in block.modulations:
                modulation[-1].weight.normal_(std=.01)
                modulation[-1].bias.fill_(.05)
    return model.requires_grad_(False)


class RecordingResponse:
    def __init__(self, model):
        self.model, self.calls = model, []

    def encode_condition(self, **kwargs):
        return self.model.encode_condition(**kwargs)

    def sample(self, reference, condition, *, noise, steps, checkpoint):
        self.calls.append((reference.clone(), noise.clone()))
        return self.model.sample(reference, condition, noise=noise, steps=steps, checkpoint=checkpoint)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("step,member", [(0, 0), (2, 1)])
def test_generation_preserves_index_noise_streams_and_partial_batches(batch_size, step, member):
    shape, _, data = config_and_data(batch_size)
    model = response_model(shape, data)
    recording = RecordingResponse(model)
    owner = SimpleNamespace(device=torch.device("cpu"), shape=shape,
                            inference_microbatch=batch_size,
                            protocol=SimpleNamespace(horizon_ticks=96, ticks_per_hour=1))
    batch = data.sample_response("T-0", 5, np.random.default_rng(926), "T")
    expression = torch.as_tensor(batch["control"])
    seed, elapsed = 31, 24.
    actual = world.BiologicalWorld._generate_member(owner, recording, expression, batch,
                                                   seed, step, member, elapsed, 5, 2)
    expected, noises = [], []
    with torch.no_grad():
        for index in range(5):
            generator = torch.Generator(device="cpu").manual_seed(stream_seed(seed, "noise", step, member, index))
            noise = torch.randn((1, shape.n_genes), generator=generator)
            condition = model.encode_condition(**condition_kwargs(batch, 1, "cpu", elapsed, 96.))
            expected.append(model.sample(expression[index:index+1], condition, noise, steps=2, checkpoint=False))
            noises.append(noise)
    torch.testing.assert_close(torch.cat([row[0] for row in recording.calls]), expression, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([row[1] for row in recording.calls]), torch.cat(noises), rtol=0, atol=0)
    assert [len(row[0]) for row in recording.calls] == [min(batch_size, 5-i) for i in range(0, 5, batch_size)]
    torch.testing.assert_close(actual, torch.cat(expected), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_encode_vectors_keeps_cpu_default_and_matches_single_cell_encoding(batch_size):
    shape, _, data = config_and_data()
    model = response_model(shape, data)
    values = np.random.default_rng(32).normal(size=(5, shape.n_genes))
    with torch.no_grad():
        expected = torch.cat([model.encode_cells(torch.tensor(row[None], dtype=torch.float32))[1]
                              for row in values])
    actual = auxiliary.encode_vectors(model, values, "cpu", batch_size=batch_size)
    assert actual.device.type == "cpu" and not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_tensor_encoding_never_uses_numpy_conversion(monkeypatch):
    shape, _, data = config_and_data()
    model = response_model(shape, data)
    expression = torch.randn(5, shape.n_genes, requires_grad=True)
    expected = auxiliary.encode_vectors(model, expression.detach().numpy(), "cpu", batch_size=3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor input must never take the NumPy conversion path")

    monkeypatch.setattr(auxiliary, "tensor", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    actual = auxiliary.encode_vectors(model, expression, "cpu", batch_size=3, output_device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad


@pytest.fixture
def world_factory(monkeypatch):
    shape, config, data = config_and_data()
    members = [response_model(shape, data), response_model(shape, data)]
    auxiliaries = {"pool": PopulationPool(shape.d_model), "state": StateClassifier(shape.d_model),
                   "backward": BackwardModel(shape.n_genes, shape.d_model),
                   "correction": CorrectionModel(shape.n_genes, shape.d_model)}
    for module in auxiliaries.values():
        module.eval().requires_grad_(False)
    pipeline = {"response": ["member-0", "member-1"], **{key: key for key in auxiliaries}}
    monkeypatch.setattr(world, "load_response", lambda path, *_: members[int(path[-1])])
    monkeypatch.setattr(world, "load_auxiliary", lambda path, *_, **kwargs: auxiliaries[path])
    monkeypatch.setattr(world, "file_hash", lambda value: "hash-" + str(value))
    monkeypatch.setattr(auxiliary, "file_hash", lambda value: "hash-" + str(value))

    def make(batch_size):
        updated = deepcopy(config)
        updated["response"]["inference_microbatch"] = batch_size
        return world.BiologicalWorld(data, updated, pipeline), updated, data, pipeline

    return make


def test_world_identity_and_feature_cache_binding_include_batching(world_factory):
    first, config1, data, pipeline = world_factory(1)
    second, config2, _, _ = world_factory(2)
    third, config3, _, _ = world_factory(3)
    assert len({first.biological_identity, second.biological_identity, third.biological_identity}) == 3
    for stage in ("pool", "state"):
        assert len({auxiliary.pipeline_identity(stage, data, config, pipeline)
                    for config in (config1, config2, config3)}) == 3
    legacy = deepcopy(config1)
    del legacy["response"]["inference_microbatch"]
    assert auxiliary.pipeline_identity("pool", data, legacy, pipeline) == auxiliary.pipeline_identity("pool", data, config1, pipeline)


def test_world_tensor_features_correction_and_score_use_configured_batches(world_factory, monkeypatch):
    owner, _, _, _ = world_factory(3)
    monkeypatch.setattr(owner, "_gate", lambda _: True)
    calls = []
    original = world.encode_vectors

    def record(reference, expression, device, **kwargs):
        assert isinstance(expression, torch.Tensor)
        assert kwargs == {"batch_size": 3, "output_device": owner.device}
        calls.append(len(expression))
        return original(reference, expression, device, **kwargs)

    monkeypatch.setattr(world, "encode_vectors", record)
    root = owner.roots("T")[0]
    prefix = (next(iter(owner.actions)),)
    state = owner.observe(root, prefix, 17)
    score = owner.score_one(root, prefix, 17)
    assert state.features.isfinite().all() and np.isfinite(score)
    # Correction encoding, state encoding, and scoring encoding all stay tensors.
    assert calls == [10, 10, 10, 5]


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("member_count", [1, 2])
def test_batched_world_fixed_sequence_score_matches_single_cell_world(world_factory, gate, member_count):
    baseline, _, _, _ = world_factory(1)
    baseline.members = baseline.members[:member_count]
    baseline._gate = lambda _: gate
    with torch.no_grad():
        baseline.correction.network[-1].bias.fill_(.05)
    root = baseline.roots("T")[0]
    prefix = (next(iter(baseline.actions)),)
    seeds = (51, 52)
    expected = [baseline.score_one(root, prefix, seed) for seed in seeds]
    for batch_size in (2, 3):
        owner, _, _, _ = world_factory(batch_size)
        owner.members = owner.members[:member_count]
        owner._gate = lambda _: gate
        actual = [owner.score_one(root, prefix, seed) for seed in seeds]
        assert actual == pytest.approx(expected, abs=2e-6, rel=2e-5)
        batched = owner._score_seed_batch(root, prefix, seeds, owner.n_pop, owner.steps)
        assert batched == pytest.approx(expected, abs=2e-6, rel=2e-5)


@pytest.mark.parametrize("correction", ["absent", "disabled", "enabled"])
def test_world_skips_inactive_correction_and_reuses_projection(world_factory, monkeypatch, correction):
    owner, _, _, _ = world_factory(3)
    monkeypatch.setattr(owner, "_gate", lambda _: correction == "enabled")

    def forbidden(*args, **kwargs):
        raise AssertionError("Inactive correction must not run backward or correction diagnostics")

    if correction == "absent":
        owner.backward = owner.correction = None
    elif correction == "disabled":
        owner.backward = owner.correction = forbidden
    calls = []
    original = world.projected

    def project(raw, data):
        calls.append(raw)
        return original(raw, data)

    monkeypatch.setattr(world, "projected", project)
    root = owner.roots("T")[0]
    prefix = (next(iter(owner.actions)),)
    expression, _ = owner.propagate(root, prefix, 51)
    expected_calls = len(owner.members) * (2 if correction == "enabled" else 1)
    assert expression.isfinite().all() and len(calls) == expected_calls
    calls.clear()
    scores = owner._score_seed_batch(root, prefix, (51, 52), owner.n_pop, owner.steps)
    assert np.isfinite(scores).all() and len(calls) == 2 * expected_calls


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_encoding_batch_sizes_are_rejected(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        auxiliary.encode_vectors(None, torch.zeros(1, 7), "cpu", batch_size=batch_size)
