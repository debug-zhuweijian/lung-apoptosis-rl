"""Assay scope and actual scalar/batched calls remain explicit in schema2."""

from types import SimpleNamespace

import pytest
import torch

from test_world_execution_v1 import world_factory, cpu_thread_and_seed
from lung.engine.config import StageUnavailable
from lung.engine.world import INITIAL_STATE_VERSION, endpoint_inference_scope


def scope_fixture():
    config = {"endpoint_protocol_id": "assay-a", "population": 4,
              "history_support": "single_exposure_only",
              "response": {"inference_ode_steps": 2, "inference_microbatch": 1}}
    generation = {"population": 4, "inference_ode_steps": 2, "inference_microbatch": 1,
                  "initial_state_interface": INITIAL_STATE_VERSION,
                  "history_support": "single_exposure_only"}
    model = SimpleNamespace(endpoint_schema_version=2, endpoint_protocol_ids=("assay-a",),
                            feature_generation={"world_inference": generation}, feature_pipeline_stages=())
    return config, model


def test_protocol_and_sampling_change_are_rejected():
    config, model = scope_fixture()
    assert endpoint_inference_scope(model, config, {})["endpoint_protocol_id"] == "assay-a"
    with pytest.raises(StageUnavailable, match="PROTOCOL"):
        endpoint_inference_scope(model, {**config, "endpoint_protocol_id": "assay-b"}, {})
    with pytest.raises(StageUnavailable, match="GENERATION_SCOPE"):
        endpoint_inference_scope(model, config, {"inference_ode_steps": 3})
    model.feature_generation = {}
    with pytest.raises(StageUnavailable, match="GENERATION_SCOPE"):
        endpoint_inference_scope(model, config, {})


def test_schema1_requires_no_new_fields():
    assert endpoint_inference_scope(SimpleNamespace(), {}, {}) == {}


def test_new_feature_stage_cannot_be_added_after_endpoint_training():
    config, model = scope_fixture()
    with pytest.raises(StageUnavailable, match="STAGE_SET_MISMATCH"):
        endpoint_inference_scope(model, config, {"correction": "new-correction.pt"})


def test_both_world_paths_pass_explicit_protocol(world_factory):
    world, _, _, _ = world_factory(1)
    seen = []

    class ScopedEndpoint(torch.nn.Module):
        def forward(self, features, history, lengths, *, endpoint_protocol_id):
            assert endpoint_protocol_id == "assay-a"
            seen.append(endpoint_protocol_id)
            return features.new_full((len(features),), .25)

    world.endpoint = ScopedEndpoint()
    world.endpoint.endpoint_schema_version = 2
    world.endpoint.feature_generation = {"world_inference": {
        "population": world.n_pop, "inference_ode_steps": world.steps}}
    world.endpoint_call_kwargs = {"endpoint_protocol_id": "assay-a"}
    root = world.roots("T")[0]
    sequence = (next(iter(world.actions)),)
    assert world.score_one(root, sequence, 42) == .25
    assert torch.equal(world.score(root, sequence, (43, 44)), torch.tensor([.25, .25], dtype=torch.float64))
    assert seen == ["assay-a", "assay-a", "assay-a"]
    with pytest.raises(StageUnavailable, match="RESOLUTION_OUTSIDE"):
        world.score_one(root, sequence, 45, n=world.n_pop + 1)
    with pytest.raises(StageUnavailable, match="RESOLUTION_OUTSIDE"):
        world.score(root, sequence, (45,), ode_steps=world.steps + 1)
    v_root = world.roots("V")[0]
    with pytest.raises(StageUnavailable, match="RESOLUTION_OUTSIDE"):
        world.score(v_root, sequence, (46,), n=world.n_pop + 1, ode_steps=world.steps + 1)
