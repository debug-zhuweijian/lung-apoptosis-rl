"""Updated tiny synthetic components exercise modular wiring, not biological efficacy."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung.engine import diagnostics
from lung.engine.auxiliary import (calibration_audit_binding, feature_cache, load_auxiliary,
                                   pipeline_identity, train_auxiliary,
                                   verify_auxiliary_dependencies)
from lung.engine.checkpoint import atomic_save, load_checkpoint, save_training
from lung.engine.config import StageUnavailable, file_hash
from lung.engine.latent_pipeline import _load_shared, load_latent_adapter
from lung.engine.representation import load_representation
from lung.engine.representation_cache import encoder_identity
from lung.engine.response import condition_kwargs, tensor
from lung.engine.response_loader import load_response
from lung.engine.response_research import _load_for_comparison
from lung.engine.residuals import prediction_rows
from lung.engine.synthetic import SyntheticData
from lung.engine.world import BiologicalWorld
from lung.models import ModelConfig, ResponseModel
from lung.models.latent_staged import LatentMember, SharedAutoencoder
from lung.rl import ModelRollout, PotentialNetwork


class Monitor:
    def __init__(self, root):
        self.weights, self.experiment = root / "weights", root / "experiment"
        self.weights.mkdir(parents=True)
        self.experiment.mkdir(parents=True)

    def progress(self, iterable, **kwargs):
        return iterable

    def record(self, *args, **kwargs):
        pass


class TrackedData(SyntheticData):
    def __init__(self, config):
        super().__init__(config)
        self.roles = []

    def sample_response(self, condition_id, n, rng, role):
        self.roles.append(role)
        return super().sample_response(condition_id, n, rng, role)

    def state_arrays(self, fold, role):
        self.roles.append("state-" + role)
        return super().state_arrays(fold, role)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(107)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def components(tmp_path):
    shape = ModelConfig(n_genes=8, d_model=8, latent_dim=4, n_heads=2,
                        encoder_layers=1, pad_layers=1, context_dim=2,
                        backbone="compact_full_gene", checkpoint_blocks=False)
    config = {"device": "cpu", "task": "PROXY-A549-V1", "epochs": 1,
              "seed": 107, "population": 3, "synthetic": True,
              "history_support": "single_exposure_only", "aux_batch": 2,
              "state_batch": 6, "aux_lr": 1e-3, "save_period": 1,
              "calibration_steps": 2, "state_fold": "1_to_8",
              "research": {"latent": {"d_model": 8, "latent_dim": 4}},
              "response": {"model": asdict(shape), "inference_ode_steps": 2,
                           "inference_microbatch": 2}}
    data, monitor = TrackedData(config), Monitor(tmp_path)
    shared = SharedAutoencoder(shape, torch.as_tensor(data.neighbors))
    optimizer = torch.optim.Adam(shared.parameters(), lr=1e-3)
    batch = data.sample_response("T-0", 3, np.random.default_rng(107), "T")
    shared.reconstruction_loss(tensor(np.concatenate([batch["control"], batch["target"]]), "cpu")).backward()
    optimizer.step()
    path = tmp_path / "ae.pt"
    save_training(path, shared, optimizer, epoch=1, identity="SYNTHETIC-modular-ae",
                  metadata={"stage": "latent_ae", "model_config": asdict(shape),
                            "data_identity": data.identity, "synthetic": True},
                  counters={"optimizer_step": 1})
    reference = load_representation(path, data, "cpu")
    data.roles.clear()
    return data, config, monitor, {"representation": str(path)}, reference


def member_checkpoint(components, name="response", *, stage="latent_cfm", old=False):
    data, config, monitor, pipeline, representation = components
    shared, shape = representation.shared, representation.config
    member = LatentMember(shape, shared)
    member.init_decoder_from_shared()
    member.decoder.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in member.parameters() if p.requires_grad], lr=1e-3)
    batch = data.sample_response("T-0", 3, np.random.default_rng(207), "T")
    with torch.no_grad():
        h = shared.encode_h(tensor(batch["control"], "cpu"))[1]
        z = shared.project(shared.encode_h(tensor(batch["target"], "cpu"))[1])
    member.cfm_loss(z, h, member.condition_encoder(**condition_kwargs(batch, 3, "cpu"))).backward()
    optimizer.step()
    path = monitor.weights / f"{name}.pt"
    metadata = {"stage": stage, "model_config": asdict(shape),
                "data_identity": data.identity, "synthetic": True}
    if not old:
        metadata.update(representation_hash=representation.checkpoint_hash,
                        representation_identity=representation.identity,
                        cache=encoder_identity(shared, shape))
    save_training(path, member, optimizer, epoch=1, identity="SYNTHETIC-modular-member",
                  metadata=metadata, counters={"optimizer_step": 1})
    saved = load_checkpoint(path)
    atomic_save(path, {**saved, "shared": shared.state_dict()})
    return str(path)


def train_heads(components):
    data, config, monitor, pipeline, _ = components
    for stage in ("pool", "state"):
        pipeline = train_auxiliary(stage, data, config, monitor, pipeline)
    return pipeline


def test_independent_h_representation_trains_pool_state_and_preserves_roles(components):
    data, config, monitor, source, reference = components
    tokens, h = reference.encode_cells(torch.randn(3, 8))
    assert tokens.shape == (3, 1, 8) and h.shape == (3, 8)
    assert reference.shared.project(h).shape == (3, 4)
    pipeline = train_heads(components)
    assert "response" not in pipeline
    assert set(data.roles) == {"T", "V", "state-fit", "state-selection", "state-calibration"}
    assert load_auxiliary(pipeline["pool"], data, "cpu").width == 8
    for stage in ("pool", "state"):
        saved = load_checkpoint(pipeline[stage])
        assert saved["metadata"]["input_coordinate"] == "h"
        assert saved["metadata"]["dependencies"]["pipeline"] == {
            "representation": [file_hash(source["representation"])]}
        assert verify_auxiliary_dependencies(pipeline[stage], pipeline)
    changed = {**pipeline, "response": [member_checkpoint(components)]}
    assert pipeline_identity("pool", data, config, changed) == pipeline_identity("pool", data, config, pipeline)
    assert verify_auxiliary_dependencies(pipeline["state"], changed)
    changed_config = deepcopy(config)
    changed_config["response"]["inference_ode_steps"] *= 2
    assert pipeline_identity("state", data, changed_config, changed) == pipeline_identity("state", data, config, pipeline)


def test_changed_ae_rejects_auxiliary_and_feature_cache_even_with_same_tensors(components):
    data, config, monitor, source, _ = components
    pipeline = train_heads(components)
    identity = pipeline_identity("pool", data, config, pipeline)
    cache = monitor.experiment / "fixed-cache.pt"
    feature_cache(cache, identity, lambda: ["cached"])
    changed_path = monitor.weights / "changed-ae.pt"
    saved = load_checkpoint(source["representation"])
    atomic_save(changed_path, {**saved, "extra": {"new_metadata": True}})
    changed = {**pipeline, "representation": str(changed_path)}
    assert pipeline_identity("pool", data, config, changed) != identity
    with pytest.raises(StageUnavailable, match="DEPENDENCY_MISMATCH"):
        verify_auxiliary_dependencies(pipeline["pool"], changed)
    with pytest.raises(ValueError, match="FEATURE_CACHE_IDENTITY_MISMATCH"):
        feature_cache(cache, pipeline_identity("pool", data, config, changed), lambda: None)
    assert calibration_audit_binding(data, config, changed) != calibration_audit_binding(data, config, pipeline)


@pytest.mark.parametrize("change,error", [
    ("updates", "TRAINED_REPRESENTATION_REQUIRED"),
    ("data", "MODEL_DATA_IDENTITY_MISMATCH"),
    ("stage", "TRAINED_REPRESENTATION_REQUIRED"),
    ("nonfinite", "NONFINITE_MODEL_WEIGHTS"),
])
def test_representation_loader_rejects_untrained_incompatible_or_nonfinite(components, change, error):
    data, _, monitor, pipeline, _ = components
    saved = load_checkpoint(pipeline["representation"])
    if change == "updates":
        saved["counters"]["optimizer_step"] = 0
    elif change == "data":
        saved["metadata"]["data_identity"] = "another-gene-axis-and-normalization"
    elif change == "stage":
        saved["metadata"]["stage"] = "response"
    else:
        saved["model"]["latent_projection.weight"][0, 0] = float("nan")
    changed = monitor.weights / "invalid-ae.pt"
    atomic_save(changed, saved)
    with pytest.raises(ValueError, match=error):
        load_representation(changed, data, "cpu")


def test_shared_training_loader_checks_ae_and_requested_dimensions(components):
    data, config, _, pipeline, _ = components
    shared = _load_shared(config, data, "cpu", pipeline["representation"])
    assert shared.representation_hash == file_hash(pipeline["representation"])
    incompatible = deepcopy(config)
    incompatible["research"]["latent"]["latent_dim"] = 8
    with pytest.raises(ValueError, match="REPRESENTATION_MODEL_CONFIG_MISMATCH"):
        _load_shared(incompatible, data, "cpu", pipeline["representation"])


@pytest.mark.parametrize("stage", ["latent_cfm", "latent_members"])
def test_latent_loader_and_comparison_forward_h_and_condition_encoder(components, stage):
    data, _, _, _, reference = components
    path = member_checkpoint(components, stage=stage)
    for loader in (load_response, _load_for_comparison):
        model = loader(path, data, "cpu")
        assert model.condition_encoder is model.member.condition_encoder
        x = torch.randn(2, 8)
        assert model.encode_cells(x)[1].shape == (2, 8)
        assert model.sample_noise(x).shape == (2, 4)
        assert all(not p.requires_grad for p in model.shared.parameters())
        torch.testing.assert_close(model.encode_cells(x)[1], reference.encode_cells(x)[1])
    saved = load_checkpoint(path)
    saved["counters"]["optimizer_step"] = 0
    atomic_save(path, saved)
    with pytest.raises(ValueError, match="TRAINED_RESPONSE_REQUIRED"):
        load_latent_adapter(path, data, "cpu")
    saved["metadata"]["stage"] = "unknown"
    atomic_save(path, saved)
    with pytest.raises(ValueError, match="UNSUPPORTED_STAGE"):
        load_response(path, data, "cpu")


@pytest.mark.parametrize("members", [1, 5])
def test_latent_world_observe_scores_and_single_step_environment(components, members):
    data, config, monitor, _, _ = components
    pipeline = train_heads(components)
    pipeline["response"] = [member_checkpoint(components, name=f"member-{i}",
                                             stage="latent_members" if members > 1 else "latent_cfm")
                            for i in range(members)]
    world = BiologicalWorld(data, config, pipeline)
    assert all(member.shared is world.representation.shared for member in world.members)
    root = world.roots("T")[0]
    state = world.adapter.observe(root, (), 5)
    assert state.features.shape == (world.registry.spec.state_dim,)
    sequence = (next(iter(world.actions)),)
    seeds = (7, 11, 13)
    scalar = torch.tensor([world.score_one(root, sequence, seed) for seed in seeds], dtype=torch.float64)
    batched = world.adapter.score_many(root, ((sequence, seeds),))[0]
    torch.testing.assert_close(batched, scalar, rtol=2e-5, atol=2e-6)
    before_cache = len(world._score_cache.values)
    torch.testing.assert_close(world.adapter.score(root, sequence, seeds), batched, rtol=0, atol=0)
    assert len(world._score_cache.values) == before_cache
    potential = PotentialNetwork(world.registry.spec)
    optimizer = torch.optim.Adam(potential.parameters(), lr=1e-3)
    (potential([state])[0] - batched.mean().float()).square().backward()
    optimizer.step()
    potential_path = monitor.weights / "potential.pt"
    save_training(potential_path, potential, optimizer, epoch=1, identity="SYNTHETIC-potential",
                  counters={"optimizer_step": 1}, metadata={"stage": "potential",
                  "biological_identity": world.biological_identity, "synthetic": True})
    pipeline["potential"] = str(potential_path)
    final = BiologicalWorld(data, config, pipeline)
    rollout = ModelRollout(final.adapter, root, seed=9, score_repeats=2)
    mask = rollout.legal_mask
    transition = rollout.step(int(torch.nonzero(mask)[0]), mask.float() / mask.sum())
    assert transition.terminated and transition.next_state is None and transition.phi_next == 0
    assert np.isfinite(transition.reward) and 0 <= transition.base_reward <= 1
    assert all(not p.requires_grad for member in final.members for p in member.member.parameters())
    with pytest.raises(StageUnavailable, match="POPULATION_RESOLUTION_UNRESOLVED"):
        BiologicalWorld(data, {**config, "synthetic": False}, {k: v for k, v in pipeline.items() if k != "potential"})


def test_world_rejects_inconsistent_shared_ae_and_accepts_old_embedded_weights(components):
    data, config, monitor, _, _ = components
    pipeline = train_heads(components)
    path = member_checkpoint(components, old=True)
    pipeline["response"] = [path]
    assert BiologicalWorld(data, config, pipeline).members
    saved = load_checkpoint(path)
    saved["shared"]["latent_projection.weight"] += .1
    atomic_save(path, saved)
    with pytest.raises(ValueError, match="LATENT_REPRESENTATION_IDENTITY_MISMATCH"):
        BiologicalWorld(data, config, pipeline)


def test_response_velocity_change_preserves_heads_but_changes_world_identity(components):
    data, config, monitor, _, _ = components
    pipeline = train_heads(components)
    source = member_checkpoint(components)
    pipeline["response"] = [source]
    old = BiologicalWorld(data, config, pipeline)
    saved = load_checkpoint(source)
    saved["model"]["velocity_model.state_projection.weight"] += .1
    changed_path = monitor.weights / "velocity-changed.pt"
    atomic_save(changed_path, saved)
    changed = {**pipeline, "response": [str(changed_path)]}
    for stage in ("pool", "state"):
        assert verify_auxiliary_dependencies(pipeline[stage], changed)
        assert pipeline_identity(stage, data, config, pipeline) == pipeline_identity(stage, data, config, changed)
    new = BiologicalWorld(data, config, changed)
    assert old.biological_identity != new.biological_identity
    assert old.adapter.versions != new.adapter.versions


def test_residual_predictions_use_independent_h_features(components):
    data, config, monitor, _, _ = components
    pipeline = train_heads(components)
    pipeline["response"] = [member_checkpoint(components)]
    rows = prediction_rows("R", data, config, monitor, pipeline)
    assert len(rows) == 5
    saved = torch.load(rows[0]["path"], weights_only=False)
    assert saved["population"].shape == (8,)
    assert saved["observed_population"].shape == (8,)
    assert saved["raw"].shape == (3, 8)


def test_original_response_checkpoint_still_dispatches(components):
    data, _, monitor, _, reference = components
    shape = replace(reference.config, backbone="gene_token")
    model = ResponseModel(shape, torch.as_tensor(data.neighbors))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = data.sample_response("T-0", 3, np.random.default_rng(41), "T")
    condition = model.encode_condition(**condition_kwargs(batch, 3, "cpu"))
    model.cfm_loss(tensor(batch["control"], "cpu"), tensor(batch["target"], "cpu"), condition).backward()
    optimizer.step()
    path = monitor.weights / "legacy.pt"
    save_training(path, model, optimizer, epoch=1, identity="SYNTHETIC-legacy",
                  metadata={"stage": "response", "data_identity": data.identity,
                            "model_config": asdict(shape)}, counters={"optimizer_step": 1})
    restored = load_response(path, data, "cpu")
    assert isinstance(restored, ResponseModel)
    assert diagnostics.load_response is load_response
    assert restored.encode_cells(tensor(batch["control"], "cpu"))[1].shape == (3, 8)
