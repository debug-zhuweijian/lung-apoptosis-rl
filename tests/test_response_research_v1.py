"""Research-entry contracts using synthetic data and temporary checkpoints."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse as sp  # used by _stub_cache_data (S11.1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.engine.numerics import PrecisionPolicy, policy_autocast
from lung.engine.response_training import condition_batch_mode, train_conditions
from lung.models import build_response_model
from test_models_v1 import tiny_config, neighbors, conditions

@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(311)
    yield
    torch.set_num_threads(old)


def test_cfm_stage_grouping_draws_no_mmd_samples_or_rng():
    """cfm warm-up groups conditions and never draws MMD batches (task 7.2-1)."""
    from lung.engine.config import load_config
    from synthetic_data import smoke_config, SyntheticData
    from lung.engine.response import response_rngs
    from lung.models import ModelConfig
    config = smoke_config(load_config())
    cfg = {**config['response'], 'coupling': 'random_product_control',
           'condition_microbatch': 2, '_training_stage': 'cfm'}
    data = SyntheticData(config)
    model = build_response_model(ModelConfig(**{**cfg['model'], 'backbone': 'compact_full_gene'}),
                                 torch.as_tensor(data.neighbors))
    assert condition_batch_mode(model, cfg)[1] == 'grouped_compact_observed_cfm_direct'
    torch.manual_seed(317)
    rng, mmd_rng, generator = response_rngs(317, 'cpu')
    with patch.object(data, 'sample_response', wraps=data.sample_response) as sample:
        rows = train_conditions(model, data, ['T-0', 'T-1'], rng, cfg, 'cpu', 2,
                                 mmd_rng=mmd_rng, mmd_generator=generator)
    assert sample.call_count == 2  # one OT draw per condition, no MMD draw
    assert all(row['mmd_loss'] is None and row['mmd_computed'] == 0. for row in rows)


def test_grouped_lazy_mmd_when_prefetched_without_second():
    """A stage transition mid-epoch reloads MMD draws on demand (task W2)."""
    from lung.engine.config import load_config
    from synthetic_data import smoke_config, SyntheticData
    from lung.engine.response import response_rngs
    from lung.models import ModelConfig
    config = smoke_config(load_config())
    cfg = {**config['response'], 'coupling': 'random_product_control',
           'condition_microbatch': 2, '_training_stage': 'full',
           'cfm_samples': 4, 'mmd_samples': 4, 'microbatch': 4, 'mmd_microbatch': 4,
           'train_ode_steps': 2}
    data = SyntheticData(config)
    model = build_response_model(ModelConfig(**{**cfg['model'], 'backbone': 'compact_full_gene'}),
                                 torch.as_tensor(data.neighbors))
    torch.manual_seed(319)
    rng, mmd_rng, generator = response_rngs(319, 'cpu')
    ot_batches = [(data.sample_response(key, cfg['ot_samples'], rng, role='T'), None)
                  for key in ('T-0', 'T-1')]
    with patch.object(data, 'sample_response', wraps=data.sample_response) as sample:
        rows = train_conditions(model, data, ['T-0', 'T-1'], rng, cfg, 'cpu', 2,
                                 mmd_rng=mmd_rng, mmd_generator=generator, batches=ot_batches)
    assert any(row['mmd_loss'] is not None for row in rows)
    mmd_draws = [call for call in sample.call_args_list
                 if len(call.args) > 2 and call.args[1] == cfg['mmd_samples']
                 and call.args[2] is mmd_rng]
    assert len(mmd_draws) == 2




def test_precision_policy_identity_and_neutral_legacy_autocast():
    assert PrecisionPolicy('bf16').identity() != PrecisionPolicy('fp32').identity()
    assert PrecisionPolicy('fp32').identity()['tf32_enabled'] is False
    assert PrecisionPolicy('tf32').identity()['tf32_enabled'] is True
    model = Mock(spec=['config'])
    with policy_autocast(model, torch.device('cpu')):  # unset policy -> no-op context
        assert torch.equal(torch.ones(1) + 1, torch.tensor([2.]))


def test_cache_manifest_is_the_only_completion_marker(tmp_path):
    from lung.engine.representation_cache import load_representation_cache
    h = np.random.default_rng(5).normal(size=(5, 8)).astype(np.float32)
    np.savez(tmp_path / 'fit_shard_0000.npz', h=h,
             cell_id=np.array([f'c{i}' for i in range(5)], dtype=str),
             condition_id=np.array(['T-0'] * 5, dtype=str))
    digest = hashlib.sha256((tmp_path / 'fit_shard_0000.npz').read_bytes()).hexdigest()
    manifest = {'schema_version': 1, 'protocol': 'latent-h-cache-v1', 'status': 'complete',
                'data_identity': 'data-A', 'encoder': {'weights': 'enc-A'},
                'roles': {'fit': {'cells': 5, 'conditions': 1,
                                  'shards': [{'file': 'fit_shard_0000.npz', 'rows': 5,
                                              'sha256': digest}]}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    cache = load_representation_cache(tmp_path, 'cpu', expected_data_identity='data-A',
                                      expected_encoder={'weights': 'enc-A'})
    assert cache.h['fit'].shape == (5, 8) and len(cache.pool('fit', 'T-0')) == 5
    with pytest.raises(ValueError, match='CACHE_DATA_IDENTITY_MISMATCH'):
        load_representation_cache(tmp_path, 'cpu', expected_data_identity='data-B',
                                  expected_encoder={'weights': 'enc-A'})
    with pytest.raises(ValueError, match='CACHE_ENCODER_IDENTITY_MISMATCH'):
        load_representation_cache(tmp_path, 'cpu', expected_data_identity='data-A',
                                  expected_encoder={'weights': 'enc-B'})
    unfinished = {**manifest, 'status': 'building'}
    (tmp_path / 'manifest.json').write_text(json.dumps(unfinished), encoding='utf-8')
    with pytest.raises(ValueError, match='CACHE_INCOMPLETE'):
        load_representation_cache(tmp_path, 'cpu')






def test_condition_only_reference_groups_with_explicit_reason():
    model = build_response_model(replace(tiny_config(False), backbone='compact_full_gene',
                                         reference_mode='condition_only'), neighbors())
    width, reason = condition_batch_mode(model, {'condition_microbatch': 3,
                                                 '_training_stage': 'full',
                                                 'mmd_backward': 'auto'})
    assert (width, reason) == (3, 'grouped_compact_condition_only_full_direct')




def test_encoder_identity_supports_sparse_graph_and_versioned_cache(tmp_path):
    """Real sparse graph buffer digests; rebuilds land in a versioned dir (R2)."""
    from lung.engine.representation_cache import (_manifest_matches,
                                                   build_representation_cache,
                                                   encoder_identity,
                                                   load_representation_cache)
    from lung.models.latent_staged import SharedAutoencoder
    from scipy import sparse as sp
    shape = replace(tiny_config(False), backbone='compact_full_gene', latent_dim=3)
    shared = SharedAutoencoder(shape, neighbors())
    assert any(value.is_sparse for value in shared.state_dict().values())
    first = encoder_identity(shared, shape)
    assert first == encoder_identity(shared, shape)

    data = _stub_cache_data(n_conditions=2, cells=3, n_genes=shape.n_genes)
    requested = tmp_path / "cache"
    manifest = build_representation_cache(data, shared, requested, "cpu",
                                          roles=("fit",), batch_size=2)
    assert manifest["encoder"] == first
    assert _manifest_matches(requested, data.identity, first)
    load_representation_cache(requested, "cpu", expected_data_identity=data.identity,
                              expected_encoder=first)
    changed = SharedAutoencoder(shape, neighbors())
    with torch.no_grad():
        changed.decoder[0].weight.add_(1.)
    second = encoder_identity(changed, shape)
    assert second != first
    directory, status = _ensure_stub(changed, data, requested)
    assert status == "built" and directory != requested
    assert directory.name.startswith("cache-")
    assert _manifest_matches(requested, data.identity, first)  # old marker untouched


def _ensure_stub(shared, data, requested):
    from lung.engine.representation_cache import ensure_representation_cache
    return ensure_representation_cache(data, shared, requested, "cpu", roles=("fit",),
                                       batch_size=2)


def _stub_cache_data(n_conditions, cells, n_genes):
    """Minimal reader-shaped stub: real CSR rows, canonical normalize, controls."""
    rng = np.random.default_rng(7)
    rows = []
    for condition in range(n_conditions):
        for cell in range(cells):
            rows.append({"cell_id": f"c{condition}-{cell}",
                         "matrix_path": "shard.npz", "matrix_row": condition * cells + cell,
                         "condition_id": f"T-{condition}", "role": "fit"})
    frame = pd.DataFrame(rows)
    counts = sp.csr_matrix(rng.integers(0, 4, size=(len(frame), n_genes)).astype(np.float64) + 1)
    reader = SimpleNamespace(
        condition_ids=lambda role, include_controls=False: [f"T-{i}" for i in range(n_conditions)],
        role_cells=lambda condition, role="fit": frame.loc[frame.condition_id.eq(condition)].copy(),
        controls=pd.DataFrame({"condition_id": [f"T-{i}" for i in range(n_conditions)],
                               "control_condition_id": [f"T-{i}" for i in range(n_conditions)]}),
        rows=SimpleNamespace(read=lambda part: counts[part["matrix_row"].to_numpy()]),
        normalize=lambda block: ((np.log1p(block.toarray()) - 0.) / 1.).astype(np.float32),
    )
    return SimpleNamespace(identity="stub-data", manifest={"signature": "sig"}, reader=reader)


def _cache_stub_with_pool(pools):
    """Fake cache exposing the interface draw_control_h actually uses (S11.2)."""
    cache = SimpleNamespace(
        h={"fit": torch.arange(sum(len(pool) for pool in pools.values()),
                               dtype=torch.float32).reshape(-1, 1)},
        condition_pools={("fit", name): pool for name, pool in pools.items()})
    cache.pool = lambda role, condition: cache.condition_pools[(role, condition)]
    return cache


def test_control_draw_mirrors_reader_two_level_sampling():
    """Pool draw then in-pool draw, original batch order — incl. small pools (R3)."""
    from lung.engine.latent_pipeline import control_ids, draw_control_h
    cache = _cache_stub_with_pool({"T-A": np.arange(5), "T-B": np.arange(5, 12)})
    data = SimpleNamespace(reader=SimpleNamespace(controls=pd.DataFrame(
        {"condition_id": ["T-0"] * 2,
         "control_condition_id": ["T-A", "T-B"]})))
    assert control_ids(data, "T-0") == ("T-A", "T-B")
    n = 8
    reference, actual = np.random.default_rng(41), np.random.default_rng(41)
    # Reference: the reader's algorithm (pool picks, then per-pick cell locations).
    pools = [np.arange(5), np.arange(5, 12)]
    picks = reference.integers(len(pools), size=n)
    locations = np.asarray([reference.integers(len(pools[index])) for index in picks])
    positions = [np.flatnonzero(picks == index) for index in range(len(pools))]
    expected = np.concatenate([pool[locations[position]]
                               for pool, position in zip(pools, positions)])[
        np.argsort(np.concatenate(positions))]
    drawn = draw_control_h(data, cache, "T-0", n, actual)
    np.testing.assert_array_equal(drawn.flatten().numpy(), expected.astype(np.float64))
    single_cache = _cache_stub_with_pool({"T-A": np.arange(5)})
    single = draw_control_h(data, single_cache, "T-0", 6, np.random.default_rng(3))
    assert int(single.min()) >= 0 and int(single.max()) < 5  # in-pool indices, not first-n


def test_latent_coupling_solves_ot_and_pairs_consistently():
    """coupling_space=latent really solves OT; h/z pairs share coupling indices (R3)."""
    from lung.engine import latent_pipeline as pipeline

    calls = []
    real = pipeline.entropic_ot

    def counting(source, target):
        calls.append((tuple(source.shape), tuple(target.shape)))
        return real(source, target)

    h_pool = torch.randn(6, 4)
    cache = SimpleNamespace(
        h={"fit": h_pool},
        condition_pools={("fit", "T-0"): np.arange(6), ("fit", "ctl"): np.arange(6)},
        pool=lambda role, condition: np.arange(6),
        draw=Mock(return_value=(h_pool[2:], None)))
    data = SimpleNamespace(
        reader=SimpleNamespace(controls=pd.DataFrame(
            {"condition_id": ["T-0"], "control_condition_id": ["ctl"]})),
        action_features=lambda condition: {"fingerprint": np.zeros(2048, np.float32),
                                           "dose": 1., "duration": 24., "background_id": 0,
                                           "source_id": 0, "protocol_id": 0,
                                           "condition_id": condition, "is_vehicle": False})
    shared = SimpleNamespace(project=lambda h: h[:, :2])
    with patch.object(pipeline, "entropic_ot", side_effect=counting):
        h_reference, z_target, features = pipeline._couple(
            "T-0", data, cache, shared, 4, np.random.default_rng(9), "cpu", "latent")
    assert calls == [((4, 2), (4, 2))]  # solver receives both projected pools
    assert h_reference.shape == (4, 4) and z_target.shape == (4, 2)
    # Every paired target z is a row of the projected target pool (same indices).
    pool_z = h_pool[2:, :2]
    for row in z_target:
        assert any(torch.allclose(row, candidate) for candidate in pool_z)






def test_entry_load_shared_binding(tmp_path, monkeypatch):
    """The CFM entry binds device= and ae_path= explicitly."""
    from lung.engine import latent_pipeline

    ae = tmp_path / "best.pt"
    ae.write_bytes(b"x")
    captured = []

    class Sentinel(RuntimeError):
        pass

    def spy(config, data, device, ae_path=None):
        captured.append((device, ae_path))
        raise Sentinel

    monkeypatch.setattr(latent_pipeline, "_load_shared", spy)
    monitor = SimpleNamespace(weights=tmp_path / "w", logdir=tmp_path / "l",
                              experiment=tmp_path / "e",
                              say=lambda *_: None, record=lambda *a, **k: None,
                              progress=lambda *a, **k: None, config={})
    config = {"response": {"members": 1, "lr": 1e-3,
                           "weight_decay": 0., "grad_clip": 1., "ot_samples": 4,
                           "microbatch": 4, "conditions_per_update": 2,
                           "train_ode_steps": 2, "inference_ode_steps": 2},
              "epochs": 1, "early_stopping": False, "device": "cpu",
              "research": {"latent": {"ae_checkpoint": str(ae)}, "precision": {"mode": "fp32"}}}
    with pytest.raises(Sentinel):
        latent_pipeline.run_latent_cfm(config, monitor, Mock())
    assert captured[-1] == (torch.device("cpu"), str(ae))



def test_unchecked_condition_encoder_matches_checked():
    """S03: the unchecked dense path matches the checked one on legal inputs."""
    from lung.models.encoders import ConditionEncoder

    shape = replace(tiny_config(False), backbone="compact_full_gene")
    encoder = ConditionEncoder(shape)
    encoder.eval()
    base = conditions(5)
    with torch.no_grad():
        checked = encoder(**base)
        unchecked = encoder.forward_unchecked(**base)
    torch.testing.assert_close(checked, unchecked, rtol=0, atol=0)
