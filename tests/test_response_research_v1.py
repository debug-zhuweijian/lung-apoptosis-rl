"""Research-entry contracts using synthetic data and temporary checkpoints."""
import copy
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

shape_n_genes = tiny_config(False).n_genes  # AE stub axis (S06 test)
neighbors_np = neighbors()


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
    from lung.engine.synthetic import smoke_config, SyntheticData
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
    from lung.engine.synthetic import smoke_config, SyntheticData
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


def test_budget_exhaustion_is_resumable_not_converged(tmp_path):
    """budget_exhausted force-saves and never masquerades as completion (task 7.2-7)."""
    from lung.engine.config import load_config
    from lung.engine.synthetic import smoke_config, SyntheticData
    from lung.engine.response import train_response
    from lung.engine.checkpoint import load_checkpoint
    from lung.engine.training_control import checkpoint_completion
    from test_response_resume_v3 import Monitor
    config = smoke_config(load_config())
    config.update(epochs=3, early_stopping=False)
    config['response'].update(members=1, member_workers=1, coupling='random_product_control',
                              conditions_per_update=4, train_ode_steps=2)
    data = SyntheticData(config)
    monitor = Monitor(tmp_path)
    with patch('lung.engine.response.evaluate_response', return_value=({'loss': 1.}, [])):
        train_response(data, config, monitor, resume=False,
                       stop_check=lambda step, epoch: step >= 1)
    saved = load_checkpoint(next((tmp_path / 'weights').rglob('last.pt')))
    assert saved['extra']['budget_exhausted'] is True
    assert saved['extra'].get('update_resume') is not None  # mid-epoch cursor preserved (R4.1)
    with pytest.raises(ValueError, match='TRAINING_NOT_COMPLETE'):
        checkpoint_completion(saved, config['epochs'])
    with patch('lung.engine.response.evaluate_response', return_value=({'loss': 1.}, [])):
        train_response(data, config, monitor, resume=True)  # no budget this time
    resumed = load_checkpoint(next((tmp_path / 'weights').rglob('last.pt')))
    assert resumed['epoch'] == config['epochs']


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


def test_background_saver_tracks_and_escalates(tmp_path):
    """flush/close raise on writer failure; the assertion wraps the raising call (C3)."""
    from lung.engine.async_checkpoint import BackgroundSaver
    saver = BackgroundSaver(max_queue=2)
    target = tmp_path / 'step.pt'
    saver.observe_computed(1)
    saver.submit(1, lambda: target.write_text('one'))
    saver.flush()
    assert saver.durable_step == 1 and saver.status()['computed_step'] == 1

    def boom():
        raise OSError('disk full')

    saver.submit(2, boom)
    with pytest.raises(RuntimeError, match='BACKGROUND_SAVE_FAILED'):
        saver.flush()
    with pytest.raises(RuntimeError, match='BACKGROUND_SAVE_FAILED'):
        saver.close()  # drain failure still joins the writer, then propagates (R7)


def test_member_batch_requires_single_process():
    from lung.engine.member_batch import train_latent_members
    with pytest.raises(ValueError, match='SINGLE_PROCESS'):
        train_latent_members({'response': {'members': 5, 'member_workers': 2}},
                             Mock(), Mock(), Mock())


def test_condition_only_reference_groups_with_explicit_reason():
    model = build_response_model(replace(tiny_config(False), backbone='compact_full_gene',
                                         reference_mode='condition_only'), neighbors())
    width, reason = condition_batch_mode(model, {'condition_microbatch': 3,
                                                 '_training_stage': 'full',
                                                 'mmd_backward': 'auto'})
    assert (width, reason) == (3, 'grouped_compact_condition_only_full_direct')


@pytest.mark.parametrize("device,member_count", [
    ("cpu", 2),
    pytest.param("cuda", 5,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
])
def test_member_batched_forward_matches_solo_member(device, member_count):
    """Stacked member forward == solo forward; grads on trainable whitelist only (R1)."""
    from lung.engine.member_batch import _batched_forward
    from lung.models.latent_staged import LatentMember, SharedAutoencoder
    shape = replace(tiny_config(False), backbone='compact_full_gene', latent_dim=3)
    shared = SharedAutoencoder(shape, neighbors()).to(device)
    solo = LatentMember(shape, shared).to(device)
    members = [solo, *[copy.deepcopy(solo) for _ in range(member_count - 1)]]
    z = torch.randn(4, shape.latent_dim, device=device)
    h = torch.randn(4, shape.d_model, device=device)
    base = {**{key: value.to(device) for key, value in conditions(4).items()},
            "vehicle": torch.zeros(4, dtype=torch.bool, device=device)}
    args = {key: base[key] for key in ("fingerprint", "dose", "duration", "background",
                                       "source", "protocol", "elapsed", "horizon", "vehicle")}
    noise = torch.randn(4, shape.latent_dim, device=device)
    tau = torch.rand(4, device=device)
    solo_loss = solo(z, h, noise=noise, tau=tau, **args)
    params, buffers = torch.func.stack_module_state(members)
    forward = _batched_forward(solo, params, buffers)
    stacked = {key: torch.stack([tensor] * member_count) for key, tensor in
               {"z": z, "h": h, "noise": noise, "tau": tau, **args}.items()}
    losses = forward(*[stacked[key] for key in
                       ("z", "h", "fingerprint", "dose", "duration", "background",
                        "source", "protocol", "elapsed", "horizon", "vehicle", "noise", "tau")])
    assert losses.shape == (member_count,) and torch.isfinite(losses).all()
    for loss in losses:
        torch.testing.assert_close(loss, solo_loss, rtol=1e-6, atol=1e-8)
    losses.sum().backward()
    solo_loss.backward()
    for name, parameter in solo.named_parameters():
        stacked_grad = params[name].grad
        if name.startswith(("velocity_model.", "condition_encoder.")):
            assert parameter.grad is not None, name
            assert torch.isfinite(stacked_grad).all(), name
            for gradient in stacked_grad:
                torch.testing.assert_close(gradient, parameter.grad,
                                           rtol=1e-5, atol=1e-7, msg=name)
        else:
            assert parameter.grad is None, name  # frozen decoder stays untouched
            assert stacked_grad is None, name


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


def test_budget_cli_total_overrides_saved_state(tmp_path, monkeypatch):
    """--budget-seconds is the NEW overall cap; consumed time is kept (R4.4)."""
    from lung.engine.response_research import ResearchBudget
    monkeypatch.setattr("time.monotonic", lambda: 100.)
    state = tmp_path / "research_state.json"
    state.write_text(json.dumps({"schema_version": 1, "budget_total_seconds": 10.,
                                 "budget_consumed_seconds": 9., "branches": {}}),
                     encoding="utf-8")
    expanded = ResearchBudget(100., state)
    assert expanded.total == 100. and expanded.consumed == 9.
    assert json.loads(state.read_text(encoding="utf-8"))["budget_total_seconds"] == 100.
    reloaded = ResearchBudget(None, state)
    assert reloaded.total == 100. and reloaded.consumed == 9.


def test_stacked_inputs_keep_reference_and_target_content():
    """S01: stacked 'h' rows come from CONTROL cells, 'z' from TREATED targets.

    Same-dim case must be distinguished by content, not shape: the control pool
    and the projected targets use disjoint numeric ranges here.
    """
    from lung.engine.member_batch import _stacked_inputs
    from lung.models.latent_staged import LatentMember, SharedAutoencoder

    def couple_stub(d_model, latent_dim):
        h_pool = torch.arange(1, 7, dtype=torch.float32).reshape(6, 1).repeat(1, d_model) / 1000.
        z_pool = torch.arange(1, 7, dtype=torch.float32).reshape(6, 1).repeat(1, latent_dim) * 100.
        return (h_pool[:4], z_pool[2:], {"fingerprint": np.zeros(2048, np.float32),
                                         "dose": 1., "duration": 24., "background_id": 0,
                                         "source_id": 0, "protocol_id": 0,
                                         "condition_id": "T-0", "is_vehicle": False})

    shape = replace(tiny_config(False), backbone="compact_full_gene",
                    d_model=6, latent_dim=4)
    shared = SharedAutoencoder(shape, neighbors())
    members = [LatentMember(shape, shared)]
    generators = [torch.Generator().manual_seed(0)]
    draws = [couple_stub(6, 4)]
    piece = slice(0, 4)
    stacked = _stacked_inputs(members, generators, draws, piece, 4, torch.device("cpu"))
    assert stacked["z"].shape == (1, 4, 4) and stacked["h"].shape == (1, 4, 6)  # S01 shapes
    # Content: h rows belong to the small-valued control pool, z rows to the
    # large-valued target pool — the pre-fix swap fails both assertions.
    assert stacked["h"].max() < 1. and stacked["z"].min() > 10.
    for row in stacked["z"][0]:
        assert any(torch.allclose(row, candidate) for candidate in draws[0][1])


def test_entry_load_shared_binding(tmp_path, monkeypatch):
    """S02: both real entries bind device= and ae_path=, not a positional path."""
    from lung.engine import latent_pipeline, member_batch

    ae = tmp_path / "best.pt"
    ae.write_bytes(b"x")
    captured = []

    class Sentinel(RuntimeError):
        pass

    def spy(config, data, device, ae_path=None):
        captured.append((device, ae_path))
        raise Sentinel

    monkeypatch.setattr(latent_pipeline, "_load_shared", spy)
    monkeypatch.setattr(latent_pipeline, "_resolve_ae_checkpoint", lambda config, monitor: ae)
    monitor = SimpleNamespace(weights=tmp_path / "w", logdir=tmp_path / "l",
                              experiment=tmp_path / "e",
                              say=lambda *_: None, record=lambda *a, **k: None,
                              progress=lambda *a, **k: None, config={})
    config = {"response": {"members": 1, "member_workers": 1, "lr": 1e-3,
                           "weight_decay": 0., "grad_clip": 1., "ot_samples": 4,
                           "microbatch": 4, "conditions_per_update": 2,
                           "train_ode_steps": 2, "inference_ode_steps": 2},
              "epochs": 1, "early_stopping": False, "device": "cpu",
              "research": {"latent": {}, "precision": {"mode": "fp32"}}}
    with pytest.raises(Sentinel):
        latent_pipeline.run_latent_cfm(config, monitor, Mock(), Mock())
    assert captured[-1] == (torch.device("cpu"), str(ae))

    monkeypatch.setattr(member_batch, "_load_shared", spy)
    monkeypatch.setattr(member_batch, "_resolve_ae_checkpoint", lambda config, monitor: ae)
    with pytest.raises(Sentinel):
        member_batch.train_latent_members(config, monitor, Mock(), Mock())
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


def test_validate_stacked_inputs_rejects_illegal():
    """S03: illegal stacked inputs are rejected OUTSIDE the vmap domain."""
    from lung.engine.member_batch import _validate_stacked_inputs
    from lung.models.latent_staged import LatentMember, SharedAutoencoder

    shape = replace(tiny_config(False), backbone="compact_full_gene", latent_dim=3)
    member = LatentMember(shape, SharedAutoencoder(shape, neighbors()))
    base = conditions(4)
    vehicle = base.get("vehicle", torch.zeros(4, dtype=torch.bool))
    good = {**base, "fingerprint": torch.randn(2, 4, 2048),
            "dose": base["dose"].repeat(2), "duration": base["duration"].repeat(2),
            "background": base["background"].repeat(2), "source": base["source"].repeat(2),
            "protocol": base["protocol"].repeat(2),
            "elapsed": base["elapsed"].repeat(2), "horizon": base["horizon"].repeat(2),
            "vehicle": vehicle.repeat(2)}
    _validate_stacked_inputs(good, member)  # legal passes

    def rejected(**changes):
        with pytest.raises(ValueError):
            _validate_stacked_inputs({**good, **changes}, member)

    rejected(fingerprint=torch.full((2, 4, 2048), float("nan")))
    rejected(dose=good["dose"] - 10.)
    rejected(duration=good["duration"] * 0)
    rejected(horizon=good["horizon"] * 0)
    rejected(background=torch.full_like(good["background"], 99))


def test_ae_mid_epoch_resume_consumes_from_cursor(tmp_path, monkeypatch):
    """S06: mid-epoch budget exit saves the cursor; resume continues after it."""
    import time as time_module
    from lung.engine import latent_pipeline as pipeline
    from lung.engine.response_research import ResearchBudget
    from lung.engine.checkpoint import load_checkpoint

    clock = SimpleNamespace(t=0.)
    monkeypatch.setattr(time_module, "monotonic", lambda: clock.t)
    keys = ["T-0", "T-1", "T-2", "T-3"]
    calls = []

    def sample_response(condition, n, rng, role):
        calls.append((role, condition))
        block = np.abs(np.random.default_rng(hash(condition) % 1000).normal(
            size=(n, shape_n_genes))).astype(np.float32) + 0.1
        return {"control": block, "target": block * 1.1,
                "control_cell_ids": np.array([f"{condition}-c{i}" for i in range(n)]),
                "target_cell_ids": np.array([f"{condition}-t{i}" for i in range(n)]),
                "condition": {"fingerprint": np.zeros(2048, np.float32), "dose_um": 1.,
                              "duration_h": 24., "source_id": 0, "background_id": 0,
                              "protocol_id": 0, "condition_id": condition,
                              "is_vehicle": False, "chemical_id": "g"},
                "role": role}
    data = SimpleNamespace(n_genes=shape_n_genes, neighbors=neighbors_np,
                           identity="stub", manifest={"signature": "s"},
                           condition_ids=lambda role: (["V-0"] if role == "V" else keys),
                           iter_response_epoch=lambda role, seed: iter(keys),
                           sample_response=sample_response)
    monitor = SimpleNamespace(weights=tmp_path / "w", logdir=tmp_path / "l",
                              experiment=tmp_path / "e",
                              say=lambda *_: None, record=lambda *a, **k: None,
                              progress=lambda *a, **k: None, config={})
    config = {"response": {"lr": 1e-3, "weight_decay": 0., "grad_clip": 10.,
                           "conditions_per_update": 2, "ot_samples": 4},
              "epochs": 1, "early_stopping": False, "device": "cpu", "seed": 7,
              "research": {"latent": {"ae": {"lr": 1e-3, "batch": 4}},
                           "precision": {"mode": "fp32"}}}

    class ExhaustNow:
        def __init__(self):
            self.done = False

        def __call__(self, step, epoch):
            if not self.done:
                self.done = True
                clock.t += 10.
            return clock.t > 5.

    budget = ResearchBudget(5., tmp_path / "b.json")
    budget.check = ExhaustNow()
    pipeline.run_latent_ae(config, monitor, data, budget)
    saved = load_checkpoint(tmp_path / "w" / "latent_ae" / "last.pt")
    position = saved["extra"]["resume"]
    assert saved["epoch"] == 0 and position["next_group_start"] == 2  # mid-epoch (S06)
    assert position["keys"] == keys and "numpy_rng" in position

    calls.clear()
    clock.t = 0.
    resumed = pipeline.run_latent_ae(config, monitor, data,
                                     ResearchBudget(None, tmp_path / "b2.json"))
    train_calls = [condition for role, condition in calls if role == "T" and condition in keys]
    assert train_calls[:2] == ["T-2", "T-3"]  # continues AFTER the cursor


def test_budget_settles_on_interrupt(tmp_path, monkeypatch):
    """S07: an interrupt inside an arm settles that arm's budget file."""
    import time as time_module
    from lung.engine import response_research as research

    clock = SimpleNamespace(t=0.)
    monkeypatch.setattr(time_module, "monotonic", lambda: clock.t)

    def interrupted(*args, **kwargs):
        clock.t += 50.
        raise KeyboardInterrupt

    data = Mock()
    weights, logdir, experiment = (tmp_path / "w", tmp_path / "l", tmp_path / "e")
    for folder in (weights, logdir, experiment):
        folder.mkdir(parents=True, exist_ok=True)
    monitor = SimpleNamespace(weights=weights, logdir=logdir, experiment=experiment,
                              say=lambda *_: None, record=lambda *a, **k: None,
                              progress=lambda *a, **k: None, config={})
    config = {"research": {"arms": ["full"], "budget_seconds": 100.},
              "epochs": 1, "device": "cpu",
              "response": {"lr": 1e-3, "weight_decay": 0., "grad_clip": 1.,
                           "conditions_per_update": 2, "members": 1, "member_workers": 1}}
    budget = research.ResearchBudget(100., tmp_path / "parent.json")
    # run_route_compare imports train_response INSIDE the function, so patch the
    # source module, not the research namespace (S11 signature check).
    with patch("lung.engine.response.train_response", side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt):
            research.run_route_compare(config, monitor, data, budget)
    arm_state = json.loads((experiment / "budget_full.json").read_text(encoding="utf-8"))
    assert arm_state["budget_consumed_seconds"] == 50.
    parent_state = json.loads((tmp_path / "parent.json").read_text(encoding="utf-8"))
    assert parent_state["budget_consumed_seconds"] == 50.


def test_saver_group_durable_semantics():
    """S10: durable_step advances only when every file of a group is written."""
    import threading
    from lung.engine.async_checkpoint import BackgroundSaver

    first_done, release = threading.Event(), threading.Event()
    saver = BackgroundSaver(max_queue=4)
    saver.submit(1, first_done.set, group=2)
    saver.submit(1, lambda: release.wait(timeout=5), group=2)
    first_done.wait(timeout=5)
    assert saver.status()["durable_step"] is None  # one of two done: NOT durable (S10)
    release.set()
    saver.flush()
    assert saver.durable_step == 1 and not saver.status()["pending_group_jobs"]

    def boom():
        raise OSError("member file failed")

    saver.submit(2, boom, group=2)
    with pytest.raises(RuntimeError, match="BACKGROUND_SAVE_FAILED"):
        saver.flush()
    assert saver.durable_step != 2  # a failed group is never claimed durable (S10)
    with pytest.raises(RuntimeError, match="BACKGROUND_SAVE_FAILED"):
        saver.close()
