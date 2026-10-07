"""CPU semantic equivalence and bounded exact-cache contract; no efficacy claims."""
from dataclasses import replace
import pytest
import torch
from test_world_execution_v1 import world_factory, cpu_thread_and_seed
from test_rl_v1 import system
from test_decision_engine_v1 import synthetic_world
from lung.engine.scoring import ScalarScoreCache
from lung.rl import DLPAPlanner, DLPAConfig, JointActor


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batched_score_matches_scalar_all_members_and_rng(world_factory, batch_size, monkeypatch):
    owner, _, _, _ = world_factory(2)
    owner.score_batch_size = batch_size
    root = owner.roots("T")[0]
    sequence = (next(iter(owner.actions)),)
    seeds = (11, 17, 31, 51, 63)
    expected = torch.tensor([owner.score_one(root, sequence, seed) for seed in seeds], dtype=torch.float64)
    rng = torch.random.get_rng_state().clone()
    calls = [[] for _ in owner.members]
    for i, member in enumerate(owner.members):
        original = member.sample
        def record(reference, *args, _i=i, _fn=original, **kwargs):
            calls[_i].append(len(reference))
            return _fn(reference, *args, **kwargs)
        monkeypatch.setattr(member, "sample", record)
    actual = owner.score(root, sequence, seeds)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(len(values) == ((len(seeds)+batch_size-1)//batch_size)*3 for values in calls)
    before = [len(values) for values in calls]
    torch.testing.assert_close(owner.score(root, sequence, seeds), actual, rtol=0, atol=0)
    assert before == [len(values) for values in calls]
    assert owner._score_cache.hits == len(seeds)
    # Same action with a new seed is a new simulation, never a cache-derived replicate.
    owner.score(root, sequence, (1001,))
    assert all(len(values) > count for values, count in zip(calls, before))


def test_full_cache_key_and_eviction(world_factory):
    owner, _, _, _ = world_factory(2)
    root = owner.roots("T")[0]
    sequence = (next(iter(owner.actions)),)
    owner.score(root, sequence, (12,))
    old = tuple(owner._score_cache.values)
    owner.steps += 1
    owner.score(root, sequence, (12,))
    assert len(owner._score_cache.values) == 2
    owner.adapter.versions = replace(owner.adapter.versions, reward="new-calibration")
    owner.score(root, sequence, (12,))
    assert len(owner._score_cache.values) == 3
    cache = ScalarScoreCache(2)
    for index in range(3):
        cache.put((index,), .5)
    assert len(cache.values) == 2 and cache.get((0,)) is None
    assert old[0] in owner._score_cache.values


def test_dlpa_uses_batched_callback_and_keeps_independent_search_streams(system):
    registry, adapter, root, calls = system
    batches = []
    def score_many(root, requests):
        batches.append(requests)
        return [adapter.score_fn(root, sequence, seeds) for sequence, seeds in requests]
    adapter.score_many_fn = score_many
    actor = JointActor(registry)
    planner = DLPAPlanner(adapter, actor, DLPAConfig(rounds=2, candidates_per_round=12,
                                                   elite_count=2, score_repeats=3))
    result = planner.plan(root, 19)
    assert len(batches) == 2 and result.teachers
    streams = [seed for batch in batches for _, seeds in batch for seed in seeds]
    assert len(set(streams)) == len(streams)
    with pytest.raises(ValueError, match="distinct seed"):
        adapter.score_many(root, ((("STOP",), (1, 1)),))


@pytest.mark.parametrize("population", [1, 5, 6])
def test_time_mixture_batch_matches_scalar_predictor(population, monkeypatch):
    from types import SimpleNamespace
    from test_world_execution_v1 import config_and_data, response_model
    from lung.timecourse import scoring, nested, response
    shape, _, data = config_and_data(2)
    models = [(response_model(shape, data), {"inference_microbatch": 2, "inference_ode_steps": 2})
              for _ in range(2)]
    data.action_features = lambda cid: data.sample_response(cid, 1, __import__('numpy').random.default_rng(1), "T")
    predictor = nested.FrozenTimePredictor.__new__(nested.FrozenTimePredictor)
    predictor.data, predictor.models = data, models
    predictor.cfg = {"device": "cpu"}
    predictor.artifacts = {"response": ["a", "b"]}
    predictor.regressors = {"B": "B", "K": "K", "U": "U"}
    predictor.gate = {"enabled": True}
    predictor.calibration = {"on": {"status": "fixture", "quantile": 1., "eps": .01}}
    for module in (scoring, nested):
        monkeypatch.setattr(module, "project", lambda x, _: x)
        monkeypatch.setattr(module, "chain_features", lambda data, cid, x, ref, b, device: x.mean(0))
        monkeypatch.setattr(module, "infer", lambda model, features, device:
                            features*.01 if model == "K" else torch.tensor([.1, .2]))
    reference = torch.randn(population, shape.n_genes)
    seeds = (7, 13, 21)
    expected = [predictor.predict("T-0", population, seed, reference)["projected"] for seed in seeds]
    before = torch.random.get_rng_state().clone()
    actual = scoring.projected_seed_batch(predictor, "T-0", reference, seeds)
    assert torch.equal(before, torch.random.get_rng_state())
    for value, target in zip(actual, expected):
        assert len(value) == population
        torch.testing.assert_close(value, target, rtol=2e-5, atol=2e-6)


def test_time_cache_roles_root_calibration_population_and_steps(monkeypatch):
    from test_timecourse_decisions import fixture_data, Predictor, Proxy
    from lung.timecourse.decision_world import TimeDecisionWorld
    data = fixture_data()
    predictor = Predictor(data)
    ids = data.condition_ids(drugs=data.drug_groups["D"])
    owner = TimeDecisionWorld(predictor, Proxy(), ids, ("rep1", "plate49"), "T", "env",
                              {"device": "cpu", "population": 4, "projection_width": 4})
    root, sequence = owner.root, (ids[0],)
    owner.raw_score(root, sequence, (7, 8), 4)
    owner.raw_score(root, sequence, (7, 8), 4)
    assert len(predictor.calls) == 2 and owner._score_cache.hits == 2
    owner.raw_score(replace(root, role="E"), sequence, (7,), 4)
    owner.raw_score(root, sequence, (7,), 8)
    predictor.calibration = {"version": "new"}
    owner.raw_score(root, sequence, (7,), 4)
    assert len(predictor.calls) == 5


@pytest.mark.parametrize("mode", ["population_mean", "condition_only"])
def test_input_ablation_respects_population_boundaries(world_factory, mode):
    owner, _, _, _ = world_factory(2)
    for member in owner.members:
        member.config = replace(member.config, reference_mode=mode)
    root = owner.roots("T")[0]
    sequence, seeds = (next(iter(owner.actions)),), (2, 3, 5)
    expected = torch.tensor([owner.score_one(root, sequence, seed) for seed in seeds], dtype=torch.float64)
    torch.testing.assert_close(owner.score(root, sequence, seeds), expected, rtol=2e-5, atol=2e-6)


def test_checkpoint_throttle_has_exact_resume(synthetic_world, tmp_path, monkeypatch):
    from test_decision_engine_v1 import Monitor
    from lung.engine import decision
    from lung.engine.checkpoint import load_checkpoint
    _, config, data, _ = synthetic_world
    config = {**config, "training_control": {"checkpoint_every_updates": 100, "checkpoint_every_seconds": 0}}
    writes = []
    original = decision.atomic_save
    def recording(path, payload):
        writes.append((str(path), payload["extra"]["cursor"].copy()))
        return original(path, payload)
    monkeypatch.setattr(decision, "atomic_save", recording)
    full = Monitor(tmp_path / "full")
    decision.train_decision("dlpa", data, config, full, {})
    assert not any(row["phase"] == "distill" and row["update"] == 1 for _, row in writes)
    resumed = Monitor(tmp_path / "resume", fail=("distill_step", 1))
    with pytest.raises(KeyboardInterrupt):
        decision.train_decision("dlpa", data, config, resumed, {})
    resumed.fail = None
    decision.train_decision("dlpa", data, config, resumed, {}, resume=True)
    a = load_checkpoint(full.weights / "dlpa/seed_0/last.pt")
    b = load_checkpoint(resumed.weights / "dlpa/seed_0/last.pt")
    assert a["counters"] == b["counters"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    assert b["extra"]["training_control"]["status"] == "complete"
    assert b["extra"]["training_control"]["updates"] == b["counters"]["distill_step"]


@pytest.mark.parametrize("endpoint", [False, True])
def test_multistep_correction_and_endpoint_remain_request_local(world_factory, monkeypatch, endpoint):
    owner, _, _, _ = world_factory(2)
    monkeypatch.setattr(owner, "_gate", lambda context: True)
    if endpoint:
        owner.endpoint = lambda features, history, lengths: features[:, 0].sigmoid()
    root = owner.roots("T")[0]
    aid = next(iter(owner.actions))
    sequence, seeds = (aid, aid), (4, 19, 23)
    expected = torch.tensor([owner.score_one(root, sequence, seed) for seed in seeds], dtype=torch.float64)
    torch.testing.assert_close(owner.score(root, sequence, seeds), expected, rtol=2e-5, atol=2e-6)
