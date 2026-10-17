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


@pytest.mark.parametrize("mode", ["population_mean", "condition_only"])
def test_input_ablation_respects_population_boundaries(world_factory, mode):
    owner, _, _, _ = world_factory(2)
    for member in owner.members:
        member.config = replace(member.config, reference_mode=mode)
    root = owner.roots("T")[0]
    sequence, seeds = (next(iter(owner.actions)),), (2, 3, 5)
    expected = torch.tensor([owner.score_one(root, sequence, seed) for seed in seeds], dtype=torch.float64)
    torch.testing.assert_close(owner.score(root, sequence, seeds), expected, rtol=2e-5, atol=2e-6)




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
