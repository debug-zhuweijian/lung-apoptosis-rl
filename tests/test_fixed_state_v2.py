"""Fixed initial-state and STOP contract tests; synthetic, no efficacy claims."""
from dataclasses import FrozenInstanceError, replace
import json
import pickle

import numpy as np
import pytest
import torch

from test_world_execution_v1 import world_factory, cpu_thread_and_seed, RecordingResponse
from lung.engine import world
from lung.rl import RootContext, RLUnavailable


def explicit_root(owner, values):
    return owner.root("T-0", "T", expression=values,
                      gene_ids=tuple(str(i) for i in range(owner.shape.n_genes)),
                      data_identity=owner.data.identity)


def test_default_root_observe_score_and_restore_share_fixed_start(world_factory):
    owner, _, _, _ = world_factory(3)
    root = owner.roots("T")[0]
    action = (next(iter(owner.actions)),)
    recorders = [RecordingResponse(member) for member in owner.members]
    owner.members = recorders
    first = owner.observe(root, (), 11)
    second = owner.observe(root, (), 19)
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
    expected, _ = owner._initial(root, owner.n_pop)
    scores, noises = [], []
    for seed in (71, 72):
        for recorder in recorders:
            recorder.calls.clear()
        scores.append(owner.score_one(root, action, seed))
        for recorder in recorders:
            torch.testing.assert_close(torch.cat([row[0] for row in recorder.calls]), expected, rtol=0, atol=0)
        noises.append(torch.cat([row[1] for row in recorders[0].calls]))
    assert not torch.equal(*noises)
    restored, _, _, _ = world_factory(3)
    restored_root = pickle.loads(pickle.dumps(root))
    assert restored.roots("T")[0] == restored_root
    assert restored.score_one(restored_root, action, 71) == scores[0]
    nested, _ = owner._initial(root, 16)
    torch.testing.assert_close(nested[:owner.n_pop], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="POPULATION_SIZE"):
        owner._initial(root, 17)


def test_two_explicit_observations_reach_response_unchanged(world_factory):
    owner, _, _, _ = world_factory(3)
    action = (next(iter(owner.actions)),)
    recorders = [RecordingResponse(member) for member in owner.members]
    owner.members = recorders
    inputs = [np.full((owner.n_pop, owner.shape.n_genes), value, np.float32) for value in (0., .5)]
    roots = [explicit_root(owner, value) for value in inputs]
    assert roots[0].root_id != roots[1].root_id
    states = [owner.observe(root, (), 5).features for root in roots]
    assert not torch.equal(*states)
    for root, expression in zip(roots, inputs):
        for recorder in recorders:
            recorder.calls.clear()
        owner.score_one(root, action, 93)
        for recorder in recorders:
            torch.testing.assert_close(torch.cat([row[0] for row in recorder.calls]),
                                       torch.tensor(expression), rtol=0, atol=0)
    inputs[0][:] = 99
    values, _ = owner._initial(roots[0], owner.n_pop)
    assert torch.count_nonzero(values) == 0
    with pytest.raises(FrozenInstanceError):
        roots[0].payload.initial_seed = 0
    with pytest.raises(ValueError, match="POPULATION_SIZE"):
        owner._initial(roots[0], owner.n_pop + 1)


@pytest.mark.parametrize("kind", ["short", "axis", "nan", "inf", "rank", "overflow"])
def test_explicit_expression_invalid_shapes_and_values(world_factory, kind):
    owner, _, _, _ = world_factory(2)
    values = np.ones((owner.n_pop, owner.shape.n_genes))
    if kind == "short":
        values = values[:-1]
    elif kind == "axis":
        values = values[:, :-1]
    elif kind == "rank":
        values = values[0]
    else:
        values[0, 0] = {"nan": np.nan, "inf": np.inf, "overflow": 1e100}[kind]
    with pytest.raises(ValueError, match="EXPRESSION_INVALID"):
        explicit_root(owner, values)


def test_root_rejects_legacy_mismatch_role_gene_axis_and_version(world_factory, monkeypatch):
    owner, config, data, pipeline = world_factory(2)
    root = owner.roots("T")[0]
    for bad in (RootContext("T-0", "T", {"condition_id": "T-0"}), replace(root, role="E"),
                replace(root, payload=replace(root.payload, data_identity="old"))):
        with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
            owner.observe(bad, (), 1)
    with pytest.raises(ValueError, match="ROLE_MISMATCH"):
        owner.root("T-0", "E")
    values = np.zeros((owner.n_pop, owner.shape.n_genes))
    with pytest.raises(ValueError, match="GENE_AXIS_MISMATCH"):
        owner.root("T-0", "T", expression=values)
    with pytest.raises(ValueError, match="GENE_AXIS_MISMATCH"):
        owner.root("T-0", "T", expression=values, data_identity=data.identity,
                   gene_ids=tuple(str(i) for i in reversed(range(owner.shape.n_genes))))
    monkeypatch.setattr(world, "INITIAL_STATE_VERSION", "future-fixed-interface")
    changed = world.BiologicalWorld(data, config, pipeline)
    assert owner.biological_identity != changed.biological_identity


def test_population_mean_reference_is_computed_before_microbatches(world_factory):
    class PopulationMeanResponse(RecordingResponse):
        def condition_reference(self, x):
            return x.mean(0, keepdim=True).expand_as(x)

    owner, _, _, _ = world_factory(2)
    root = owner.roots("T")[0]
    expression, batch = owner._initial(root, owner.n_pop)
    recording = PopulationMeanResponse(owner.reference)
    expected = expression.mean(0, keepdim=True).expand_as(expression)
    outputs = []
    for size in (1, 3):
        owner.inference_microbatch = size
        recording.calls.clear()
        outputs.append(owner._generate_member(recording, expression, batch, 83, 0, 0, 0., owner.n_pop, 2))
        torch.testing.assert_close(torch.cat([row[0] for row in recording.calls]), expected, rtol=0, atol=0)
    torch.testing.assert_close(*outputs, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("explicit", [False, True])
def test_initial_cache_reads_once_evicts_and_isolates_returned_values(world_factory, monkeypatch, explicit):
    owner, _, data, _ = world_factory(3)
    roots = owner.roots("T")
    if explicit:
        roots[0] = explicit_root(owner, np.arange(16 * owner.shape.n_genes, dtype=np.float32).reshape(
            16, owner.shape.n_genes) / 100.)
    reads = []
    original = data.sample_controls

    def sample(condition_id, n, rng, role):
        reads.append((condition_id, n, role))
        return original(condition_id, n, rng, role)

    monkeypatch.setattr(data, "sample_controls", sample)
    action = (next(iter(owner.actions)),)
    first_score = owner.score_one(roots[0], action, 31)
    owner.score_one(roots[0], action, 32)
    owner.observe(roots[0], (), 33)
    assert reads == [("T-0", 16, "T")]
    whole, context = owner._initial(roots[0], 16)
    expected = whole.clone()
    whole.fill_(99)
    context["fingerprint"][:] = 99
    prefix, clean_context = owner._initial(roots[0], owner.n_pop)
    torch.testing.assert_close(prefix, expected[:owner.n_pop], rtol=0, atol=0)
    assert np.max(clean_context["fingerprint"]) == 1
    assert owner.score_one(roots[0], action, 31) == first_score
    assert len(reads) == 1
    # Validation still runs on a cache hit.
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        owner._initial(replace(roots[0], role="E"), owner.n_pop)
    with pytest.raises(ValueError, match="POPULATION_SIZE"):
        owner._initial(roots[0], 17)
    assert len(reads) == 1
    owner._initial(roots[1], owner.n_pop)
    assert reads[-1] == ("T-1", 16, "T")
    restored, _ = owner._initial(roots[0], 16)
    assert reads == [("T-0", 16, "T"), ("T-1", 16, "T"), ("T-0", 16, "T")]
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tail_kind", ["valid", "illegal_switch", "illegal_terminal", "too_short"])
def test_stop_tails_validate_complete_state_machine(world_factory, tmp_path, tail_kind):
    owner, config, data, pipeline = world_factory(2)
    a, b = tuple(owner.actions)
    manifest = {"semantics": "complete_replacement_no_carryover_assumption", "ticks_per_hour": 1,
                "horizon_ticks": 48, "initial_state": "start", "terminal_states": ["end"],
                "transitions": [{"from": "start", "action_id": a, "to": "mid"},
                                {"from": "mid", "action_id": a, "to": "end"}],
                "stop_tails": [{"state": "start", "elapsed_ticks": 0, "rule_id": "full"},
                               {"state": "mid", "elapsed_ticks": 24, "rule_id": "half"}],
                "tail_rules": {"full": [a, a], "half": [a]}}
    if tail_kind == "illegal_switch":
        manifest["tail_rules"]["full"] = [b, b]
    elif tail_kind == "illegal_terminal":
        manifest["transitions"][1]["to"] = "wrong"
    elif tail_kind == "too_short":
        manifest["tail_rules"]["full"] = [a]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {**config, "protocol": str(path), "history_support": "model_assumption"}
    if tail_kind != "valid":
        with pytest.raises(RLUnavailable):
            world.BiologicalWorld(data, config, pipeline)
        return
    owner = world.BiologicalWorld(data, config, pipeline)
    assert owner._expanded(("STOP",)) == (a, a)
    assert owner._expanded((a, "STOP")) == (a, a)
    root = owner.roots("T")[0]
    assert owner.score_one(root, ("STOP",), 8) == owner.score_one(root, (a, a), 8)
    with pytest.raises(RLUnavailable):
        owner._expanded(("STOP", a))
    owner.tail_rules["full"] = [b, b]
    with pytest.raises(RLUnavailable):
        owner._expanded(("STOP",))
