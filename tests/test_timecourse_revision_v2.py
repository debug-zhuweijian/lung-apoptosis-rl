"""Fixed state, actual actor completion, transfer and frozen tensor-chain regressions."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.timecourse.decision_world import TimeDecisionWorld
from lung.timecourse.decisions import _train, _transfer
from lung.timecourse.response import train_response, predict_response, load_frozen_response
from lung.timecourse.auxiliary import architecture, infer, load_frozen_regressor, population_error
from lung.timecourse.state_proxy import convert_coordinates
from lung.rl.networks import JointActor, ConditionalActor
from test_timecourse_decisions import fixture_data, Predictor, Proxy
from test_timecourse_data_response import fixture_data as response_data, tiny_config


def world():
    data = fixture_data()
    return TimeDecisionWorld(Predictor(data), Proxy(), data.condition_ids(drugs=['D1'], time=72),
                             ('rep1', 'plate49'), 'T', 'env', {'population':4, 'projection_width':4})


def test_fixed_state_is_immutable_and_drives_scoring():
    w = world()
    a = np.ones((4, w.data.n_genes), np.float32)
    root = w.fixed_root(a, 'a', 'T')
    a[:] = 100
    np.testing.assert_array_equal(w.reference(root), 1)
    torch.testing.assert_close(w.observe(root, (), 1).features, w.observe(root, (), 99).features)
    other = w.fixed_root(np.full_like(a, 2), 'b', 'T')
    assert w.fixed_root(np.full_like(a, 2), 'a', 'T').root_id != root.root_id
    first_reference = w._reference_tensor(root)
    first_reference.zero_()
    torch.testing.assert_close(w._reference_tensor(root), torch.ones_like(first_reference))
    with patch.object(w.data, 'sample_controls', side_effect=AssertionError('resampled start')):
        first = w.score(root, (w.ids[0],), (7, 8))
        second = w.score(other, (w.ids[0],), (7, 8))
        assert torch.all(second > first)
        w.raw_score(root, (w.ids[0],), (7, 8), 16)
    assert not torch.equal(w.observe(root, (), 1).features, w.observe(other, (), 1).features)


def test_critic_only_short_run_and_completed_resume_fail(tmp_path):
    torch.set_num_threads(1)
    w = world()
    w.adapter.population_resolution_passed = True
    cfg = {'device':'cpu', 'epochs':1, 'scoring_repeats':1,
           'daac':{'episodes_per_epoch':2, 'warmup_transitions':1, 'batch_size':1,
                   'actor_delay':9, 'replay_capacity':8}}
    for resume in (False, True):
        with pytest.raises(ValueError, match='NO_ACTUAL_POLICY_TRAINING_UPDATES'):
            _train(w, cfg, 'daac', 5, tmp_path/'short', resume=resume)
    state = torch.load(tmp_path/'short/last.pt', weights_only=False)['learner']
    assert state['counters']['critic_step'] == 1
    assert state['counters']['actor_step'] == 0


@pytest.mark.parametrize('kind', [JointActor, ConditionalActor])
def test_transfer_preserves_policy_class_weights_and_rng(kind):
    w = world()
    actor = kind(w.registry)
    state = torch.get_rng_state().clone()
    copied = _transfer(actor, w.registry, 'cpu')
    assert type(copied) is kind and torch.equal(torch.get_rng_state(), state)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, copied.state_dict()[key])


def test_cached_response_same_outputs_rng_and_no_reference_resampling(tmp_path):
    torch.set_num_threads(1)
    data, cfg = response_data(), tiny_config()
    cfg['epochs'] = 1
    result = train_response(data, cfg, tmp_path/'p')
    paths = result['checkpoints']
    cid = data.condition_ids(drugs=['e'])[0]
    before = torch.get_rng_state().clone()
    models = [load_frozen_response(path, data, 'cpu') for path in paths]
    assert torch.equal(before, torch.get_rng_state())
    cold = predict_response(paths, data, cid, 3, 17, 'cpu')
    with patch('lung.timecourse.response.load_checkpoint', side_effect=AssertionError('reloaded')):
        hot = predict_response(paths, data, cid, 3, 17, 'cpu', models=models, as_tensor=True)
    np.testing.assert_array_equal(cold, hot.numpy())
    fixed = np.ones((3, data.n_genes), np.float32)
    with patch.object(data, 'sample_controls', side_effect=AssertionError('resampled')):
        left = predict_response(paths, data, cid, 3, 17, 'cpu', reference=fixed, models=models, as_tensor=True)
        right = predict_response(paths, data, cid, 3, 18, 'cpu', reference=fixed, models=models, as_tensor=True)
        seen = []
        original = models[0][0].sample
        def record(start, *args, **kwargs):
            seen.append(start.clone())
            return original(start, *args, **kwargs)
        with patch.object(models[0][0], 'sample', side_effect=record):
            mixture = predict_response(paths*2, data, cid, 3, 17, 'cpu', reference=fixed,
                                       models=models*2, as_tensor=True)
        torch.testing.assert_close(torch.cat(seen), torch.tensor(fixed))
        assert mixture.shape == (3, data.n_genes)
    assert left.shape == right.shape == (3, data.n_genes)
    assert not torch.equal(left, right)
    assert torch.equal(before, torch.get_rng_state())


def test_cached_aux_tensor_bridge_and_error(tmp_path):
    model = architecture(6, 2, 4)
    path = tmp_path/'b.pt'
    torch.save({'shape':[6,2,4], 'model':model.state_dict()}, path)
    before = torch.get_rng_state().clone()
    cached = load_frozen_regressor(path)
    features = np.arange(6, dtype=np.float32)
    expected = infer(path, features)
    with patch('torch.load', side_effect=AssertionError('reloaded')):
        actual = infer(cached, torch.tensor(features))
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert torch.equal(before, torch.get_rng_state())
    x = np.random.default_rng(7).normal(size=(67, 5)).astype(np.float32)
    y = np.random.default_rng(8).normal(size=(71, 5)).astype(np.float32)
    assert population_error(torch.tensor(x), y) == population_error(x, y)
    torch.testing.assert_close(convert_coordinates(torch.tensor(x), np.zeros(5), np.ones(5),
                                                   np.ones(5), np.full(5, 2)), torch.tensor((x-1)/2))
