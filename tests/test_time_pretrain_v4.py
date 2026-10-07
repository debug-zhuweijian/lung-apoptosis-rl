from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from test_timecourse_data_response import fixture_data, tiny_config
from lung.timecourse.response import train_response
from lung.engine.checkpoint import load_checkpoint


def test_known24_and_validation_permissions():
    base = fixture_data()
    data = base.training_view(known24_only=True, validation_fraction=.2)
    assert all(data.conditions.loc[key].time_point == 24 for key in data.condition_ids('T') + data.condition_ids('V'))
    for key in data.condition_ids('T'):
        train = data.sample_target(key, role='T')
        val = data.sample_target(key, role='V')
        assert len(train) == 2 and len(val) == 1
        assert not any(np.array_equal(a, b) for a in train for b in val)
    with pytest.raises(PermissionError, match='PRETRAIN24_ONLY'):
        data.sample_target(base.condition_ids(drugs=['d'])[0], role='T')
    for drug in ('c', 'e'):
        with pytest.raises(PermissionError):
            base.training_view(validation_fraction=.2).sample_target(base.condition_ids(drugs=[drug])[0], role='V')
    assert len(base.sample_target(base.condition_ids('T')[0], role='T')) == 3


def test_member_pretrain_reuse_and_real_early_completion(tmp_path):
    torch.set_num_threads(1)
    data, cfg = fixture_data(), tiny_config()
    cfg['response']['members'] = 2
    cfg['response']['model']['backbone'] = 'compact_full_gene'
    cfg['early_stopping'] = {'patience': 1, 'min_updates': 1}
    cfg['epochs'] = 1
    start = train_response(data.view([], [], 'start').training_view(known24_only=True), cfg, tmp_path / 'start')
    weights = [load_checkpoint(path)['model'] for path in start['checkpoints']]
    assert any(not torch.equal(weights[0][key], weights[1][key]) for key in weights[0] if not weights[0][key].is_sparse)
    cfg['early_stopping'] = {'patience': 1, 'min_updates': 1}
    cfg['epochs'] = 10
    cfg['early_stopping'] = {'patience': 1, 'min_updates': 1}
    with patch('lung.timecourse.response.validation_loss', return_value=1.):
        a = train_response(data, cfg, tmp_path / 'a', initial_checkpoints=start['checkpoints'])
        b = train_response(data, cfg, tmp_path / 'b', initial_checkpoints=start['checkpoints'])
    assert all(item['actual_epochs'] == 2 and item['reason'] == 'validation_plateau' for item in a['completion'])
    for path1, path2 in zip(a['checkpoints'], b['checkpoints']):
        x, y = load_checkpoint(path1), load_checkpoint(path2)
        assert x['metadata']['validation_scope'] == 'within_legal_fit_population_holdout_not_outer_oof'
        for key in x['model']:
            if not x['model'][key].is_sparse:
                assert torch.equal(x['model'][key], y['model'][key])
    with pytest.raises(ValueError, match='ILLEGAL_PRETRAIN_INITIALIZATION'):
        train_response(data, cfg, tmp_path / 'illegal', initial_checkpoints=a['checkpoints'])


def test_real_fold_topology_retains_seventeen_independent_fits():
    from lung.timecourse.nested import folds
    drugs = [f'd{i}' for i in range(28)]
    fit_sets = []
    for held in folds(drugs, 4):
        fit = [d for d in drugs if d not in held]
        for inner_held in folds(fit, 3):
            inner_fit = [d for d in fit if d not in inner_held]
            assert not set(inner_fit) & set(held + inner_held)
            fit_sets.append(inner_fit)
        fit_sets.append(fit)
    fit_sets.append(drugs)
    assert len(fit_sets) == 17
    assert sorted(map(len, fit_sets)) == [14] * 12 + [21] * 4 + [28]


def test_time_stage_transition_has_full_objective_validation(tmp_path):
    torch.set_num_threads(1)
    data, cfg = fixture_data(), tiny_config()
    cfg['epochs'] = 3
    cfg['response']['model']['backbone'] = 'compact_full_gene'
    cfg['response']['validation_population'] = 2
    cfg['response']['training_schedule'] = {'mode': 'cfm_then_full', 'transition': 'validation_plateau', 'warmup_epochs': 1, 'patience': 1}
    result = train_response(data, cfg, tmp_path / 'staged')
    state = load_checkpoint(result['checkpoints'][0])['extra']['training_control']
    assert state['stage'] == 'full' and state['full_updates'] >= 1 and state['full_validations'] >= 1
    assert result['completion'][0]['reason'] == 'max_epochs'
    assert result['completion'][0]['actual_epochs'] == 3


def test_nested_shared_start_is_reused_without_fold_carryover(tmp_path):
    from lung.timecourse.nested import run_nested
    from test_timecourse_nested import fixture_data as nested_fixture
    import json
    torch.set_num_threads(1)
    data, cfg = nested_fixture(), tiny_config()
    cfg['epochs'] = 1
    cfg['response']['model']['backbone'] = 'compact_full_gene'
    cfg['timecourse'] = {'shared24_pretrain': {'enabled': True, 'epochs': 1}}
    cfg['time_auxiliary'] = {'epochs': 1, 'hidden': 4, 'population': 2, 'outer_folds': 2, 'inner_folds': 2}
    calls = []
    real_train = train_response
    def track(view, config, output, **kwargs):
        calls.append((view, kwargs.get('initial_checkpoints')))
        return real_train(view, config, output, **kwargs)
    with patch('lung.timecourse.nested.TimeData', side_effect=lambda root, config, fit_drugs, allowed_target_drugs, role: data.view(fit_drugs, allowed_target_drugs, role)), patch('lung.timecourse.nested.train_response', side_effect=track):
        result = run_nested(tmp_path, cfg, tmp_path / 'nested')
    assert len(calls) == 8 and calls[0][1] is None
    assert calls[0][0].known24_only
    assert all(item[1] == calls[1][1] for item in calls[1:])
    assert all('shared24_pretrain' in path for path in calls[1][1])
    audit = json.loads(Path(result['audit']).read_text())
    p = [item for item in audit['models'] if item['role'].endswith('/P')]
    assert len(p) == 7 and all(item['initial_checkpoints'] for item in p)


def test_time_best_selection_retains_real_epoch_and_completed_last(tmp_path):
    from lung.timecourse import response
    import json
    torch.set_num_threads(1)
    data, cfg = fixture_data(), tiny_config()
    cfg['epochs'] = 10
    cfg['early_stopping'] = {'patience': 1}
    cfg['response']['model']['backbone'] = 'compact_full_gene'
    def stop(phase, current, total):
        if phase == 'time_response_member_1' and current == 1:
            raise KeyboardInterrupt('selection committed, fit incomplete')
    with patch.object(response, 'validation_loss', return_value=1.):
        with pytest.raises(KeyboardInterrupt):
            response.train_response(data, cfg, tmp_path / 'fit', progress=stop)
    candidate = next((tmp_path / 'fit/member_1').glob('best-update-*.pt'))
    with pytest.raises(ValueError, match='FINAL_TIME_RESPONSE_REQUIRED'):
        response.load_frozen_response(candidate, data, 'cpu')
    with patch.object(response, 'validation_loss', return_value=2.):
        result = response.train_response(data, cfg, tmp_path / 'fit', resume=True)
    selected = load_checkpoint(result['checkpoints'][0])
    last = load_checkpoint(tmp_path / 'fit/member_1/last.pt')
    assert selected['epoch'] == 1 and last['epoch'] == 2
    assert result['completion'][0]['actual_epochs'] == 2
    assert selected['extra']['training_control']['status'] == 'running'
    assert last['extra']['training_control']['status'] == 'complete'
    response.load_frozen_response(result['checkpoints'][0], data, 'cpu')
    assert response.train_response(data, cfg, tmp_path / 'fit', resume=True)['checkpoints'] == result['checkpoints']
    receipt_path = tmp_path / 'fit/member_1/completion.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['completion']['actual_updates'] == last['counters']['optimizer_step']
    receipt['last_hash'] = 'invalid'
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='FINAL_TIME_RESPONSE_REQUIRED'):
        response.load_frozen_response(result['checkpoints'][0], data, 'cpu')


@pytest.mark.parametrize('helper', ['response_training.py', 'inference_noise.py'])
def test_time_helper_source_change_rejects_resume(tmp_path, helper):
    from lung.timecourse import response
    torch.set_num_threads(1)
    cfg = tiny_config()
    cfg['epochs'] = 1
    cfg['response']['model']['backbone'] = 'compact_full_gene'
    response.train_response(fixture_data(), cfg, tmp_path / 'fit')
    real_hash = response.file_hash
    with patch.object(response, 'file_hash', side_effect=lambda path: 'changed-helper' if Path(path).name == helper else real_hash(path)):
        with pytest.raises(ValueError, match='TIME_RESPONSE_RESUME_IDENTITY_MISMATCH'):
            response.train_response(fixture_data(), cfg, tmp_path / 'fit', resume=True)
