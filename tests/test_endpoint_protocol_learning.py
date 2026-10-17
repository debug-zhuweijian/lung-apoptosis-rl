"""Pure small arrays and hand-written fixed checkpoint parameters only."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.lung.engine import endpoint_protocol_learning as learner


def test_fixed_parameters_bound_fraction():
    model = learner.FixedRidge({'mean': [0., 0.], 'scale': [1., 2.],
                               'beta': [1., 1.], 'intercept': .5})
    assert model(np.array([[0., 0.], [-100., 0.], [100., 0.]])).tolist() == [.5, 0., 1.]
    with pytest.raises(ValueError):
        model([[float('nan'), 0.]])


def test_ridge_pure_array_constant_and_translation():
    x = np.array([[1., 4.], [2., 4.], [3., 4.]])
    a = learner.ridge_parameters(x, [.2, .4, .6])
    b = learner.ridge_parameters(x + 1000, [.2, .4, .6])
    assert np.allclose(a['beta'], b['beta'])
    assert a['scale'][1] == 1.


def test_duplicate_y_or_nonfit_or_generated_weight_rejected():
    base = {'observation_id': 'one', 'role': 'fit', 'weight': 1.,
            'label_kind': 'measured_published_group_mean', 'label_fraction': .3}
    for rows in ([base, base], [{**base, 'role': 'final_test'}],
                 [{**base, 'weight': 500}], [{**base, 'label_fraction': 2}]):
        with pytest.raises(ValueError):
            learner.validate_units(rows)


def handwritten_checkpoint():
    scope = {'endpoint_definition': {'gate': 'test_only'}, 'source_id': 'test_source',
             'cell_line': 'A549', 'state_id': 'test', 'state_kind': 'test', 'dose_scenario': 'test'}
    m = {'schema': learner.SCHEMA, 'scorer_kind': 'endpoint_G', 'algorithm': 'fixed_lambda_ridge_baseline',
         'lambda': learner.LAMBDA, 'feature_names': ['x'], 'scope': scope,
         'value_scale': 'terminal_Annexin_fraction', 'bindings': {'test': 'mock', 'legal_fit_observation_ids': ['a', 'b']},
         'fit_observation_ids': ['a', 'b'], 'fit_count': 2, 'normalization_role': 'fit',
         'weight_per_source_observation': 1, 'synthetic': False, 'actual_fit_completed': True,
         'model_effectiveness_validated': False,
         'reward_qualification': 'pending_independent_measured_validation'}
    value = {'metadata': m, 'parameters': {'mean': [0.], 'scale': [1.], 'beta': [.1], 'intercept': .4}}
    return {**value, 'content_identity': learner.identity(value)}


def test_handwritten_roundtrip_without_label_read(tmp_path, monkeypatch):
    value = handwritten_checkpoint()
    path = tmp_path / 'fixed.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    original = Path.read_text
    def guard(p, *args, **kw):
        assert p == path
        return original(p, *args, **kw)
    monkeypatch.setattr(Path, 'read_text', guard)
    scorer = learner.load_g(path, expected_bindings={'test': 'mock', 'legal_fit_observation_ids': ['a', 'b']}, expected_scope=value['metadata']['scope'])
    assert scorer.scorer_kind == 'endpoint_G'
    assert np.allclose(scorer.scorer([[1.]]), [.5])
    with pytest.raises(ValueError, match='BINDING'):
        learner.load_g(path, expected_bindings={'test': 'changed'}, expected_scope=value['metadata']['scope'])


def test_checkpoint_tamper(tmp_path):
    value = handwritten_checkpoint()
    value['parameters']['beta'] = [.9]
    path = tmp_path / 'fixed.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='CONTENT_CHANGED'):
        learner.load_g(path, expected_bindings={'test': 'mock', 'legal_fit_observation_ids': ['a', 'b']}, expected_scope=value['metadata']['scope'])

def test_cli_default_never_enters_training(monkeypatch, capsys):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'scripts/train_endpoint_protocol_models.py'
    spec = importlib.util.spec_from_file_location('learner_cli_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        pytest.fail('default entered real fit')
    monkeypatch.setattr(module, 'fit_g', forbidden)
    assert module.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['actual_fit_performed'] is False
    assert report['model_execution_performed'] is False
    assert 'audited_real_standard_P_cache' in report['missing']


def test_g_load_requires_exact_fit_role_ids(tmp_path):
    value = handwritten_checkpoint()
    value['metadata']['fit_observation_ids'] = ['forged_a', 'forged_b']
    value['content_identity'] = learner.identity({k: value[k] for k in ('metadata', 'parameters')})
    path = tmp_path / 'g.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='BINDING'):
        learner.load_g(path, expected_bindings=value['metadata']['bindings'], expected_scope=value['metadata']['scope'])


def test_loaded_wrapper_old_rank_rejects_and_new_rank_checks_scope(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.lung.engine.endpoint_protocol_model import rank_protocol_candidates
    value = handwritten_checkpoint()
    path = tmp_path / 'fixed.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    wrapper = learner.load_g(path, expected_bindings=value['metadata']['bindings'], expected_scope=value['metadata']['scope'])
    with pytest.raises(ValueError, match='SCORER_KIND'):
        rank_protocol_candidates(None, ['c'], scorer=wrapper, cache_manifest='mock')
    monkeypatch.setattr(learner, 'plan_bindings', lambda *a: value['metadata']['bindings'])
    monkeypatch.setattr(learner, 'ProtocolConditionAdapter', lambda: SimpleNamespace(encode=lambda packet: SimpleNamespace(scopes={'wrong_state': True})))
    plan = SimpleNamespace(identity=lambda *a: {}, metadata_units=lambda *a, **k: [])
    bridge = SimpleNamespace(require_response_features=lambda *a, **k: {})
    with pytest.raises(ValueError, match='RANK_SCOPE'):
        learner.rank_checkpoint_candidates(bridge, ['c'], checkpoint_scorer=wrapper, plan=plan,
            fold_id='f', cache_manifest='mock')
