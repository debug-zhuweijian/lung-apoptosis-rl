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


def test_checkpoint_tamper_and_q_pending(tmp_path):
    value = handwritten_checkpoint()
    value['parameters']['beta'] = [.9]
    path = tmp_path / 'fixed.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='CONTENT_CHANGED'):
        learner.load_g(path, expected_bindings={'test': 'mock', 'legal_fit_observation_ids': ['a', 'b']}, expected_scope=value['metadata']['scope'])
    assert learner.terminal_q_preflight()['private_Q_implemented'] is True

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


def test_final_test_cannot_qualify_q_reward(tmp_path):
    g = handwritten_checkpoint()
    path = tmp_path / 'qualification.json'
    path.write_text(json.dumps({'schema': 'protocol-G-reward-qualification-v1',
                               'role': 'final_test', 'measured_validation_passed': True}), encoding='utf-8')
    with pytest.raises(ValueError, match='QUALIFICATION_PENDING'):
        learner._qualification(path, g)


def test_private_q_handwritten_parameter_roundtrip(tmp_path, monkeypatch):
    artifact = tmp_path / 'fixed_artifact.json'
    artifact.write_text('{}', encoding='utf-8')
    scope = handwritten_checkpoint()['metadata']['scope']
    q = {'schema': learner.Q_SCHEMA, 'bindings': {'mock': 'identity'}, 'scope': scope,
         'feature_names': ['x'], 'parameters': {'mean': [0.], 'scale': [1.], 'beta': [.2], 'intercept': .3},
         'evidence': 'model_simulated', 'actual_fit_completed': True, 'private_parameters': True,
         'target': 'terminal_reward_only', 'lambda': learner.LAMBDA,
         'value_scale': 'terminal_Annexin_fraction', 'biological_effectiveness_validated': False,
         'g_checkpoint': {'path': str(artifact), 'sha256': learner.sha(artifact)},
         'qualification': {'path': str(artifact), 'sha256': learner.sha(artifact)}}
    q['content_identity'] = learner.identity(q)
    path = tmp_path / 'q.json'
    path.write_text(json.dumps(q), encoding='utf-8')
    from types import SimpleNamespace
    monkeypatch.setattr(learner, '_qualification', lambda *args: {'mock': True})
    monkeypatch.setattr(learner, 'load_g', lambda *args, **kw: SimpleNamespace(feature_names=('x',)))
    scorer = learner.load_terminal_q(path, expected_bindings={'mock': 'identity'}, expected_scope=scope)
    assert scorer.scorer_kind == 'private_candidate_Q'
    assert np.allclose(scorer.scorer([[1.]]), [.5])


@pytest.mark.parametrize('value', [-.1, True, float('nan'), 2])
def test_rule_rejects_negative_bool_or_nonfinite(value):
    with pytest.raises(ValueError, match='REWARD_RULE'):
        learner._rule({'max_mae_fraction': value, 'require_pairwise_order_match': False})


def test_g_load_requires_exact_fit_role_ids(tmp_path):
    value = handwritten_checkpoint()
    value['metadata']['fit_observation_ids'] = ['forged_a', 'forged_b']
    value['content_identity'] = learner.identity({k: value[k] for k in ('metadata', 'parameters')})
    path = tmp_path / 'g.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='BINDING'):
        learner.load_g(path, expected_bindings=value['metadata']['bindings'], expected_scope=value['metadata']['scope'])


def test_program_qualification_roundtrip_and_forged_metric_recomputed(tmp_path, monkeypatch):
    checkpoint = handwritten_checkpoint()
    checkpoint['metadata']['bindings']['role_contract'] = {'fold_id': 'f', 'branch': 'main'}
    checkpoint['metadata']['bindings']['legal_selection_observation_ids'] = ['s1', 's2']
    checkpoint['metadata']['bindings']['reserved_test_observation_ids'] = ['test']
    g_path, cache_path, rule_path = [tmp_path / name for name in ('g.json', 'cache.json', 'rule.json')]
    g_path.write_text(json.dumps(checkpoint), encoding='utf-8')
    cache_path.write_text('{}', encoding='utf-8')
    rule_path.write_text(json.dumps({'max_mae_fraction': .2, 'require_pairwise_order_match': True}), encoding='utf-8')
    units = [{'candidate_id': 'c1', 'observation_id': 's1'}, {'candidate_id': 'c2', 'observation_id': 's2'}]
    block = {'block_id': 'selection_block', 'role': 'selection', 'candidate_ids': ['c1', 'c2']}
    predictions = {'c1': .3, 'c2': .7}
    class Plan:
        def metadata_units(self, fold, branch, role):
            return units if role == 'selection' else [{'observation_id': 'test'}] if role == 'final_test' else []
        def freeze_predictions(self, fold, bid, scores, *, branch, artifacts):
            from src.lung.data.endpoint_protocol_evaluation import digest
            return {'identity': {'fold_id': 'f', 'branch': 'main'}, 'block_id': bid,
                    'predictions': scores, 'predictions_sha256': digest(scores), 'artifacts': artifacts}
        def evaluate_frozen(self, receipt, *, endpoint_definition):
            # Pure stand-in for the real selection-only endpoint API.
            return {'identity': receipt['identity'], 'block_id': receipt['block_id'],
                    'role': 'selection', 'mean_absolute_error': .1, 'pairwise_order_matches': True}
    plan = Plan()
    monkeypatch.setattr(learner, '_selection_prediction', lambda *a: (checkpoint, block, units, predictions))
    out = tmp_path / 'validation'
    qualification = learner.validate_reward(None, plan, fold_id='f', cache_manifest=cache_path,
        g_path=g_path, rule_path=rule_path, output=out)
    assert qualification['measured_validation_passed'] is True
    learner._revalidate_reward(out / 'qualification.json', checkpoint, bridge=None, plan=plan,
        fold_id='f', branch='main', cache_manifest=cache_path, g_path=g_path)
    # Self-consistent file/hash forgery passes byte checks but fails recomputation.
    evaluation_path = out / 'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    evaluation['mean_absolute_error'] = 0.
    evaluation_path.write_text(json.dumps(evaluation), encoding='utf-8')
    qualification['evaluation_artifact']['sha256'] = learner.sha(evaluation_path)
    (out / 'qualification.json').write_text(json.dumps(qualification), encoding='utf-8')
    with pytest.raises(ValueError, match='ACTUAL_SELECTION_METRICS'):
        learner._revalidate_reward(out / 'qualification.json', checkpoint, bridge=None, plan=plan,
            fold_id='f', branch='main', cache_manifest=cache_path, g_path=g_path)


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


@pytest.mark.parametrize('mae', [-999, True, float('nan'), 2])
def test_metrics_cannot_be_negative_bool_nonfinite_or_over_one(mae):
    with pytest.raises(ValueError, match='SELECTION_METRICS'):
        learner._passes({'max_mae_fraction': .2, 'require_pairwise_order_match': False},
            {'role': 'selection', 'mean_absolute_error': mae, 'pairwise_order_matches': True})


@pytest.mark.parametrize('change', [{'terminated': 1}, {'weight': True}, {'reward': False},
                                    {'target': True}, {'vector': [True]}, {'extra': 1},
                                    {'vector': [[1]]}, {'target': .9}])
def test_terminal_decision_strict_types(change):
    row = {'candidate_id': 'c', 'vector': [1.], 'terminated': True,
           'reward': .4, 'target': .4, 'weight': 1.}
    with pytest.raises(ValueError, match='ROW_SCHEMA'):
        learner.validate_decision_rows([{**row, **change}])
