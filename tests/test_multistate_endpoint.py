"""Synthetic metadata/typed-cache doubles only. Never run a production model."""
import json
from pathlib import Path

import numpy as np
import pytest

from lung.data.martin_state_function import BACKGROUND_ORDER, DOMAIN_IDS, SCALE, MartinStateFunctionInputs
from lung.data.multistate_role_plan import MultiStateRolePlan, build_role_plan, digest, write_new
from lung.engine import multistate_endpoint_model as model
from lung.engine.multistate_endpoint_learning import NonnegativeRidge, fit_g, load_g, ridge_parameters, terminal_q_preflight
from lung.engine.native_response_producer import VerifiedNativeDescriptor


class MockSource(MartinStateFunctionInputs):
    def __init__(self, directory):
        self.directory = directory
        self.references = {b: {'state_reference_id': b, 'cell_background': b} for b in BACKGROUND_ORDER}
        self.domains = {d: {'measurement_domain_id': d, 'value_scale': SCALE,
                          'allowed_state_references': list(BACKGROUND_ORDER),
                          'endpoint_definition': {'signal': d + ':caspase'}} for d in DOMAIN_IDS}
        self.candidates, self.blocks = {}, {}
        labels = []
        for bi, b in enumerate(BACKGROUND_ORDER):
            for di, d in enumerate(DOMAIN_IDS):
                block = b + ':' + d
                ids = [block + ':' + str(j) for j in range(3 if di == 0 else 2)]
                self.blocks[block] = {'decision_block_id': block, 'measurement_domain_id': d,
                                     'state_reference_id': b, 'observation_window_h': 72,
                                     'candidate_ids': ids, 'value_scale': SCALE}
                for j, cid in enumerate(ids):
                    self.candidates[cid] = {'candidate_id': cid, 'observation_id': cid,
                        'measurement_domain_id': d, 'state_reference_id': b,
                        'cell_background': b, 'decision_block_id': block,
                        'actual_protocol': {'observation_window_h': 72, 'reported_exposure_h': 72},
                        'chemical': {'chemical_id': 'compound:' + str(j)},
                        'nominal_dose_uM': .16, 'response_request': {'producer_status': 'pending'}}
                    labels.append({'candidate_id': cid, 'observation_id': cid,
                        'value': float(100 + bi * 10 + j), 'weight': 1, 'value_scale': SCALE,
                        'label_kind': 'digitized_published_group_mean_not_raw_replicate'})
        directory.mkdir()
        write_new(directory / 'metadata.json', {'references': list(self.references.values()),
            'measurement_domains': list(self.domains.values()), 'candidates': list(self.candidates.values()),
            'decision_blocks': list(self.blocks.values())})
        write_new(directory / 'observations.json', labels)
        write_new(directory / 'manifest.json', {'synthetic': True})

    def decision_inputs(self, candidate_ids):
        rows = [self.candidates[c] for c in candidate_ids]
        blocks = {r['decision_block_id'] for r in rows}
        if len(blocks) != 1 or set(candidate_ids) != set(self.blocks[next(iter(blocks))]['candidate_ids']):
            raise ValueError('COMPLETE_SINGLE_BLOCK_REQUIRED')


@pytest.fixture
def lane(tmp_path):
    source = MockSource(tmp_path / 'source')
    build_role_plan(source, tmp_path / 'roles')
    return source, MultiStateRolePlan(tmp_path / 'roles', source), tmp_path


def mock_native(lane, monkeypatch):
    source, plan, tmp = lane
    fold, head = 'cycle_final:HCC2935', DOMAIN_IDS[0]
    monkeypatch.setattr(model, '_native_bindings', lambda source: {'test_only': 'mock_coordinates'})
    descriptors = {}
    for cid, c in source.candidates.items():
        if c['measurement_domain_id'] != head:
            continue
        path = tmp / (cid.replace(':', '_') + '.json')
        write_new(path, {'synthetic_descriptor_stub': cid})
        descriptors[cid] = {'path': path.name, 'sha256': model.sha(path),
                           'bindings': model.required_bindings(source, plan, fold, head, cid),
                           'request_binding': model._function_request(source, cid)}
    index = {'schema': model.CACHE_INDEX_SCHEMA, 'descriptors': descriptors}
    index['content_identity'] = digest(index)
    index_path = tmp / 'cache.json'
    write_new(index_path, index)

    def fake_loader(path, bindings):
        cid = json.loads(Path(path).read_text())['synthetic_descriptor_stub']
        c = source.candidates[cid]
        v = np.array([float(BACKGROUND_ORDER.index(c['cell_background'])), float(cid.rsplit(':', 1)[1]), 7.])
        v.setflags(write=False)
        return VerifiedNativeDescriptor(v, ('native.mean:a', 'native.var:a', 'native.constant'),
            {'function_candidate_binding': model._function_request(source, cid),
             'role_plan_identity': plan.identity(fold, head), 'measurement_level': 'bulk'},
            bindings, 'native', 'test_only_mock_execution', {})
    monkeypatch.setattr(model, '_load_producer_descriptor', fake_loader)
    return fold, head, index_path


def test_metadata_cycle_has_disjoint_whole_background_roles(lane):
    source, plan, _ = lane
    for fold in plan.metadata['folds']:
        for head, counts in zip(DOMAIN_IDS, ((6, 3, 3), (4, 2, 2))):
            sets = [{u['candidate_id'] for u in plan.metadata_units(fold['fold_id'], head, role)}
                    for role in ('fit', 'selection', 'final_test')]
            assert tuple(map(len, sets)) == counts
            assert not any(a & b for i, a in enumerate(sets) for b in sets[i + 1:])
            assert len(plan.training_domain(fold['fold_id'], head)['fit_state_references']) == 2
    assert plan.metadata['old_LOBO_roles_reused'] is False


def test_real_preflight_and_fit_stop_before_labels(lane, monkeypatch):
    source, plan, tmp = lane
    monkeypatch.setattr(plan, 'fit_units', lambda *a: pytest.fail('Y parsed before pending backend'))
    result = model.preflight(source, plan, fold_id='cycle_final:HCC2935', head_id=DOMAIN_IDS[0])
    assert result['status'] == 'pending_native_backend' and result['labels_parsed'] is False
    with pytest.raises(ValueError, match='NATIVE_BULK72_BACKEND_PENDING'):
        fit_g(source, plan, fold_id='cycle_final:HCC2935', head_id=DOMAIN_IDS[0],
              native_cache_index=tmp / 'does_not_exist', output=tmp / 'g.json')
    assert not (tmp / 'g.json').exists()


def test_mock_source_ridge_fit_and_single_state_complete_ranking(lane, monkeypatch):
    source, plan, tmp = lane
    fold, head, index = mock_native(lane, monkeypatch)
    cp = fit_g(source, plan, fold_id=fold, head_id=head, native_cache_index=index, output=tmp / 'g.json')
    assert len(cp['metadata']['fit_observation_ids']) == 6
    assert len(cp['metadata']['training_domain']['fit_state_references']) == 2
    result = model.rank_block(source, plan, fold_id=fold, head_id=head,
        block_id='HCC2935:' + head, role='final_test', native_cache_index=index, checkpoint=tmp / 'g.json')
    assert len(result['predictions']) == 3 and result['state_in_fit'] is False
    assert all(r['value'] > 1 for r in result['predictions'])
    assert cp['parameters']['mean'] == [2.5, 1., 7.]
    assert cp['parameters']['scale'][2] == 1.


@pytest.mark.parametrize('bad', ('subset', 'state', 'head', 'duplicate'))
def test_reject_mixed_or_incomplete_ranking_before_cache(lane, bad):
    source, plan, tmp = lane
    head = DOMAIN_IDS[0]
    ids = list(source.blocks['HCC2935:' + head]['candidate_ids'])
    if bad == 'subset':
        ids = ids[:2]
    elif bad == 'state':
        ids[0] = source.blocks['PC9:' + head]['candidate_ids'][0]
    elif bad == 'head':
        ids[0] = source.blocks['HCC2935:' + DOMAIN_IDS[1]]['candidate_ids'][0]
    else:
        ids[0] = ids[1]
    with pytest.raises(ValueError, match='COMPLETE_SINGLE_STATE'):
        model.rank_block(source, plan, fold_id='cycle_final:HCC2935', head_id=head,
            block_id='HCC2935:' + head, role='final_test', native_cache_index=tmp / 'missing',
            checkpoint=tmp / 'missing', candidate_ids=ids)


def test_caspase_nonnegative_not_fraction_clip():
    p = ridge_parameters([[0., 7.], [1., 7.]], [100., 200.])
    scorer = NonnegativeRidge(p)
    assert scorer([[0., 7.], [1., 7.]])[1] > 1
    assert scorer([[-100., 7.]])[0] == 0
    assert p['scale'][1] == 1


@pytest.mark.parametrize('field', ('bindings', 'request_binding', 'sha256'))
def test_mock_native_cache_rejects_changed_binding_or_bytes(lane, monkeypatch, field):
    source, plan, tmp = lane
    fold, head, index = mock_native(lane, monkeypatch)
    value = json.loads(index.read_text())
    cid = plan.metadata_units(fold, head, 'fit')[0]['candidate_id']
    value['descriptors'][cid][field] = {} if field != 'sha256' else '0' * 64
    value['content_identity'] = digest({k: v for k, v in value.items() if k != 'content_identity'})
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='NATIVE_'):
        model.NativeMultiStateAdapter().encode(cid, source=source, plan=plan, fold_id=fold,
                                              head_id=head, native_cache_index=index)


def test_mock_loader_rejects_audit_mock_execution(lane, monkeypatch):
    source, plan, _ = lane
    fold, head, index = mock_native(lane, monkeypatch)
    real = model._load_producer_descriptor
    def audit(path, bindings):
        d = real(path, bindings)
        return VerifiedNativeDescriptor(d.vector, d.feature_names, d.request, d.bindings,
                                        'audit_mock', d.cache_identity, d.reference)
    monkeypatch.setattr(model, '_load_producer_descriptor', audit)
    with pytest.raises(ValueError, match='CAPABILITY_PENDING'):
        model.NativeMultiStateAdapter().encode(plan.metadata_units(fold, head, 'fit')[0]['candidate_id'],
            source=source, plan=plan, fold_id=fold, head_id=head, native_cache_index=index)


def test_reject_old_fold_and_wrong_role_head(lane):
    _, plan, _ = lane
    with pytest.raises(ValueError, match='UNKNOWN_MULTISTATE_FOLD'):
        plan.identity('leave_background_out:HCC2935', DOMAIN_IDS[0])
    with pytest.raises(ValueError, match='ROLE_OR_HEAD'):
        plan.decision_block('cycle_final:HCC2935', DOMAIN_IDS[1], 'HCC2935:' + DOMAIN_IDS[0], 'final_test')


def test_mutated_in_memory_source_metadata_rejected(lane):
    source, plan, _ = lane
    source.domains[DOMAIN_IDS[0]]['endpoint_definition'] = {'signal': 'Annexin'}
    with pytest.raises(ValueError, match='MUTATED_SOURCE_METADATA'):
        plan.verify()


def test_checkpoint_content_tamper_rejected(lane, monkeypatch):
    source, plan, tmp = lane
    fold, head, index = mock_native(lane, monkeypatch)
    path = tmp / 'g.json'
    cp = fit_g(source, plan, fold_id=fold, head_id=head, native_cache_index=index, output=path)
    cp['parameters']['intercept'] += 1
    path.write_text(json.dumps(cp))
    with pytest.raises(ValueError, match='CONTENT_CHANGED'):
        load_g(path, source=source, plan=plan, fold_id=fold, head_id=head, native_cache_index=index)


def test_q_stays_pending_with_new_or_old_roles():
    q = terminal_q_preflight()
    assert q['reward_qualified'] is False and q['Q_fit_implemented'] is False
    assert q['old_LOBO_no_selection_role_eligible'] is False
    assert 'single_terminal' in q['target']


@pytest.mark.parametrize('change', ('scorer_kind', 'normalization_role', 'output_policy', 'training_domain'))
def test_rehashed_checkpoint_cannot_switch_scorer_scope_or_scale(lane, monkeypatch, change):
    source, plan, tmp = lane
    fold, head, index = mock_native(lane, monkeypatch)
    path = tmp / 'g.json'
    cp = fit_g(source, plan, fold_id=fold, head_id=head, native_cache_index=index, output=path)
    cp['metadata'][change] = {} if change == 'training_domain' else 'forged'
    cp['content_identity'] = digest({k: v for k, v in cp.items() if k != 'content_identity'})
    path.write_text(json.dumps(cp))
    with pytest.raises(ValueError, match='BINDINGS_OR_MEASUREMENT'):
        load_g(path, source=source, plan=plan, fold_id=fold, head_id=head, native_cache_index=index)


def test_freeze_is_complete_role_bound_and_not_reward_qualification(lane):
    source, plan, tmp = lane
    head, block = DOMAIN_IDS[0], 'HCC827:' + DOMAIN_IDS[0]
    ids = source.blocks[block]['candidate_ids']
    predictions = dict.fromkeys(ids, 150.)
    artifacts = {str(source.directory / 'metadata.json'): model.sha(source.directory / 'metadata.json')}
    receipt = plan.freeze_predictions('cycle_final:HCC2935', head, block, 'selection', predictions, artifacts)
    assert receipt['execution_certified'] is False and receipt['reward_qualified'] is False
    with pytest.raises(ValueError, match='INVALID_COMPLETE_FROZEN'):
        plan.freeze_predictions('cycle_final:HCC2935', head, block, 'selection', {ids[0]: 150.}, artifacts)
    with pytest.raises(ValueError, match='ROLE_OR_HEAD'):
        plan.freeze_predictions('cycle_final:HCC2935', head, block, 'final_test', predictions, artifacts)


def test_cli_has_no_training_or_weight_loading_switch():
    from scripts.train_multistate_endpoint import main
    with pytest.raises(SystemExit) as error:
        main(['--train'])
    assert error.value.code == 2


@pytest.mark.parametrize('bad', ([1., True], [1., 2 + 3j], np.array([True, False]),
                                np.array([1 + 2j, 3 + 4j])))
@pytest.mark.parametrize('field', ('mean', 'scale', 'beta'))
def test_frozen_parameters_reject_bool_complex_before_coercion(field, bad):
    params = {'mean': [0., 0.], 'scale': [1., 1.], 'beta': [1., 1.], 'intercept': 0.}
    params[field] = bad
    with pytest.raises(ValueError, match='NO_BOOL_OR_COMPLEX'):
        NonnegativeRidge(params)


@pytest.mark.parametrize('bad', ([[0., True], [1., 2.]], [[0., 1j], [1., 2.]],
                                np.array([[True, False], [False, True]]),
                                np.array([[1j, 2.], [3., 4.]])))
def test_fit_and_prediction_x_reject_bool_complex(bad):
    with pytest.raises(ValueError, match='NO_BOOL_OR_COMPLEX'):
        ridge_parameters(bad, [100., 200.])
    scorer = NonnegativeRidge({'mean': [0., 0.], 'scale': [1., 1.],
                              'beta': [1., 1.], 'intercept': 0.})
    with pytest.raises(ValueError, match='NO_BOOL_OR_COMPLEX'):
        scorer(bad)


@pytest.mark.parametrize('bad', ([100., True], [100., 2j], np.array([True, False]),
                                np.array([100., 200j])))
def test_fit_y_reject_bool_complex(bad):
    with pytest.raises(ValueError, match='NO_BOOL_OR_COMPLEX'):
        ridge_parameters([[0., 1.], [1., 2.]], bad)


@pytest.mark.parametrize('bad', ([1., True, 3.], [1., 2j, 3.], np.array([True, False, True]),
                                np.array([1., 2j, 3.])))
def test_typed_native_features_reject_bool_complex(lane, monkeypatch, bad):
    source, plan, _ = lane
    fold, head, index = mock_native(lane, monkeypatch)
    original = model._load_producer_descriptor
    def invalid_vector(path, bindings):
        d = original(path, bindings)
        return VerifiedNativeDescriptor(bad, d.feature_names, d.request, d.bindings,
                                        d.execution_kind, d.cache_identity, d.reference)
    monkeypatch.setattr(model, '_load_producer_descriptor', invalid_vector)
    with pytest.raises(ValueError, match='NO_BOOL_OR_COMPLEX'):
        model.NativeMultiStateAdapter().encode(plan.metadata_units(fold, head, 'fit')[0]['candidate_id'],
            source=source, plan=plan, fold_id=fold, head_id=head, native_cache_index=index)
