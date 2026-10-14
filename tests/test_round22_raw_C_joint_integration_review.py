"""Real raw-C JSON reader -> real G preparation on tiny synthetic source stubs.

No C reader/declaration validator is mocked. Source/P production registration
and completed AE provenance are controlled missing-dependency fixtures. Asset
files are text bytes, never PT parameters. Nothing is fitted or forwarded.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lung.engine import full_protocol_response as producer
from lung.engine import joint_selection_learning as joint
from lung.engine import raw_state_scoring as raw
from lung.models.full_protocol import FullPhysicalProtocolBatch


@pytest.fixture
def real_reader_bridge(tmp_path, monkeypatch):
    current_dependencies = raw.execution_dependencies()
    monkeypatch.setattr(joint, 'ROOT', tmp_path)
    monkeypatch.setattr(producer, 'ROOT', tmp_path)
    monkeypatch.setattr(raw, 'ROOT', tmp_path)
    monkeypatch.setattr(raw, 'execution_dependencies', lambda: deepcopy(current_dependencies))
    calls = {'functional_Y': 0, 'PT_loads': 0, 'nn_initializations': 0, 'model_forwards': 0,
        'fit_calls': 0, 'parameter_readers': 0, 'P_fixture_reader': 0}
    def deny(key):
        def forbidden(*args, **kwargs):
            calls[key] += 1
            pytest.fail('FORBIDDEN_BRIDGE_OPERATION:' + key)
        return forbidden
    monkeypatch.setattr(torch, 'load', deny('PT_loads'))
    monkeypatch.setattr(torch.nn.Module, '__init__', deny('nn_initializations'))
    monkeypatch.setattr(raw, 'load_classifier', deny('parameter_readers'))
    monkeypatch.setattr(raw, 'load_state_for_prediction', deny('parameter_readers'))
    monkeypatch.setattr(raw, 'score_prediction', deny('model_forwards'))
    monkeypatch.setattr(joint, 'fit_models', deny('fit_calls'))
    monkeypatch.setattr(joint, 'ridge_parameters', deny('fit_calls'))
    view = SimpleNamespace(identity=lambda: {'synthetic_fixture': True, 'tiny_real_C_reader_bridge': 1},
        measured_units=deny('functional_Y'))
    bindings = {'data_identity': 'synthetic_three_gene_source_not_production'}
    genes = ['ENSG_FIXTURE_0', 'ENSG_FIXTURE_1', 'ENSG_FIXTURE_2']
    root = tmp_path / 'experiments'
    ae_root, c_root = root / 'AE', root / 'C'
    ae_root.mkdir(parents=True)
    c_root.mkdir()
    origin_path = ae_root / 'best.pt'
    origin_path.write_bytes(b'SYNTHETIC AE source bytes; not a PT or completed training\n')
    origin = {'run': str(ae_root), 'weights': str(origin_path), 'checkpoint_sha256': joint.sha(origin_path)}
    def controlled_ae_origin(run, identity):
        assert Path(run).resolve() == ae_root and identity == bindings['data_identity']
        assert joint.sha(origin_path) == origin['checkpoint_sha256']
        return deepcopy(origin)
    monkeypatch.setattr(raw, 'verify_ae_origin', controlled_ae_origin)
    cfg = {'task': 'SYNTHETIC_FIXTURE_ONLY', 'synthetic': False, 'seed': 77, 'device': 'cpu', 'epochs': 2,
        'aux_lr': .01, 'calibration_steps': 3, 'calibration_lr': .01, 'state_batch': 2,
        'state_fold': '1_to_8', 'early_stopping': False, 'state_loss': 'weighted_ce',
        'state_aggregate_lambda': 0., 'response': {'inference_microbatch': 2}, 'training_control': {}}
    manifest = {'schema': raw.SCHEMA + ':run', 'data_identity': bindings['data_identity'],
        'config': cfg, 'AE_origin': origin, 'synthetic': False, 'execution_dependencies': current_dependencies}
    joint.write(c_root / 'run_manifest.json', manifest)
    assets = {}
    for key, filename in (('serving', 'state_weights.pt'), ('calibrated', 'weights/state/calibrated.pt'),
        ('best', 'weights/state/best.pt'), ('last', 'weights/state/last.pt'),
        ('temperature', 'weights/state/temperature-last.pt')):
        p = c_root / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(('SYNTHETIC C asset bytes only:' + key + '\n').encode())
        assets[key] = {'path': str(p), 'sha256': joint.sha(p)}
    training_id = 'fixture_C_not_a_trained_model'
    cache = c_root / 'cache' / ('state-calibration-' + training_id + '.pt')
    cache.parent.mkdir()
    cache.write_bytes(b'SYNTHETIC calibration cache bytes; no RNA or parameters\n')
    representation = {'weights': 'fixture_h_tensor_identity_not_actual_weights', 'n_genes': 3,
        'd_model': 4, 'latent_dim': 3, 'checkpoint_sha256': origin['checkpoint_sha256']}
    declaration = {'schema': raw.SCHEMA + ':completed', 'synthetic': False,
        'data_identity': bindings['data_identity'], 'AE_origin': origin, 'representation_identity': representation,
        'width': 4, 'class_order': list(raw.LABELS), 'state_fold': '1_to_8',
        'training_completion': {'status': 'complete', 'reason': 'max_epochs', 'actual_epochs': 2,
            'actual_updates': 4, 'target_epochs': 2}, 'selected_epoch': 2, 'selected_optimizer_step': 4,
        'calibration_completion': {'status': 'complete', 'cursor': 3, 'target_steps': 3,
            'dependencies': {'role': 'calibration', 'objective': 'unweighted_three_class_nll', 'lr': .01,
                'target_steps': 3, 'state_identity': training_id,
                'uncalibrated_best_sha256': assets['best']['sha256'], 'feature_cache_sha256': joint.sha(cache)}},
        'training_identity': training_id, 'manifest_sha256': joint.sha(c_root / 'run_manifest.json'),
        'assets': assets, **raw.FLAGS}
    declaration['state_identity'] = raw.config_hash(declaration)
    declaration_path = c_root / 'state_declaration.json'
    joint.write(declaration_path, declaration)
    # The false synthetic tag deliberately exercises default metadata validation
    # under our controlled AE provenance. It supplies no production authority.
    assert raw.validate_state_declaration(declaration_path) == declaration
    ids = ['fixture_candidate_' + format(i, '02d') for i in range(20)]
    units, requests, p_records, c_records, expected = [], {}, [], [], []
    roles = {r: [] for r in ('fit', 'selection', 'final_test')}
    for i, cid in enumerate(ids):
        role = 'fit' if i < 8 else ('selection' if i < 12 else 'final_test')
        obs = 'fixture_observation_' + str(i)
        roles[role].append(obs)
        candidate = {'dose_uM': .1 + i / 100, 'actual_dose_verified': True, 'state_covariates': {},
            'protocol': {'exposure_duration_h': 24., 'washout_at_h': None,
                'drug_free_recovery_h': None, 'terminal_time_h': 24.}}
        units.append({'candidate_id': cid, 'observation_id': obs, 'role': role,
            'weight': 1, 'complete_candidate': candidate})
        requests[cid] = {'candidate_id': cid, 'physical': {'events': ['one_synthetic_interval']}}
        p_body = {'schema': producer.VERSION, 'candidate_id': cid, 'request': requests[cid], 'bindings': bindings,
            'model_metadata': {'representation': {'checkpoint_sha256': 'fixture_safe_P_AE_export_SHA'},
                'response': {'synthetic_source_stub': True}}, 'production_registration': {'fixture_only': True},
            'reference': {'shape': [2431, 3], 'synthetic_shape_only_no_reference_RNA': True},
            'seed': 77, 'steps': 32, 'outlet': producer.OUTLET, 'summary': 'synthetic_P_summary_only',
            'training_protocol_coverage': {'fixture_only': True}, 'candidate_protocol_OOD': True,
            'candidate_protocol_scientifically_validated': False, 'reference_target_cell_pairing': False,
            'source_state_numerically_encoded': False, 'cross_source_transfer_validated': False,
            'expression_mean': [.1 + i / 100] * 3, 'expression_variance': [.01] * 3}
        p_body['prediction_identity'] = producer.digest(p_body)
        pp = root / 'P' / cid / 'prediction.json'
        joint.write(pp, p_body)
        p_records.append({'candidate_id': cid, **joint.binding(pp), 'prediction_identity': p_body['prediction_identity']})
        moments = [.2 + i / 1000, .3 - i / 1000, .5]
        expected.append(moments)
        c_body = {'schema': raw.SIDECAR_SCHEMA, 'candidate_id': cid,
            'prediction': {'path': str(pp), 'sha256': joint.sha(pp), 'prediction_identity': p_body['prediction_identity']},
            'state_asset': raw.state_asset(declaration_path, declaration), 'class_order': list(raw.LABELS),
            'P_AE_bridge': {'export_checkpoint_sha256': 'fixture_safe_P_AE_export_SHA',
                'source_checkpoint_sha256': origin['checkpoint_sha256'],
                'representation_identity': {k: v for k, v in representation.items() if k != 'checkpoint_sha256'}},
            'probability_mean': moments, 'probability_variance': [.01, .01, .01], 'sample_count': 2431,
            'predict_domain': producer.OUTLET, 'sampling': {'seed': 77, 'steps': 32, 'microbatch': 128,
                'device': 'cpu', 'policy': raw.REPLAY_POLICY}, 'P_replay_moments_exact': True,
            'C_input_coordinate': 'same_AE_standardized_common_log1p_to_h',
            'probability_summary': 'equal_reference_cell_weighted_original_three_class_probabilities',
            'raw_label_semantics': 'original_TIDYI_sorting_labels_not_apoptosis_outcomes', **raw.FLAGS}
        c_body['sidecar_identity'] = raw.config_hash(c_body)
        cp = root / 'Cside' / cid / 'state_sidecar.json'
        joint.write(cp, c_body)
        c_records.append({'candidate_id': cid, **joint.binding(cp)})
    p_index, c_index = root / 'P-index.json', root / 'C-index.json'
    joint.write(p_index, {'schema': joint.INDEX_SCHEMA, 'predictions': p_records})
    joint.write(c_index, {'schema': joint.STATE_INDEX_SCHEMA, 'view_identity': view.identity(), 'sidecars': c_records})
    def controlled_P_reader(path):
        calls['P_fixture_reader'] += 1
        body = joint.read(path)
        assert body['prediction_identity'] == producer.digest({k: v for k, v in body.items() if k != 'prediction_identity'})
        return body
    monkeypatch.setattr(producer, 'read_prediction', controlled_P_reader)
    monkeypatch.setattr(joint, 'metadata_context', lambda *a: (
        view, 'synthetic_source_reader', deepcopy(units), deepcopy(requests), {'gene_ids': genes}, deepcopy(bindings), deepcopy(roles)))
    def controlled_numeric(reader, cid, count):
        assert reader == 'synthetic_source_reader' and cid in ids and count == 1
        values, known = torch.zeros(1, 1, 8), torch.zeros(1, 1, 8, dtype=torch.bool)
        values[0, 0, :4] = torch.tensor([units[ids.index(cid)]['complete_candidate']['dose_uM'], 0., 24., 24.])
        known[0, 0, :4] = True
        physical = FullPhysicalProtocolBatch(torch.zeros(1, 1, 2048), torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.long), values, known, torch.ones(1, dtype=torch.long),
            torch.full((1,), 24.), torch.ones(1, dtype=torch.bool))
        physical.validate()
        return {'fingerprint': physical.fingerprint[:, 0]}, physical
    monkeypatch.setattr(producer, 'numeric_batch', controlled_numeric)
    return SimpleNamespace(root=tmp_path, ids=ids, p_index=p_index, c_index=c_index,
        expected=np.asarray(expected), serving=Path(assets['serving']['path']), calls=calls,
        real_raw_reader=raw.read_state_sidecar, real_declaration_validator=raw.validate_state_declaration)


def test_real_C_reader_twenty_means_enter_G_without_Y_PT_or_model(real_reader_bridge):
    env = real_reader_bridge
    assert raw.read_state_sidecar is env.real_raw_reader
    assert raw.validate_state_declaration is env.real_declaration_validator
    prepared = joint.prepare_inputs(env.p_index, state_sidecar_index=env.c_index)
    assert np.array_equal(prepared.full[:, -3:], env.expected)
    assert prepared.full_names[-3:] == tuple('raw_sorting_C.probability_mean.' + c for c in raw.LABELS)
    assert len(prepared.identity['raw_state_C']['sidecars']) == 20
    assert env.calls['P_fixture_reader'] == 40  # 20 G reads + 20 original raw-C reader reads
    assert all(env.calls[k] == 0 for k in env.calls if k != 'P_fixture_reader')
    assert joint.information_record(prepared)['C_is_measured_Annexin'] is False


@pytest.mark.parametrize('change,reason', (
    ('C_asset_bytes', 'STATE_ASSET_CHANGED'),
    ('AE_bridge', 'SIDECAR_P_C_AE_BRIDGE_CHANGED'),
    ('class_order', 'SIDECAR_CLASS_OR_SCIENCE_CHANGED')))
def test_real_C_reader_rejects_asset_bridge_and_class_drift_before_G(real_reader_bridge, change, reason):
    env = real_reader_bridge
    if change == 'C_asset_bytes':
        with env.serving.open('ab') as stream:
            stream.write(b'SYNTHETIC_C_BYTE_DRIFT')
    else:
        index = joint.read(env.c_index)
        record = index['sidecars'][0]
        path = env.root / record['path']
        body = joint.read(path)
        if change == 'AE_bridge':
            body['P_AE_bridge']['source_checkpoint_sha256'] = 'different_synthetic_AE_origin'
        else:
            body['class_order'] = list(reversed(body['class_order']))
        body['sidecar_identity'] = raw.config_hash({k: v for k, v in body.items() if k != 'sidecar_identity'})
        path.write_text(joint.canonical(body), encoding='utf-8')
        record['sha256'] = joint.sha(path)
        env.c_index.write_text(joint.canonical(index), encoding='utf-8')
    with pytest.raises(ValueError, match=reason):
        joint.prepare_inputs(env.p_index, state_sidecar_index=env.c_index)
    assert raw.read_state_sidecar is env.real_raw_reader
    assert all(env.calls[k] == 0 for k in env.calls if k != 'P_fixture_reader')
