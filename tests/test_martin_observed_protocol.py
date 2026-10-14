"""Fixed metadata and CPU physical tensors only; no model modules or forward."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lung.data import martin_observed_protocol as module


class FakeRNA:
    def __init__(self, rows, allowed, allow_final):
        if 'final_test' in allowed and not allow_final:
            raise PermissionError('FINAL_NOT_AUTHORIZED')
        self.rows, self.allowed = rows, allowed
        self.target_reads = 0

    def close(self):
        pass

    def requests(self, *, system, fold_id, role):
        if role not in self.allowed or system not in self.rows:
            raise PermissionError('ROLE_SYSTEM_NOT_ALLOWED')
        return json.loads(json.dumps([r for r in self.rows[system] if r['fold_id'] == fold_id and r['role'] == role]))

    def reference(self, request_id, **kwargs):
        return {'values': np.array([[-1., 2.]], dtype=np.float64), 'target_expression_used': False,
                'functional_Y': None, 'coordinate': kwargs['coordinate'], 'domain': kwargs['domain']}

    def supervision(self, request_id, **kwargs):
        self.target_reads += 1
        return {'reference': self.reference(request_id, **kwargs),
                'target': {'values': np.array([[3., 4.]], dtype=np.float64), 'target_expression_used': True}}


@pytest.fixture
def lane(tmp_path, monkeypatch):
    chemical = {'chemical_id': 'pubchem:71496458', 'fingerprint_sha256': 'mock_nominal_fp'}
    rows, groups, links = {'lobo': [], 'cycle': []}, {}, {'lobo': [], 'cycle': []}
    backgrounds = ['HCC2935', 'HCC827', 'H1975', 'PC9']
    for system in rows:
        for i, final_bg in enumerate(backgrounds):
            fold = system + ':' + final_bg
            for j, background in enumerate(backgrounds):
                role = ('final_test' if j == i else 'selection' if system == 'cycle' and j == (i + 1) % 4 else 'fit')
                for state in ['acute_post_treatment', 'DTP', 'short_washout', 'long_washout']:
                    exposure = 24 if state == 'acute_post_treatment' else 504
                    recovery = 24 if state == 'short_washout' else {'H1975': 72, 'HCC827': 96, 'PC9': 168, 'HCC2935': 240}[background] if state == 'long_washout' else 0
                    request = {'request_id': system + ':' + background + ':' + state + ':' + fold,
                        'fold_id': fold, 'role': role, 'background_id': background,
                        'dose_uM': .5, 'chemical_id': chemical['chemical_id'], 'protocol_id': 'Martin2022_RNA_' + state,
                        'target_sample_ids': [background + state + str(n) for n in range(3)],
                        'target_column_indices': [3, 4, 5], 'physical_reference_ids': [background + 'ref' + str(n) for n in range(3)],
                        'reference_column_indices': [0, 1, 2], 'function_72h_four_drug_target': False,
                        'actual_protocol': {'long_short_history': state, 'exposure_duration_h': exposure,
                                            'washout_h': recovery, 'observation_time_h': exposure + recovery}}
                    rows[system].append(request)
                    group = module._group(request, chemical)
                    groups[group['group_id']] = group
                    links[system].append({'request_id': request['request_id'], 'fold_id': fold,
                                          'role': role, 'group_id': group['group_id']})
    snapshot = {'groups': list(groups.values()), 'role_links': links, 'RNA_coordinates': {'test_only': True},
                'source_bindings': {'test_only': 'fixed_metadata_double'}, 'fingerprint': np.r_[np.ones(70, dtype=np.uint8), np.zeros(1978, dtype=np.uint8)]}
    def source(root, allowed, allow_final):
        return FakeRNA(rows, allowed, allow_final), snapshot
    monkeypatch.setattr(module, '_reconstruct', source)
    output = tmp_path / 'package'
    module.prepare(tmp_path, output)
    reader = module.MartinObservedProtocol(tmp_path, output, allowed_roles=module.ROLES, allow_final=True)
    return tmp_path, output, rows, reader


def test_sixteen_groups_and_forty_documented_events(lane):
    _, _, _, reader = lane
    assert reader.manifest['groups'] == 16 and reader.manifest['events'] == 40
    assert sum(len(g['events']) for g in reader._groups.values()) == 40
    assert reader.manifest['role_requests'] == {'lobo': 64, 'cycle': 64}


@pytest.mark.parametrize('background,endpoint', [('H1975', 576), ('HCC827', 600), ('PC9', 672), ('HCC2935', 744)])
def test_each_actual_long_clock_and_unknown_operation_duration(lane, background, endpoint):
    _, _, rows, reader = lane
    row = next(r for r in rows['cycle'] if r['background_id'] == background and r['actual_protocol']['long_short_history'] == 'long_washout')
    condition = reader.condition(row['request_id'], system='cycle', fold_id=row['fold_id'], role=row['role'])
    group = condition['group']
    assert group['endpoint_hours'] == endpoint
    assert [e['event_type'] for e in group['events']] == ['drug_interval', 'wash', 'media_change', 'recovery']
    assert group['events'][1]['values']['wash_count'] == 2
    assert group['events'][1]['values']['start_hours'] == group['events'][1]['values']['end_hours'] == 504
    assert group['unknown']['wash_operation_duration'] is None
    assert group['events'][3]['values']['prior_retention'] is None
    assert group['reference_own_clock'] == {'drug_free_preculture_days': 21, 'vehicle_DMSO_hours': 24, 'same_cell_t0': False}


def test_cpu_batch_has_500nM_fp_and_unknown_masks(lane):
    _, _, rows, reader = lane
    selected = reader.requests(system='cycle', fold_id='cycle:HCC2935', role='fit')
    batch = reader.physical_batch([r['request_id'] for r in selected], system='cycle', fold_id='cycle:HCC2935', role='fit')
    assert batch.validate() is None and batch.values.device.type == 'cpu'
    assert torch.all(batch.values[:, 0, 0] == 500)
    assert torch.all(batch.fingerprint[:, 0].sum(-1) == 70)
    assert not batch.known[..., module.SCALAR_FIELDS.index('prior_retention')].any()
    assert not batch.fingerprint_known[:, 1:].any()
    assert all(g['unknown']['maintenance_media_events'] is None for g in reader._groups.values())


@pytest.mark.parametrize('bad', ('role', 'fold', 'system'))
def test_role_fold_and_system_mismatch_blocked(lane, bad):
    _, _, rows, reader = lane
    r = next(r for r in rows['cycle'] if r['role'] == 'selection')
    kwargs = {'system': 'cycle', 'fold_id': r['fold_id'], 'role': r['role']}
    kwargs['fold_id' if bad == 'fold' else bad] = 'fit' if bad == 'role' else 'other_fold' if bad == 'fold' else 'lobo'
    with pytest.raises(PermissionError):
        reader.condition(r['request_id'], **kwargs)


def test_reference_only_does_not_open_supervision(lane):
    _, _, rows, reader = lane
    r = rows['cycle'][0]
    packet = reader.reference(r['request_id'], system='cycle', fold_id=r['fold_id'], role=r['role'])
    assert packet['target_expression_used'] is False and reader._rna.target_reads == 0
    reader.supervision(r['request_id'], system='cycle', fold_id=r['fold_id'], role=r['role'])
    assert reader._rna.target_reads == 1


def test_design_base_contract_stays_pending_and_metadata_copy(lane):
    _, _, rows, reader = lane
    r = rows['cycle'][0]
    p = reader.condition(r['request_id'], system='cycle', fold_id=r['fold_id'], role=r['role'])
    base = p['group']['base_design']
    assert base['design_choice_not_paper_instruction'] is True and base['elapsed_h'] == 0
    assert base['numeric_model_kwargs_ready'] is False and base['source_index'] is None
    p['group']['events'][0]['values']['dose'] = 160
    assert reader.condition(r['request_id'], system='cycle', fold_id=r['fold_id'], role=r['role'])['group']['events'][0]['values']['dose'] == 500


@pytest.mark.parametrize('field', ('dose_uM', 'observation_time_h', 'chemical_id', 'function_72h_four_drug_target'))
def test_changed_rna_facts_not_accepted(lane, field):
    _, _, rows, _ = lane
    request = json.loads(json.dumps(rows['cycle'][0]))
    if field == 'observation_time_h':
        request['actual_protocol'][field] = 72
    else:
        request[field] = True if field == 'function_72h_four_drug_target' else .16 if field == 'dose_uM' else 'another_drug'
    with pytest.raises(ValueError, match='RNA_DOSE_CLOCK_OR_CHEMICAL'):
        module._group(request, {'chemical_id': 'pubchem:71496458', 'fingerprint_sha256': 'mock'})


def test_rehashed_modified_event_package_rejected(lane):
    root, path, _, reader = lane
    reader.close()
    groups = module.read_json(path / 'groups.json')
    groups[0]['events'][0]['values']['dose'] = 160
    (path / 'groups.json').write_text(json.dumps(groups))
    m = module.read_json(path / 'manifest.json')
    m['files']['groups.json'] = module.sha(path / 'groups.json')
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='RECONSTRUCTION_CHANGED'):
        module.MartinObservedProtocol(root, path)


def test_existing_package_not_overwritten(lane):
    root, path, _, _ = lane
    with pytest.raises(FileExistsError):
        module.prepare(root, path)


def test_real_registry_rejects_changed_source_before_rna_access(tmp_path):
    relative = module.ARTICLE
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('caller replacement paper and own hash')
    with pytest.raises(ValueError, match='FIXED_PROTOCOL_SOURCE_CHANGED'):
        module.prepare(tmp_path, tmp_path / 'forged')
    assert not (tmp_path / 'forged').exists()


def test_final_role_requires_closed_reader_permission(lane):
    root, path, _, _ = lane
    with pytest.raises(PermissionError, match='FINAL_NOT_AUTHORIZED'):
        module.MartinObservedProtocol(root, path, allowed_roles=('fit', 'final_test'))


@pytest.mark.parametrize('dtype', (torch.bool, torch.int64, torch.complex64))
def test_cpu_physical_numeric_dtype_rejected(lane, dtype):
    _, _, _, reader = lane
    rows = reader.requests(system='cycle', fold_id='cycle:HCC2935', role='fit')
    batch = reader.physical_batch([rows[0]['request_id']], system='cycle', fold_id='cycle:HCC2935', role='fit')
    object.__setattr__(batch, 'values', batch.values.to(dtype))
    with pytest.raises(ValueError, match='real floating point'):
        batch.validate()


def test_rehashed_role_link_cannot_replace_source_role(lane):
    root, path, _, reader = lane
    reader.close()
    links = module.read_json(path / 'role_links.json')
    links['cycle'][0]['role'] = 'final_test' if links['cycle'][0]['role'] != 'final_test' else 'fit'
    (path / 'role_links.json').write_text(json.dumps(links))
    m = module.read_json(path / 'manifest.json')
    m['files']['role_links.json'] = module.sha(path / 'role_links.json')
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='RECONSTRUCTION_CHANGED'):
        module.MartinObservedProtocol(root, path)


def test_prepare_cli_has_no_training_switch():
    from scripts.prepare_martin_observed_protocol import main
    with pytest.raises(SystemExit) as error:
        main(['--train'])
    assert error.value.code == 2


@pytest.mark.parametrize('field', ('functional_Y_read', 'model_or_function_qualified'))
def test_integer_zero_cannot_impersonate_manifest_false(lane, field):
    root, path, _, reader = lane
    reader.close()
    m = module.read_json(path / 'manifest.json')
    m[field] = 0
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='RECONSTRUCTION_CHANGED'):
        module.MartinObservedProtocol(root, path)


@pytest.mark.parametrize('field', ('same_cell_t0', 'numeric_model_kwargs_ready'))
def test_integer_zero_cannot_impersonate_rehashed_group_false(lane, field):
    root, path, _, reader = lane
    reader.close()
    groups = module.read_json(path / 'groups.json')
    target = groups[0]['base_design'] if field == 'numeric_model_kwargs_ready' else groups[0]
    target[field] = 0
    (path / 'groups.json').write_text(json.dumps(groups))
    m = module.read_json(path / 'manifest.json')
    m['files']['groups.json'] = module.sha(path / 'groups.json')
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='RECONSTRUCTION_CHANGED'):
        module.MartinObservedProtocol(root, path)


@pytest.mark.parametrize('bad', (float('nan'), float('inf'), float('-inf')))
def test_rehashed_nonfinite_source_metadata_rejected(lane, bad):
    root, path, _, reader = lane
    reader.close()
    groups = module.read_json(path / 'groups.json')
    groups[0]['endpoint_hours'] = bad
    (path / 'groups.json').write_text(json.dumps(groups))
    m = module.read_json(path / 'manifest.json')
    m['files']['groups.json'] = module.sha(path / 'groups.json')
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='NONFINITE_OR_INVALID'):
        module.MartinObservedProtocol(root, path)


def test_canonical_metadata_types_and_list_order_are_strict():
    assert not module.same_json({'flag': False}, {'flag': 0})
    assert not module.same_json({'number': 1}, {'number': 1.})
    assert not module.same_json(['wash', 'media_change'], ['media_change', 'wash'])
    assert module.same_json({'a': 1, 'b': True}, {'b': True, 'a': 1})
