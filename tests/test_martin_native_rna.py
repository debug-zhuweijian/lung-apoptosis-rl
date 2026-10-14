"""Tiny observed-array tests; real full readback is recorded separately."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from lung.data import martin_native_rna as module


@pytest.fixture
def package(tmp_path, monkeypatch):
    base = tmp_path / 'registered'
    (base / 'rna').mkdir(parents=True)
    log = (np.arange(240, dtype=np.float64).reshape(4, 60) - 100.) / 7.
    counts = np.arange(240, dtype=np.float64).reshape(4, 60) / 3.
    for domain, values in (('log2TPM', log), ('estimated_counts', counts)):
        np.save(base / ('rna/' + domain + '.npy'), values, allow_pickle=False)
    observed = np.array([True, False, True, False, False, True])
    rows = np.array([0, 2, 3], dtype=np.int64)
    request = {'request_id': 'LOBO:fit', 'fold_id': 'LOBO:a', 'role': 'fit',
               'physical_reference_ids': ['G0', 'G1', 'G2'], 'target_sample_ids': ['G3', 'G4', 'G5'],
               'reference_column_indices': [0, 1, 2], 'target_column_indices': [3, 4, 5],
               'actual_protocol': {'exposure_duration_h': 504., 'washout_h': 72., 'observation_time_h': 576.},
               'function_72h_four_drug_target': False}
    source = {'base': base, 'bindings': {'registered_matrix': 'test_only_source_binding'},
              'matrices': {'log2TPM': log, 'estimated_counts': counts},
              'observed': observed, 'source_rows': rows,
              'axes': {'source_symbols': ['a', 'b', 'c', 'd'],
                       'joint_ensembl_ids': ['e' + str(i) for i in range(6)],
                       'units': {'log2TPM': 'log2(TPM+0.01)', 'estimated_counts': 'lengthScaledTPM counts'}},
              'samples': [{'gsm': 'G' + str(i), 'column_index': i} for i in range(60)],
              'lobo': [request, {**request, 'request_id': 'LOBO:final', 'role': 'final_test'}],
              'cycle': [{**request, 'request_id': 'cycle:fit', 'fold_id': 'cycle:a'},
                        {**request, 'request_id': 'cycle:selection', 'fold_id': 'cycle:a', 'role': 'selection'},
                        {**request, 'request_id': 'cycle:final', 'fold_id': 'cycle:a', 'role': 'final_test'}]}
    monkeypatch.setattr(module, '_source', lambda root: source)
    output = tmp_path / 'package'
    reader = module.prepare(tmp_path, output)
    return tmp_path, output, source, reader


@pytest.mark.parametrize('domain', module.DOMAINS)
@pytest.mark.parametrize('coordinate', ('source', 'joint'))
def test_observed_values_exact_unrounded_and_masked(package, domain, coordinate):
    _, _, source, reader = package
    result = reader.supervision('cycle:fit', system='cycle', role='fit', domain=domain, coordinate=coordinate)
    for target, columns in ((False, [0, 1, 2]), (True, [3, 4, 5])):
        part = result['target' if target else 'reference']
        raw = source['matrices'][domain][:, columns].T
        if coordinate == 'source':
            np.testing.assert_array_equal(part['values'], raw)
            assert part['observed_mask'].all()
        else:
            np.testing.assert_array_equal(part['values'][:, source['observed']], raw[:, source['source_rows']])
            assert np.isnan(part['values'][:, ~source['observed']]).all()
            assert not part['observed_mask'][:, ~source['observed']].any()
        assert part['values'].dtype == np.dtype('float64')
        assert not part['values'].flags.writeable and not part['observed_mask'].flags.writeable
        assert part['functional_Y'] is None and part['same_cell_t0'] is False
        assert part['normalization'] == 'none_source_values_preserved'
        assert part['function_72h_four_drug_target'] is False
        assert part['target_expression_used'] == target
    if domain == 'log2TPM':
        assert np.nanmin(result['target']['values']) < 0
    else:
        assert any(x != np.floor(x) for x in result['target']['values'][np.isfinite(result['target']['values'])])


def test_reference_only_never_selects_target(package, monkeypatch):
    _, _, _, reader = package
    original = reader._read
    def require_reference(*args, **kwargs):
        assert kwargs['target'] is False
        return original(*args, **kwargs)
    monkeypatch.setattr(reader, '_read', require_reference)
    reader.reference('cycle:fit', system='cycle', role='fit')


@pytest.mark.parametrize('system,request_id,role', [('lobo', 'cycle:fit', 'fit'),
                                              ('cycle', 'LOBO:fit', 'fit'),
                                              ('cycle', 'cycle:final', 'fit'),
                                              ('cycle', 'cycle:selection', 'fit')])
def test_role_and_system_cannot_be_mixed(package, system, request_id, role):
    _, _, _, reader = package
    with pytest.raises(PermissionError, match='REQUEST_ROLE_OR_SYSTEM'):
        reader.reference(request_id, system=system, role=role)


def test_final_requires_explicit_permission_and_request_copy(package):
    root, path, _, _ = package
    with pytest.raises(PermissionError, match='FINAL_OR_UNKNOWN'):
        module.MartinNativeRNA(root, path, allowed_roles=('fit', 'final_test'))
    reader = module.MartinNativeRNA(root, path)
    with pytest.raises(PermissionError, match='ROLE_OR_ROLE_SYSTEM'):
        reader.reference('cycle:final', system='cycle', role='final_test')
    requests = reader.requests(system='cycle', fold_id='cycle:a', role='fit')
    requests[0]['actual_protocol']['washout_h'] = 0
    assert reader.requests(system='cycle', fold_id='cycle:a', role='fit')[0]['actual_protocol']['washout_h'] == 72


@pytest.mark.parametrize('bad', (np.ones((4, 60), dtype=bool), np.ones((4, 60), dtype=complex),
                                 np.ones((4, 60), dtype=np.float32)))
def test_illegal_source_dtype_rejected_without_coercion(bad):
    with pytest.raises(ValueError, match='FLOAT64_OBSERVATIONS'):
        module.validate_values(bad, 'log2TPM', (4, 60))
    with pytest.raises(ValueError, match='ORDERED_MAPPING'):
        module.align_observed(bad, np.array([0]), np.array([True, False]))


@pytest.mark.parametrize('bad', (float('nan'), float('inf'), -1.))
def test_illegal_estimated_counts_rejected(bad):
    values = np.ones((4, 60), dtype=np.float64)
    values[0, 0] = bad
    with pytest.raises(ValueError, match='FLOAT64_OBSERVATIONS'):
        module.validate_values(values, 'estimated_counts', (4, 60))


def test_rehashed_fake_matrix_still_rejected_against_registered_source(package):
    root, path, _, reader = package
    reader.close()
    changed = np.load(path / 'source_log2TPM.npy', allow_pickle=False)
    changed[0, 0] += 1
    np.save(path / 'source_log2TPM.npy', changed, allow_pickle=False)
    m = module.read_json(path / 'manifest.json')
    m['files']['source_log2TPM.npy'] = module.sha(path / 'source_log2TPM.npy')
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='EXACT_SOURCE_OBSERVATIONS'):
        module.MartinNativeRNA(root, path)


def test_rehashed_fake_role_metadata_still_rejected(package):
    root, path, _, _ = package
    requests = module.read_json(path / 'cycle_requests.json')
    requests[0]['role'] = 'final_test'
    (path / 'cycle_requests.json').write_text(json.dumps(requests))
    m = module.read_json(path / 'manifest.json')
    m['files']['cycle_requests.json'] = module.sha(path / 'cycle_requests.json')
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='SOURCE_RECONSTRUCTION'):
        module.MartinNativeRNA(root, path)


def test_zero_filled_missing_gene_rejected_even_when_rehashed(package):
    root, path, source, reader = package
    reader.close()
    changed = np.load(path / 'joint_log2TPM.npy', allow_pickle=False)
    changed[:, ~source['observed']] = 0
    np.save(path / 'joint_log2TPM.npy', changed, allow_pickle=False)
    m = module.read_json(path / 'manifest.json')
    m['files']['joint_log2TPM.npy'] = module.sha(path / 'joint_log2TPM.npy')
    (path / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='EXACT_SOURCE_OBSERVATIONS'):
        module.MartinNativeRNA(root, path)


def test_existing_package_never_overwritten(package):
    root, path, _, _ = package
    with pytest.raises(FileExistsError, match='NEW_RNA_INPUT_VERSION'):
        module.prepare(root, path)


def test_public_coordinate_metadata_is_copy_and_cannot_relabel_values(package):
    _, _, _, reader = package
    view = reader.axes
    view['units']['log2TPM'] = 'library10000'
    view['source_symbols'][0] = 'fake'
    manifest = reader.manifest
    manifest['model_or_function_qualified'] = True
    result = reader.reference('cycle:fit', system='cycle', role='fit')
    assert result['units'] == 'log2(TPM+0.01)' and result['gene_ids'][0] == 'a'
    assert reader.manifest['model_or_function_qualified'] is False


def test_rehashed_cycle_roles_in_clone_rejected_before_data_read(tmp_path):
    actual_root = Path(module.__file__).resolve().parents[3]
    clone = tmp_path / 'clone'
    for relative in (module.NATIVE + '/manifest.json', module.CYCLE + '/manifest.json', module.CYCLE + '/plan.json'):
        destination = clone / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual_root / relative, destination)
    plan_path, manifest_path = clone / module.CYCLE / 'plan.json', clone / module.CYCLE / 'manifest.json'
    plan = module.read_json(plan_path)
    roles = plan['folds'][0]['background_roles']
    roles['selection'], roles['final_test'] = roles['final_test'], roles['selection']
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    manifest = module.read_json(manifest_path)
    manifest['plan_sha256'] = module.sha(plan_path)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='FIXED_REGISTERED_SOURCE_CHANGED'):
        module.prepare(clone, tmp_path / 'forged-package')
    assert not (tmp_path / 'forged-package').exists()


@pytest.mark.parametrize('relative', (module.BASE + '/rna/rna_samples.csv',
                                    module.BASE + '/rna/log2TPM.npy',
                                    '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/genes.parquet',
                                    'scripts/materialize_martin2022_rna.py'))
def test_fixed_sample_clock_array_axis_generator_anchor_rejects_changed_bytes(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'caller_replacement_with_self_computed_hash')
    assert module.sha(path) != module.REGISTERED_SOURCE_HASHES[relative]
    with pytest.raises(ValueError, match='FIXED_REGISTERED_SOURCE_CHANGED'):
        module._check_anchor(tmp_path, relative, module.REGISTERED_SOURCE_HASHES[relative])


def test_identical_registered_content_copy_is_permitted(tmp_path):
    actual_root = Path(module.__file__).resolve().parents[3]
    relative = module.CYCLE + '/plan.json'
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(actual_root / relative, path)
    module._check_anchor(tmp_path, relative, module.REGISTERED_SOURCE_HASHES[relative])


def test_anchor_registry_cannot_be_mutated_by_caller():
    with pytest.raises(TypeError):
        module.REGISTERED_SOURCE_HASHES[module.CYCLE + '/plan.json'] = '0' * 64
