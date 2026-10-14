"""Synthetic schema tests and pinned protocol metadata; no RNA body or model."""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lung.data import bothe_observation_inputs as data


def matrix():
    return (['ENSG00000000003', 'ENSG00000000419'], ['7105', None],
            [(0, 2**53 + 17, *range(10))] * 1 + [(None,) * 12], [(True,) * 12, (False,) * 12])


def test_exact_large_integer_zero_and_none_transpose():
    genes, ncbi, values, masks = matrix()
    native, observed = data.compile_matrix(genes, ncbi, data.SAMPLES, values, masks)
    assert native[0] == (0, None) and observed[0] == (True, False)
    assert type(native[1][0]) is int and native[1][0] == 2**53 + 17


@pytest.mark.parametrize('bad', [False, 1.0, -1, 2**63])
def test_non_exact_or_negative_counts_rejected(bad):
    genes, ncbi, values, masks = matrix()
    values[0] = (bad, *values[0][1:])
    with pytest.raises(ValueError, match='EXACT_INT64'):
        data.compile_matrix(genes, ncbi, data.SAMPLES, values, masks)


@pytest.mark.parametrize('value,mask', [(0, False), (None, True), (0, 1)])
def test_missing_observed_and_boolean_mask_disagreement(value, mask):
    genes, ncbi, values, masks = matrix()
    values[0] = (value, *values[0][1:]); masks[0] = (mask, *masks[0][1:])
    with pytest.raises(ValueError, match='MASK_DISAGREE'):
        data.compile_matrix(genes, ncbi, data.SAMPLES, values, masks)


def test_sample_order_is_identity_not_count_processing_order():
    genes, ncbi, values, masks = matrix()
    with pytest.raises(ValueError, match='SAMPLE_AXIS'):
        data.compile_matrix(genes, ncbi, tuple(reversed(data.SAMPLES)), values, masks)


def test_exact_joint_text_and_empty_missing_readback(tmp_path):
    counts, masks = tmp_path / 'counts.tsv', tmp_path / 'mask.tsv'
    counts.write_text('\t'.join(['ensembl_id', 'ncbi_gene_id', *data.SAMPLES]) + '\n'
                      + '\t'.join(['ENSG00000000003', '7105', '0', str(2**53+17), *['3']*10]) + '\n'
                      + '\t'.join(['ENSG00000000419', '', *['']*12]) + '\n', encoding='utf8')
    masks.write_text('\t'.join(['ensembl_id', *data.SAMPLES]) + '\n'
                     + '\t'.join(['ENSG00000000003', *['1']*12]) + '\n'
                     + '\t'.join(['ENSG00000000419', *['0']*12]) + '\n', encoding='utf8')
    _, _, native, observed = data.parse_joint_tables(counts, masks, [('ENSG00000000003', '7105'), ('ENSG00000000419', None)])
    assert native[1] == (2**53 + 17, None) and observed[1] == (True, False)
    counts.write_text(counts.read_text().replace('9007199254741009', '9007199254741009.0'), encoding='utf8')
    with pytest.raises(ValueError, match='INTEGER_TEXT'):
        data.parse_joint_tables(counts, masks, [('ENSG00000000003', '7105'), ('ENSG00000000419', None)])


@pytest.fixture
def protocol_metadata():
    mapping = data.read_json(data.ROOT / data.SOURCE / 'raw_sample_mapping.json')
    physical = [r for r in data.read_json(data.ROOT / data.PHYSICAL / 'records.json') if r['source_id'] == 'Bothe2021:E-MTAB-9923']
    with (data.ROOT / data.SOURCE / 'E-MTAB-9923.sdrf.txt').open(encoding='utf8', newline='') as stream:
        sdrf = list(csv.DictReader(stream, delimiter='\t'))
    return mapping, physical, sdrf


def test_actual_pinned_context_ids_four_histories_three_replicates(protocol_metadata):
    contexts = data.build_contexts(*protocol_metadata)
    assert len(contexts) == 12 and len({r['history_id'] for r in contexts}) == 4
    assert {r['replicate_index'] for r in contexts} == {1, 2, 3}
    assert contexts[0]['run_id'] == 'ERR5000295' and contexts[10]['run_id'] == 'ERR5000305'
    assert all(r['reference_role'] == 'parallel_T32_double_vehicle_control' and r['observation_hours'] == 32
               and r['same_cell_t0'] is False and r['paired_reference_target_identified'] is False
               and r['split_role'] is None and r['domain'] == data.DOMAIN for r in contexts)
    contexts[0]['reference_sample_ids'].append('fixture-mutated')
    assert len(contexts[1]['reference_sample_ids']) == 3


@pytest.mark.parametrize('field', ['role', 'endpoint', 'recovery', 'vehicle_basis'])
def test_context_role_clock_or_vehicle_basis_cannot_be_invented(protocol_metadata, field):
    mapping, physical, sdrf = json.loads(json.dumps(protocol_metadata))
    if field == 'role': physical[0]['role'] = 'final_test'
    elif field == 'endpoint': physical[0]['endpoint_hours'] = 0
    elif field == 'recovery': physical[0]['events'][3]['values']['recovery_hours'] = 0
    else: physical[0]['source_vehicle_dose']['unit'] = 'percent_v_v'
    with pytest.raises(ValueError):
        data.build_contexts(mapping, physical, sdrf)


def test_native_domain_cannot_be_relabelled_umi(tmp_path):
    joint = tmp_path / data.BASE / 'processed' / data.VERSION / 'joint-axis-v1'
    joint.mkdir(parents=True)
    data.write_json(joint / 'manifest.json', {'schema': 'bothe-actual-joint-count-export-v1', 'domain': 'single_cell_UMI'})
    with pytest.raises(ValueError, match='APPROVED_ACTUAL_PROJECT'):
        data.preflight(tmp_path)


def test_missing_production_is_pending_no_placeholder_or_directory(tmp_path):
    result = data.preflight(tmp_path)
    assert result['source_data_ready'] is False and result['sample_pack_created'] is False
    assert result['status'] == 'pending_source_bound_joint_count_export'
    assert not (tmp_path / data.OUTPUT).exists()
    with pytest.raises(ValueError, match='NO_PLACEHOLDER'):
        data.prepare(tmp_path)
    assert not (tmp_path / data.OUTPUT).exists()


def builder():
    path = data.ROOT / 'scripts/prepare_bothe_observation_inputs.py'
    spec = importlib.util.spec_from_file_location('observation_builder_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_waiter_chinese_utf8_exact_identity_and_pid_boundaries(monkeypatch):
    module = builder(); wrapper = data.ROOT / data.RECOVERY / 'finalize_joint_counts.py'
    command = '"F:/anaconda/envs/study/python.exe" -X utf8 -u "' + str(wrapper).replace('\\', '/') + '" --execute'
    launch = {'finalizer_PID': 57396, 'commandline': command, 'creation_date': '2026-10-11T17:50:00.234448+08:00'}
    observed = {'pid': 57396, 'name': 'python.exe', 'commandline': command, 'creation_date': '2026-10-11T09:50:00.234448+00:00'}
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=b'\xef\xbb\xbf'+json.dumps(observed, ensure_ascii=False).encode('utf8')))
    assert module.process_alive(launch, 'finalizer_PID', wrapper)
    observed['commandline'] += ' extra'
    assert not module.process_alive(launch, 'finalizer_PID', wrapper)
    with pytest.raises(ValueError, match='POSITIVE_WIN32'):
        module.process_alive({**launch, 'finalizer_PID': True}, 'finalizer_PID', wrapper)


@pytest.mark.parametrize('change', ['python_c', 'other_interpreter', 'extra_argument'])
def test_waiter_requires_the_executed_reviewed_script(monkeypatch, change):
    module = builder(); wrapper = data.ROOT / data.RECOVERY / 'finalize_joint_counts.py'
    command = f'"{Path(module.sys.executable).resolve().as_posix()}" -X utf8 -u "{wrapper.resolve().as_posix()}" --execute'
    if change == 'python_c': command = command.replace('-X utf8 -u', '-c "pass"')
    elif change == 'other_interpreter': command = command.replace('python.exe', 'other_python.exe')
    else: command += ' extra'
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('invalid launch must fail before CIM'))
    with pytest.raises(ValueError, match='WRAPPER_COMMAND'):
        module.process_alive({'finalizer_PID': 123, 'commandline': command}, 'finalizer_PID', wrapper)


def publication_fixture(root):
    directory = root / data.BASE / 'processed' / data.VERSION
    evidence = root / data.RECOVERY / 'count-resume-fixture'
    directory.mkdir(parents=True); evidence.mkdir(parents=True)
    manifest = {'schema': 'bothe-source-bound-counts-v1', 'evidence': 'actual_registered_RNA_native_execution',
                'domain': 'bulk_total_RNA_unique_intron_fragment_counts', 'version': data.VERSION,
                'status': 'counts_complete', 'counts_complete': True, 'runs': list(data.RUNS),
                'readyRNA': False, 'model_eligible': False, 'function_eligible': False}
    data.write_json(directory / 'manifest.json', manifest)
    marker = {'counts_complete': True, 'runs': list(data.RUNS), 'ML_PT_training': False,
              'manifest_sha256': data.sha(directory / 'manifest.json')}
    data.write_json(evidence / 'publication.json', marker)
    lifecycle = {'status': 'complete', 'runs': list(data.RUNS), 'completed_runs': list(data.RUNS),
                 'recovery_provenance': str(evidence / 'provenance.json')}
    data.write_json(directory / 'driver.json', lifecycle)
    return directory, evidence, manifest, marker, lifecycle


def test_fresh_complete_count_publication_without_rna_body(tmp_path):
    publication_fixture(tmp_path)
    assert data.closed_count_publication(tmp_path)['counts_complete'] is True
    assert not (tmp_path / data.OUTPUT).exists()


@pytest.mark.parametrize('field', ['partial_marker', 'unfinished_driver', 'false_qualification', 'hash', 'foreign_path'])
def test_count_driver_alone_or_incomplete_marker_cannot_replace_closed_publication(tmp_path, field):
    directory, evidence, manifest, marker, lifecycle = publication_fixture(tmp_path)
    if field == 'partial_marker':
        (evidence / 'publication.json').write_text('{"counts_complete":', encoding='utf8')
        assert data.closed_count_publication(tmp_path) is None
        return
    if field == 'unfinished_driver':
        lifecycle['status'] = 'running'
        (directory / 'driver.json').write_text(json.dumps(lifecycle), encoding='utf8')
        assert data.closed_count_publication(tmp_path) is None
        return
    if field == 'false_qualification':
        manifest['model_eligible'] = True
        (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
        marker['manifest_sha256'] = data.sha(directory / 'manifest.json')
        (evidence / 'publication.json').write_text(json.dumps(marker), encoding='utf8')
    elif field == 'hash':
        marker['manifest_sha256'] = '0' * 64
        (evidence / 'publication.json').write_text(json.dumps(marker), encoding='utf8')
    else:
        lifecycle['recovery_provenance'] = str(tmp_path / 'foreign-count-resume/provenance.json')
        (directory / 'driver.json').write_text(json.dumps(lifecycle), encoding='utf8')
    with pytest.raises(ValueError):
        data.closed_count_publication(tmp_path)


def test_data_truth_boundaries_never_grant_models_or_function():
    assert all(value is False for value in data.LIMITS.values())
    assert data.LIMITS['E_D_P_dependencies_resolved'] is False and data.LIMITS['Annexin_link_validated'] is False


def test_arbitrary_constructor_cannot_bless_synthetic_or_interface_counts():
    with pytest.raises(ValueError, match='SOURCE_BOUND_LOADER'):
        data.BotheObservationInputs({}, {'actual': True})
