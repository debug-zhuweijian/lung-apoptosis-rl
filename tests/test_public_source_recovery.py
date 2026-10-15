"""Source contracts; tiny data test implementations, not scientific effects."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lung.data import public_source_recovery as recovery


def test_soft_samples_preserve_shared_protocol_without_inventing_a_pair(tmp_path):
    p = tmp_path / 'source.soft'
    p.write_text('^SAMPLE = GSM1\n!Sample_title = A375_NC_1\n'
                 '!Sample_characteristics_ch1 = cell line: A375\n'
                 '!Sample_treatment_protocol_ch1 = All conditions: control, A, A then B.\n'
                 '^SAMPLE = GSM2\n!Sample_title = A375_A_1\n'
                 '!Sample_characteristics_ch1 = cell line: A375\n')
    samples = recovery.read_geo_samples(p)
    assert samples['GSM1']['title'] == ['A375_NC_1']
    assert samples['GSM1']['treatment_protocol_ch1'] == ['All conditions: control, A, A then B.']
    assert 'drug' not in samples['GSM1']


def test_duplicate_geo_accession_is_rejected(tmp_path):
    p = tmp_path / 'source.soft'
    p.write_text('^SAMPLE = GSM1\n!Sample_title = a\n^SAMPLE = GSM1\n!Sample_title = b\n')
    with pytest.raises(ValueError, match='Duplicate'):
        recovery.read_geo_samples(p)


def test_sample_binding_checks_identity_background_and_exact_column_set():
    records = {'GSM1': {'title': ['A375_NC_1'], 'characteristics_ch1': ['cell line: A375']}}
    bindings = [{'sample_id': 'GSM1', 'source_column': 'S1_NC', 'source_title': 'A375_NC_1', 'cell_line': 'A375'}]
    frame = pd.DataFrame({'gene': ['ENSG1'], 'S1_NC': [0.25]})
    rows = recovery.bind_geo_samples(frame, records, bindings, identifier='gene')
    assert rows.iloc[0].sample_id == 'GSM1'
    assert rows.iloc[0].role == 'unassigned'
    assert not rows.iloc[0].paired_to_function_replicate
    with pytest.raises(ValueError, match='background'):
        recovery.bind_geo_samples(frame, records, [dict(bindings[0], cell_line='A549')], identifier='gene')
    with pytest.raises(ValueError, match='column'):
        recovery.bind_geo_samples(frame.assign(extra=[3]), records, bindings, identifier='gene')
    with pytest.raises(ValueError, match='unique'):
        recovery.bind_geo_samples(frame, records, bindings * 2, identifier='gene')


def test_source_lock_rejects_changed_bytes_and_missing_files(tmp_path):
    path = tmp_path / 'input.txt'; path.write_bytes(b'actual')
    records = [{'path': 'input.txt', 'bytes': 6, 'sha256': hashlib.sha256(b'actual').hexdigest(), 'url': 'https://example.org/input'}]
    assert recovery.verify_source_lock(tmp_path, records)[0]['verified']
    path.write_bytes(b'edited')
    with pytest.raises(ValueError, match='checksum'):
        recovery.verify_source_lock(tmp_path, records)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        recovery.verify_source_lock(tmp_path, records)


def test_writer_roundtrips_values_masks_units_and_rejects_overwrite(tmp_path):
    axis = pd.DataFrame({'common_gene_index': [0, 1], 'ensembl_id': ['E1', 'E2']})
    frame = pd.DataFrame({'ensembl_id': ['E2.7'], 'raw_col': [3.25]})
    samples = pd.DataFrame([{'source_column': 'raw_col', 'sample_id': 'GSM1', 'role': 'unassigned'}])
    result = recovery.write_expression_packet(tmp_path / 'new', frame, axis, samples,
        identifier='ensembl_id', scale='bulk_estimated_counts', metadata={'source_id': 'test'})
    assert result['source_value_count'] == 1
    with np.load(tmp_path / 'new' / 'expression.npz', allow_pickle=False) as data:
        assert data['expression'][0, 1] == 3.25
        assert np.isnan(data['expression'][0, 0])
        assert data['observed_mask'].tolist() == [[False, True]]
        assert data['sample_ids'].tolist() == ['GSM1']
    loaded = recovery.load_expression_packet(tmp_path / 'new', expected_scale='bulk_estimated_counts')
    assert loaded['expression'][0, 1] == 3.25
    with pytest.raises(ValueError, match='domain'):
        recovery.load_expression_packet(tmp_path / 'new', expected_scale='bulk_raw_counts')
    with pytest.raises(FileExistsError):
        recovery.write_expression_packet(tmp_path / 'new', frame, axis, samples,
            identifier='ensembl_id', scale='bulk_estimated_counts', metadata={})


def test_reader_rejects_tampering_not_merely_same_dimensions(tmp_path):
    axis = pd.DataFrame({'common_gene_index': [0], 'ensembl_id': ['E1']})
    samples = pd.DataFrame([{'source_column': 'r', 'sample_id': 'S1'}])
    recovery.write_expression_packet(tmp_path / 'new', pd.DataFrame({'ensembl_id': ['E1'], 'r': [1]}), axis, samples,
        identifier='ensembl_id', scale='bulk_raw_counts', metadata={})
    (tmp_path / 'new' / 'samples.csv').write_text('sample_id\nOTHER\n')
    with pytest.raises(ValueError, match='checksum'):
        recovery.load_expression_packet(tmp_path / 'new', expected_scale='bulk_raw_counts')


def test_recovery_cli_imports_from_outside_repository(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts/recover_public_expression_bundle.py'
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--source-root' in result.stdout


def test_nominal_chemical_comparison_does_not_desalt_or_drop_stereo():
    compare=recovery.compare_full_structures
    assert compare('CCO','OCC')['full_graph_equal']
    assert not compare('CCO.Cl','CCO')['full_graph_equal']
    result=compare('N[C@H](C)O','N[C@@H](C)O')
    assert not result['full_graph_equal'] and result['connectivity_equal']
    with pytest.raises(ValueError,match='parse'):
        compare('not a SMILES','CCO')


def test_reader_requires_identity_file_hashes_and_declared_background(tmp_path):
    axis=pd.DataFrame({'common_gene_index':[0,1],'ensembl_id':['E1','E2']})
    samples=pd.DataFrame([{'source_column':'r','sample_id':'S1','cell_line':'A375'}])
    recovery.write_expression_packet(tmp_path/'new',pd.DataFrame({'ensembl_id':['E1'],'r':[1.]}),axis,samples,
        identifier='ensembl_id',scale='bulk_TPM',metadata={})
    with pytest.raises(ValueError,match='background'):
        recovery.load_expression_packet(tmp_path/'new',expected_scale='bulk_TPM',expected_background='A549')
    with pytest.raises(ValueError,match='Ordered'):
        recovery.load_expression_packet(tmp_path/'new',expected_scale='bulk_TPM',expected_gene_ids=['E2','E1'])
    p=tmp_path/'new'/'manifest.json'; obj=json.loads(p.read_text());del obj['files']['samples.csv'];p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='required'):
        recovery.load_expression_packet(tmp_path/'new',expected_scale='bulk_TPM')


def test_empty_sample_id_is_not_a_verified_identity(tmp_path):
    with pytest.raises(ValueError,match='identities'):
        recovery.write_expression_packet(tmp_path/'new',pd.DataFrame({'ensembl_id':['E1'],'r':[1.]}),
            pd.DataFrame({'common_gene_index':[0],'ensembl_id':['E1']}),
            pd.DataFrame([{'source_column':'r','sample_id':' '}]),identifier='ensembl_id',scale='bulk_TPM',metadata={})


def test_sciplex_paper_dose_grid_preserves_native_unit_provenance():
    frame = pd.DataFrame({'dose':[0,10,100,1000,10000]})
    result = recovery.sciplex_nominal_dose_sidecar(frame)
    assert result.dose.tolist() == [0,10,100,1000,10000]
    assert result.dose_um.tolist() == [0,.01,.1,1,10]
    assert result.source_dose_unit_explicit_in_file.isna().all()
    assert result.dose_unit_from_primary_paper.eq('nM').all()
    assert result.dose_unit_evidence.str.contains('10.1126/science.aax6234',regex=False).all()
    assert result.active_exposure_verified_h.isna().all()
    with pytest.raises(ValueError, match='dose grid'):
        recovery.sciplex_nominal_dose_sidecar(pd.DataFrame({'dose':[50]}))
