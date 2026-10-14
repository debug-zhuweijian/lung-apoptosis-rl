"""Native metadata/role/unit boundaries; no model or functional labels."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.lung.data.native_response_units import NativeResponseUnits, build, sha

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'experiments/20261011-data-alignment-round17/native-response-units-v1'

@pytest.fixture(scope='module')
def native():
    return NativeResponseUnits(ROOT, ASSET, allowed_roles=('fit', 'final_test'), allow_test=True)

def test_counts(native):
    assert len(native._requests) == 3240
    assert len(native.requests()) == 1498
    assert len(native.requests('final_test')) == 367

@pytest.mark.parametrize('background', ['PC9', 'H1975', 'HCC827', 'HCC2935'])
def test_bulk_fold(native, background):
    fold = 'leave_background_out:' + background
    fit, final = native.requests('fit', fold_id=fold), native.requests('final_test', fold_id=fold)
    assert len(fit) == 12 and len(final) == 4
    assert all(r['background_id'] != background for r in fit)
    assert all(r['background_id'] == background for r in final)
    assert set(x for r in fit for x in r['target_sample_ids']).isdisjoint(x for r in final for x in r['target_sample_ids'])

def test_clocks_and_identity(native):
    single = native.requests()
    assert {r['actual_protocol']['duration_h'] for r in single} == {24., 72.}
    assert all(r['unique_functional_Y'] is None for r in native._requests)
    assert all(r['reference_semantics'] != 'same_cell_t0' for r in native._requests)
    long = [r for r in native._requests if r['measurement_level'] == 'bulk' and r['actual_protocol']['long_short_history'] == 'long_washout']
    assert {r['background_id']: r['actual_protocol']['washout_h'] for r in long} == {'PC9':168., 'H1975':72., 'HCC827':96., 'HCC2935':240.}

def test_role_gate(native):
    with pytest.raises(PermissionError): native.requests('selection')
    req = native.requests('final_test')[0]
    with pytest.raises(PermissionError): native.reference(req['request_id'], role='fit', expected_gene_ids=[])

def test_explicit_test_gate():
    with pytest.raises(PermissionError): NativeResponseUnits(ROOT, ASSET, allowed_roles=('final_test',))

@pytest.mark.parametrize('method', ['read', 'reference'])
def test_gene_axis_guard(native, method):
    req = native.requests()[0]
    genes = native.axes[req['gene_axis_id']]
    with pytest.raises(ValueError, match='ORDERED_GENE'): getattr(native, method)(req['request_id'], role='fit', expected_gene_ids=list(reversed(genes)))

def test_request_copy(native):
    req = native.requests()[0]
    req['actual_protocol']['duration_h'] = -999
    assert native.requests()[0]['actual_protocol']['duration_h'] != -999

def test_missing_mapping(native):
    mapping = native.bulk_common_axis_mask()
    missing = mapping.loc[~mapping.source_observed]
    assert len(missing) == 16294
    assert missing.source_row_index.isna().all()

def test_asset_tamper(native, tmp_path):
    import shutil
    shutil.copytree(ASSET, tmp_path / 'asset')
    (tmp_path / 'asset/requests.json').write_text('[]')
    with pytest.raises(ValueError, match='ASSET_CHANGED'): NativeResponseUnits(ROOT, tmp_path / 'asset')

def test_builder_never_deserializes_arrays(monkeypatch, tmp_path):
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata builder loaded expression'))
    result = build(ROOT, tmp_path / 'new')
    assert result['requests'] == 3240 and not result['function_labels_read']

def test_source_hash_guard(tmp_path):
    import shutil
    shutil.copytree(ASSET, tmp_path / 'asset')
    manifest_path = tmp_path / 'asset/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['source_bindings']['08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/genes.parquet'] = '0'*64
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='SOURCE_BINDING_CHANGED'): NativeResponseUnits(ROOT, tmp_path / 'asset')

def test_full_reference_weights_and_role_only_metadata(native, monkeypatch):
    from scipy import sparse
    from src.lung.data import reader
    captures = []
    class MockRows:
        def read(self, rows):
            captures.append(rows.copy())
            return sparse.csr_matrix((len(rows), 32461))
        def close(self):
            pass
    monkeypatch.setattr(reader, 'training_rows', lambda *a, **k: MockRows())
    req = native.requests()[0]
    result = native.reference(req['request_id'], role='fit', expected_gene_ids=native.axes[req['gene_axis_id']], max_rows=None)
    assert result['selection'] == 'full_reference'
    assert np.isclose(sum(result['reference_weights']), 1.)
    assert len(result['reference_rows']) == len(result['reference_ids'])
    assert set(captures[0].condition_id) == set(req['physical_reference_ids'])
    assert captures[0].role.eq('fit').all()
    assert req['condition_id'] not in set(captures[0].condition_id)
    for cid, part in captures[0].groupby('condition_id'):
        indices = [i for i,r in enumerate(result['reference_rows']) if r['condition_id']==cid]
        assert np.isclose(sum(result['reference_weights'][i] for i in indices), 1/len(req['physical_reference_ids']))

def test_bulk_reference_never_indexes_target(native, monkeypatch):
    req = native.requests('fit', fold_id='leave_background_out:H1975')[0]
    selected = []
    class Matrix:
        shape = (19712,60)
        def __getitem__(self, index):
            selected.extend(index[1])
            return np.zeros((19712,len(index[1])))
    monkeypatch.setattr(np,'load',lambda *a,**k: Matrix())
    result = native.reference(req['request_id'],role='fit',expected_gene_ids=native.axes[req['gene_axis_id']],max_rows=None)
    assert selected == req['reference_column_indices']
    assert set(selected).isdisjoint(req['target_column_indices'])
    assert np.isclose(result['weights_sum'],1.)
