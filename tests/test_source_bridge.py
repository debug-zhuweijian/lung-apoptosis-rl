"""Contract tests for real-source recovery; these fixtures are not biological evidence."""
import importlib
import numpy as np
import pandas as pd
import pytest

from lung.data.external_gene_axis import align_gene_axis


def axis():
    return pd.DataFrame({'common_gene_index': [0, 1], 'ensembl_id': ['ENSG1', 'ENSG2'], 'gene_symbol': ['A', 'B']})


def test_duplicate_requested_sample_cannot_duplicate_one_measurement():
    frame = pd.DataFrame({'ensembl_id': ['ENSG1'], 's': [3]})
    with pytest.raises(ValueError, match='sample'):
        align_gene_axis(frame, axis(), identifier='ensembl_id', samples=['s', 's'], scale='bulk_raw_counts')


@pytest.mark.parametrize('bad_id', ['', ' ', None])
def test_unknown_target_identity_cannot_become_an_observed_gene(bad_id):
    target = axis()
    target.loc[0, 'ensembl_id'] = bad_id
    frame = pd.DataFrame({'ensembl_id': [bad_id], 's': [3]})
    with pytest.raises(ValueError, match='identity'):
        align_gene_axis(frame, target, identifier='ensembl_id', samples=['s'], scale='bulk_raw_counts')


def test_duplicate_source_sample_columns_are_rejected():
    frame = pd.DataFrame([['ENSG1', 3, 4]], columns=['ensembl_id', 's', 's'])
    with pytest.raises(ValueError, match='column'):
        align_gene_axis(frame, axis(), identifier='ensembl_id', samples=['s'], scale='bulk_raw_counts')


def test_blank_source_symbol_is_missing_not_a_gene_match():
    target = axis()
    target.loc[0, 'gene_symbol'] = ''
    frame = pd.DataFrame({'gene_symbol': [''], 's': [3]})
    values, mask, ledger = align_gene_axis(frame, target, identifier='gene_symbol', samples=['s'], scale='bulk_raw_counts')
    assert not mask.any() and np.isnan(values).all()
    assert ledger.mapping_status.tolist() == ['missing_source_identifier']


def test_public_axis_uses_token_order_and_original_source_columns():
    mod = importlib.import_module('lung.data.source_bridge')
    tahoe = pd.DataFrame({'token_id': [6, 3, 9], 'ensembl_id': ['ENSG2', 'ENSG1', 'ENSG3'], 'gene_symbol': ['B', 'A', 'C']})
    common = mod.reconstruct_public_axis(tahoe, ['ENSG3', 'ENSG2', 'ENSG4'])
    assert common.ensembl_id.tolist() == ['ENSG2', 'ENSG3']
    assert common.tahoe_token_id.tolist() == [6, 9]
    assert common.tahoe_column.tolist() == [1, 2]
    assert common.tidyi_column.tolist() == [1, 0]
    assert common.common_gene_index.tolist() == [0, 1]


def test_equal_axis_length_does_not_certify_reference_identity():
    mod = importlib.import_module('lung.data.source_bridge')
    original = axis()
    changed = axis()
    changed.ensembl_id = ['ENSG2', 'ENSG1']
    assert mod.axis_identity(original)['ordered_ensembl_sha256'] != mod.axis_identity(changed)['ordered_ensembl_sha256']
    with pytest.raises(ValueError, match='identity/order'):
        mod.verify_axis_reference(original, changed)
    assert mod.verify_axis_reference(original, original)['reference_axis_match_verified'] is True


def test_public_axis_refuses_duplicate_tokens_and_unknown_source_ids():
    mod = importlib.import_module('lung.data.source_bridge')
    tahoe = pd.DataFrame({'token_id': [3, 3], 'ensembl_id': ['ENSG1', 'ENSG2'], 'gene_symbol': ['A', 'B']})
    with pytest.raises(ValueError, match='token'):
        mod.reconstruct_public_axis(tahoe, ['ENSG1'])
    tahoe.token_id = [3, 4]
    with pytest.raises(ValueError, match='identity'):
        mod.reconstruct_public_axis(tahoe, ['ENSG1', 'ENSG1'])


def test_tidyi_readback_reorders_only_by_cell_identity(tmp_path):
    mod = importlib.import_module('lung.data.source_bridge')
    counts = tmp_path/'counts.csv'
    counts.write_text(',ENSG1,ENSG2\nc1,0,2\nc2,3,0\n')
    metadata = tmp_path/'metadata.csv'
    pd.DataFrame({'cell_id': ['c2', 'c1'], 'Cell_line': ['A549', None],
                  'cell_state': ['PI-APC-', 'PI+APC+'], 'Plate': [8,8], 'Well': ['A1','A2'],
                  'total_counts': [3,3], 'n_genes_by_counts': [1,2]}).to_csv(metadata,index=False)
    result = mod.read_tidyi_export(counts, metadata)
    np.testing.assert_array_equal(result['expression'], [[3,0], [0,2]])
    assert result['gene_ids'] == ['ENSG1','ENSG2']
    assert result['metadata'].cell_id.tolist() == ['c2','c1']
    assert result['strict_a549'].tolist() == [True,False]
    assert result['qc_discrepancies'].cell_id.tolist() == ['c1']
    counts.write_text(',ENSG1,ENSG1\nc1,0,2\nc2,3,0\n')
    with pytest.raises(ValueError, match='identity'):
        mod.read_tidyi_export(counts, metadata)


@pytest.mark.parametrize('value', ['nan','-1','1.5'])
def test_tidyi_export_rejects_invalid_counts(tmp_path, value):
    mod = importlib.import_module('lung.data.source_bridge')
    counts=tmp_path/'counts.csv';counts.write_text(',ENSG1\nc1,'+value+'\n')
    meta=tmp_path/'meta.csv'
    pd.DataFrame({'cell_id':['c1'],'Cell_line':['A549'],'cell_state':['PI-APC-'],
                  'Plate':[8],'Well':['A1'],'total_counts':[0],'n_genes_by_counts':[0]}).to_csv(meta,index=False)
    with pytest.raises(ValueError, match='count'):
        mod.read_tidyi_export(counts,meta)


def _recovery_module():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('recovery_bridge_test', Path(__file__).parents[1]/'scripts/recover_public_expression.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_recovery_does_not_certify_arbitrary_axis_as_frozen(tmp_path, monkeypatch):
    import gzip, json
    mod = _recovery_module()
    soft = tmp_path/'meta.soft.gz'
    with gzip.open(soft,'wt') as h:
        h.write('^SAMPLE = GSM1\n!Sample_title = A549_Notreatment_rep1\n'
                '!Sample_treatment_protocol_ch1 = Cells were treated with 1 μM HM97662 or left untreated for 6 days.\n')
    matrix = tmp_path/'matrix.tsv'
    pd.DataFrame({'gene_id':['ENSG1'],'gene_name':['A'],'A549_NT-1':[1.25]}).to_csv(matrix,sep='\t',index=False)
    target=tmp_path/'axis.parquet';axis().to_parquet(target,index=False)
    out=tmp_path/'out'
    monkeypatch.setattr('sys.argv',['recover','hm97662','--soft',str(soft),'--matrix',str(matrix),
                                    '--gene-axis',str(target),'--output',str(out)])
    mod.main()
    manifest=json.loads((out/'manifest.json').read_text())
    v=manifest['validation']
    assert v['common_axis_verified'] is False
    assert v['axis_projection_verified'] is True
    assert v['reference_axis_match_verified'] is False
    assert v['common_axis_identity']['gene_count'] == 2
    with np.load(out/'common_axis_expression.npz',allow_pickle=False) as saved:
        assert saved['value_scale'].item()=='bulk_estimated_counts'
        assert saved['sample_ids'].tolist()==['GSM1']
        np.testing.assert_array_equal(saved['observed_mask'],[[True,False]])


def test_reference_disagreement_rejected_before_output_publication(tmp_path, monkeypatch):
    mod=_recovery_module()
    target=tmp_path/'axis.parquet';axis().to_parquet(target,index=False)
    bad=axis();bad.ensembl_id=['ENSG2','ENSG1']
    reference=tmp_path/'reference.csv';bad.to_csv(reference,index=False)
    out=tmp_path/'not-created'
    monkeypatch.setattr('sys.argv',['recover','hm97662','--soft',str(tmp_path/'not-read.gz'),
        '--gene-axis',str(target),'--reference-gene-axis',str(reference),'--output',str(out)])
    with pytest.raises(ValueError,match='identity/order'):
        mod.main()
    assert not out.exists()


def test_tidyi_export_accepts_original_blank_index_header(tmp_path):
    mod=importlib.import_module('lung.data.source_bridge')
    counts=tmp_path/'counts.csv';counts.write_text(',ENSG1\ncell1,2\n')
    meta=tmp_path/'meta.csv';meta.write_text(',Cell_line,cell_state,Plate,Well,total_counts,n_genes_by_counts\ncell1,A549,PI-APC-,8,A1,2,1\n')
    result=mod.read_tidyi_export(counts,meta)
    assert result['metadata'].cell_id.tolist()==['cell1']
    assert result['expression'].tolist()==[[2]]
