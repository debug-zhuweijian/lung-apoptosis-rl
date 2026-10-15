import pytest
from scripts.recover_a549_state_context import protocol,bind_samples


def blocks():
    result=[];cols=[]
    for c in ['ctr','pb','doxo','ink']:
        for label in ['1','2','2-2','3']:
            sid=c+label;col='col_'+sid;cols.append(col)
            result.append((sid,dict(title=[f'A549 {c} {label}'],description=[col],characteristics_ch1=['cell line: A549','replicate: '+label.split('-')[0],'treatment: '+c]),''))
    return result,cols


def test_palbo_conflict_never_silently_becomes_single_duration():
    r=protocol('pb');assert r['protocol_conflict'] and r['exposure_h'] is None
    assert r['paper_specific_exposure_h']==168 and r['GEO_generic_exposure_h']==72


def test_pulse_clock_and_collection_clock_are_distinct():
    r=protocol('doxo');assert (r['dose_uM'],r['exposure_h'],r['recovery_h'],r['collection_h'])==(.1,72,96,168)
    assert protocol('ink')['exposure_h']==168


def test_repeated_replicate_label_is_grouped_without_merging_libraries():
    r=bind_samples(*blocks());assert len(r)==16 and r.conservative_split_group.nunique()==3
    assert r.loc[r.library_label.isin(['2','2-2']),'conservative_split_group'].nunique()==1
    assert not r.four_independent_biological_replicates_verified.any()


def test_missing_explicit_sample_column_binding_rejected():
    b,c=blocks()
    with pytest.raises(ValueError,match='column'):bind_samples(b,c[1:])


def test_unknown_protocol_not_guessed():
    with pytest.raises(ValueError,match='Unknown'):protocol('new')


def test_source_background_disagreement_rejected():
    b,c=blocks();b[0][1]['characteristics_ch1'][0]='cell line: SK-MEL-147'
    with pytest.raises(ValueError,match='background'):bind_samples(b,c)


def test_suffix_collision_cannot_silently_merge_gene_rows():
    import pandas as pd
    from scripts.recover_a549_state_context import align_gene_axis
    f=pd.DataFrame({'ensembl_id':['ENSG1','ENSG1.1'],'sample':[3,5]})
    a=pd.DataFrame({'common_gene_index':[0],'ensembl_id':['ENSG1']})
    x,mask,ledger=align_gene_axis(f,a,identifier='ensembl_id',samples=['sample'],scale='bulk_raw_counts')
    assert not mask.any() and set(ledger.mapping_status)=={'duplicate_source_identifier'}


@pytest.mark.parametrize('reverse_axis', [False, True])
def test_context_export_keeps_gene_ids_values_and_mask_aligned(tmp_path, monkeypatch, reverse_axis):
    import sys
    import numpy as np
    import pandas as pd
    from scripts import recover_a549_state_context as recovery

    records,columns=blocks()
    source=tmp_path/'source';source.mkdir()
    for name in ['counts.xlsx','GSE246690.soft.gz','paper.xml']:
        (source/name).write_bytes(b'synthetic source fixture')
    frame=pd.DataFrame({'ensembl_id':['ENSG1','ENSG2'], **{c:[0,11] for c in columns}})
    monkeypatch.setattr(recovery.pd,'read_excel',lambda *args,**kwargs:frame.copy())
    monkeypatch.setattr(recovery,'read_soft',lambda path:('',records))
    axis=pd.DataFrame({'common_gene_index':[0,1,2], 'ensembl_id':['ENSG2','ENSG3','ENSG1']})
    if reverse_axis:axis=axis.iloc[::-1]
    axis_path=tmp_path/'axis.csv';axis.to_csv(axis_path,index=False)
    output=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['recover','--source',str(source),'--gene-axis',str(axis_path),'--output',str(output)])
    recovery.main()
    with np.load(output/'common_axis_counts.npz',allow_pickle=False) as result:
        np.testing.assert_array_equal(result['gene_ids'],['ENSG2','ENSG3','ENSG1'])
        np.testing.assert_array_equal(result['expression'],np.tile([11,np.nan,0],(16,1)))
        np.testing.assert_array_equal(result['observed_mask'],np.tile([True,False,True],(16,1)))
        np.testing.assert_array_equal(result['sample_ids'],[r[0] for r in records])
