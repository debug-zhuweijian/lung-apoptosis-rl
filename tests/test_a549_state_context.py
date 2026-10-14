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
