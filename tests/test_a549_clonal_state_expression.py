import datetime
import numpy as np
import pandas as pd
import pytest
from scripts.recover_a549_clonal_state_expression import combine,sample_metadata,SHEETS


def frames():
    result={}
    for s,cols in SHEETS.items():
        f=pd.DataFrame({'ensembl_gene_id':['ENSG00000000001','ENSG00000000002'],
                        'hgnc_symbol':['A',datetime.datetime(2001,9,1)]})
        for c in cols:f[c]=[1,2]
        result[s]=f.iloc[::-1].reset_index(drop=True) if s.endswith('mero') else f
    return result


def test_join_ids_not_differential_expression_row_order():
    raw,damage=combine(frames())
    assert raw.shape==(2,13) and len(damage)==3
    np.testing.assert_array_equal(raw.Sample_P1,[1,2])
    assert not damage.guessed_symbol_repair.any()


def test_repeated_parental_counts_cannot_be_conflicting():
    x=frames();x['parental_vs_mero'].loc[0,'Sample_P1']=99
    with pytest.raises(ValueError,match='Repeated'):combine(x)


def test_normalized_or_duplicate_gene_values_rejected():
    x=frames();x['parental_vs_holo']['Sample_1_1']=x['parental_vs_holo']['Sample_1_1'].astype(float);x['parental_vs_holo'].loc[0,'Sample_1_1']=.5
    with pytest.raises(ValueError,match='integer'):combine(x)
    x=frames();x['parental_vs_holo'].loc[0,'ensembl_gene_id']='ENSG00000000002'
    with pytest.raises(ValueError,match='duplicate'):combine(x)


def test_clone_one_point_one_is_not_one_point_ten():
    cols=list(dict.fromkeys(c for v in SHEETS.values() for c in v));s=sample_metadata(cols)
    assert len(s)==12 and s.clone_ID_also_listed_in_functional_panel.sum()==7
    assert not s.paired_RNA_functional_replicate.any()
    assert not s.loc[s.source_clone_id.isin(['1.2','3.7']),'clone_ID_also_listed_in_functional_panel'].any()
    assert s.explicit_RNA_collection_h.isna().all()
    assert '1.10' not in set(s.source_clone_id.dropna())


def test_library_count_totals_validate_all_source_reads():
    from scripts.recover_a549_clonal_state_expression import validate_library_totals
    raw,_=combine(frames());cols=[c for c in raw if c.startswith('Sample_')]
    q=pd.DataFrame({'sample':[c.removeprefix('Sample_') for c in cols], '# reads mapping to a gene':[3]*12})
    assert validate_library_totals(raw,q).difference.eq(0).all()
    q.loc[0,'# reads mapping to a gene']=4
    with pytest.raises(ValueError,match='reported'):validate_library_totals(raw,q)
