import numpy as np
import pandas as pd
import pytest
from scripts.recover_nci_a549_factorial import probe_gene_map,sample_protocol
from lung.data.external_gene_axis import align_gene_axis


def test_adjacent_multiple_ensembl_ids_remain_ambiguous():
    a=pd.DataFrame({'ID':['p1','p2','p3'],'Gene symbol':['A','B','C'],'Gene ID':['1','2','1///2']})
    g=pd.DataFrame({'GeneID':['1','2'],'dbXrefs':['HGNC:1|Ensembl:ENSG00001|Ensembl:ENSG00002|X:3','Ensembl:ENSG00003']})
    r=probe_gene_map(a,g)
    assert r.ensembl_id.isna().tolist()==[True,False,True]
    assert r.ensembl_id.iloc[1]=='ENSG00003'


def test_duplicate_probe_annotation_rejected():
    a=pd.DataFrame({'ID':['p','p'],'Gene symbol':['A','B'],'Gene ID':['1','2']})
    g=pd.DataFrame({'GeneID':['1'],'dbXrefs':['Ensembl:ENSG00001']})
    with pytest.raises(ValueError,match='Duplicate'):probe_gene_map(a,g)


def test_rma_scale_preserves_negative_log_intensity_not_counts():
    a=pd.DataFrame({'common_gene_index':[0,1],'ensembl_id':['ENSG1','ENSG2'],'gene_symbol':['A','B']})
    x=pd.DataFrame({'ensembl_id':['ENSG1'],'s':[-.25]})
    v,m,_=align_gene_axis(x,a,identifier='ensembl_id',samples=['s'],scale='microarray_log2_RMA')
    assert v[0,0]==-.25 and np.isnan(v[0,1]) and m.tolist()==[[True,False]]


def test_source_typo_preserved_and_explicit_alias_recorded():
    m={'platform_id':['GPL571'],'source_name_ch1':['human tumor cell line A549']}
    r=sample_protocol('A549_gemcitibine_1000nM_24h',m)
    assert r['source_drug_label']=='gemcitibine' and r['drug_name']=='gemcitabine' and r['dose_uM']==1
    assert not r['chemical_structure_verified']


@pytest.mark.parametrize('title',['H460_cisplatin_3000nM_24h','A549_cisplatin_3uM_24h','A549_cisplatin_3000nM_48h'])
def test_unreviewed_protocol_rejected(title):
    with pytest.raises(ValueError):sample_protocol(title,{'platform_id':['GPL571'],'source_name_ch1':['human tumor cell line A549']})


def grid():
    rows=[dict(drug_name='x',dose_nM=d,exposure_h=h,sample_id=f's{i}') for i,(d,h) in enumerate((d,h) for d in [0,100,1000] for h in [2,6,24])]
    return pd.DataFrame(rows)


@pytest.mark.parametrize('fold',['dose','time','protocol'])
def test_held_axes_do_not_select_genes(fold):
    from scripts.recover_nci_a549_factorial import axis_diagnostics
    s=grid();v=np.random.default_rng(5).normal(7,1,(9,30))
    a=axis_diagnostics(v,s,5);key='x:'+fold
    v[s.sample_id.isin(a['folds'][key]['held_sample_ids']),0]+=10000
    b=axis_diagnostics(v,s,5)
    assert a['folds'][key]['observed_gene_feature_indices']==b['folds'][key]['observed_gene_feature_indices']


def test_incomplete_grid_is_explicit_not_filled():
    from scripts.recover_nci_a549_factorial import axis_diagnostics
    s=grid().iloc[:-1]
    d=axis_diagnostics(np.ones((8,10)),s)
    assert len(d['excluded_incomplete_drugs'])==1 and d['results']==[]


def test_static_rma_has_no_artificial_response():
    from scripts.recover_nci_a549_factorial import axis_diagnostics
    d=axis_diagnostics(np.tile(np.arange(20),(9,1)),grid())
    assert all(r['delta_log2_RMA_MSE']==0 for r in d['results'])
