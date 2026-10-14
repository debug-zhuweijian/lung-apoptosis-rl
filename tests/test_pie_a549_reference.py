import pandas as pd
import numpy as np
import pytest
from scripts.recover_pie_a549_functional_reference import CachedRanges,align_means


def test_unfetched_bytes_never_become_zero_filled_data():
    r=CachedRanges(100,[(5,b'abc'),(90,b'tail123456')])
    r.seek(5);assert r.read(3)==b'abc'
    with pytest.raises(ValueError,match='Unfetched'):r.read(1)
    with pytest.raises(ValueError,match='Whole'):r.read()
    r.seek(-10,2);assert r.read(20)==b'tail123456';assert r.read(1)==b''


def fixture():
    axis=pd.DataFrame(dict(common_gene_index=[0,1,2],ensembl_id=['E1','E2','E3']))
    meta=pd.DataFrame(dict(gene_symbol=['A','B','C'],ensembl_id=['E1','E2','E3']))
    f=pd.DataFrame(dict(target=['drug']*2,feature=['A','B'],target_mean=[0.,1.5],ref_mean=[.2,2.],target_ncells=[10,10],ref_ncells=[30,30]))
    return f,axis,meta


def test_native_group_mean_zero_and_missing_remain_distinct():
    x,c,m,l,s=align_means(*fixture())
    assert x[0,0]==0 and x[0,1]==1.5 and np.isnan(x[0,2])
    assert m.tolist()==[[True,True,False]]
    assert not s.raw_individual_cells_downloaded.any() and not s.plate_matched_reference.any()


def test_ambiguous_full_reference_symbol_is_not_chosen_on_restricted_axis():
    f,a,m=fixture();m=pd.concat([m,pd.DataFrame(dict(gene_symbol=['A'],ensembl_id=['outside_axis']))])
    x,c,mask,l,s=align_means(f,a,m);assert np.isnan(x[0,0]) and not mask[0,0]


@pytest.mark.parametrize('column,value',[('target_mean',-1),('ref_mean',float('nan')),('target_ncells',0),('ref_ncells',0)])
def test_invalid_abundance_or_population_rejected(column,value):
    f,a,m=fixture();f.loc[0,column]=value
    with pytest.raises(ValueError):align_means(f,a,m)


def test_duplicated_gene_is_not_summed_into_a_new_measurement():
    f,a,m=fixture();f.loc[1,'feature']='A'
    with pytest.raises(ValueError,match='Duplicate tested'):align_means(f,a,m)
