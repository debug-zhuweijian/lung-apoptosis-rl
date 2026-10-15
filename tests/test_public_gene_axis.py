import numpy as np
import pandas as pd
import pytest
from scripts.recover_public_gene_axis import build_axis,save_aligned
from lung.data.external_gene_axis import align_gene_axis


def sources():
    genes=pd.DataFrame({'ensembl_id':['E3','E1','E2','E4'],'gene_symbol':['C','A','B','D'],'token_id':[3,4,5,6]})
    counts=pd.DataFrame([[2,0,8],[0,1,9]],columns=['E2','E1','outside'],index=['c1','c2'])
    meta=pd.DataFrame({'Cell_line':['A549',None]},index=['c1','c2'])
    return genes,counts,meta


def test_identity_intersection_sorted_not_position_or_expression_selected():
    g,x,m=sources();a,meta=build_axis(g,x,m.iloc[::-1])
    assert a.ensembl_id.tolist()==['E1','E2']
    assert a.tidyi_source_column.tolist()==[1,0]
    assert meta.index.tolist()==['c1','c2']
    assert meta.Cell_line.isna().sum()==1


@pytest.mark.parametrize('defect',['gene_duplicate','cell_duplicate','cell_mismatch','negative','fraction','nan'])
def test_invalid_axis_source_rejected(defect):
    g,x,m=sources()
    if defect=='gene_duplicate':g.loc[1,'ensembl_id']='E3'
    elif defect=='cell_duplicate':x.index=['c1','c1']
    elif defect=='cell_mismatch':m.index=['c1','other']
    elif defect=='negative':x.iloc[0,0]=-1
    elif defect=='fraction':x=x.astype(float);x.iloc[0,0]=.5
    else:x=x.astype(float);x.iloc[0,0]=np.nan
    with pytest.raises(ValueError):build_axis(g,x,m)


def test_TPM_alignment_preserves_scale_zero_and_missingness():
    g,x,m=sources();axis,_=build_axis(g,x,m)
    frame=pd.DataFrame({'ensembl_id':['E1'],'s':[0.]})
    values,mask,_=align_gene_axis(frame,axis,identifier='ensembl_id',samples=['s'],scale='bulk_TPM')
    assert values[0,0]==0 and np.isnan(values[0,1])
    assert mask.tolist()==[[True,False]]
    frame.loc[0,'s']=-1
    with pytest.raises(ValueError):align_gene_axis(frame,axis,identifier='ensembl_id',samples=['s'],scale='bulk_TPM')


def test_sample_metadata_binding_cannot_be_position_guessed(tmp_path):
    g,x,m=sources();axis,_=build_axis(g,x,m)
    frame=pd.DataFrame({'ensembl_id':['E1'],'s':[1.]})
    with pytest.raises(ValueError):save_aligned(tmp_path,'bad',frame,axis,['s'],'bulk_TPM',pd.DataFrame({'sample_id':['wrong']}),'sha')


def test_position_only_audit_can_expose_sorted_labels():
    from scripts.recover_public_gene_axis import audit_plate_layout
    rows=[dict(Cell_line='A549',Plate=8,Well=f'{r}{c}',cell_state='a' if c==1 else 'b') for r in ['A','B','C'] for c in [1,2]]
    result,pred=audit_plate_layout(pd.DataFrame(rows))
    assert result['accuracy']==1 and result['RNA_used'] is False
    assert result['expression_model_exploits_position_proven'] is False


def test_position_audit_rejects_duplicate_wells():
    from scripts.recover_public_gene_axis import audit_plate_layout
    f=pd.DataFrame([dict(Cell_line='A549',Plate=8,Well='A1',cell_state='a')]*2)
    with pytest.raises(ValueError,match='Duplicate'):audit_plate_layout(f)


@pytest.mark.parametrize('reverse_axis', [False, True])
def test_export_keeps_gene_ids_values_and_missing_mask_aligned(tmp_path, reverse_axis):
    axis=pd.DataFrame({'common_gene_index':[0,1,2], 'ensembl_id':['ENSG2','ENSG3','ENSG1']})
    if reverse_axis:axis=axis.iloc[::-1]
    source=pd.DataFrame({'ensembl_id':['ENSG1','ENSG2'], 'sample':[0,11]})
    save_aligned(tmp_path,'aligned',source,axis,['sample'],'bulk_raw_counts',
                 pd.DataFrame({'sample_id':['sample']}),'synthetic_fixture')
    with np.load(tmp_path/'aligned/aligned_expression.npz',allow_pickle=False) as result:
        np.testing.assert_array_equal(result['gene_ids'],['ENSG2','ENSG3','ENSG1'])
        np.testing.assert_array_equal(result['expression'],[[11,np.nan,0]])
        np.testing.assert_array_equal(result['observed_mask'],[[True,False,True]])
        np.testing.assert_array_equal(result['sample_ids'],['sample'])
