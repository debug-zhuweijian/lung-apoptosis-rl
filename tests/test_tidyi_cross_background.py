import numpy as np
import pandas as pd
import pytest
from scripts.audit_tidyi_cross_background import recover_counts,score,evaluate


def test_gene_and_cell_alignment_and_unknown_exclusion():
    x=pd.DataFrame([[2,1],[4,3]],index=['b','a'],columns=['g2','g1'])
    m=pd.DataFrame({'Cell_line':['unknown','MDA_MB_231_A']},index=['a','b'])
    a=pd.DataFrame({'ensembl_id':['g1','g2']})
    out,meta,excluded=recover_counts(x,m,a)
    np.testing.assert_array_equal(out,[[1,2]])
    assert meta.cell_id.tolist()==['b'] and excluded.cell_id.tolist()==['a']


def test_missing_gene_cannot_be_zero_filled():
    with pytest.raises(ValueError,match='Common genes'):
        recover_counts(pd.DataFrame([[1]],index=['a'],columns=['g']),pd.DataFrame({'Cell_line':['MDA_MB_231_A']},index=['a']),pd.DataFrame({'ensembl_id':['missing']}))


def test_metric_known_perfect_prediction():
    r=score(np.eye(3),np.arange(3),['a','b','c'])
    assert r['accuracy']==1 and r['balanced_accuracy']==1 and r['mean_log_loss']==0


def test_duplicate_cells_across_sources_rejected():
    with pytest.raises(ValueError,match='overlap'):
        evaluate(np.ones((2,2)),pd.DataFrame({'cell_id':['same','same']}),pd.DataFrame())


def test_fixed_transfer_design_has_no_subject_overlap_or_endpoint_claim():
    rng=np.random.default_rng(12);rows=[]
    for bg,p in [('A549',8),('MDA_MB_231_A',6),('MDA_MB_231_A',14)]:
        for row in 'AB':
            for col,state in [(1,'PI-APC-'),(9,'PI-APC+'),(17,'PI+APC+')]:
                rows.append(dict(cell_id=f'{p}_{row}{col}',Well=f'{row}{col}',Cell_line=bg,Plate=p,cell_state=state))
    x=rng.poisson(10,(18,20)).astype(float);a=pd.DataFrame({'ensembl_id':[f'g{i}' for i in range(20)],'gene_symbol':['MT-X']+['g']*19})
    r,p=evaluate(x,pd.DataFrame(rows),a)
    assert len(r['folds'])==3 and len(p)==18
    for f in r['folds']:assert not set(f['training_cells'])&set(f['test_cells'])
    assert not r['future_fate_validated'] and not r['drug_endpoint_validated'] and not r['therapeutic_target_changed']
