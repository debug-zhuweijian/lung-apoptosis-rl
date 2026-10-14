import numpy as np
import pandas as pd
import pytest
from scripts.audit_tidyi_within_plate_increment import training_pca,fit_probability,diagnostic


def fixture():
    rng=np.random.default_rng(41);x=rng.poisson(10,(9,12)).astype(float)
    m=pd.DataFrame([dict(cell_id=f'{r}{c}',Well=f'{r}{c}',Plate=8,Cell_line='A549',cell_state=s,
                         log1p_total_counts=5.,pct_counts_mt=10.,log1p_n_genes_by_counts=3.)
                    for r in 'ABC' for c,s in [(1,'PI-APC-'),(9,'PI-APC+'),(17,'PI+APC+')]])
    return x,m


def test_held_out_values_cannot_change_training_projection():
    rng=np.random.default_rng(0);x=rng.normal(size=(8,6));test=rng.normal(size=(2,6))
    a=training_pca(x,test,genes=4,components=2);b=training_pca(x,test*1000,genes=4,components=2)
    for k in [0,2,3,4]:np.testing.assert_array_equal(a[k],b[k])
    assert not np.allclose(a[1],b[1])


def test_classifier_probabilities_are_finite_and_normalized():
    x=np.eye(3).repeat(3,axis=0);y=np.repeat(np.arange(3),3)
    p=fit_probability(x,y,np.eye(3));assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1),1);assert np.array_equal(p.argmax(axis=1),np.arange(3))


def test_fold_separation_and_scientific_boundaries():
    x,m=fixture();r,p=diagnostic(x,m,genes=6,components=2)
    assert len(p)==9 and len(r['folds'])==3
    for f in r['folds']:assert not set(f['training_cell_ids'])&set(f['test_cell_ids'])
    assert not r['independent_culture_validation'] and not r['future_fate_prediction']
    assert not r['causal_state_effect_identified']


def test_flow_label_measurements_are_not_predictor_inputs():
    x,m=fixture();a,_=diagnostic(x,m,6,2)
    m['dead APC-A Mean']=np.arange(9)*999999;m['singlets DAPI-A Mean']=1000000
    b,_=diagnostic(x,m,6,2);assert a['metrics']==b['metrics']


def test_background_mixture_rejected():
    x,m=fixture();m.loc[0,'Cell_line']='unknown'
    with pytest.raises(ValueError,match='strictly'):diagnostic(x,m)


def test_no_zero_library_imputation():
    x,m=fixture();x[0]=0
    with pytest.raises(ValueError,match='Zero common'):diagnostic(x,m)
