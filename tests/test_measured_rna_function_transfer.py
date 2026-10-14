import numpy as np
import pytest
from scripts.audit_measured_rna_function_transfer import evaluate,response_features


def inputs():
    r=np.random.default_rng(14)
    return r.normal(size=(5,8)),r.integers(0,2,(5,12)),r.uniform(.1,.8,(5,3)),[f'd{i}' for i in range(5)]


def test_held_labels_do_not_change_their_predictions():
    x,f,y,n=inputs();a,p=evaluate(x,f,y,n,4);y[0,:]=.99;y[1:,0]=.98;b,q=evaluate(x,f,y,n,4)
    key=p.held_drug.eq('d0')&p.held_experiment.eq(1)
    np.testing.assert_array_equal(p.loc[key,'prediction'],q.loc[key,'prediction'])
    for fold in a['folds']:
        assert fold['held_drug'] not in fold['training_drugs']
        assert fold['held_experiment'] not in fold['training_experiments']


def test_missing_test_expression_is_training_mean_imputed_only_inside_predictor():
    train=np.arange(24).reshape(4,6).astype(float);test=np.full((1,6),np.nan)
    z,t,selected,missing=response_features(train,test,3)
    assert missing==3 and np.all(t==0) and np.isnan(test).all()


def test_test_values_cannot_select_training_features():
    x,f,y,n=inputs();a=response_features(x[1:],x[:1],4);b=response_features(x[1:],x[:1]*1e6,4)
    np.testing.assert_array_equal(a[2],b[2]);np.testing.assert_array_equal(a[0],b[0])


def test_constant_function_has_no_invented_improvement():
    x,f,y,n=inputs();y[:]=.3;r,p=evaluate(x,f,y,n,4)
    np.testing.assert_allclose(p.prediction,.3,atol=1e-12)
    assert not r['initial_state_increment'] and not r['apoptosis_calibration'] and not r['selection_benefit_validated']


def test_out_of_range_measured_fraction_is_not_clipped_to_pass():
    x,f,y,n=inputs();y[0,0]=1.1
    with pytest.raises(ValueError,match='Valid measured'):evaluate(x,f,y,n,4)
