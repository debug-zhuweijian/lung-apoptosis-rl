import numpy as np
import pytest
from scripts.audit_public_erlotinib_transfer import compare_vectors,one_index


def test_level_correlation_does_not_imply_effect_agreement():
    baseline=np.arange(100,dtype=float)*10
    effect=np.tile([-1.,1.],50)
    assert compare_vectors(baseline+effect,baseline-effect)['spearman']>.99
    assert compare_vectors(effect,-effect)['spearman']==pytest.approx(-1)
    result=compare_vectors(effect,-effect)
    assert result['cross_source_copy_MSE']==4 and result['A_zero_effect_MSE']==1
    assert not result['gene_level_pvalue_computed']


def test_constant_vectors_do_not_create_nan_json_or_imputed_correlation():
    assert compare_vectors([0.,0.],[1.,2.])['spearman'] is None


@pytest.mark.parametrize('a,b', [([1,np.nan],[2,3]),([1,2],[1]),([[1,2]],[[1,2]])])
def test_invalid_observed_vectors_rejected(a,b):
    with pytest.raises(ValueError):compare_vectors(a,b)


def test_missing_or_duplicated_conditions_are_not_averaged_silently():
    for x in [[False,False],[True,True]]:
        with pytest.raises(ValueError):one_index(x)
    assert one_index([False,True])==1
