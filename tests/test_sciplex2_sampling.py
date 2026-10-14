import importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('sampling',Path(__file__).resolve().parents[1]/'scripts/audit_sciplex2_sampling.py')
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)


def test_selected_gene_order_and_excluded_cells(tmp_path):
    f=tmp_path/'x.tsv';f.write_text('1\t1\t2\n3\t1\t7\n2\t2\t8\n3\t3\t4\n')
    panel=MOD.extract_panel(f,np.array([1,-1,0]),np.array([2,0]),3)
    np.testing.assert_array_equal(panel,[[4,0],[7,2]])
    with pytest.raises(ValueError,match='panel'):
        MOD.extract_panel(f,np.array([1,-1,0]),np.array([0,0]),3)


def test_full_library_denominator_and_identical_cells():
    # Same proportional expression with unequal library sizes: zero sampling variation.
    p=np.array([[1,2],[2,4],[3,6]])
    result=MOD.resample_noise(p,np.array([10,20,30]),9,50,np.random.default_rng(1))
    assert result['resampling_mse_mean']<1e-20
    with pytest.raises(ValueError,match='inconsistent'):
        MOD.resample_noise(p,np.array([1,1,1]),9,50,np.random.default_rng(1))


def test_reproducible_conditional_variation_not_biological_ci():
    p=np.array([[0,8],[7,0],[2,3]])
    a=MOD.resample_noise(p,np.array([10,10,10]),3,100,np.random.default_rng(3))
    b=MOD.resample_noise(p,np.array([10,10,10]),3,100,np.random.default_rng(3))
    assert a==b and a['resampling_mse_mean']>0
