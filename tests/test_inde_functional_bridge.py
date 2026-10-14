import numpy as np
import pandas as pd
import pytest
from scripts.recover_inde_functional_bridge import lf_from_counts,ranking_repeatability,align_gene_axis


def test_running_max_dead_not_current_green_controls_denominator():
    q,valid,peak,live=lf_from_counts([[10],[5]],[[100],[90]],[[0],[0]])
    np.testing.assert_allclose(q[:,0],[10/110,10/100]);assert valid.all()
    np.testing.assert_array_equal(peak[:,0],[10,10])


def test_overlap_is_subtracted_once_from_live():
    q,valid,_,live=lf_from_counts([[10]],[[100]],[[20]])
    assert live[0,0]==80 and q[0,0]==10/90 and valid[0,0]


def test_above_one_source_quantity_not_clipped_to_success():
    q,valid,_,_=lf_from_counts([[10]],[[1]],[[2]])
    assert q[0,0]>1 and not valid[0,0]


def test_missing_green_history_stays_flagged_after_numeric_recovery():
    q,valid,_,_=lf_from_counts([[10],[np.nan],[12]],[[100],[100],[100]],[[0],[0],[0]])
    assert np.isfinite(q[2,0]) and not valid[2,0]


def test_negative_source_counts_not_accepted():
    _,valid,_,_=lf_from_counts([[1]],[[-1]],[[-2]])
    assert not valid.any()


def test_small_or_incomplete_panel_not_pretended_to_be_258_drugs():
    f=pd.DataFrame([dict(arm='single_agent',time_h=24,role='library_compound',author_autofluorescence_exclusion=False,valid_measured_fraction=True)])
    with pytest.raises(ValueError,match='coverage'):ranking_repeatability(f)


def test_FPKM_preserved_without_count_conversion_or_zero_imputation():
    f=pd.DataFrame({'ensembl_id':['ENSG1'],'a':[.125]});g=pd.DataFrame({'common_gene_index':[0,1],'ensembl_id':['ENSG1','ENSG2']})
    x,mask,_=align_gene_axis(f,g,identifier='ensembl_id',samples=['a'],scale='bulk_FPKM')
    assert x[0,0]==.125 and mask.tolist()==[[True,False]] and np.isnan(x[0,1])


def test_negative_FPKM_is_rejected():
    f=pd.DataFrame({'ensembl_id':['ENSG1'],'a':[-.1]});g=pd.DataFrame({'common_gene_index':[0],'ensembl_id':['ENSG1']})
    with pytest.raises(ValueError,match='negative'):align_gene_axis(f,g,identifier='ensembl_id',samples=['a'],scale='bulk_FPKM')
