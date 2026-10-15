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


@pytest.mark.parametrize('reverse_axis', [False, True])
def test_RNA_export_keeps_gene_ids_values_and_mask_aligned(tmp_path, monkeypatch, reverse_axis):
    from scripts import recover_inde_functional_bridge as recovery

    source=tmp_path/'source';source.mkdir()
    records=[];frame=pd.DataFrame({'geneID':['ENSG1','ENSG2']})
    for i,dose in enumerate(['N/A','N/A','500 nM','500 nM','5 microM','5 microM']):
        name=f'condition_{i}';frame[name]=[0,.125]
        records.append((f'GSM{i}',{'title':[name], 'characteristics_ch1':['cell line: A549','pimasertib dose: '+dose]},''))
    frame.to_csv(source/'fpkm_A549.csv.gz',index=False)
    monkeypatch.setattr(recovery,'read_soft',lambda path:('',records))
    axis=pd.DataFrame({'common_gene_index':[0,1,2], 'ensembl_id':['ENSG2','ENSG3','ENSG1']})
    if reverse_axis:axis=axis.iloc[::-1]
    output=tmp_path/'out';output.mkdir()
    recovery.recover_rna(source,axis,output)
    with np.load(output/'RNA_common_axis_FPKM.npz',allow_pickle=False) as result:
        np.testing.assert_array_equal(result['gene_ids'],['ENSG2','ENSG3','ENSG1'])
        np.testing.assert_array_equal(result['expression'],np.tile([.125,np.nan,0],(6,1)))
        np.testing.assert_array_equal(result['observed_mask'],np.tile([True,False,True],(6,1)))
        np.testing.assert_array_equal(result['sample_ids'],[r[0] for r in records])
