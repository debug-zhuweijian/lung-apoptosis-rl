import pandas as pd
import pytest
from scripts.recover_erlotinib_quinacrine_reference import sample_protocol,parse_assignments,functional_reference


def meta(title, genotype='mutant KRAS (G12S)'):
    return {'title':[title], 'platform_id':['GPL17692'], 'characteristics_ch1':[genotype]}


def test_combination_is_simultaneous_not_a_sequence():
    r=sample_protocol('id',meta('A549 cells, erlotinib 1 μM + quinacrine 5 μM, 48 h'))
    assert (r['erlotinib_uM'],r['quinacrine_uM'],r['exposure_hours'])==(1,5,48)
    assert r['design']=='simultaneous_combination'
    assert not r['parallel_time_matched_control'] and not r['paired_functional_replicate']


def test_genotype_conflict_retained_not_silently_corrected():
    r=sample_protocol('id',meta('A549 cells, erlotinib 1 μM, 12 h','mutant KRAS (G61H)'))
    assert r['source_genotype_conflict'] and 'G61H' in r['source_characteristics']
    assert r['paper_A549_genotype_conflicts_with_canonical_background']


def test_unknown_dose_units_rejected():
    with pytest.raises(ValueError):sample_protocol('id',meta('A549 cells, erlotinib 1 mM, 24 h'))


def test_annotation_does_not_choose_one_of_multiple_genes():
    f=pd.DataFrame([{'ID':'p','gene_assignment':'NM_1 // A // a // x // 1 /// NM_2 // B // b // y // ---','crosshyb_type':'3','category':'main'}])
    p=parse_assignments(f)
    assert p.iloc[0]['Gene ID']=='ambiguous_source_assignments'


def test_annexin_positive_is_not_all_non_live_events():
    f=functional_reference();q=f[(f.panel=='left_1uM')&(f.condition=='quinacrine')].iloc[0]
    assert q.Annexin_positive_percent==pytest.approx(41.5)
    assert q.non_double_negative_percent==pytest.approx(59.8)
    assert len(f)==8 and f.matched_RNA_sample_id.notna().sum()==2
    assert f.loc[f.condition=='quinacrine','matched_RNA_sample_id'].isna().all()
    assert not f.raw_replicates_available.any()
