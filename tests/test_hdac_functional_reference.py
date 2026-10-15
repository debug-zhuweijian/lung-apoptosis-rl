import numpy as np
import pandas as pd
import pytest
from scripts.recover_hdac_functional_reference import extract_viability,extract_crossmodal,DRUGS,DOSES


def frame():
    return pd.DataFrame([dict(Line='A549',Drug=d,Concentration=c,**{'Ratio 1':1.1,'Ratio 2':.9,'Ratio 3':1.,'Average':1.,'SD':.1,'SE':.1}) for d in sorted(DRUGS) for c in sorted(DOSES)])


def test_ratios_not_apoptosis_and_anchors_not_repeats():
    a=extract_viability(frame())
    assert len(a)==147 and a.control_normalization_anchor.sum()==21
    assert (a.value>1).sum()==49
    assert not a.annexin_label.any() and a.apoptosis_fraction.isna().all()
    assert a.cross_condition_repeat_identity.isna().all()
    assert a.source_reported_SE.eq(.1).all()


@pytest.mark.parametrize('change',['missing','duplicate','dose','negative'])
def test_invalid_source_domain_fails(change):
    f=frame()
    if change=='missing':f=f.iloc[:-1]
    if change=='duplicate':f.iloc[-1]=f.iloc[0]
    if change=='dose':f.loc[0,'Concentration']=100
    if change=='negative':f.loc[0,'Ratio 1']=-.1
    with pytest.raises(ValueError):extract_viability(f)


def test_no_other_background_as_A549():
    f=frame();f['Line']='PC9'
    with pytest.raises(ValueError):extract_viability(f)


def test_effects_are_statistics_with_protein_identity():
    f=pd.DataFrame([{'Gene':str(i),'Drug':d,'Uniprot':str(i),'Line':'A549','Protein log2FC':i/1000,'RNA normalized_effect':i/2000} for d in ['Abexinostat','Belinostat','Panobinostat','TSA'] for i in range(5560)])
    output,c=extract_crossmodal(f)
    assert output.equals(f) and all(v==pytest.approx(1) for v in c.values())
    f.loc[1,'Uniprot']=f.loc[0,'Uniprot']
    with pytest.raises(ValueError):extract_crossmodal(f)
