import numpy as np
import pandas as pd
import pytest
from scripts.recover_state_context_functional_reference import read_value,bind_contexts,LANDMARKS


def samples():
    return pd.DataFrame([dict(condition=c,library_label=r,sample_id=c+r,background='A549',
        conservative_split_group='rep'+r.split('-')[0]) for c in ['ctr','pb','doxo','ink'] for r in ['1','2','2-2','3']])


def test_relative_biomass_above_one_hundred_not_clipped():
    value,lo,hi=read_value(711)
    assert 100<lo<value<hi<150


def test_raw_replicates_not_invented_from_four_libraries():
    r=bind_contexts(samples())
    assert r.RNA_libraries.eq(4).all() and r.conservative_RNA_split_groups.eq(3).all()
    assert not r.RNA_function_replicate_pairing.any()
    assert r.loc[r.context.eq('pb'),'protocol_conflict'].all()
    assert r.loc[r.context.eq('pb'),'exposure_h'].isna().all()


def test_wrong_background_or_incomplete_context_rejected():
    s=samples();s.loc[0,'background']='SK-Mel-147'
    with pytest.raises(ValueError,match='Background'):bind_contexts(s)
    with pytest.raises(ValueError,match='coverage'):bind_contexts(samples().iloc[1:])


def test_reading_bounds_widen_and_are_not_sampling_error():
    a=read_value(779,1);b=read_value(779,3)
    assert a[0]==b[0] and b[1]<a[1]<a[2]<b[2]
    with pytest.raises(ValueError):read_value(np.nan)


def test_eight_means_are_two_doses_four_contexts():
    assert len(LANDMARKS)==8 and len({(d,c) for d,c,_,_ in LANDMARKS})==8
    assert all(read_value(y)[0]>0 for _,_,_,y in LANDMARKS)
