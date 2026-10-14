import numpy as np
import pandas as pd
import pytest
from scripts.recover_mln_dose_time import bind,diagnostics


def records():
    r=[];columns=[]
    for i,(d,h,k) in enumerate((d,h,k) for d in [0,.25,.5] for h in [8,24] for k in [1,2]):
        columns.append('S'+str(i))
        r.append((f'GSM{i}',dict(characteristics_ch1=['cell line: A549',f'treatment: MLN4924 {d}µM',f'time: {h}h'],
                              title=[f'A549 Replicate {k}'],description=[str(i)]),None))
    return pd.Index(columns),r


def test_explicit_description_bind_not_geo_or_column_order():
    c,r=records();s=bind(c[::-1],r[::-1])
    assert s.iloc[0].source_column=='S11' and len(s)==12


def test_missing_contemporaneous_vehicle_rejected():
    c,r=records()
    with pytest.raises(ValueError):bind(c[1:],r[1:])


def test_unreviewed_dose_unit_rejected():
    c,r=records();r[1][1]['characteristics_ch1'][1]='treatment: MLN4924 250nM'
    with pytest.raises(ValueError,match='units'):bind(c,r)


@pytest.mark.parametrize('fold',['dose','time','protocol'])
def test_held_axis_cannot_select_features(fold):
    c,r=records();s=bind(c,r);rng=np.random.default_rng(8);v=rng.poisson(9,(12,30)).astype(float)
    a=diagnostics(v,s,5);held=set(a['folds'][fold]['held_sample_ids'])
    v[s.sample_id.isin(held),0]+=100000
    b=diagnostics(v,s,5)
    assert a['folds'][fold]['feature_indices']==b['folds'][fold]['feature_indices']


def test_static_no_response_has_zero_error():
    c,r=records();s=bind(c,r);v=np.tile(np.arange(1,31),(12,1))
    a=diagnostics(v,s,5)
    assert all(x['delta_log1pCPM_MSE']==0 for x in a['results'])
    assert a['folds']['time']['test_vehicle_role']=='evaluation_only'
