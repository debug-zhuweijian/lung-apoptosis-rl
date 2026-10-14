import pandas as pd
import pytest
from scripts.recover_trametinib_timecourse import gene_identity,bind_samples


def records():
    r=[];cols=[]
    for day in [0,2,5,10]:
        for i in [1,2,3]:
            acc=f'GSM{day}{i}';stem=f'A549_day{day}_rep{i}';cols.append(stem)
            r.append((acc,dict(characteristics_ch1=['cell line: A549',f'time: Day {day}','treatment: '+('-' if day==0 else 'trametinib (10nM)')],
                               supplementary_file_1=[f'https://example.org/{acc}_{stem}.txt.gz'],
                               title=[f'A549 cells, D{day}, biol rep{i}'],description=[stem]),None))
    return pd.Index(cols),r


def test_compound_gene_labels_and_missing_symbols():
    assert gene_identity(pd.Series(['ENSG00001|ABC','ENSG00002'])).tolist()==['ENSG00001','ENSG00002']


@pytest.mark.parametrize('labels',[['ENSG00001|A','ENSG00001|B'],['ENSG00001.1','ENSG00001.2'],['ABC|ENSG00001']])
def test_ambiguous_or_invalid_gene_labels_rejected(labels):
    with pytest.raises(ValueError):gene_identity(pd.Series(labels))


def test_baseline_not_fabricated_as_contemporaneous_control():
    c,r=records();s=bind_samples(c,r)
    assert s.elapsed_h.unique().tolist()==[0,48,120,240]
    assert not s.contemporaneous_vehicle_available.any()
    assert not s.replicate_pairing_across_time_verified.any()


def test_missing_sample_is_rejected():
    c,r=records()
    with pytest.raises(ValueError):bind_samples(c[:-1],r)


def test_test_values_cannot_change_selected_training_features():
    import numpy as np
    from scripts.recover_trametinib_timecourse import time_diagnostic
    rng=np.random.default_rng(9);v=rng.poisson(10,(12,20)).astype(float)
    days=np.repeat([0,2,5,10],3)
    _,a=time_diagnostic(v,days,5)
    v[days==5,0]+=100000
    _,b=time_diagnostic(v,days,5)
    assert a['5']==b['5']


def test_no_time_information_added_to_static_expression():
    import numpy as np
    from scripts.recover_trametinib_timecourse import time_diagnostic
    v=np.tile(np.arange(1,21),(12,1))
    r,_=time_diagnostic(v,np.repeat([0,2,5,10],3),5)
    assert all(x['mean_expression_MSE']<1e-25 for x in r)
