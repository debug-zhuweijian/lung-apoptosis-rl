import pandas as pd
import numpy as np
import pytest
from scripts.recover_trf_expression import validate,GROUPS,SAMPLES


def frame():
    d=pd.DataFrame({'gene_id':['ENSG1','ENSG2']})
    for j,s in enumerate(SAMPLES):d[s]=[j+.12,0]
    for g in GROUPS:d[g]=d[[g+str(i) for i in [1,2,3]]].mean(axis=1).round(4)
    d['Unnamed: 17']=np.nan
    return d


def test_group_means_not_pseudoreplicates():
    v=validate(frame());assert v.shape==(2,12);assert v[1].sum()==0


@pytest.mark.parametrize('defect',['mean','extra','negative','gene'])
def test_invalid_source_fails(defect):
    d=frame()
    if defect=='mean':d.loc[0,'CON']+=1
    elif defect=='extra':d['new_sample']=[1,2]
    elif defect=='negative':d.loc[0,'CON1']=-1
    else:d.loc[1,'gene_id']='ENSG1'
    with pytest.raises(ValueError):validate(d)
