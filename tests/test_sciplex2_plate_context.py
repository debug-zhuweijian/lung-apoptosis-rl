import numpy as np
import pandas as pd
import pytest
from scripts.audit_sciplex2_plate_context import evaluate


def inputs():
    g=pd.DataFrame([dict(plate_prefix=p,drug='Dex',dose_source_label=d,dose_uM=d*.5,cell_count=100)
                    for p in ['A','B'] for d in [0,1,5]])
    x=np.array([[5,10,20],[10,10,15],[15,5,15],[7,8,20],[12,8,15],[17,3,15]])
    return x,g


def test_feature_selection_excludes_held_plate():
    x,g=inputs();_,a=evaluate(x,g,n_features=2);x[3:,0]*=10000
    _,b=evaluate(x,g,n_features=2)
    assert a[1]['gene_indices']==b[1]['gene_indices']
    assert a[1]['held_plate']=='B'


def test_identical_plates_transfer_exactly():
    x,g=inputs();x[3:]=x[:3];s,_=evaluate(x,g)
    assert s[s.method!='held_plate_vehicle'].mse.max()==0
    assert set(s.method)=={'donor_same_condition','held_plate_vehicle','additive_plate_reference'}


def test_missing_condition_and_duplicate_rejected():
    x,g=inputs()
    with pytest.raises(ValueError,match='Missing'):
        evaluate(x[1:],g.iloc[1:].reset_index(drop=True))
    with pytest.raises(ValueError,match='identity'):
        evaluate(np.vstack([x,x[:1]]),pd.concat([g,g.iloc[:1]],ignore_index=True))
