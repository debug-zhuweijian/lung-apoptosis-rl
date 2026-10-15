import numpy as np
import pandas as pd
import pytest
from scripts.recover_state_summary import join_summaries


def data():
    a=pd.DataFrame(dict(gene_id=['iso1','iso2','iso3'],geneName=['A','A','-'],length=[10,20,30],
                        FPKM_Dormancy=[0,2.5,3],FPKM_NC=[1,2,3]))
    a['gene id']=['ENSG1','ENSG1','-']
    b=a.drop(columns='FPKM_NC').copy();b['FPKM_Relapse']=[3,2,1]
    return a,b


def test_preserves_transcript_rows_and_group_unit():
    a,b=data();v,ann=join_summaries(a,b.iloc[::-1].reset_index(drop=True))
    assert v.shape==(3,3) and v[1,1]==2.5
    assert ann.gene_mapping_status.tolist()==['multiple_source_transcripts_per_gene']*2+['missing_source_annotation']


@pytest.mark.parametrize('defect',['duplicate','annotation','dormancy','coverage','nonfinite'])
def test_source_disagreement_fails(defect):
    a,b=data()
    if defect=='duplicate':b.loc[1,'gene_id']='iso1'
    elif defect=='annotation':b.loc[1,'gene id']='ENSG2'
    elif defect=='dormancy':b.loc[1,'FPKM_Dormancy']=9
    elif defect=='coverage':b=b.iloc[:2]
    else:b.loc[1,'FPKM_Relapse']=np.nan
    with pytest.raises(ValueError):join_summaries(a,b)
