import numpy as np
import pandas as pd
import pytest
from scripts.recover_archs4_em2 import verify_export,SAMPLES


def inputs():
    t=pd.DataFrame(np.arange(12).reshape(3,4),index=['A','B','A'],columns=SAMPLES)
    m={s:dict(series='GSE270062',source='A549') for s in SAMPLES}
    return t,m


def test_duplicate_symbols_preserved_not_summed():
    t,m=inputs();v,a=verify_export(t,['A','B','A'],m)
    np.testing.assert_array_equal(v,t.to_numpy())
    assert a.identity_status.tolist()==['duplicate_symbol_requires_gene_ID','unique_source_symbol','duplicate_symbol_requires_gene_ID']


@pytest.mark.parametrize('case',['sample_order','gene_order','series','background','negative','nonfinite'])
def test_mismatched_export_rejected(case):
    t,m=inputs();genes=['A','B','A']
    if case=='sample_order':t=t[t.columns[::-1]]
    elif case=='gene_order':genes=['B','A','A']
    elif case=='series':m[SAMPLES[0]]['series']='GSEwrong'
    elif case=='background':m[SAMPLES[0]]['source']='AsPC-1'
    elif case=='negative':t.iloc[0,0]=-1
    else:t.iloc[0,0]=np.nan
    with pytest.raises(ValueError):verify_export(t,genes,m)


def test_provider_axis_resolves_only_unique_symbols_without_release_binding():
    from scripts.recover_archs4_em2 import attach_provider_axis
    t,m=inputs();_,a=verify_export(t,['A','B','A'],m)
    p={'symbol':['A','B','A'],'ensembl_gene':['ENSG1','ENSG2','ENSG3']}
    out=attach_provider_axis(a,p)
    assert out.resolved_ensembl_id.notna().tolist()==[False,True,False]
    assert out.loc[1,'resolved_ensembl_id']=='ENSG2'
    p['symbol']=['B','A','A']
    with pytest.raises(ValueError):attach_provider_axis(a,p)
