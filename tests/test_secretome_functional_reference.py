import pytest
from scripts.recover_secretome_functional_reference import extract_blocks,LABELS


def rows():
    out=[]
    for label in LABELS:out.extend([(label,None,None),(1.,2.,3.),(None,None,None)])
    return out[:-1]


def test_three_real_source_values_not_flow_events_or_means():
    out=extract_blocks(rows());assert len(out)==6 and all(len(v)==3 for _,v in out)


@pytest.mark.parametrize('case',['label','missing','nonfinite','extra'])
def test_changed_source_block_rejected(case):
    r=rows()
    if case=='label':r[0]=('mouse',None,None)
    elif case=='missing':r[1]=(1,None,3)
    elif case=='nonfinite':r[1]=(1,float('nan'),3)
    else:r[2]=(9,None,None)
    with pytest.raises(ValueError):extract_blocks(r)
