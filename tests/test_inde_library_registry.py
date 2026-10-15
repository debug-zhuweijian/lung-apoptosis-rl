import pandas as pd
import pytest
from scripts.audit_inde_library_registry import audit


def response(name,cid):return dict(source_name=name,queried_primary_name=name,cids=[cid],status=200)


def test_different_names_can_link_by_independent_CID_not_fuzzy_similarity():
    t=pd.DataFrame({'drug':['Selinexor'],'pubchem_cid':[42]})
    l,c=audit(['KPT-330'],[response('KPT-330',42)],t)
    assert c==[dict(source_name='KPT-330',tahoe_name='Selinexor',cid=42)]
    assert not l.experimental_form_verified.any()


def test_material_records_same_CID_are_not_silently_collapsed():
    t=pd.DataFrame({'drug':['A','A solvate'],'pubchem_cid':[42,42]})
    _,c=audit(['A'],[response('A',42)],t);assert c[0]['tahoe_name']=='A'
    l,c=audit(['Alias'],[response('Alias',42)],t);assert not c
    assert 'multiple_same_CID' in l.candidate_reason.iloc[0]


def test_duplicate_source_CID_requires_review_not_independent_drugs():
    t=pd.DataFrame({'drug':['A'],'pubchem_cid':[42]})
    l,c=audit(['A','Alias'],[response('A',42),response('Alias',42)],t)
    assert not c and l.multiple_source_names_same_CID.all()


def test_missing_query_coverage_or_changed_material_fails():
    t=pd.DataFrame({'drug':['A'],'pubchem_cid':[42]})
    with pytest.raises(ValueError,match='Incomplete'):audit(['A'],[],t)
    with pytest.raises(ValueError,match='material'):audit(['A sulfate'],[dict(response('A sulfate',42),queried_primary_name='A')],t)


def test_unresolved_or_absent_CID_not_claimed_absent_compound():
    t=pd.DataFrame({'drug':['A'],'pubchem_cid':[42]})
    l,c=audit(['B'],[response('B',99)],t)
    assert not c and 'not_proven_absent' in l.candidate_reason.iloc[0]
