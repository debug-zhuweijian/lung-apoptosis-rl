import pytest
from scripts.recover_cached_pie_subset import partition_coverage
from scripts.audit_measured_rna_function_transfer import validate_cohort


def test_a_target_spanning_two_groups_is_not_complete_with_one_group():
    result=partition_coverage({'a':{1,2},'b':{3}}, {1,3})
    assert result[0]['missing_groups']==[2]
    assert not result[0]['complete_source_groups']
    assert result[1]['complete_source_groups']
    assert len(result)==2


def test_extra_cached_groups_do_not_expand_the_declared_cohort():
    result=partition_coverage({'a':{1}}, {1,2,3})
    assert len(result)==1 and result[0]['required_groups']==[1]


def test_empty_group_proof_is_rejected():
    with pytest.raises(ValueError):partition_coverage({'a':set()},set())


def test_chemical_fingerprint_catalog_is_not_an_eligibility_table():
    import pandas as pd
    from scripts.recover_cached_pie_subset import declared_targets
    with pytest.raises(ValueError,match='nominal_structures'):
        declared_targets(pd.DataFrame({'drug_id':['a']}))


def test_complete_cached_targets_still_fail_full_cohort_evaluation():
    import pandas as pd
    from scripts.recover_cached_pie_subset import declared_targets
    c=pd.DataFrame({'tahoe_drug':['a','b'],'nominal_functional_transfer_eligible':[True,True]})
    targets=declared_targets(c)
    with pytest.raises(ValueError,match='Incomplete'):
        validate_cohort(targets[:1],c,'nominal_functional_transfer_eligible')
    validate_cohort(targets,c,'nominal_functional_transfer_eligible')
