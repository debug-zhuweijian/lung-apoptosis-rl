import pandas as pd
import pytest
from scripts.link_inde_tahoe_conditions import primary_name,parse_single_dose,bind


def test_material_suffix_not_discarded_as_alias():
    assert primary_name('Drug (hydrochloride)')=='Drug (hydrochloride)'
    assert primary_name('Drug (S)')=='Drug (S)'
    assert primary_name('Drug (HCl)')=='Drug (HCl)'
    assert primary_name('Drug (cis)')=='Drug (cis)'
    assert primary_name('Drug (bromide)')=='Drug (bromide)'
    assert primary_name('Trametinib (GSK1120212)')=='Trametinib'


def test_concentration_units_are_converted_explicitly():
    assert parse_single_dose("[('X', 5000, 'nM')]",'X')==5
    assert parse_single_dose("[('X', 0.005, 'mM')]",'X')==5


@pytest.mark.parametrize('value',["[('X', 5, 'uL')]","[('X', 5, 'uM'),('Y',1,'uM')]","[('Y',5,'uM')]","[('X',-1,'uM')]"])
def test_dose_or_identity_ambiguity_rejected(value):
    with pytest.raises(ValueError):parse_single_dose(value,'X')


def test_functional_replicates_do_not_become_cross_product_pairs():
    c=pd.DataFrame([dict(source_compound_name='X',tahoe_drug='X',drug_id='pubchem:1',strict_structure_link_eligible=True)])
    s=pd.DataFrame([dict(sample=f's{i}',drug='X',plate=f'p{i}',drugname_drugconc="[('X',5,'uM')]") for i in [1,2]])
    f=pd.DataFrame([dict(source_compound_name='X',replicate=i,source_column_1based=1,time_h=24,library_dose_uM=5,arm='single_agent') for i in [1,2,3]])
    links,observed=bind(c,s,f)
    assert len(links)==2 and len(observed)==3
    assert links.functional_group_id.nunique()==1
    assert not links.paired_RNA_functional_replicates.any() and not links.matched_culture.any()
    assert not links.a549_cells_in_sample_verified.any()


def test_target_metadata_cannot_supply_source_identity_without_independent_query():
    from scripts.link_inde_tahoe_conditions import validate_name_resolution
    c=[dict(source_name='X (alias)',cid=1)]
    validate_name_resolution(c,[dict(source_name='X (alias)',queried_primary_name='X',status=200,cids=[1])])
    for ids in [[],[2],[1,2]]:
        with pytest.raises(ValueError,match='unresolved or conflicting'):
            validate_name_resolution(c,[dict(source_name='X (alias)',queried_primary_name='X',status=200,cids=ids)])


def test_identity_match_cannot_remove_author_assay_exclusion():
    from scripts.link_inde_tahoe_conditions import qualify_functional_catalog
    c=pd.DataFrame([dict(source_compound_name='X',nominal_reference_link_eligible=True)])
    obs=pd.DataFrame([dict(source_compound_name='X',replicate=r,arm='single_agent',time_h=24,
        library_dose_uM=5,valid_measured_fraction=True,lethal_fraction=.3,
        author_autofluorescence_exclusion=True) for r in [1,2,3]])
    result=qualify_functional_catalog(c,obs)
    assert result.nominal_reference_link_eligible.all()
    assert not result.nominal_functional_transfer_eligible.any()
    assert result.functional_assay_exclusion_reason.iloc[0]=='author_autofluorescence_exclusion'
    obs['author_autofluorescence_exclusion']=False
    assert qualify_functional_catalog(c,obs).nominal_functional_transfer_eligible.all()
    obs.loc[0,'lethal_fraction']=1.1
    assert not qualify_functional_catalog(c,obs).nominal_functional_transfer_eligible.any()
