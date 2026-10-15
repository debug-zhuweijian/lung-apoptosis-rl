import pandas as pd
import pytest
from scripts.recover_nci_chemical_links import link_samples,validated_catalog


def samples():
    return pd.DataFrame([dict(drug_name='x',dose_nM=0,is_vehicle=True),dict(drug_name='x',dose_nM=100,is_vehicle=False)])


def test_vehicle_retains_block_identity_but_has_no_action_drug():
    r=link_samples(samples(),pd.DataFrame([dict(drug_name='x',drug_id='pubchem:1')]))
    assert r.experimental_block_drug_id.tolist()==['pubchem:1','pubchem:1']
    assert pd.isna(r.action_drug_id.iloc[0]) and r.action_drug_id.iloc[1]=='pubchem:1'
    assert r.nominal_action_structure_verified.tolist()==[False,True]
    assert not r.paired_RNA_endpoint.any() and not r.experimental_reagent_form_verified.any()


def test_missing_structure_is_not_filled_by_name_guess():
    with pytest.raises(ValueError,match='coverage'):
        link_samples(samples(),pd.DataFrame([dict(drug_name='y',drug_id='pubchem:1')]))


def test_nonzero_vehicle_rejected():
    s=samples();s.loc[0,'dose_nM']=1
    with pytest.raises(ValueError,match='Vehicle/dose'):
        link_samples(s,pd.DataFrame([dict(drug_name='x',drug_id='pubchem:1')]))


def test_ambiguous_registry_identity_not_promoted():
    with pytest.raises(ValueError,match='Unverified'):
        validated_catalog([dict(name_resolution_status='ambiguous')],{'PropertyTable':{'Properties':[]}},[])


def test_registry_key_conflict_requires_review():
    with pytest.raises(ValueError,match='Structure/key'):
        validated_catalog([dict(name_resolution_status='unique_pubchem_name_and_synonyms_verified',selected_cid=1,source_name='x')],
                          {'PropertyTable':{'Properties':[dict(CID=1,SMILES='CCO',InChIKey='wrong')]}},
                          [{'InformationList':{'Information':[dict(CID=1,Synonym=['x'])]}}])


def test_disconnected_platinum_registry_graph_requires_geometry_review():
    from lung.data.drug_identity import describe_structure
    smi='N.N.Cl[Pt]Cl';key=describe_structure(smi)['rdkit_inchikey']
    c=validated_catalog([dict(name_resolution_status='unique_pubchem_name_and_synonyms_verified',selected_cid=1,source_name='cisplatin')],
                        {'PropertyTable':{'Properties':[dict(CID=1,SMILES=smi,InChIKey=key)]}},
                        [{'InformationList':{'Information':[dict(CID=1,Synonym=['cisplatin'])]}}])
    assert c.coordination_geometry_review_required.tolist()==[True]
    assert c.structure_feature_use_eligible.tolist()==[False]


def test_review_flag_propagates_to_actual_sample_links():
    c=pd.DataFrame([dict(drug_name='x',drug_id='pubchem:1',structure_feature_use_eligible=False)])
    r=link_samples(samples(),c)
    assert not r.action_structure_feature_use_eligible.any()
    assert r.nominal_action_structure_verified.tolist()==[False,True]
