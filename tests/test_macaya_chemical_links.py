import pandas as pd
import pytest
from scripts.recover_macaya_chemical_links import compare_tahoe,resolve_verified_rows


def test_solvate_name_cannot_be_promoted_by_equal_parent_smiles():
    c=pd.DataFrame([dict(source_compound_id='x',canonical_full_smiles='CCO')])
    t=pd.DataFrame([dict(drug='ordinary',canonical_smiles='CCO'),dict(drug='drug DMSO solvate',canonical_smiles='CCO')])
    r=compare_tahoe(c,t)
    assert r.nominal_full_form_eligible.tolist()==[True,False]
    assert not r.paired_RNA_endpoint.any()
    assert not r.dose_time_background_matched.any()


def test_stereo_difference_not_exact_match():
    c=pd.DataFrame([dict(source_compound_id='x',canonical_full_smiles='C[C@H](O)F')])
    r=compare_tahoe(c,pd.DataFrame([dict(drug='other',canonical_smiles='C[C@@H](O)F')]))
    assert r.structure_relation.tolist()==['same_connectivity_different_stereochemistry']
    assert not r.nominal_full_form_eligible.any()


def test_unverified_name_rejected():
    with pytest.raises(ValueError,match='Unverified'):
        resolve_verified_rows([dict(name_resolution_status='ambiguous_name_multiple_cids')],{'PropertyTable':{'Properties':[]}})


def test_structure_key_disagreement_rejected():
    with pytest.raises(ValueError,match='Structure/key'):
        resolve_verified_rows([dict(name_resolution_status='unique_pubchem_name_and_synonyms_verified',selected_cid=1)],
                              {'PropertyTable':{'Properties':[dict(CID=1,SMILES='CCO',InChIKey='wrong')]}})
