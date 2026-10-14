"""Regressions for stereochemical collisions and unsafe salt equivalence."""

import numpy as np
import pandas as pd
import pytest

from src.lung.data.drug_identity import build_identity_catalog, compare_structures, describe_structure, map_conditions


DOXO = "C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O.Cl"
EPI = "C[C@H]1[C@@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O.Cl"
OLD_ANTHRACYCLINE = "CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O.Cl"


def test_anthracycline_epimers_no_longer_share_identity_or_chiral_features():
    catalog = pd.DataFrame([
        {"drug_id": "d", "drug_name": "Doxorubicin HCl", "canonical_smiles": OLD_ANTHRACYCLINE, "pubchem_cid": 443939},
        {"drug_id": "e", "drug_name": "Epirubicin HCl", "canonical_smiles": OLD_ANTHRACYCLINE, "pubchem_cid": 65348},
    ])
    # Public PubChem property values for the source's original CIDs, not names.
    references = {
        443939: {"CID": 443939, "SMILES": DOXO, "InChIKey": "MWWSFMDVAYGXBV-RUELKSSGSA-N"},
        65348: {"CID": 65348, "SMILES": EPI, "InChIKey": "MWWSFMDVAYGXBV-FGBSZODSSA-N"},
    }
    identities, features = build_identity_catalog(catalog, references)
    assert identities.source_identity_id.nunique() == 1
    assert identities.identity_id.nunique() == 2
    assert identities.parent_id.nunique() == 2
    assert identities.identity_status.eq("original_cid_structure_verified").all()
    assert not np.array_equal(features[0], features[1])
    assert not identities.experimental_material_verified.any()
    assert not identities.cross_source_pair_verified.any()


def test_paclitaxel_reference_retains_all_eleven_reported_stereocenters():
    smiles = "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    description = describe_structure(smiles)
    assert description["rdkit_inchikey"] == "RCINICONZNJXQF-MZXODVADSA-N"
    assert description["defined_stereo_count"] == 11
    assert description["unspecified_stereo_count"] == 0


def test_salt_and_free_form_are_only_parent_candidates():
    free, salt = "CNCC", "CNCC.Cl"
    a, b = describe_structure(free), describe_structure(salt)
    assert a["identity_id"] != b["identity_id"]
    assert a["parent_id"] == b["parent_id"]
    assert compare_structures(free, salt) == "same_largest_organic_fragment_only"
    assert compare_structures("C[NH2+]CC.[Cl-]", free) == "different_structures"


def test_parent_does_not_merge_stereoisomers_or_equal_size_mixtures():
    assert compare_structures(DOXO, EPI) == "same_connectivity_different_stereochemistry"
    assert describe_structure(DOXO)["parent_id"] != describe_structure(EPI)["parent_id"]
    ambiguous = describe_structure("CCO.COC")
    assert ambiguous["parent_id"] is None
    assert ambiguous["parent_status"] == "ambiguous_largest_organic_fragments"


def test_isotope_difference_is_not_classed_as_only_stereochemistry():
    assert compare_structures("[13CH3]CO", "CCO") == "different_structures"
    assert describe_structure("[13CH3]CO")["parent_id"] != describe_structure("CCO")["parent_id"]


def test_missing_cid_and_structure_conflict_never_get_eligible_features():
    catalog = pd.DataFrame([
        {"drug_id": "x", "drug_name": "Unresolved", "canonical_smiles": None, "pubchem_cid": np.nan},
        {"drug_id": "y", "drug_name": "Conflict", "canonical_smiles": "CCO", "pubchem_cid": 1},
    ])
    key = describe_structure("CCCC")["rdkit_inchikey"]
    identities, features = build_identity_catalog(catalog, {1: {"CID": 1, "SMILES": "CCCC", "InChIKey": key}})
    assert identities.identity_status.tolist() == ["source_cid_missing_or_invalid", "source_connectivity_conflict_review_required"]
    assert not identities.eligible_for_structural_features.any()
    assert not features.any()


def test_named_solvate_cannot_silently_use_free_form_reference():
    catalog = pd.DataFrame([{"drug_id": "x", "drug_name": "Compound (DMSO_TF solvate)",
                             "canonical_smiles": "CCO", "pubchem_cid": 702}])
    key = describe_structure("CCO")["rdkit_inchikey"]
    identities, features = build_identity_catalog(catalog, {702: {"CID": 702, "SMILES": "CCO", "InChIKey": key}})
    assert identities.identity_status.iloc[0] == "source_material_form_review_required"
    assert identities.material_form_review_reason.iloc[0] == "source_solvate_label_without_solvent_component_in_reference"
    assert not identities.eligible_for_structural_features.any()
    assert not features.any()


def test_condition_join_preserves_ids_concentration_and_time_without_mutation():
    conditions = pd.DataFrame({"condition_id": ["tahoe:smp_1"], "drug_id": ["x"], "dose_value": [0.5], "treatment_hours": [24]})
    original = conditions.copy(deep=True)
    catalog = pd.DataFrame([{ "drug_id": "x", "drug_name": "Unknown", "canonical_smiles": None, "pubchem_cid": np.nan}])
    identities, _ = build_identity_catalog(catalog, {})
    joined = map_conditions(conditions, identities)
    pd.testing.assert_frame_equal(conditions, original)
    pd.testing.assert_frame_equal(joined[original.columns], original)
    assert joined.source_condition_id.tolist() == original.condition_id.tolist()
    assert joined.dose_basis.iloc[0].endswith("basis_unverified")
    with pytest.raises(ValueError, match="Unmapped"):
        map_conditions(conditions.assign(drug_id="missing"), identities)
