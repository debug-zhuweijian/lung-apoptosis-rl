"""A vehicle flag cannot establish DMSO identity across experimental sources."""

import numpy as np
import pandas as pd
import pytest

from lung.data.chemistry import build_chemical_catalog


@pytest.mark.parametrize("smiles", ["CCO", "O", None, "", "CS(C)=O.O", "not_smiles"])
def test_vehicle_needs_explicit_dmso_structure(smiles):
    catalog = pd.DataFrame([{"drug_id": "control", "drug_name": "control",
                             "component_role": "vehicle", "canonical_smiles": smiles}])
    with pytest.raises(ValueError, match="DMSO"):
        build_chemical_catalog(catalog)


@pytest.mark.parametrize("smiles", ["CS(C)=O", "CS(=O)C", "O=S(C)C"])
def test_verified_dmso_preserves_legacy_contract(smiles):
    catalog = pd.DataFrame([{"drug_id": "DMSO", "drug_name": "DMSO",
                             "component_role": "vehicle", "canonical_smiles": smiles}])
    identity, fp, meta = build_chemical_catalog(catalog)
    assert identity.iloc[0].chemical_or_experiment_group_id == "vehicle:DMSO"
    assert identity.iloc[0].identity_status == "vehicle"
    assert identity.iloc[0].canonical_full_smiles is None
    assert fp.shape == (1, 2048) and fp.dtype == np.uint8 and not fp.any()
    assert meta["fingerprint"]["includeChirality"] is True


def test_ordinary_active_drug_is_not_relabelled_as_a_vehicle():
    identity, fp, _ = build_chemical_catalog(pd.DataFrame([
        {"drug_id": "a", "drug_name": "ethanol", "canonical_smiles": "CCO"},
        {"drug_id": "b", "drug_name": "unresolved", "canonical_smiles": None},
    ]))
    assert identity.iloc[0].identity_status == "parsed_full_form"
    assert identity.iloc[1].identity_status == "DRUG_IDENTITY_UNRESOLVED"
    assert fp[0].any() and not fp[1].any()
