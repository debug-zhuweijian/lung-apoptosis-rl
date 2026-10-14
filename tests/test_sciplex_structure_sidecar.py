import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sciplex_structures", ROOT / "scripts/build_sciplex_structure_sidecar.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs(smiles="CCO"):
    identity = module.describe_structure("CCO")["identity_id"]
    source = pd.DataFrame([{"cell_type": "A549", "vehicle": False, "catalog_number": "S1",
                            "name": "fixture compound", "SMILES": smiles, "CAS.Number": "fixture", "M.w.": 1.0}])
    reference = pd.DataFrame([{"identity_id": identity, "drug_id": "tahoe:fixture",
        "drug_name": "fixture", "identity_status": "original_cid_structure_verified",
        "eligible_for_structural_features": True, "chemical_or_experiment_group_id": identity}])
    conditions = pd.DataFrame([{"eligible": True, "is_vehicle": False,
                                 "chemical_or_experiment_group_id": identity, "role": "fit"}])
    return source, reference, conditions


def test_full_match_never_inherits_experimental_verification():
    rows, fp, matches = module.build_records(*inputs())
    assert fp.shape == (1, 2048) and fp.dtype == np.uint8
    assert matches.iloc[0].reference_roles == "fit"
    assert not rows.iloc[0].experimental_material_verified
    assert rows.iloc[0].identity_status == "source_reported_structure_not_independently_verified"


def test_unspecified_stereo_is_preserved_and_flagged():
    rows, _, matches = module.build_records(*inputs("CC(O)C(=O)O"))
    assert rows.iloc[0].unspecified_stereo_count == 1
    assert "@" not in rows.iloc[0].canonical_full_smiles
    assert "unspecified_stereochemistry" in rows.iloc[0].warnings
    assert matches.empty


def test_conflicting_records_for_one_catalogue_are_rejected():
    source, reference, conditions = inputs()
    conflict = source.assign(SMILES="CCN")
    with pytest.raises(ValueError, match="exactly one"):
        module.build_records(pd.concat([source, conflict]), reference, conditions)


def test_equal_structure_distinct_source_catalogues_are_not_discarded():
    source, reference, conditions = inputs()
    duplicate = source.assign(catalog_number="S2")
    rows, fp, _ = module.build_records(pd.concat([source, duplicate]), reference, conditions)
    assert list(rows.drug_id) == ["sciplex:S1", "sciplex:S2"]
    assert rows.identity_id.nunique() == 1
    np.testing.assert_array_equal(fp[0], fp[1])
