"""Reviewed nominal forms are real model inputs, never experimental pairing."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from scripts.build_reviewed_identity_v3 import (
    ADOPT, apply_endpoint_reviews, apply_tahoe_reviews, candidate_links,
)
from src.lung.data import TrainingData, prepare_training
from src.lung.data.common import sha256
from src.lung.data.drug_identity import describe_structure, map_conditions
from tests.test_identity_prepare_v2 import identity_fixture, write_json


def synthetic_review(source_id="drug3", smiles="CCCC.Cl"):
    mol = Chem.MolFromSmiles(smiles)
    structure = describe_structure(smiles)
    reference = {"CID": 777, "SMILES": smiles, "InChIKey": structure["rdkit_inchikey"],
                 "InChI": Chem.MolToInchi(mol), "Title": "synthetic nominal form",
                 "MolecularFormula": "synthetic", "MolecularWeight": "0",
                 "UndefinedAtomStereoCount": 0, "UndefinedBondStereoCount": 0}
    review = {"source_id": source_id, "recommendation": ADOPT, "nominal_identity_resolved": True,
              "experimental_reagent_batch_verified": False, "cross_source_sample_pair_verified": False,
              "proposed_pubchem_cid": 777, "proposed_pubchem_inchikey": structure["rdkit_inchikey"],
              "proposed_identity_id": structure["identity_id"], "vendor_full_structure_matches_proposed": True,
              "manufacturer": "synthetic fixture", "manufacturer_catalogue": "fixture-only",
              "manufacturer_cas": None, "evidence_path": "synthetic.json", "additional_evidence_path": None,
              "remaining_limit": "synthetic fixture; no biological sample or historical lot"}
    return review, reference


def test_reviewed_nominal_form_reaches_prepare_reader_and_preserves_v2(identity_fixture):
    root, old_config, catalog, fp = identity_fixture
    old_assets = prepare_training(root, old_config, build_preprocessing=False)
    old_hashes = {p: sha256(p) for parent in (root / "identity-v2", old_assets) for p in parent.rglob("*") if p.is_file()}
    review, reference = synthetic_review()
    original_id = catalog.set_index("drug_id").loc["drug3", "source_pubchem_cid"]
    result, new_fp = apply_tahoe_reviews(catalog, fp, pd.DataFrame([review]), {777: reference},
                                       {"path": "reference.json", "sha256": "fixture", "url": "fixture"})
    row = result.set_index("drug_id").loc["drug3"]
    assert row.source_pubchem_cid == original_id and row.reference_pubchem_cid == 777
    assert row.identity_status == "manufacturer_nominal_form_verified"
    assert row.eligible_for_structural_features and not row.experimental_material_verified
    assert not row.cross_source_pair_verified
    assert not fp[row.fingerprint_row].any() and new_fp[row.fingerprint_row].any()
    assert catalog.set_index("drug_id").loc["drug3", "identity_status"] == "source_material_form_review_required"
    directory = root / "identity-v3"
    shutil.copytree(root / "identity-v2", directory)
    result.to_parquet(directory / "drug_identity.parquet", index=False)
    conditions = map_conditions(pd.read_parquet(root / "prepared/tahoe_conditions.parquet"), result)
    conditions.assign(identity_version="identity-v3").to_parquet(directory / "condition_identity.parquet", index=False)
    np.save(directory / "morgan2048_chiral.npy", new_fp, allow_pickle=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["version"] = "identity-v3"
    manifest["outputs"] = {name: {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
                           for name in manifest["outputs"]}
    write_json(directory / "manifest.json", manifest)
    config = json.loads(old_config.read_text())
    config.update(assets="training-identity-v3", drug_identity_manifest="identity-v3/manifest.json",
                  drug_identity_manifest_sha256=sha256(directory / "manifest.json"))
    path = root / "configs/data_identity_v3.json"
    write_json(path, config)
    prepare_training(root, path, build_preprocessing=False)
    reader = TrainingData(root, path, require_preprocessing=False)
    action = reader.action_features("condition3")
    assert action["chemical_id"] == row.identity_id
    np.testing.assert_array_equal(action["fingerprint"], new_fp[row.fingerprint_row])
    assert reader.manifest["metadata"]["excluded_unresolved_cells"] == 12
    assert reader.manifest["drug_identity"]["version"] == "identity-v3"
    assert all(sha256(p) == digest for p, digest in old_hashes.items())


@pytest.mark.parametrize("field", ["experimental_reagent_batch_verified", "cross_source_sample_pair_verified"])
def test_nominal_review_cannot_promote_lot_or_sample_claims(identity_fixture, field):
    _, _, catalog, fp = identity_fixture
    review, reference = synthetic_review()
    review[field] = True
    with pytest.raises(ValueError, match="only a nominal"):
        apply_tahoe_reviews(catalog, fp, pd.DataFrame([review]), {777: reference}, {})


def test_endpoint_alias_resolution_keeps_parent_only_matches_unpaired(identity_fixture):
    _, _, catalog, fp = identity_fixture
    review, reference = synthetic_review()
    target, _ = apply_tahoe_reviews(catalog, fp, pd.DataFrame([review]), {777: reference},
                                   {"path": "reference.json", "sha256": "fixture", "url": "fixture"})
    target = target[target.drug_id.eq("drug3")]
    review, reference = synthetic_review("Inde2020:fixture", "CCCC")
    compounds = pd.DataFrame([{"source_id": review["source_id"], "compound_id": None,
        "name_resolution_status": "ambiguous_name_multiple_cids", "source_compound_name": "fixture",
        "study": "Inde2020", "candidate_tahoe_drug_name": target.drug_name.iloc[0],
        "source_dose_um": 1., "source_time_hours": 24., "tahoe_background_relation": "synthetic-only"}])
    result = apply_endpoint_reviews(compounds, pd.DataFrame([review]), {777: reference})
    assert result.nominal_structure_resolved.all() and not result.use_as_paired_training_target.any()
    assert not result.experimental_reagent_form_verified.any()
    conditions = pd.DataFrame([{"drug_id": "drug3", "condition_id": "condition3", "dose_value": 1., "treatment_hours": 24.}])
    links = candidate_links(result, target, conditions)
    assert links.structure_relation.tolist() == ["same_largest_organic_fragment_only"]
    assert not links.eligible_nominal_full_structure_link.any()
    assert not links.experimental_sample_pair_verified.any() and not links.dose_basis_verified.any()


def test_v2_does_not_silently_accept_new_review_status(identity_fixture):
    root, config, catalog, _ = identity_fixture
    held = catalog.drug_id.eq("drug3")
    catalog.loc[held, "identity_status"] = "manufacturer_nominal_form_verified"
    catalog.loc[held, "eligible_for_structural_features"] = True
    catalog.to_parquet(root / "identity-v2/drug_identity.parquet", index=False)
    from tests.test_identity_prepare_v2 import refresh_identity_manifest
    refresh_identity_manifest(root, config)
    with pytest.raises(ValueError, match="feature mask conflicts"):
        prepare_training(root, config, build_preprocessing=False)
