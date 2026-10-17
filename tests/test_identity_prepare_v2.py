"""Identity dependency binding and actual training features; synthetic fixtures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from rdkit import Chem

from src.lung.data import TrainingData, prepare_training
from src.lung.data.common import DEFAULT_ASSETS, dataset_config, digest_json, sha256
from src.lung.data.drug_identity import build_identity_catalog, map_conditions
from src.lung.data.splits import condition_roles
from tests.test_training_data_v1 import tiny_bundle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def refresh_identity_manifest(root, config_path):
    identity = root / "identity-v2"
    manifest_path = identity / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["outputs"]:
        file = identity / name
        manifest["outputs"][name] = {"sha256": sha256(file), "bytes": file.stat().st_size}
    write_json(manifest_path, manifest)
    config = json.loads(config_path.read_text())
    config["drug_identity_manifest_sha256"] = sha256(manifest_path)
    write_json(config_path, config)


@pytest.fixture
def identity_fixture(tmp_path):
    tiny_bundle(tmp_path)
    bundle = tmp_path / "prepared"
    source = pd.read_parquet(bundle / "drug_catalog.parquet")
    source["pubchem_cid"] = [100 + i for i in range(20)] + [None]
    # Same incomplete source structure, distinct reviewed stereoisomers.
    source.loc[:1, "canonical_smiles"] = "CC(O)F"
    source.to_parquet(bundle / "drug_catalog.parquet", index=False)
    prepared_manifest = json.loads((bundle / "manifest.json").read_text())
    for record in prepared_manifest["outputs"]:
        record["sha256"] = sha256(bundle / record["path"])
    write_json(bundle / "manifest.json", prepared_manifest)
    properties = {}
    for i, row in source.iloc[:20].iterrows():
        structure = ["C[C@H](O)F", "C[C@@H](O)F"][i] if i < 2 else row.canonical_smiles
        properties[100 + i] = {"CID": 100 + i, "SMILES": structure,
                               "InChIKey": Chem.MolToInchiKey(Chem.MolFromSmiles(structure))}
    del properties[102]  # An absent reference cannot become a ready chemical.
    catalog, fp = build_identity_catalog(source, properties)
    held = catalog.drug_id.eq("drug3")
    catalog.loc[held, "identity_status"] = "source_material_form_review_required"
    catalog.loc[held, "eligible_for_structural_features"] = False
    fp[catalog.loc[held, "fingerprint_row"].to_numpy()] = 0
    conditions = pd.read_parquet(bundle / "tahoe_conditions.parquet")
    directory = tmp_path / "identity-v2"
    directory.mkdir()
    catalog.to_parquet(directory / "drug_identity.parquet", index=False)
    map_conditions(conditions, catalog).to_parquet(directory / "condition_identity.parquet", index=False)
    np.save(directory / "morgan2048_chiral.npy", fp, allow_pickle=False)
    raw = directory / "reference.json"
    write_json(raw, {"fixture": True})
    write_json(directory / "sources.json", [{"path": raw.name, "sha256": sha256(raw), "bytes": raw.stat().st_size}])
    dependencies = ["drug_identity.parquet", "condition_identity.parquet", "morgan2048_chiral.npy", "sources.json"]
    write_json(directory / "manifest.json", {
        "version": "identity-v2", "status": "reference_mapping_complete_with_explicit_exclusions",
        "inputs_sha256": {**{f"prepared/{name}": sha256(bundle / name)
                          for name in ("drug_catalog.parquet", "tahoe_conditions.parquet")},
                          "src/lung/data/drug_identity.py": "0" * 64},
        "outputs": {name: {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
                    for name in dependencies}, "raw_response_count": 1,
    })
    config = yaml.safe_load((tmp_path / "configs/dataset.yaml").read_text())
    config.update(assets="training-identity-v2", drug_identity_manifest="identity-v2/manifest.json",
                  drug_identity_manifest_sha256=sha256(directory / "manifest.json"))
    config_path = tmp_path / "configs/data_identity_v2.json"
    write_json(config_path, config)
    return tmp_path, config_path, catalog, fp


def test_reviewed_structures_reach_splits_and_actual_action_features(identity_fixture):
    root, config_path, catalog, fp = identity_fixture
    frozen_source = sha256(root / "prepared/drug_catalog.parquet")
    assets = prepare_training(root, config_path, build_preprocessing=False)
    legacy = json.loads((assets / "manifest.json").read_text())
    legacy["drug_identity"]["inputs"]["src/lung/data/drug_identity.py"] = "0" * 64
    legacy["signature"] = digest_json({"config": legacy["config"],
        "source_manifest": legacy["source_manifest_sha256"], "training_schema": 2,
        "drug_identity": legacy["drug_identity"]})
    write_json(assets / "manifest.json", legacy)
    reader = TrainingData(root, config_path, require_preprocessing=False)
    assert reader.manifest["schema_version"] == 2
    assert reader.manifest["drug_identity"]["manifest_sha256"] == sha256(root / "identity-v2/manifest.json")
    first, second = reader.action_features("condition0"), reader.action_features("condition1")
    assert first["chemical_id"] != second["chemical_id"]
    assert not np.array_equal(first["fingerprint"], second["fingerprint"])
    by_drug = catalog.set_index("drug_id")
    np.testing.assert_array_equal(first["fingerprint"], fp[by_drug.loc["drug0", "fingerprint_row"]])
    np.testing.assert_array_equal(reader.fingerprints, np.load(root / "identity-v2/morgan2048_chiral.npy"))
    assert first["fingerprint"].shape == (2048,)
    for condition in ("condition2", "condition3"):
        with pytest.raises(ValueError, match="DRUG_IDENTITY_UNRESOLVED"):
            reader.action_features(condition)
        assert pd.isna(reader.condition_table.loc[condition, "role"])
    cells = pd.concat(pd.read_parquet(assets / f"{split}.parquet") for split in ("train", "val", "test"))
    assert not cells.condition_id.isin(["condition2", "condition3"]).any()
    assert cells.groupby("chemical_or_experiment_group_id").role.nunique().max() == 1
    assert reader.manifest["metadata"]["excluded_unresolved_cells"] == 24
    assert prepare_training(root, config_path, build_preprocessing=False) == assets
    assert sha256(root / "prepared/drug_catalog.parquet") == frozen_source


@pytest.mark.parametrize("relative,error", [
    ("identity-v2/manifest.json", "manifest hash mismatch"),
    ("identity-v2/drug_identity.parquet", "output hash mismatch"),
    ("identity-v2/morgan2048_chiral.npy", "output hash mismatch"),
    ("identity-v2/reference.json", "raw response hash mismatch"),
    ("prepared/drug_catalog.parquet", "input hash mismatch"),
])
def test_same_path_dependency_changes_rejected_even_when_asset_hashing_is_disabled(identity_fixture, relative, error):
    root, config_path, _, _ = identity_fixture
    prepare_training(root, config_path, build_preprocessing=False)
    with (root / relative).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match=error):
        TrainingData(root, config_path, require_preprocessing=False, verify_hashes=False)
    with pytest.raises(ValueError, match=error):
        prepare_training(root, config_path, build_preprocessing=False)


def test_review_mask_is_enforced_even_if_structure_matches_a_usable_drug(identity_fixture):
    root, _, catalog, _ = identity_fixture
    chemical = catalog.assign(chemical_or_experiment_group_id=catalog.identity_id, identity_version="identity-v2")
    chemical.loc[chemical.drug_id.eq("drug3"), "chemical_or_experiment_group_id"] = chemical.loc[
        chemical.drug_id.eq("drug0"), "chemical_or_experiment_group_id"].iloc[0]
    result = condition_roles(pd.read_parquet(root / "prepared/tahoe_conditions.parquet"), chemical)
    held = result.set_index("condition_id").loc["condition3"]
    assert not held.eligible and pd.isna(held.role) and pd.isna(held.outer_split)


@pytest.mark.parametrize("invalid", ["duplicate_fingerprint_row", "unreviewed_ready_mask"])
def test_invalid_identity_semantics_rejected_after_valid_hashes(identity_fixture, invalid):
    root, config_path, catalog, _ = identity_fixture
    if invalid == "duplicate_fingerprint_row":
        catalog.loc[1, "fingerprint_row"] = catalog.loc[0, "fingerprint_row"]
        error = "indexed binary 2048-bit"
    else:
        catalog.loc[catalog.drug_id.eq("drug3"), "eligible_for_structural_features"] = True
        error = "feature mask conflicts"
    catalog.to_parquet(root / "identity-v2/drug_identity.parquet", index=False)
    refresh_identity_manifest(root, config_path)
    with pytest.raises(ValueError, match=error):
        prepare_training(root, config_path, build_preprocessing=False)


def test_identity_config_requires_an_explicit_hash_and_a_new_asset_version(identity_fixture):
    root, config_path, _, _ = identity_fixture
    config = json.loads(config_path.read_text())
    config.pop("drug_identity_manifest_sha256")
    write_json(config_path, config)
    with pytest.raises(ValueError, match="both a manifest path"):
        dataset_config(root, config_path)
    config.update(drug_identity_manifest_sha256=sha256(root / "identity-v2/manifest.json"), assets=DEFAULT_ASSETS)
    write_json(config_path, config)
    with pytest.raises(ValueError, match="new training assets version"):
        dataset_config(root, config_path)


def test_default_schema1_signature_and_source_features_remain_unchanged(identity_fixture):
    root, _, _, _ = identity_fixture
    assets = prepare_training(root, build_preprocessing=False)
    reader = TrainingData(root, require_preprocessing=False)
    config = dataset_config(root)
    expected = digest_json({"config": config, "source_manifest": sha256(root / "prepared/manifest.json"),
                            "training_schema": 1})
    assert reader.manifest["signature"] == expected
    assert reader.manifest["schema_version"] == 1
    assert "drug_identity" not in reader.manifest
    assert assets.name == "training-v1"
    assert reader.action_features("condition0")["chemical_id"] == reader.action_features("condition1")["chemical_id"]
