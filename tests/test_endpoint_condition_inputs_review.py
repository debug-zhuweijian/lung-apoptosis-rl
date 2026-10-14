"""A chemical review cannot silently change identity or release a held structure."""

import json

import pandas as pd
import pytest

from scripts.build_endpoint_condition_inputs import reviewed_additions
from src.lung.data.drug_identity import describe_structure
from src.lung.data.endpoint_alignment import file_hash


def package(tmp_path, *, approved=True, verified=True, wrong_identity=False, duplicate=False):
    # Synthetic molecule fixture, never a published experimental observation.
    info = describe_structure("CCO")
    row = {"source_id": "synthetic:ethanol", "status": "synthetic_primary_review",
           "eligible_for_structural_features": approved,
           "primary_structure_identity_verified": verified,
           "canonical_full_smiles": info["canonical_full_smiles"],
           "identity_id": "incorrect" if wrong_identity else info["identity_id"],
           "rdkit_inchikey": info["rdkit_inchikey"]}
    source = tmp_path / "primary.txt"
    source.write_text("synthetic primary evidence", encoding="utf8")
    table = tmp_path / "chemical_candidates.csv"
    pd.DataFrame([row, row] if duplicate else [row]).to_csv(table, index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "input_base": ".", "inputs_sha256": {source.name: file_hash(source)},
        "outputs_sha256": {table.name: file_hash(table)}}), encoding="utf8")
    return source


def test_unverified_primary_identity_stays_held(tmp_path):
    package(tmp_path, approved=True, verified=False)
    rows, _ = reviewed_additions(tmp_path)
    assert rows["synthetic:ethanol"]["eligible"] is False


def test_changed_primary_source_is_rejected(tmp_path):
    source = package(tmp_path)
    rows, _ = reviewed_additions(tmp_path)
    assert rows["synthetic:ethanol"]["eligible"] is True
    source.write_text("changed evidence", encoding="utf8")
    with pytest.raises(ValueError, match="source changed"):
        reviewed_additions(tmp_path)


def test_self_consistent_file_hashes_do_not_mask_wrong_chemical_identity(tmp_path):
    package(tmp_path, wrong_identity=True)
    with pytest.raises(ValueError, match="structure identity"):
        reviewed_additions(tmp_path)


def test_duplicate_source_alias_is_rejected(tmp_path):
    package(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate source"):
        reviewed_additions(tmp_path)


def test_primary_image_in_ancestor_review_remains_bound(tmp_path):
    ancestor = tmp_path / "previous"
    ancestor.mkdir()
    source = package(ancestor)
    current = tmp_path / "current"
    current.mkdir()
    table = current / "chemical_candidates.csv"
    table.write_bytes((ancestor / table.name).read_bytes())
    (current / "manifest.json").write_text(json.dumps({
        "input_base": "..",
        "inputs_sha256": {"previous/manifest.json": file_hash(ancestor / "manifest.json")},
        "outputs_sha256": {table.name: file_hash(table)}}), encoding="utf8")
    reviewed_additions(current)
    source.write_text("changed ancestor evidence", encoding="utf8")
    with pytest.raises(ValueError, match="source changed"):
        reviewed_additions(current)
