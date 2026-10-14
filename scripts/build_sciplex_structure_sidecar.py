"""Preserve sci-Plex reported structures; comparison is not material verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.drug_identity import describe_structure, material_form_review_reason

SOURCE = ROOT / "08-datasets/sciPlex3/GSM4150378/sources/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz"
V3 = ROOT / "08-datasets/LungCancer_A549_Integration/training-identity-v3"
OUTPUT = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/chemical_inputs_v1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_records(metadata, reference, conditions):
    fields = ["catalog_number", "name", "SMILES", "CAS.Number", "M.w."]
    source = metadata.loc[metadata.cell_type.eq("A549") & ~metadata.vehicle, fields].drop_duplicates()
    if source.catalog_number.duplicated().any() or source.empty:
        raise ValueError("Each source catalogue ID must have exactly one reported record")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    records, vectors, matches = [], [], []
    role_sets = conditions.loc[conditions.eligible & ~conditions.is_vehicle].groupby("chemical_or_experiment_group_id").role.agg(lambda x: sorted(set(x)))
    for item in source.sort_values("catalog_number").to_dict("records"):
        description = describe_structure(item["SMILES"])
        if description["parse_status"] != "parsed":
            raise ValueError(f"Unparseable official structure: {item['catalog_number']}")
        vector = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(description["canonical_full_smiles"])), vector)
        warnings = []
        if description["unspecified_stereo_count"]:
            warnings.append("source_has_unspecified_stereochemistry_not_completed")
        if not description["rdkit_inchikey"]:
            description = {**description, "rdkit_inchikey": None}
            warnings.append("InChIKey_unavailable_full_SMILES_identity_retained")
        form = material_form_review_reason(item["name"], description)
        if form:
            warnings.append(form)
        source_id = "sciplex:" + item["catalog_number"]
        records.append({
            "drug_id": source_id, "catalog_number": item["catalog_number"],
            "source_name": item["name"], "source_smiles": item["SMILES"],
            "source_cas": item["CAS.Number"], "source_molecular_weight": item["M.w."],
            **description, "identity_version": "sciplex-source-structures-v1",
            "chemical_or_experiment_group_id": description["identity_id"],
            "identity_status": "source_reported_structure_not_independently_verified",
            "eligible_for_structural_features": True,
            "eligibility_scope": "encoding_of_reported_structure_not_reagent_or_stereo_verification",
            "experimental_material_verified": False, "cross_source_pair_verified": False,
            "is_vehicle": False, "fingerprint_row": len(vectors),
            "warnings": "|".join(warnings),
        })
        vectors.append(vector)
        for old in reference.loc[reference.identity_id.eq(description["identity_id"])].to_dict("records"):
            matches.append({"drug_id": source_id, "catalog_number": item["catalog_number"],
                "identity_id": description["identity_id"], "reference_drug_id": old["drug_id"],
                "reference_drug_name": old["drug_name"],
                "reference_identity_status": old["identity_status"],
                "reference_structural_features_eligible": bool(old["eligible_for_structural_features"]),
                "reference_roles": "|".join(role_sets.get(old["chemical_or_experiment_group_id"], [])),
                "relation": "same_full_recorded_structure_not_cross_source_material_or_sample_proof"})
    return pd.DataFrame(records), np.stack(vectors), pd.DataFrame(matches)


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep published versions; choose a new output for changes")
    inputs = [SOURCE, V3 / "chemicals.parquet", V3 / "conditions.parquet", V3 / "manifest.json",
              ROOT / "src/lung/data/drug_identity.py", Path(__file__)]
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    metadata = pd.read_csv(SOURCE, sep="\t")
    frame, fingerprints, matches = build_records(metadata, pd.read_parquet(V3 / "chemicals.parquet"), pd.read_parquet(V3 / "conditions.parquet"))
    if any(digest(ROOT / rel) != expected for rel, expected in before.items()):
        raise ValueError("Source changed during structure extraction")
    OUTPUT.mkdir(parents=True)
    frame.to_csv(OUTPUT / "chemical_identity.csv", index=False)
    frame.to_parquet(OUTPUT / "chemical_identity.parquet", index=False)
    np.save(OUTPUT / "fingerprints.npy", fingerprints, allow_pickle=False)
    matches.to_csv(OUTPUT / "v3_full_matches.csv", index=False)
    stats = {"source_compounds": len(frame), "unique_full_recorded_structures": int(frame.identity_id.nunique()),
             "source_unspecified_stereo_records": int(frame.unspecified_stereo_count.gt(0).sum()),
             "source_multifragment_records": int(frame.fragment_count.gt(1).sum()),
             "missing_InChIKey_records": int(frame.rdkit_inchikey.isna().sum()),
             "source_catalogues_matching_v3_full_structure": int(matches.catalog_number.nunique()),
             "independently_verified_source_materials": 0}
    manifest = {"schema_version": 1, "status": "source_reported_structures_encoded_not_material_verified",
                "input_base": ".", "inputs_sha256": before, "counts": stats,
                "source_id": "sciPlex3:GSM4150378", "fingerprint": {"radius": 2, "bits": 2048,
                    "includeChirality": True, "rdkit_version": rdBase.rdkitVersion,
                    "row_order": frame.drug_id.tolist()},
                "outputs_sha256": {p.name: digest(p) for p in sorted(OUTPUT.iterdir())},
                "input_path_base": "project_root", "formal_training_ready": False}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": stats, "manifest_sha256": digest(OUTPUT / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
