"""Keep the S4246 relative-stereo annotation out of single-structure inputs."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_sciplex_structure_sidecar as v1

REVIEW = v1.OUTPUT.parent / "chemical_review_S4246"
OUTPUT = v1.OUTPUT.parent / "chemical_inputs_v2"
SDF = REVIEW / "independent_Selleck_S4246.sdf"
RELATIVE_ID = "source_relative_stereo:sciPlex3:GSM4150378:S4246"


def relative_candidates(source_smiles, sdf_path):
    """Two absolute interpretations of the recorded relative relationship, not a mixture."""
    records = list(Chem.ForwardSDMolSupplier(io.BytesIO(Path(sdf_path).read_bytes()), removeHs=False))
    if len(records) != 1 or records[0] is None:
        raise ValueError("Expected one reviewed S4246 SDF record")
    mol = records[0]
    dummy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if (mol.GetProp("Catalog Number") != "S4246" or mol.GetProp("CAS Number") != "4548-34-9"
            or len(dummy) != 1 or dummy[0].GetDegree() != 0
            or not dummy[0].HasProp("molFileAlias")
            or dummy[0].GetProp("molFileAlias") != "relative stereochemistry"
            or Chem.MolToSmiles(mol) != Chem.MolToSmiles(Chem.MolFromSmiles(source_smiles))):
        raise ValueError("S4246 source and relative-stereo SDF evidence differ")
    edited = Chem.RWMol(mol)
    edited.RemoveAtom(dummy[0].GetIdx())
    original = edited.GetMol()
    Chem.SanitizeMol(original)
    mirror = Chem.Mol(original)
    centres = [atom for atom in mirror.GetAtoms() if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    if len(centres) != 2:
        raise ValueError("Reviewed S4246 must have two depicted centres")
    for atom in centres:
        atom.InvertChirality()
    descriptions = [v1.describe_structure(Chem.MolToSmiles(item)) for item in [original, mirror]]
    if len({row["identity_id"] for row in descriptions}) != 2:
        raise ValueError("S4246 interpretations must be distinct mirror structures")
    return descriptions


def build_records(metadata, reference, conditions, sdf_path=SDF):
    frame, fingerprints, matches = v1.build_records(metadata, reference, conditions)
    candidates, candidate_vectors = [], []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    for index, row in frame.iterrows():
        mol = Chem.MolFromSmiles(row.canonical_full_smiles)
        if not any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            continue
        if row.catalog_number != "S4246":
            raise ValueError("Unreviewed dummy atom cannot enter a structural feature")
        descriptions = relative_candidates(row.source_smiles, sdf_path)
        for number, description in enumerate(descriptions):
            vector = np.zeros(2048, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(
                Chem.MolFromSmiles(description["canonical_full_smiles"])), vector)
            candidate_vectors.append(vector)
            candidates.append({"source_drug_id": row.drug_id, "relative_entity_id": RELATIVE_ID,
                "interpretation_id": f"S4246:absolute-interpretation-{number + 1}", **description,
                "fingerprint_row": number, "mixture_weight": None, "observed_material_identity": False,
                "meaning": "alternative_absolute_interpretation_not_an_additional_experiment"})
        replacement = {"identity_id": RELATIVE_ID, "chemical_or_experiment_group_id": RELATIVE_ID,
            "identity_status": "nominal_relative_stereochemistry_verified_absolute_composition_unspecified",
            "eligible_for_structural_features": False,
            "eligibility_scope": "pending_uncertainty_aware_representation_for_single_structure_model",
            "warnings": str(row.warnings) + "|relative_stereo_annotation_preserved|absolute_composition_unspecified|CAS_source_web_conflict"}
        for key, value in replacement.items():
            frame.at[index, key] = value
    usable = frame.eligible_for_structural_features
    selected = fingerprints[frame.loc[usable, "fingerprint_row"].to_numpy(dtype=int)]
    frame = frame.assign(identity_version="sciplex-source-structures-v2", fingerprint_row=-1,
                         absolute_interpretation_candidates_file=None)
    frame.loc[usable, "fingerprint_row"] = np.arange(int(usable.sum()))
    frame.loc[~usable, "absolute_interpretation_candidates_file"] = "relative_stereo_candidates.csv"
    variants = np.stack(candidate_vectors) if candidate_vectors else np.empty((0, 2048), dtype=np.uint8)
    return frame, selected, matches, pd.DataFrame(candidates), variants


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep published versions; choose a new output for changes")
    inputs = [v1.SOURCE, v1.V3 / "chemicals.parquet", v1.V3 / "conditions.parquet", v1.V3 / "manifest.json",
              ROOT / "src/lung/data/drug_identity.py", Path(v1.__file__), Path(__file__), SDF,
              REVIEW / "independent_review.md", REVIEW / "independent_acquisition.json",
              REVIEW / "independent_structure_comparison.json", REVIEW / "PubChem_CAS4548-34-9.json"]
    before = {path.relative_to(ROOT).as_posix(): v1.digest(path) for path in inputs}
    frame, fp, matches, candidates, variants = build_records(pd.read_csv(v1.SOURCE, sep="\t"),
        pd.read_parquet(v1.V3 / "chemicals.parquet"), pd.read_parquet(v1.V3 / "conditions.parquet"))
    if any(v1.digest(ROOT / path) != expected for path, expected in before.items()):
        raise ValueError("Source changed during structure extraction")
    OUTPUT.mkdir(parents=True)
    frame.to_csv(OUTPUT / "chemical_identity.csv", index=False)
    frame.to_parquet(OUTPUT / "chemical_identity.parquet", index=False)
    np.save(OUTPUT / "fingerprints.npy", fp, allow_pickle=False)
    matches.to_csv(OUTPUT / "v3_full_matches.csv", index=False)
    candidates.to_csv(OUTPUT / "relative_stereo_candidates.csv", index=False)
    np.save(OUTPUT / "relative_stereo_candidate_fingerprints.npy", variants, allow_pickle=False)
    counts = {"source_compounds": len(frame), "single_structure_feature_eligible": int(frame.eligible_for_structural_features.sum()),
              "retained_relative_stereo_entities_pending_representation": int((~frame.eligible_for_structural_features).sum()),
              "alternative_absolute_interpretations_not_experiments": len(candidates),
              "source_catalogues_matching_v3_full_structure": int(matches.catalog_number.nunique()),
              "independently_verified_source_materials": 0}
    manifest = {"schema_version": 2, "status": "source_reported_structures_with_reviewed_relative_stereo_exception",
        "source_id": "sciPlex3:GSM4150378", "supersedes_for_consumption": "chemical_inputs_v1",
        "input_path_base": "project_root", "input_base": ".", "inputs_sha256": before, "counts": counts,
        "fingerprint": {"radius": 2, "bits": 2048, "includeChirality": True, "rdkit_version": rdBase.rdkitVersion,
                        "row_order": frame.loc[frame.eligible_for_structural_features, "drug_id"].tolist(),
                        "ineligible_row": -1, "ineligible_rows_must_not_be_indexed": True},
        "relative_stereo_policy": "retain_source_record_and_two_unweighted_interpretations; no_single_structure_substitution",
        "formal_training_ready": False, "outputs_sha256": {path.name: v1.digest(path) for path in sorted(OUTPUT.iterdir())}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "manifest_sha256": v1.digest(OUTPUT / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
