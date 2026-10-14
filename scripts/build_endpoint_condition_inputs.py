"""Join reviewed endpoint conditions to source-specific chemical structures.

Exports real input tensors and unresolved reasons, never fabricated RNA features.
The output is a data-preparation artifact, not an endpoint-head training manifest.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.endpoint_alignment import file_hash, load_alignment
from lung.data.drug_identity import describe_structure

BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1"


def reviewed_additions(directory):
    """Read a frozen primary-source chemical review, including held identities."""
    directory = directory if directory.is_absolute() else ROOT / directory
    manifest_path = directory / "manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths, checked_reviews = [], set()

    def verify_review(review_path):
        review_path = review_path.resolve()
        if review_path in checked_reviews:
            return
        checked_reviews.add(review_path)
        paths.append(review_path)
        review = json.loads(review_path.read_text(encoding="utf-8"))
        input_base = review_path.parent / review["input_base"] if "input_base" in review else ROOT
        for base, entries in [(input_base, review["inputs_sha256"]),
                              (review_path.parent, review["outputs_sha256"])]:
            for name, digest in entries.items():
                path = (base / name).resolve()
                if file_hash(path) != digest:
                    raise ValueError("Additional chemical review source changed: " + str(path))
                paths.append(path)
                if path.name == "manifest.json":
                    verify_review(path)

    verify_review(manifest_path)
    if "chemical_candidates.csv" not in meta["outputs_sha256"]:
        raise ValueError("Additional chemical candidate table is not hash-bound")
    rows = pd.read_csv(directory / "chemical_candidates.csv")
    if rows.source_id.duplicated().any():
        raise ValueError("Additional review contains duplicate source compound IDs")
    registry = {}
    for row in rows.to_dict("records"):
        flags = [row[k] for k in ("eligible_for_structural_features", "primary_structure_identity_verified")]
        if not all(isinstance(v, (bool, np.bool_)) for v in flags):
            raise ValueError("Additional review eligibility must be explicit boolean")
        eligible = all(flags)
        if eligible:
            structure = describe_structure(row["canonical_full_smiles"])
            molecule = Chem.MolFromSmiles(row["canonical_full_smiles"])
            if (structure["identity_id"] != row["identity_id"]
                    or molecule is None or Chem.MolToInchiKey(molecule) != row["rdkit_inchikey"]):
                raise ValueError("Additional primary structure identity is inconsistent")
        registry[row["source_id"]] = {
            "structure": row["canonical_full_smiles"], "eligible": eligible,
            "basis": row["status"],
            "warning": row["status"] + "; nominal primary representation, not experimental lot verification",
        }
    return registry, paths


def main(output, chemical_links=BASE / "chemical-links-v1", alignment=BASE / "curated-20261010-r3",
         additional_review=None):
    output = output if output.is_absolute() else ROOT / output
    if (output / "manifest.json").exists():
        raise FileExistsError("Condition inputs already frozen; use a new version")
    alignment = alignment if alignment.is_absolute() else ROOT / alignment
    aligned = load_alignment(alignment, ROOT)
    chemical_links = chemical_links if chemical_links.is_absolute() else ROOT / chemical_links
    external_path = chemical_links / "endpoint_compounds.parquet"
    external_manifest = chemical_links / "manifest.json"
    if not external_manifest.exists():
        raise FileNotFoundError("External chemical review has not finished")
    external_meta = json.loads(external_manifest.read_text(encoding="utf-8"))
    if file_hash(external_path) != external_meta["outputs"][external_path.name]["sha256"]:
        raise ValueError("External chemical registry changed after review")
    external = pd.read_parquet(external_path)
    registry = {}
    for row in external.to_dict("records"):
        registry[row["source_id"]] = {
            "structure": row["canonical_full_smiles"],
            "eligible": bool(row["nominal_structure_resolved"]),
            "basis": row["name_resolution_status"],
            "warning": "nominal name identity; exact experimental formulation unverified",
        }
    mph_path = BASE / "source_reviews/mphahlele/chemical_registry.csv"
    mph_review_path = BASE / "source_reviews/mphahlele/structure_transcription_review_20261010.json"
    mph_review = json.loads(mph_review_path.read_text(encoding="utf-8"))
    if file_hash(mph_path) != mph_review["reviewed_file_sha256"]:
        raise ValueError("Mphahlele registry changed after independent structure review")
    for row in pd.read_csv(mph_path).to_dict("records"):
        if file_hash(ROOT / row["source_file"]) != row["source_sha256"]:
            raise ValueError("Mphahlele structural source changed after transcription")
        registry[row["compound_id"]] = {
            "structure": row["canonical_isomeric_smiles"],
            "eligible": (bool(row["identity_approved_for_features"])
                         and row["source_compound_id"] in mph_review["approved_name_structure_transcriptions"]),
            "basis": "source_systematic_name_structure_transcription_independently_reviewed",
            "warning": row["source_quality_warning"] if pd.notna(row["source_quality_warning"]) else "",
        }
    puri_path = BASE / "source_reviews/chembl/raw/jm2c01612_si_001.csv"
    source = pd.read_csv(puri_path, sep=";", encoding="cp1252", usecols=["Compound", "SMILES"],
                         dtype=str).dropna(subset=["Compound"])
    selected = source[source.Compound.eq("24")]
    if len(selected) != 1:
        raise ValueError("Author Compound24 is not unique")
    smiles = selected.iloc[0].SMILES
    if Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) != "RAJPEVIHJCKTGX-RVDMUPIBSA-N":
        raise ValueError("Author Compound24 structure differs from reviewed primary structure")
    registry["MCT1_2023:24"] = {"structure": smiles, "eligible": True,
                                  "basis": "author_supplementary_SMILES_exactly_verified", "warning": ""}
    additional_paths = []
    if additional_review is not None:
        additions, additional_paths = reviewed_additions(additional_review)
        if set(additions) & set(registry):
            raise ValueError("Additional chemical review would override an existing registry entry")
        registry.update(additions)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    structures, fingerprints = [], []
    for compound in sorted(set(aligned["interventions"].compound_id)):
        info = registry.get(compound)
        vector = np.zeros(2048, dtype=np.uint8)
        valid = bool(info and info["eligible"] and isinstance(info["structure"], str))
        description = describe_structure(info["structure"] if valid else None)
        if valid:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(info["structure"])), vector)
        structures.append({"compound_id": compound, "fingerprint_row": len(fingerprints),
                           "structure_eligible": valid,
                           "structure_basis": info["basis"] if info else "not_reviewed",
                           "warning": info["warning"] if info else "chemical identity unresolved",
                           "identity_id": description["identity_id"],
                           "canonical_full_smiles": description["canonical_full_smiles"]})
        fingerprints.append(vector)
    compounds = pd.DataFrame(structures)
    components = aligned["interventions"].merge(compounds, on="compound_id", validate="many_to_one")
    observations = aligned["observations"]
    usable = set(observations.loc[observations.exclusion_reason.fillna("").eq("") &
                                  observations.measurement_status.eq("observed"), "condition_id"])
    requests = []
    for row in aligned["conditions"].to_dict("records"):
        parts = components[components.condition_id == row["condition_id"]]
        reasons = []
        if row["condition_id"] not in usable:
            reasons.append("no_eligible_measured_endpoint")
        if not row["is_control"] and not parts.structure_eligible.all():
            reasons.append("unresolved_chemical_identity")
        if len(parts) > 1:
            reasons.append("simultaneous_combination_not_current_single_drug_generator")
        if row["treatment_hours"] != 24:
            reasons.append("time_not_supported_by_current_Tahoe24h_generator")
        if row["is_control"]:
            reasons.append("source_control_to_reference_control_transfer_not_validated")
        requests.append({**row, "component_count": len(parts),
                         "chemical_inputs_available": bool(len(parts) and parts.structure_eligible.all()),
                         "input_contract_blockers": ";".join(reasons),
                         "requires_background_transfer_validation": row["cell_line"] != "A549",
                         "expression_feature_status": "not_generated",
                         "generator_checkpoint": None,
                         "label_feature_pairing": "condition_level_proposed_prediction_not_measured_RNA_pair",
                         "formal_training_ready": False})
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in [("compounds", compounds), ("components", components),
                        ("condition_requests", pd.DataFrame(requests))]:
        path = output / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
        files[path.name] = file_hash(path)
    path = output / "fingerprints.npy"
    np.save(path, np.stack(fingerprints), allow_pickle=False)
    files[path.name] = file_hash(path)
    dependencies = [alignment / "manifest.json", external_manifest, external_path,
                    mph_path, mph_review_path, puri_path, *additional_paths]
    result = {"schema_version": 1, "status": "condition_input_alignment_with_explicit_unsupported_cases",
              "conditions": len(requests), "compounds": len(compounds),
              "structure_resolved_compounds": int(compounds.structure_eligible.sum()),
              "conditions_with_chemical_inputs": sum(r["chemical_inputs_available"] for r in requests),
              "formal_training_ready": False, "files": files,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in dependencies}}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"files", "inputs_sha256"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "condition-inputs-v2")
    parser.add_argument("--chemical-links", type=Path, default=BASE / "chemical-links-v1")
    parser.add_argument("--alignment", type=Path, default=BASE / "curated-20261010-r3")
    parser.add_argument("--additional-chemical-review", type=Path)
    args = parser.parse_args()
    main(args.output, args.chemical_links, args.alignment, args.additional_chemical_review)
