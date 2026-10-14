"""Build the narrow, source-bound GSE279184 nominal chemical-condition sidecar."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gse279184_drug_identity", ROOT / "src/lung/data/drug_identity.py")
_identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_identity)
describe_structure = _identity.describe_structure

BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round7"
PACKAGE = BASE / "GSE279184_condition_inputs_v1"
EXPRESSION = BASE / "GSE279184_common_axis_v1"
EXPECTED_EXPRESSION_MANIFEST = "ee4af0127093cc6792d7470da6454671fb9797ce693d156aaffe49b9ca59ee52"
SCALE = "author_deposited_processed_count_table"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_hashes(manifest, output_base):
    count = 0
    for key, base in (("inputs_sha256", ROOT), ("outputs_sha256", output_base)):
        for name, expected in manifest[key].items():
            if sha256(base / name) != expected:
                raise ValueError(f"Hash mismatch: {name}")
            count += 1
    return count


def official_structures():
    medkoo = BeautifulSoup((PACKAGE / "raw/MedKoo_201931_Alisertib.html").read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)
    mce = (PACKAGE / "MCE_official_web_chemical_excerpt.txt").read_text(encoding="utf-8")
    assert "201931" in medkoo and "1028486-01-2 (free acid)" in medkoo
    assert "C27H20ClFN4O4" in medkoo and "53003-10-4" in mce and "C_{42}H_{70}O_{11}" in mce
    return [
        {"material_id": "MedKoo:201931", "compound": "alisertib", "supplier": "MedKoo Biosciences Inc", "catalog_number": "201931", "CAS": "1028486-01-2",
         "source_smiles": re.search(r"SMILES Code\s+(\S+)", medkoo).group(1), "vendor_formula": "C27H20ClFN4O4", "vendor_molecular_weight": 518.90,
         "vendor_inchikey": re.search(r"InChi Key\s+(\S+)", medkoo).group(1), "source_url": "https://www.medkoo.com/products/5202",
         "structure_source_file": "raw/MedKoo_201931_Alisertib.html", "structure_locator": "Technical Data / SMILES Code; Theoretical Analysis / CAS free acid",
         "source_capture_kind": "original_official_HTML", "formula_confirmation_file": "raw/MedKoo_Alisertib_product_data.pdf", "form_basis": "official catalog explicitly free acid; related sodium and sodium hydrate are separate CAS entries"},
        {"material_id": "MCE:HY-15597", "compound": "salinomycin", "supplier": "MedChemExpress", "catalog_number": "HY-15597", "CAS": "53003-10-4",
         "source_smiles": re.search(r"^L5263:\s+(\S+)", mce, re.M).group(1), "vendor_formula": "C42H70O11", "vendor_molecular_weight": 751.00,
         "vendor_inchikey": None, "source_url": "https://www.medchemexpress.com/Salinomycin.html",
         "structure_source_file": "MCE_official_web_chemical_excerpt.txt", "structure_locator": "Chemical Information / SMILES; official datasheet page 1 for catalog, CAS, formula",
         "source_capture_kind": "official_page_text_extracted_by_web_tool_not_raw_HTML", "formula_confirmation_file": "raw/MCE_HY15597_Salinomycin_datasheet.pdf", "form_basis": "official formula and full neutral carboxylic-acid SMILES; no Na or separate water fragment represented"},
    ]


def build():
    if (PACKAGE / "manifest.json").exists():
        raise FileExistsError("The delivered v1 exists; use --verify-only or build a new version.")
    assert sha256(EXPRESSION / "manifest.json") == EXPECTED_EXPRESSION_MANIFEST
    upstream = json.loads((EXPRESSION / "manifest.json").read_text(encoding="utf-8"))
    upstream_hash_count = verify_hashes(upstream, EXPRESSION)
    acquisition = json.loads((PACKAGE / "acquisition.json").read_text(encoding="utf-8"))
    for item in acquisition:
        assert sha256(PACKAGE / item["file"]) == item["sha256"]
    for name in ("MedKoo_201931_Alisertib.html", "MedKoo_Alisertib_product_data.pdf", "MCE_HY15597_Salinomycin_datasheet.pdf"):
        assert any(a["file"] == "raw/" + name and a["status"] == 200 for a in acquisition)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    chemicals, fingerprints = [], []
    for row_index, source in enumerate(official_structures()):
        described = describe_structure(source["source_smiles"])
        mol = Chem.MolFromSmiles(source["source_smiles"])
        assert described["parse_status"] == "parsed" and described["fragment_count"] == 1
        assert described["formal_charge"] == 0 and described["unspecified_stereo_count"] == 0
        assert all(atom.GetAtomicNum() != 0 for atom in mol.GetAtoms())
        assert rdMolDescriptors.CalcMolFormula(mol) == source["vendor_formula"]
        if source["vendor_inchikey"]:
            assert source["vendor_inchikey"] == described["rdkit_inchikey"]
        vector = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
        fingerprints.append(vector)
        chemicals.append({**source, **described, "computed_formula": rdMolDescriptors.CalcMolFormula(mol),
                          "nominal_material_form": "free_acid", "counterion_in_vendor_structure": "none", "hydrate_in_vendor_structure": "none",
                          "source_structure_sha256": sha256(PACKAGE / source["structure_source_file"]),
                          "formula_confirmation_sha256": sha256(PACKAGE / source["formula_confirmation_file"]),
                          "material_identity_scope": "current_official_nominal_product_for_paper_catalog_number",
                          "historical_lot_verified": False, "batch_solvent_or_water_content_verified": False,
                          "eligible_for_structural_features": True, "fingerprint_row": row_index})
    matrix = np.stack(fingerprints)
    np.save(PACKAGE / "morgan2048_chiral.npy", matrix, allow_pickle=False)
    write_json(PACKAGE / "chemical_identities.json", chemicals)
    pd.DataFrame(chemicals).to_csv(PACKAGE / "chemical_identities.csv", index=False)
    write_json(PACKAGE / "fingerprint_metadata.json", {"algorithm": "RDKit Morgan bit vector", "radius": 2, "fp_size": 2048,
               "include_chirality": True, "dtype": "uint8", "shape": [2, 2048], "rdkit_version": rdBase.rdkitVersion,
               "input_form": "complete_official_SMILES_no_fragment_removal_or_reionization", "rows": [c["material_id"] for c in chemicals]})

    samples = pd.read_csv(EXPRESSION / "samples.csv", keep_default_na=False)
    assert samples.sample_id.tolist() == [f"GSM{n}" for n in range(8563730, 8563736)]
    assert samples.deposited_count_column.tolist() == ["ut_r1", "ut_r2", "ut_r3", "tr_r1", "tr_r2", "tr_r3"]
    assert set(samples.state) == {"alisertib_induced_senescent"}
    records = []
    for row in samples.to_dict("records"):
        treated = row["deposited_count_column"].startswith("tr_")
        assert float(row["induction_uM"]) == 0.5 and int(row["induction_days"]) == 7
        assert float(row["salinomycin_uM"]) == (2.5 if treated else 0.0)
        assert (float(row["salinomycin_treatment_h"]) == 48) if treated else row["salinomycin_treatment_h"] == ""
        records.append({"sample_id": row["sample_id"], "expression_column": row["deposited_count_column"],
                        "author_title": row["author_title"], "biological_replicate_author_label": int(row["biological_replicate_author_label"]),
                        "column_mapping_basis": row["column_mapping_basis"], "cell_line": "A549", "state_label": row["state"],
                        "state_label_basis": "study_protocol_and_GEO_metadata_not_measured_prechallenge_RNA",
                        "initial_induction_history": {"material_id": chemicals[0]["material_id"], "identity_id": chemicals[0]["identity_id"],
                            "fingerprint_row": 0, "dose_uM": 0.5, "duration_days": 7, "timing_precision": "nominal_study_protocol_7_days_not_per_sample_timestamp", "evidence": "PMC11670111 s10,s14"},
                        "pre_challenge_handling": {"protocol": ["PBS_wash", "trypsinization", "reseeding"] if treated else None,
                            "evidence": "PMC11670111 s14: in case of additional treatment", "per_sample_execution_verified": False,
                            "UT_handling_not_independently_reported": not treated, "drug_clearance_measured": False},
                        "expression_collection_condition": {"group": "salinomycin_treated" if treated else "UT_parallel_reference",
                            "material_id": chemicals[1]["material_id"] if treated else None, "identity_id": chemicals[1]["identity_id"] if treated else None,
                            "fingerprint_row": 1 if treated else None, "dose_uM": 2.5 if treated else 0.0,
                            "dose_interpretation": "author_reported_nominal_salinomycin" if treated else "no_salinomycin_not_vehicle_concentration",
                            "duration_h": 48.0 if treated else None, "duration_status": "reported" if treated else "unknown_not_imputed",
                            "time_provenance": row["time_provenance"], "vehicle_identity": None},
                        "reference_relation": "three_UT_parallel_group_references_no_R1_to_R1_pairing", "measured_t0": False,
                        "RNA_to_WB_or_Annexin_pair_created": False, "measurement_scale": SCALE,
                        "raw_unmodified_count_matrix_verified": False, "transform_applied": "none"})
    write_json(PACKAGE / "condition_inputs.json", records)
    flat = [{"sample_id": r["sample_id"], "expression_column": r["expression_column"], "state_label": r["state_label"],
             "induction_material_id": r["initial_induction_history"]["material_id"], "induction_dose_uM": 0.5, "induction_duration_days": 7,
             **{f"challenge_{k}": v for k, v in r["expression_collection_condition"].items()}} for r in records]
    pd.DataFrame(flat).to_csv(PACKAGE / "sample_conditions.csv", index=False)

    saved = np.load(PACKAGE / "morgan2048_chiral.npy", allow_pickle=False)
    for row, c in enumerate(json.loads((PACKAGE / "chemical_identities.json").read_text(encoding="utf-8"))):
        independent = np.asarray(generator.GetFingerprint(Chem.MolFromSmiles(c["canonical_full_smiles"])), dtype=np.uint8)
        assert np.array_equal(saved[row], independent)
    roundtrip = json.loads((PACKAGE / "condition_inputs.json").read_text(encoding="utf-8"))
    assert roundtrip == records and all(r["expression_collection_condition"]["duration_h"] is None for r in roundtrip[:3])
    assert all(r["expression_collection_condition"]["fingerprint_row"] is None for r in roundtrip[:3])
    assert all(r["expression_collection_condition"]["material_id"] == "MCE:HY-15597" for r in roundtrip[3:])
    validation = {"upstream_expression_manifest_sha256": EXPECTED_EXPRESSION_MANIFEST, "upstream_dependency_hashes_verified": upstream_hash_count,
                  "chemical_materials": 2, "fingerprint_shape": list(saved.shape), "fingerprints_recomputed_from_canonical_full_structures_equal": True,
                  "samples": 6, "UT_samples": 3, "treated_samples": 3, "UT_durations_remain_null": True,
                  "sample_json_roundtrip_exact": True, "functional_response_labels_added": 0, "normalization_or_training_performed": False}
    write_json(PACKAGE / "validation.json", validation)
    shutil.copy2(__file__, PACKAGE / "build_gse279184_condition_inputs_snapshot.py")
    helper = ROOT / "src/lung/data/drug_identity.py"
    shutil.copy2(helper, PACKAGE / "drug_identity_snapshot.py")
    inputs = [EXPRESSION / "manifest.json", EXPRESSION / "samples.csv", BASE / "induced/PMC11670111.xml", helper, Path(__file__),
              PACKAGE / "acquisition.json", PACKAGE / "MCE_official_web_chemical_excerpt.txt"] + [PACKAGE / a["file"] for a in acquisition]
    outputs = ["chemical_identities.json", "chemical_identities.csv", "morgan2048_chiral.npy", "fingerprint_metadata.json", "condition_inputs.json", "sample_conditions.csv",
               "validation.json", "source_evidence.json", "README.md", "build_gse279184_condition_inputs_snapshot.py", "drug_identity_snapshot.py"]
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "nominal_chemical_condition_identity_ready_with_UT_timing_unknown",
                "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
                "outputs_sha256": {name: sha256(PACKAGE / name) for name in outputs}, "validation": validation,
                "expression_condition_chemical_identity_ready": True, "all_protocol_timing_known": False,
                "initial_state_expression_measured_at_t0": False, "single_cell_consumer_ready": False,
                "Annexin_pairing_ready": False, "formal_training_ready": False, "historical_material_lot_verified": False,
                "measurement_scale": SCALE, "raw_unmodified_count_matrix_verified": False, "transform_applied": "none",
                "current_product_structures_are_nominal_references_not_historical_lot_identification": True}
    write_json(PACKAGE / "manifest.json", manifest)
    print(json.dumps({"manifest_sha256": sha256(PACKAGE / "manifest.json"), **validation}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps({"bound_hashes_verified": verify_hashes(manifest, PACKAGE), "manifest_sha256": sha256(PACKAGE / "manifest.json")}, ensure_ascii=False))
    else:
        build()
