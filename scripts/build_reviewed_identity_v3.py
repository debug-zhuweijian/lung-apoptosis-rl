"""Apply five reviewed nominal references in new versions; keep lots unverified."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.common import inside, sha256, data_file_bindings
from src.lung.data.drug_identity import compare_structures, describe_structure, map_conditions

BASE = ROOT / "08-datasets/LungCancer_A549_Integration"
REVIEW = BASE / "endpoint-alignment-v1/source_reviews/chemical_disambiguation_round2"
REVIEW_HASH = "07e544f38fb14df8133da92599339e6d46ee797fd623f4def626fbc341344c00"
OLD_IDENTITY = BASE / "identity-v2"
OLD_LINKS = BASE / "endpoint-alignment-v1/chemical-links-v1"
NEW_IDENTITY = BASE / "identity-v3"
NEW_LINKS = BASE / "endpoint-alignment-v1/chemical-links-v2"
ADOPT = "adopt_nominal_vendor_reference_in_new_version"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_manifest(path, expected=None):
    if expected and sha256(path) != expected:
        raise ValueError(f"Pinned manifest changed: {path.name}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dependencies = {path.relative_to(ROOT).as_posix(): sha256(path)}
    for name, record in data_file_bindings(manifest["outputs"]).items():
        file = inside(path.parent, name)
        if sha256(file) != record["sha256"] or file.stat().st_size != record["bytes"]:
            raise ValueError(f"Frozen output changed: {file}")
        dependencies[file.relative_to(ROOT).as_posix()] = record["sha256"]
    for name, expected_hash in data_file_bindings(manifest.get("inputs_sha256", manifest.get("frozen_inputs_sha256", {}))).items():
        file = inside(ROOT, name)
        if sha256(file) != expected_hash:
            raise ValueError(f"Frozen input changed: {file}")
        dependencies[file.relative_to(ROOT).as_posix()] = expected_hash
    return manifest, dependencies


def reviewed_reference(review, properties):
    if (review["recommendation"] != ADOPT or not review["nominal_identity_resolved"]
            or review["experimental_reagent_batch_verified"] or review["cross_source_sample_pair_verified"]):
        raise ValueError("Review does not authorize only a nominal identity correction")
    cid = int(review["proposed_pubchem_cid"])
    reference = properties[cid]
    description = describe_structure(reference["SMILES"])
    if (reference["CID"] != cid or description["parse_status"] != "parsed"
            or description["rdkit_inchikey"] != reference["InChIKey"]
            or reference["InChIKey"] != review["proposed_pubchem_inchikey"]
            or description["identity_id"] != review["proposed_identity_id"]):
        raise ValueError("Reviewed reference structure and identifiers disagree")
    return reference, description


def reference_fields(reference, description):
    return {**description, "pubchem_smiles": reference["SMILES"], "pubchem_inchikey": reference["InChIKey"],
            "pubchem_inchi": reference["InChI"], "pubchem_title": reference["Title"],
            "pubchem_molecular_formula": reference["MolecularFormula"],
            "pubchem_molecular_weight": reference["MolecularWeight"],
            "pubchem_undefined_atom_stereo_count": reference["UndefinedAtomStereoCount"],
            "pubchem_undefined_bond_stereo_count": reference["UndefinedBondStereoCount"]}


def evidence_fields(review):
    return {"nominal_manufacturer": review["manufacturer"],
            "nominal_manufacturer_catalogue": review["manufacturer_catalogue"],
            "nominal_manufacturer_cas": review["manufacturer_cas"],
            "nominal_review_manifest": (REVIEW / "manifest.json").relative_to(ROOT).as_posix(),
            "nominal_review_manifest_sha256": REVIEW_HASH,
            "nominal_review_evidence": review["evidence_path"],
            "nominal_review_additional_evidence": review["additional_evidence_path"],
            "nominal_identity_limit": review["remaining_limit"]}


def apply_tahoe_reviews(catalog, fingerprints, reviews, properties, response):
    """Change reviewed full forms only; preserve original material and source IDs."""
    result, fp = catalog.copy(), fingerprints.copy()
    result["previous_identity_id"] = result["identity_id"]
    result["previous_reference_pubchem_cid"] = result["returned_pubchem_cid"]
    result["previous_identity_status"] = result["identity_status"]
    result["previous_material_form_review_reason"] = result["material_form_review_reason"]
    result["reference_pubchem_cid"] = result["returned_pubchem_cid"]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    for review in reviews.to_dict("records"):
        matched = result.index[result.drug_id.eq(review["source_id"])].tolist()
        if len(matched) != 1 or result.loc[matched[0], "identity_status"] != "source_material_form_review_required":
            raise ValueError("Tahoe review must resolve exactly one previously held material")
        reference, description = reviewed_reference(review, properties)
        if not review["vendor_full_structure_matches_proposed"]:
            raise ValueError("Tahoe full form must match the reviewed manufacturer structure")
        index = matched[0]
        updates = {**reference_fields(reference, description), **evidence_fields(review),
                   "returned_pubchem_cid": reference["CID"], "reference_pubchem_cid": reference["CID"],
                   "identity_status": "manufacturer_nominal_form_verified",
                   "eligible_for_structural_features": True, "reference_inchikey_agrees": True,
                   "material_form_review_reason": "nominal_form_resolved_original_lot_not_verified",
                   "source_to_reference_relation": compare_structures(result.loc[index, "source_smiles"], reference["SMILES"]),
                   "source_response_path": response["path"], "source_response_sha256": response["sha256"],
                   "source_url": response["url"], "experimental_material_verified": False,
                   "cross_source_pair_verified": False}
        for key, value in updates.items():
            if key not in result:
                result[key] = pd.Series(None, index=result.index, dtype=object)
            result.at[index, key] = value
        vector = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(reference["SMILES"])), vector)
        fp[int(result.loc[index, "fingerprint_row"])] = vector
    return result, fp


def apply_endpoint_reviews(compounds, reviews, properties):
    result = compounds.copy()
    result["previous_compound_id"] = result["compound_id"]
    result["previous_name_resolution_status"] = result["name_resolution_status"]
    for review in reviews.to_dict("records"):
        matched = result.index[result.source_id.eq(review["source_id"])].tolist()
        if len(matched) != 1:
            raise ValueError("Endpoint review source_id is not unique in original compounds")
        reference, description = reviewed_reference(review, properties)
        updates = {**reference_fields(reference, description), **evidence_fields(review),
                   "compound_id": f"pubchem:{reference['CID']}", "PubChemCID": reference["CID"],
                   "selected_cid": reference["CID"], "name_resolution_status": "manufacturer_nominal_reference_verified",
                   "resolution_route": "reviewed_original_supplier_product_and_CAS",
                   "nominal_structure_resolved": True, "experimental_reagent_form_verified": False,
                   "use_as_paired_training_target": False}
        for key, value in updates.items():
            if key not in result:
                result[key] = pd.Series(None, index=result.index, dtype=object)
            result.at[matched[0], key] = value
    return result


def candidate_links(compounds, tahoe, conditions):
    """Recompute chemical candidates; no link is a paired biological sample."""
    links = []
    for source in compounds.to_dict("records"):
        found = False
        for target in tahoe.loc[~tahoe.is_vehicle].to_dict("records"):
            relation = compare_structures(source.get("canonical_full_smiles"), target.get("canonical_full_smiles"))
            expected = target["drug_name"] == source["candidate_tahoe_drug_name"]
            if relation in ("different_structures", "unresolved") and not expected:
                continue
            found = True
            selected = conditions[conditions.drug_id.eq(target["drug_id"])]
            if source["study"] == "Inde2020":
                selected = selected[selected.dose_value.eq(source["source_dose_um"]) & selected.treatment_hours.eq(source["source_time_hours"])]
            links.append({"source_id": source["source_id"], "source_compound_name": source["source_compound_name"],
                          "study": source["study"], "source_compound_id": source["compound_id"],
                          "tahoe_drug_id": target["drug_id"], "tahoe_drug_name": target["drug_name"],
                          "structure_relation": relation, "original_name_candidate": expected,
                          "source_name_resolution_status": source["name_resolution_status"],
                          "source_nominal_structure_resolved": source["nominal_structure_resolved"],
                          "tahoe_identity_status": target["identity_status"],
                          "tahoe_eligible_for_structural_features": target["eligible_for_structural_features"],
                          "identity_id": source["identity_id"], "tahoe_identity_id": target["identity_id"],
                          "parent_id": source["parent_id"], "tahoe_parent_id": target["parent_id"],
                          "candidate_tahoe_condition_ids": ";".join(selected.condition_id),
                          "condition_selection_scope": "nominal_5uM_24h" if source["study"] == "Inde2020" else "chemical_only_all_doses_times_require_endpoint_join",
                          "background_relation": source["tahoe_background_relation"],
                          "experimental_sample_pair_verified": False, "dose_basis_verified": False,
                          "use_as_paired_training_target": False})
        if not found:
            links.append({"source_id": source["source_id"], "source_compound_name": source["source_compound_name"],
                          "study": source["study"], "source_compound_id": source["compound_id"], "structure_relation": "no_tahoe_structure_link",
                          "source_name_resolution_status": source["name_resolution_status"],
                          "source_nominal_structure_resolved": source["nominal_structure_resolved"],
                          "experimental_sample_pair_verified": False, "dose_basis_verified": False,
                          "use_as_paired_training_target": False})
    result = pd.DataFrame(links)
    result["eligible_nominal_full_structure_link"] = (result.source_nominal_structure_resolved.eq(True)
        & result.tahoe_eligible_for_structural_features.eq(True) & result.structure_relation.eq("same_full_form"))
    return result


def freeze(directory, version, status, inputs, **extra):
    outputs = {p.relative_to(directory).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
               for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"}
    for name, expected in inputs.items():
        if sha256(inside(ROOT, name)) != expected:
            raise ValueError(f"Source changed while building: {name}")
    save_json(directory / "manifest.json", {"version": version, "status": status,
              "created_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": inputs,
              "outputs": outputs, **extra})


def build():
    for output in (NEW_IDENTITY, NEW_LINKS):
        if output.exists():
            raise FileExistsError(f"New version already exists; refuse overwrite: {output}")
    old_manifest, old_inputs = verify_manifest(OLD_IDENTITY / "manifest.json")
    _, link_inputs = verify_manifest(OLD_LINKS / "manifest.json")
    _, review_inputs = verify_manifest(REVIEW / "manifest.json", REVIEW_HASH)
    reviews = pd.read_parquet(REVIEW / "proposed_identity_corrections.parquet")
    accepted = reviews[reviews.recommendation.eq(ADOPT)]
    expected = {"tahoe_drug:dc132dbaa1e619ee": 50992434, "tahoe_drug:188f50973e570f3a": 129893299,
                "tahoe_drug:7f5c4efd913d4b7d": 38101, "Inde2020:Dexamethasone (DHAP)": 5743,
                "Inde2020:Temsirolimus (CCI-779, NSC 683864)": 6918289}
    if dict(zip(accepted.source_id, accepted.proposed_pubchem_cid.astype(int))) != expected:
        raise ValueError("Pinned review does not contain exactly the five approved corrections")
    proposal = json.loads((REVIEW / "proposed_reference_properties.json").read_text(encoding="utf-8"))
    properties = {item["CID"]: item for item in proposal["properties"]}
    response = {**proposal["evidence"], "path": "raw_pubchem/reviewed_nominal_properties.json"}
    catalog, fp = apply_tahoe_reviews(pd.read_parquet(OLD_IDENTITY / "drug_identity.parquet"),
        np.load(OLD_IDENTITY / "morgan2048_chiral.npy", allow_pickle=False),
        accepted[accepted.source_namespace.eq("Tahoe-100M")], properties, response)
    original_conditions = pd.read_parquet(BASE / "prepared/v1/tahoe/conditions.parquet")
    conditions = map_conditions(original_conditions, catalog).assign(identity_version="identity-v3")
    pd.testing.assert_frame_equal(conditions[original_conditions.columns], original_conditions)
    compounds = apply_endpoint_reviews(pd.read_parquet(OLD_LINKS / "endpoint_compounds.parquet"),
        accepted[accepted.source_namespace.eq("Inde2020")], properties)
    links = candidate_links(compounds, catalog, conditions)
    for directory in (NEW_IDENTITY, NEW_LINKS):
        directory.mkdir(parents=True)
    shutil.copytree(OLD_IDENTITY / "raw_pubchem", NEW_IDENTITY / "raw_pubchem")
    shutil.copy2(inside(REVIEW, proposal["evidence"]["path"]), NEW_IDENTITY / response["path"])
    sources = json.loads((OLD_IDENTITY / "sources.json").read_text(encoding="utf-8")) + [response]
    for name, frame in (("drug_identity", catalog), ("condition_identity", conditions)):
        frame.to_parquet(NEW_IDENTITY / f"{name}.parquet", index=False)
        frame.to_csv(NEW_IDENTITY / f"{name}.csv", index=False, encoding="utf-8-sig")
    np.save(NEW_IDENTITY / "morgan2048_chiral.npy", fp, allow_pickle=False)
    catalog[~catalog.eligible_for_structural_features & ~catalog.is_vehicle].to_csv(NEW_IDENTITY / "identity_review_required.csv", index=False)
    summary = {"version": "identity-v3", "mapped_conditions": len(conditions), "source_condition_ids_preserved": True,
               "feature_eligible_drugs": int(catalog.eligible_for_structural_features.sum()),
               "identity_status_counts": catalog.identity_status.value_counts().to_dict(),
               "held_treated_conditions": int((conditions.condition_type.eq("treated") & ~conditions.eligible_for_structural_features).sum()),
               "new_nominal_references": 3, "experimental_lots_verified": 0, "experimental_sample_pairs_verified": 0}
    save_json(NEW_IDENTITY / "sources.json", sources)
    save_json(NEW_IDENTITY / "summary.json", summary)
    inputs = {**old_inputs, **review_inputs}
    freeze(NEW_IDENTITY, "identity-v3", old_manifest["status"], inputs, raw_response_count=len(sources),
           review_manifest_sha256=REVIEW_HASH, prior_identity_manifest_sha256=sha256(OLD_IDENTITY / "manifest.json"))
    # Preserve old lookup paths; new manufacturer evidence is bound by inputs.
    shutil.copytree(OLD_LINKS / "raw_pubchem", NEW_LINKS / "raw_pubchem")
    for name, frame in (("endpoint_compounds", compounds), ("tahoe_candidate_links", links)):
        frame.to_parquet(NEW_LINKS / f"{name}.parquet", index=False)
        frame.to_csv(NEW_LINKS / f"{name}.csv", index=False, encoding="utf-8-sig")
    link_summary = {"version": "chemical-links-v2", "source_compounds": len(compounds),
                    "nominal_structure_resolved": int(compounds.nominal_structure_resolved.sum()),
                    "eligible_nominal_full_structure_sources_by_study": links[links.eligible_nominal_full_structure_link].groupby("study").source_id.nunique().to_dict(),
                    "link_relation_counts": links.structure_relation.value_counts().to_dict(),
                    "experimental_sample_pairs_verified": 0, "review_manifest_sha256": REVIEW_HASH,
                    "tahoe_identity_manifest_sha256": sha256(NEW_IDENTITY / "manifest.json")}
    save_json(NEW_LINKS / "summary.json", link_summary)
    _, new_inputs = verify_manifest(NEW_IDENTITY / "manifest.json")
    freeze(NEW_LINKS, "chemical-links-v2", "nominal_reference_links_with_explicit_evidence_boundaries",
           {**link_inputs, **review_inputs, **new_inputs}, review_manifest_sha256=REVIEW_HASH)
    config = json.loads((ROOT / "configs/data_identity_v2.json").read_text(encoding="utf-8"))
    config = {key: value.replace("training-identity-v2", "training-identity-v3") if isinstance(value, str) else value
              for key, value in config.items()}
    config.update(drug_identity_manifest=(NEW_IDENTITY / "manifest.json").relative_to(ROOT).as_posix(),
                  drug_identity_manifest_sha256=sha256(NEW_IDENTITY / "manifest.json"))
    config_path = ROOT / "configs/data_identity_v3.json"
    if config_path.exists():
        raise FileExistsError("Refuse to overwrite existing identity-v3 data config")
    save_json(config_path, config)
    return {"identity": summary, "links": link_summary, "identity_manifest_sha256": sha256(NEW_IDENTITY / "manifest.json"),
            "links_manifest_sha256": sha256(NEW_LINKS / "manifest.json")}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
