"""Fetch original Tahoe CIDs and build an isolated, auditable identity version."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import requests
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.data.drug_identity import build_identity_catalog, map_conditions

REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
DEFAULT_OUTPUT = ROOT / "08-datasets/LungCancer_A549_Integration/identity-v2"
PROPERTIES = ("IsomericSMILES,CanonicalSMILES,InChIKey,InChI,Title,MolecularFormula,"
              "MolecularWeight,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,"
              "BondStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_properties(cids: list[int], output: Path, offline: bool):
    raw = output / "raw_pubchem"
    raw.mkdir(parents=True, exist_ok=True)
    properties, sources = {}, []
    for offset in range(0, len(cids), 50):
        batch = cids[offset:offset + 50]
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, batch))}/property/{PROPERTIES}/JSON"
        path = raw / f"properties_{offset:04d}.json"
        metadata = path.with_suffix(".provenance.json")
        if path.exists() and metadata.exists():
            source = json.loads(metadata.read_text(encoding="utf-8"))
            if source["requested_cids"] != batch or source["sha256"] != sha256(path):
                raise ValueError(f"Cached response provenance/hash mismatch: {path}")
        else:
            if offline:
                raise FileNotFoundError(f"Offline source response missing: {path}")
            for attempt in range(3):
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    break
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
            response.raise_for_status()
            payload = response.json()
            if "PropertyTable" not in payload:
                raise ValueError("PubChem response lacks PropertyTable")
            path.write_bytes(response.content)
            source = {
                "url": url, "requested_cids": batch, "http_status": response.status_code,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "server_date": response.headers.get("Date"), "sha256": sha256(path),
                "bytes": path.stat().st_size, "path": str(path.relative_to(output)),
            }
            write_json(metadata, source)
            time.sleep(0.25)
        rows = json.loads(path.read_text(encoding="utf-8"))["PropertyTable"]["Properties"]
        returned = {int(row["CID"]): row for row in rows}
        if len(returned) != len(rows) or set(returned) != set(batch):
            raise ValueError(f"Returned CIDs differ from requested batch: {offset}")
        properties.update(returned)
        sources.append(source)
        print(f"PubChem original CIDs {min(offset + 50, len(cids))}/{len(cids)}", flush=True)
    return properties, sources


def duplicate_groups(frame: pd.DataFrame, column: str) -> list[dict]:
    groups = []
    for key, group in frame.dropna(subset=[column]).groupby(column, sort=True):
        if len(group) > 1:
            groups.append({column: key, "drug_names": group.drug_name.tolist(),
                           "source_cids": [None if pd.isna(x) else int(x) for x in group.source_pubchem_cid],
                           "new_identity_ids": group.identity_id.tolist()})
    return groups


def build(output: Path, offline: bool = False) -> dict:
    source_path = ROOT / f"08-datasets/Tahoe-100M/{REVISION}/metadata/drug_metadata.parquet"
    prepared = ROOT / "08-datasets/LungCancer_A549_Integration/prepared/v1"
    catalog_path, condition_path = prepared / "drug_catalog.parquet", prepared / "tahoe/conditions.parquet"
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in (source_path, catalog_path, condition_path)}
    catalog, conditions, official = (pd.read_parquet(p) for p in (catalog_path, condition_path, source_path))
    official = official.assign(drug_name=official.drug.str.strip()).sort_values("drug_name")
    drugs = catalog[catalog.component_role.eq("drug")].sort_values("drug_name")
    for column in ("drug_name", "pubchem_cid", "canonical_smiles"):
        if not drugs[column].reset_index(drop=True).equals(official[column].reset_index(drop=True)):
            raise ValueError(f"Prepared source mapping differs from pinned official column: {column}")
    cids = sorted({int(x) for x in catalog.pubchem_cid.dropna()})
    output.mkdir(parents=True, exist_ok=True)
    properties, provenance = fetch_properties(cids, output, offline)
    identities, fingerprints = build_identity_catalog(catalog, properties)
    source_map = {cid: source for source in provenance for cid in source["requested_cids"]}
    identities["source_response_path"] = identities.source_pubchem_cid.map(lambda cid: source_map.get(cid, {}).get("path"))
    identities["source_response_sha256"] = identities.source_pubchem_cid.map(lambda cid: source_map.get(cid, {}).get("sha256"))
    identities["source_url"] = identities.source_pubchem_cid.map(lambda cid: source_map.get(cid, {}).get("url"))
    linked = map_conditions(conditions, identities)
    source_groups = duplicate_groups(identities, "source_identity_id")
    new_groups = duplicate_groups(identities, "identity_id")
    resolved_collisions = [g for g in source_groups if len(set(g["new_identity_ids"])) > 1]
    selected = identities[identities.drug_id.isin(conditions.drug_id)]
    summary = {
        "version": "identity-v2", "status": "reference_mapping_complete_with_explicit_exclusions",
        "tahoe_revision": REVISION, "source_catalog_rows": len(catalog), "source_drug_labels": len(drugs),
        "requested_unique_cids": len(cids), "fetched_unique_cids": len(properties),
        "mapped_conditions": len(linked), "condition_ids_preserved": linked.condition_id.equals(conditions.condition_id),
        "identity_status_counts": identities.identity_status.value_counts().to_dict(),
        "relation_counts": identities.source_to_reference_relation.value_counts().to_dict(),
        "multifragment_reference_drugs": int(identities.fragment_count.gt(1).sum()),
        "reference_drugs_with_unspecified_stereo": int(identities.unspecified_stereo_count.gt(0).sum()),
        "pubchem_reference_drugs_with_undefined_stereo": int((identities.pubchem_undefined_atom_stereo_count.fillna(0)
                                                            + identities.pubchem_undefined_bond_stereo_count.fillna(0)).gt(0).sum()),
        "source_stereo_information_gained": int(identities.defined_stereo_count.gt(identities.source_defined_stereo_count).sum()),
        "material_form_review_drugs": identities.loc[identities.material_form_review_reason.notna(), "drug_name"].tolist(),
        "feature_eligible_drugs": int(identities.eligible_for_structural_features.sum()),
        "selected_feature_eligible_drugs": int(selected.eligible_for_structural_features.sum()),
        "selected_treated_conditions_needing_identity_review": int((linked.condition_type.eq("treated") & ~linked.eligible_for_structural_features).sum()),
        "source_full_structure_duplicate_groups": source_groups,
        "resolved_source_collisions": resolved_collisions,
        "new_full_structure_duplicate_groups": new_groups,
        "parent_fragment_duplicate_groups": duplicate_groups(identities, "parent_id"),
        "interpretation": "Original CID reference verified, not vendor material or cross-source experimental equivalence",
        "dose_basis": "Nominal recorded uM retained; salt-versus-active-moiety basis not established",
        "production_training_assets_modified": False, "rdkit_version": rdBase.rdkitVersion,
        "fingerprint": {"type": "Morgan_binary", "radius": 2, "bits": 2048, "includeChirality": True,
                        "row_key": "drug_id via fingerprint_row", "invalid_and_vehicle_rows": "all-zero with explicit mask"},
    }
    identities.to_parquet(output / "drug_identity.parquet", index=False)
    identities.to_csv(output / "drug_identity.csv", index=False, encoding="utf-8-sig")
    linked.to_parquet(output / "condition_identity.parquet", index=False)
    linked.to_csv(output / "condition_identity.csv", index=False, encoding="utf-8-sig")
    np.save(output / "morgan2048_chiral.npy", fingerprints, allow_pickle=False)
    review = identities[(~identities.eligible_for_structural_features & ~identities.is_vehicle)
                        | identities.unspecified_stereo_count.gt(0) | identities.fragment_count.gt(1)]
    review.to_csv(output / "identity_review_required.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summary.json", summary)
    write_json(output / "sources.json", provenance)
    artifacts = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in output.iterdir() if p.is_file() and p.suffix in (".csv", ".parquet", ".npy", ".json") and p.name != "manifest.json"}
    if inputs != {name: sha256(ROOT / name) for name in inputs}:
        raise ValueError("Original inputs changed during build")
    write_json(output / "manifest.json", {"version": "identity-v2", "status": summary["status"],
               "created_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": inputs,
               "outputs": artifacts, "raw_response_count": len(provenance)})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="Rebuild from hash-checked saved responses only")
    args = parser.parse_args()
    result = build(args.output.resolve(), args.offline)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, ensure_ascii=False, indent=2))
