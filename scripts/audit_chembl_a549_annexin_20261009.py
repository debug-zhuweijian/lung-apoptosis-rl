"""Read public ChEMBL records; build a candidate inventory, never training labels."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08-datasets/ChEMBL_A549_Annexin/ChEMBL_37/audit_2026-10-09"
BASE = "https://www.ebi.ac.uk/chembl/api/data/"


def get_pages(endpoint, key, params, name):
    path = OUT / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))[key]
    records, pages = [], []
    offset = 0
    while True:
        response = requests.get(BASE + endpoint, params={**params, "limit": 1000, "offset": offset}, timeout=45)
        response.raise_for_status()
        data = response.json()
        pages.append({"url": response.url, "page_meta": data["page_meta"]})
        records.extend(data[key])
        if not data["page_meta"]["next"]:
            break
        offset += data["page_meta"]["limit"]
    path.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(), "pages": pages, key: records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def structure_key(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(molecule) if molecule is not None else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    response = requests.get(BASE + "status.json", timeout=30)
    response.raise_for_status()
    status = response.json()
    if status["chembl_db_version"] != "ChEMBL_37":
        raise RuntimeError("Source release changed; use a separately named audit version")
    status_path = OUT / "source_status.json"
    if not status_path.exists():
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    assays = get_pages("assay.json", "assays", {"assay_cell_type": "A549", "description__icontains": "Annexin"}, "assays_all.json")
    chosen = [a for a in assays if re.search(r"\b24\s*(?:h|hr|hrs|hours)\b", a["description"], flags=re.I)]
    by_id = {a["assay_chembl_id"]: a for a in chosen}
    ids = sorted(by_id)
    activities = []
    for start in range(0, len(ids), 60):
        activities.extend(get_pages("activity.json", "activities", {"assay_chembl_id__in": ",".join(ids[start:start + 60])}, f"activities_24h_{start:04d}.json"))
        print(f"assays_processed={min(start + 60, len(ids))}/{len(ids)} activities={len(activities)}", flush=True)
    assert len({a["activity_id"] for a in activities}) == len(activities)
    rows = []
    for a in activities:
        assay = by_id[a["assay_chembl_id"]]
        description = assay["description"]
        lower = description.lower()
        flags = []
        for label, pattern in {
            "light_or_radiation": r"irradiat|light dose|photo|laser",
            "combination_or_pretreatment": r"co.?treat|pre.?treat|combination|in presence|in the presence|followed by|transfect|knockdown",
            "non_apoptosis_fraction": r"necrotic cells|necrosis|viable cells|live cells|dead cells",
            "relative_or_fold_readout": r"fold|relative|compared to|increase in|reduction in",
            "dose_uses_ic50": r"ic50|ic80|ic90|ic20",
            "modified_cell_background": r"overexpress|transfect|knockdown|resistan|silenc|gene.?edited",
            "genotype_mentioned_requires_review": r"mutant|wild.?type|genotyp",
            "time_range_mentions_24h": r"\b\d+\s*(?:to|[-–])\s*24\s*(?:h|hr|hrs|hours)\b|\b24\s*(?:to|[-–])\s*\d+\s*(?:h|hr|hrs|hours)\b",
        }.items():
            if re.search(pattern, lower):
                flags.append(label)
        dose = re.search(r"\bat\s+(\d+(?:\.\d+)?)\s*(nM|uM|mM|µM|μM)\b", description, flags=re.I)
        dose_um = None
        if dose:
            unit = dose[2].lower()
            dose_um = float(dose[1]) * (0.001 if unit == "nm" else 1000 if unit == "mm" else 1)
        else:
            flags.append("explicit_dose_not_parsed")
        value = pd.to_numeric(a.get("standard_value"), errors="coerce")
        if a.get("standard_units") != "%" or a.get("standard_relation") != "=" or pd.isna(value) or not 0 <= value <= 100:
            flags.append("not_exact_percentage_0_100")
        if a.get("data_validity_comment"):
            flags.append("database_validity_comment")
        if "early" in lower and "late" in lower:
            event = "combined_early_late_candidate"
            flags.append("combined_gate_definition_requires_review")
        else:
            event = "early" if "early apoptotic" in lower else "late" if "late apoptotic" in lower else "total_explicit" if "total apoptotic" in lower else "unspecified"
        if "apoptotic cells" not in lower and "apoptosis" not in lower:
            flags.append("apoptosis_event_not_explicit")
        rows.append({
            "activity_id": a["activity_id"], "assay_chembl_id": a["assay_chembl_id"],
            "document_chembl_id": a["document_chembl_id"], "molecule_chembl_id": a["molecule_chembl_id"],
            "molecule_name": a.get("molecule_pref_name"), "canonical_smiles": a.get("canonical_smiles"),
            "full_inchikey": structure_key(a.get("canonical_smiles")), "description": description,
            "event": event, "dose_um_candidate": dose_um, "time_hours_mentioned": 24,
            "literal_24h_exposure": bool(re.search(r"\b(?:for|after)\s+24\s*(?:h|hr|hrs|hours)\b", lower)) and "time_range_mentions_24h" not in flags,
            "standard_type": a.get("standard_type"), "standard_value": a.get("standard_value"),
            "standard_units": a.get("standard_units"), "standard_relation": a.get("standard_relation"),
            "flags": ";".join(flags), "manual_primary_review": "pending", "use_for_training": False,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "activities_24h_candidates.csv", index=False, encoding="utf-8-sig")
    conditions_path = ROOT / "08-datasets/LungCancer_A549_Integration/prepared/v1/tahoe/conditions.parquet"
    conditions = pd.read_parquet(conditions_path)
    drug_keys = conditions[["drug_name", "canonical_smiles"]].drop_duplicates().copy()
    drug_keys["full_inchikey"] = drug_keys.canonical_smiles.map(structure_key)
    match = frame.dropna(subset=["full_inchikey"]).merge(drug_keys.dropna(subset=["full_inchikey"])[["drug_name", "full_inchikey"]], on="full_inchikey")
    exact = match.merge(conditions[["condition_id", "drug_name", "dose_value", "dose_unit", "treatment_hours"]], on="drug_name")
    exact = exact[(exact.dose_unit == "uM") & (exact.treatment_hours == 24) & exact.literal_24h_exposure & (exact.dose_value == exact.dose_um_candidate)]
    match.to_csv(OUT / "tahoe_structure_candidates.csv", index=False, encoding="utf-8-sig")
    exact.to_csv(OUT / "tahoe_structure_dose_time_candidates.csv", index=False, encoding="utf-8-sig")
    docs = sorted(frame.document_chembl_id.unique())
    for start in range(0, len(docs), 80):
        get_pages("document.json", "documents", {"document_chembl_id__in": ",".join(docs[start:start + 80])}, f"documents_24h_{start:04d}.json")
    summary = {
        "status": "candidate_inventory_only_not_curated_training_data", "source_version": status["chembl_db_version"],
        "assays_annexin_a549": len(assays), "assays_mentioning_24h": len(chosen),
        "activity_records_24h": len(frame), "molecules_24h": frame.molecule_chembl_id.nunique(),
        "documents_24h": frame.document_chembl_id.nunique(), "no_automatic_flag_records": int((frame["flags"] == "").sum()),
        "no_automatic_flag_molecules": frame.loc[frame["flags"] == "", "molecule_chembl_id"].nunique(),
        "tahoe_structure_matched_activities": match.activity_id.nunique(), "tahoe_structure_matched_names": sorted(match.drug_name.unique()),
        "tahoe_structure_dose_time_matched_activities": exact.activity_id.nunique(),
        "tahoe_structure_dose_time_matched_names": sorted(exact.drug_name.unique()),
        "units_warning": "assay IDs, activity rows, quadrants, compounds and biological replicates are different units",
        "conditions_sha256": hashlib.sha256(conditions_path.read_bytes()).hexdigest(),
    }
    summary["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "summary.json"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "files_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
