"""Resolve nominal endpoint compounds and record tiered Tahoe structure links."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.drug_identity import compare_structures, describe_structure

OUT = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/chemical-links-v1"
IDENTITY = ROOT / "08-datasets/LungCancer_A549_Integration/identity-v2"
INDE = ROOT / "08-datasets/Inde2020_FractionalKilling/9bccjb75wp-v1/tahoe_name_dose_time_candidates.csv"
MACAYA = ROOT / "08-datasets/GSE161218/GSE161218/article_extracted.md"
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class PublicCache:
    def __init__(self, output: Path, offline: bool):
        self.output, self.offline, self.sources = output, offline, {}
        (output / "raw_pubchem").mkdir(parents=True, exist_ok=True)

    def get(self, kind: str, query: str, url: str):
        name = f"{kind}_{hashlib.sha256(url.encode()).hexdigest()[:24]}"
        path = self.output / "raw_pubchem" / f"{name}.json"
        meta = path.with_suffix(".provenance.json")
        if path.exists() and meta.exists():
            record = json.loads(meta.read_text(encoding="utf-8"))
            if record["url"] != url or record["sha256"] != digest(path):
                raise ValueError(f"Cached source changed: {path}")
        else:
            if self.offline:
                raise FileNotFoundError(path)
            for attempt in range(3):
                response = requests.get(url, timeout=60)
                if response.status_code in (200, 404):
                    break
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
            if response.status_code not in (200, 404):
                response.raise_for_status()
            response.json()
            path.write_bytes(response.content)
            record = {"url": url, "query": query, "kind": kind,
                      "http_status": response.status_code, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                      "path": str(path.relative_to(self.output)), "sha256": digest(path), "bytes": path.stat().st_size}
            save_json(meta, record)
            time.sleep(0.25)
        self.sources[url] = record
        return json.loads(path.read_text(encoding="utf-8")), record

    def name_cids(self, name: str):
        data, evidence = self.get("name", name, f"{BASE}/name/{quote(name, safe='')}/cids/JSON")
        return data.get("IdentifierList", {}).get("CID", []), evidence


def name_parts(original: str):
    """Only split explicit parenthetical aliases; retain any salt suffix."""
    text = original.strip()
    match = re.fullmatch(r"(.+?)\s*\(([^()]+)\)\s*(.*)", text)
    if not match:
        return text, []
    before, inside, suffix = match.groups()
    # Form names are not aliases to be stripped from molecular identity.
    if re.search(r"hydrate|chloride|bromide|sulfate|citrate|solvate|maleate|mesylate|%", inside, re.I):
        return text, []
    primary = " ".join((before + " " + suffix).split())
    aliases = [" ".join((alias.strip() + " " + suffix).split()) for alias in inside.split(",")]
    return primary, aliases


def normalize_name(name: str) -> str:
    return re.sub(r"[\s\-(),]", "", name).casefold()


def resolve_name(original: str, cache: PublicCache) -> dict:
    primary, aliases = name_parts(original)
    cids, evidence = cache.name_cids(original.strip())
    records = [evidence]
    route = "literal_source_label"
    literal_cids, primary_cids, literal_conflict = list(cids), list(cids), False
    if primary != original.strip():
        primary_cids, evidence = cache.name_cids(primary)
        records.append(evidence)
        common = sorted(set(cids).intersection(primary_cids))
        if common:
            cids, route = common, "literal_and_primary_name_intersection"
        elif not cids:
            cids, route = primary_cids, "explicit_alias_split_preserving_form_suffix"
        elif primary_cids:
            literal_conflict = True
    result = {"selected_cid": None, "name_resolution_status": "unresolved_name", "resolution_route": route,
              "primary_query": primary, "explicit_aliases": aliases, "lookup_evidence": records,
              "literal_candidate_cids": literal_cids, "primary_candidate_cids": primary_cids}
    if len(cids) != 1:
        result["name_resolution_status"] = "ambiguous_name_multiple_cids" if cids else "unresolved_name"
        result["candidate_cids"] = cids
        return result
    cid = cids[0]
    synonyms, evidence = cache.get("synonyms", str(cid), f"{BASE}/cid/{cid}/synonyms/JSON")
    records.append(evidence)
    items = synonyms.get("InformationList", {}).get("Information", [])
    if len(items) != 1 or items[0].get("CID") != cid:
        raise ValueError(f"Synonym response mismatch for CID {cid}")
    names = {normalize_name(n) for n in items[0]["Synonym"]}
    primary_supported = normalize_name(primary) in names or normalize_name(original.strip()) in names
    conflicts, unconfirmed = [], []
    for alias in aliases:
        if normalize_name(alias) in names:
            continue
        alias_cids, evidence = cache.name_cids(alias)
        records.append(evidence)
        if alias_cids and cid not in alias_cids:
            conflicts.append({"alias": alias, "cids": alias_cids})
        elif not alias_cids:
            unconfirmed.append(alias)
    status = "unique_pubchem_name_and_synonyms_verified"
    if literal_conflict:
        status = "literal_primary_structure_conflict_review_required"
    elif conflicts:
        status = "explicit_alias_conflict_review_required"
    elif not primary_supported:
        status = "unique_name_result_exact_synonym_unconfirmed"
    elif unconfirmed:
        status = "unique_primary_name_unconfirmed_parenthetical_alias"
    return {**result, "selected_cid": cid, "name_resolution_status": status,
            "primary_synonym_supported": primary_supported, "alias_conflicts": conflicts,
            "unconfirmed_aliases": unconfirmed, "candidate_cids": cids}


def inputs():
    frame = pd.read_csv(INDE, keep_default_na=False)
    rows = [{"source_id": "Inde2020:" + r["source_compound_name"],
             "source_compound_name": r["source_compound_name"], "study": "Inde2020",
             "nominal_supplier": "Selleck Chemicals", "source_catalogue_record": "L2000 library",
             "reagents_evidence": "PMC7409774 Methods: Chemicals and reagents; Chemical library screening",
             "background": "A549_N_nuclear_mKate2", "tahoe_background_relation": "engineered_to_parental_background_transfer",
             "candidate_tahoe_drug_name": r["tahoe_drug_name"], "candidate_tahoe_condition_ids": r["tahoe_condition_ids"],
             "source_dose_um": r["dose_um"], "source_time_hours": r["time_hours"]} for r in frame.to_dict("records")]
    for drug, supplier, catalogue in [("Trametinib", "Selleckchem", "GSK1120212 (alias; catalogue number not stated)"),
                                      ("Lestaurtinib", "LC Laboratories", "L6307 as written in paper"),
                                      ("Midostaurin", "MedChemExpress", "PKC412 (alias; catalogue number not stated)")]:
        rows.append({"source_id": "Macaya2023:" + drug, "source_compound_name": drug, "study": "Macaya2023",
                     "nominal_supplier": supplier, "source_catalogue_record": catalogue,
                     "reagents_evidence": "10.1038/s41467-023-41828-z Methods: Reagents; article_extracted.md line 184",
                     "background": "multiple_LUAD_backgrounds_endpoint_table_must_select", "tahoe_background_relation": "requires_endpoint_background_and_culture_check",
                     "candidate_tahoe_drug_name": drug, "candidate_tahoe_condition_ids": None,
                     "source_dose_um": None, "source_time_hours": None})
    return rows


def run(output: Path, offline: bool):
    source_paths = [INDE, MACAYA, MACAYA.parent / "sources/s41467-023-41828-z.html",
                    INDE.parent / "PMC7409774.xml", IDENTITY / "manifest.json", IDENTITY / "drug_identity.parquet",
                    IDENTITY / "condition_identity.parquet", ROOT / "src/lung/data/drug_identity.py"]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    cache = PublicCache(output, offline)
    resolved = []
    for index, row in enumerate(inputs(), 1):
        result = resolve_name(row["source_compound_name"], cache)
        resolved.append({**row, **result})
        print(f"Names {index}/40: {row['source_compound_name']} -> {result['selected_cid']} {result['name_resolution_status']}", flush=True)
    cids = sorted({cid for r in resolved for cid in [*r["literal_candidate_cids"], *r["primary_candidate_cids"],
                   *[c for conflict in r.get("alias_conflicts", []) for c in conflict["cids"]]]})
    fields = "IsomericSMILES,InChI,InChIKey,Title,MolecularFormula,MolecularWeight,UndefinedAtomStereoCount,UndefinedBondStereoCount"
    properties, prop_evidence = cache.get("properties", ",".join(map(str, cids)), f"{BASE}/cid/{','.join(map(str, cids))}/property/{fields}/JSON")
    prop_map = {r["CID"]: r for r in properties["PropertyTable"]["Properties"]}
    if set(prop_map) != set(cids):
        raise ValueError("CID property response did not cover exact requested set")
    records, name_candidates = [], []
    for row in resolved:
        for cid in sorted(set(row["literal_candidate_cids"] + row["primary_candidate_cids"])):
            candidate = prop_map[cid]
            name_candidates.append({"source_id": row["source_id"], "source_compound_name": row["source_compound_name"],
                                    "PubChemCID": cid, "pubchem_title": candidate.get("Title"),
                                    "pubchem_inchikey": candidate.get("InChIKey"),
                                    "pubchem_smiles": candidate.get("SMILES", candidate.get("IsomericSMILES")),
                                    "literal_name_result": cid in row["literal_candidate_cids"],
                                    "primary_name_result": cid in row["primary_candidate_cids"],
                                    "selected_reference": cid == row["selected_cid"],
                                    "name_resolution_status": row["name_resolution_status"],
                                    "properties_evidence_path": prop_evidence["path"]})
        ref = prop_map.get(row["selected_cid"], {})
        structure = describe_structure(ref.get("SMILES", ref.get("IsomericSMILES")))
        key_agrees = bool(ref) and structure.get("rdkit_inchikey") == ref.get("InChIKey")
        if ref and not key_agrees:
            raise ValueError(f"SMILES/InChIKey mismatch: {row['source_id']}")
        evidence = [*row["lookup_evidence"], prop_evidence] if ref else row["lookup_evidence"]
        records.append({**{k: v for k, v in row.items() if k not in ("lookup_evidence", "alias_conflicts", "unconfirmed_aliases", "candidate_cids", "explicit_aliases")},
                        "compound_id": f"pubchem:{row['selected_cid']}" if ref else None,
                        "PubChemCID": row["selected_cid"], "pubchem_title": ref.get("Title"),
                        "pubchem_inchikey": ref.get("InChIKey"), "pubchem_smiles": ref.get("SMILES", ref.get("IsomericSMILES")),
                        "pubchem_undefined_atom_stereo_count": ref.get("UndefinedAtomStereoCount"),
                        "pubchem_undefined_bond_stereo_count": ref.get("UndefinedBondStereoCount"),
                        "nominal_structure_resolved": key_agrees and row["name_resolution_status"] == "unique_pubchem_name_and_synonyms_verified",
                        "experimental_reagent_form_verified": False, "use_as_paired_training_target": False,
                        "dose_basis": "nominal_source_uM_salt_or_active_moiety_basis_unverified",
                        "lookup_evidence": json.dumps(evidence, ensure_ascii=False),
                        "alias_conflicts": json.dumps(row.get("alias_conflicts", []), ensure_ascii=False),
                        "unconfirmed_aliases": json.dumps(row.get("unconfirmed_aliases", []), ensure_ascii=False),
                        **structure})
    compounds = pd.DataFrame(records)
    tahoe = pd.read_parquet(IDENTITY / "drug_identity.parquet")
    conditions = pd.read_parquet(IDENTITY / "condition_identity.parquet")
    links = []
    for source in records:
        found = False
        for target in tahoe.loc[~tahoe.is_vehicle].to_dict("records"):
            relation = compare_structures(source.get("canonical_full_smiles"), target.get("canonical_full_smiles"))
            expected = target["drug_name"] == source["candidate_tahoe_drug_name"]
            if relation in ("different_structures", "unresolved") and not expected:
                continue
            selected = conditions[conditions.drug_id.eq(target["drug_id"])]
            if source["study"] == "Inde2020":
                selected = selected[selected.dose_value.eq(source["source_dose_um"]) & selected.treatment_hours.eq(source["source_time_hours"])]
            found = True
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
    linked = pd.DataFrame(links)
    linked["eligible_nominal_full_structure_link"] = (linked.source_nominal_structure_resolved.fillna(False)
        & linked.tahoe_eligible_for_structural_features.fillna(False)
        & linked.structure_relation.eq("same_full_form"))
    compounds.to_csv(output / "endpoint_compounds.csv", index=False, encoding="utf-8-sig")
    compounds.to_parquet(output / "endpoint_compounds.parquet", index=False)
    linked.to_csv(output / "tahoe_candidate_links.csv", index=False, encoding="utf-8-sig")
    linked.to_parquet(output / "tahoe_candidate_links.parquet", index=False)
    pd.DataFrame(name_candidates).to_csv(output / "name_resolution_candidates.csv", index=False, encoding="utf-8-sig")
    raw_inventory = []
    for path in sorted((output / "raw_pubchem").glob("*.provenance.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        if digest(output / source["path"]) != source["sha256"]:
            raise ValueError(f"Archived response hash mismatch: {source['path']}")
        raw_inventory.append({**source, "used_in_final_resolution": source["url"] in cache.sources})
    summary = {"version": "chemical-links-v1", "source_compounds": len(compounds),
               "name_resolution_status_counts": compounds.name_resolution_status.value_counts().to_dict(),
               "nominal_structure_resolved": int(compounds.nominal_structure_resolved.sum()),
               "link_relation_counts": linked.structure_relation.value_counts().to_dict(),
               "eligible_nominal_full_structure_sources_by_study": linked.loc[linked.eligible_nominal_full_structure_link].groupby("study").source_id.nunique().to_dict(),
               "raw_responses": len(cache.sources), "raw_responses_archived": len(raw_inventory), "experimental_sample_pairs_verified": 0,
               "tahoe_identity_manifest_sha256": source_hashes[str((IDENTITY / "manifest.json").relative_to(ROOT))]}
    save_json(output / "sources.json", list(cache.sources.values()))
    save_json(output / "raw_response_inventory.json", raw_inventory)
    save_json(output / "name_resolution_details.json", resolved)
    save_json(output / "summary.json", summary)
    artifacts = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in output.iterdir()
                 if p.is_file() and p.suffix in (".json", ".csv", ".parquet") and p.name != "manifest.json"}
    if source_hashes != {name: digest(ROOT / name) for name in source_hashes}:
        raise ValueError("Original inputs changed during build")
    save_json(output / "manifest.json", {"version": "chemical-links-v1", "status": "nominal_reference_links_with_explicit_evidence_boundaries",
               "created_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": source_hashes,
               "builder_sha256": digest(Path(__file__)), "outputs": artifacts})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--offline", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output.resolve(), arguments.offline)
