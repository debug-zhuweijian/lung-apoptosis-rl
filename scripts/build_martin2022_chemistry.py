"""Source-specific nominal parent chemistry; no experimental stock certification."""
from __future__ import annotations
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration"
PACKAGE = BASE / "state_function_martin2022_v1"
OUT = PACKAGE / "chemistry"
NAMES = ["osimertinib", "venetoclax", "AZD4320", "K-975"]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def fingerprint(smiles, expected_key):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Invalid or multi-fragment parent reference")
    if Chem.MolToInchiKey(mol) != expected_key:
        raise ValueError("SMILES and official InChIKey disagree")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    vector = np.zeros(2048, dtype=np.uint8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), vector)
    return canonical, vector

def dose_um(value, unit):
    if unit != "nM": raise ValueError("Expected source nominal nM")
    nm = Decimal(str(value))
    if not nm.is_finite() or nm <= 0: raise ValueError("Invalid dose")
    return format(nm / Decimal(1000), "f")

def bind_singles(frame, items):
    lookup = {x["source_drug_name"]: x for x in items}
    rows = []
    for row in frame.to_dict("records"):
        if row["candidate_type"] != "single": continue
        components = json.loads(row["components_json"])
        if len(components) != 1: raise ValueError("Single candidate must have one component")
        component = components[0]
        if component["drug"] not in lookup: raise ValueError("Unknown chemical component")
        expected = {"S2F": {"osimertinib":160,"venetoclax":300,"AZD4320":300}, "S2M":{"osimertinib":160,"K-975":100}}
        if row["panel"] not in expected or expected[row["panel"]].get(component["drug"]) != component["concentration"] or row["exposure_h"] != 72:
            raise ValueError("Source panel dose/exposure disagrees with Supplement Figure S2F/M")
        item = lookup[component["drug"]]
        rows.append({"observation_id": row["observation_id"], "decision_block_id": row["decision_block_id"], "measurement_domain_id": row["measurement_domain_id"], "cell_background": row["cell_background"], "source_drug_name": component["drug"], "source_nominal_dose": str(component["concentration"]), "source_dose_unit": component["unit"], "nominal_dose_uM": dose_um(component["concentration"],component["unit"]), "exposure_h": row["exposure_h"], "chemical_id": item["chemical_id"], "fingerprint_row": item["fingerprint_row"], "chemical_status": item["status"], "structure_type": "parent_normalized_not_experiment_stock_form_certified", "experimental_stock_form": "unknown", "experimental_stock_form_certified": False})
    if len({x["observation_id"] for x in rows}) != len(rows): raise ValueError("Duplicate observation IDs")
    return pd.DataFrame(rows)

def main():
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    existing_path = BASE / "identity-v3/drug_identity.csv"
    existing = pd.read_csv(existing_path, low_memory=False)
    ven = existing[existing.drug_name.eq("venetoclax")]
    if len(ven) != 1: raise ValueError("Ambiguous existing venetoclax")
    v = ven.iloc[0]
    cached = BASE / "identity-v3" / v.source_response_path
    if sha(cached) != v.source_response_sha256: raise ValueError("Existing PubChem source changed")
    target = raw / "venetoclax_properties_existing.json"
    shutil.copyfile(cached, target)
    props = json.loads(target.read_text())["PropertyTable"]["Properties"]
    venprops = [p for p in props if p["CID"] == int(v.returned_pubchem_cid)]
    if len(venprops) != 1: raise ValueError("Cached CID missing/ambiguous")
    (raw / "venetoclax_request.json").write_text(json.dumps({"url":v.source_url,"query":str(int(v.returned_pubchem_cid)),"status":200,"reuse_from":str(cached.relative_to(ROOT)),"sha256":sha(cached)}))
    items, vectors = [], []
    for name in NAMES:
        if name == "venetoclax":
            response = target; prop = venprops[0]; request = json.loads((raw / "venetoclax_request.json").read_text()); synonyms = None
        else:
            response = raw / f"{name}_properties.json"
            request = json.loads((raw / f"{name}_request.json").read_text())
            props = json.loads(response.read_text())["PropertyTable"]["Properties"]
            if len(props) != 1: raise ValueError(f"Ambiguous nominal PubChem reference: {name}")
            prop = props[0]
            synonyms = json.loads((raw / f"{name}_synonyms.json").read_text())["InformationList"]["Information"][0]
            if synonyms["CID"] != prop["CID"] or not any(s.lower().replace("-", "") == name.lower().replace("-", "") for s in synonyms["Synonym"]): raise ValueError("Official synonyms do not support exact source name")
        smiles = prop.get("SMILES", prop.get("IsomericSMILES"))
        canonical, vector = fingerprint(smiles, prop["InChIKey"])
        if name == "venetoclax" and prop["InChIKey"] != v.pubchem_inchikey: raise ValueError("Existing reference mismatch")
        row = len(vectors); vectors.append(vector)
        items.append({"source_drug_name":name,"chemical_id":f"pubchem:{prop['CID']}","pubchem_cid":prop["CID"],"pubchem_title":prop["Title"],"pubchem_inchikey":prop["InChIKey"],"pubchem_inchi":prop["InChI"],"pubchem_smiles":smiles,"canonical_isomeric_parent_smiles":canonical,"molecular_formula":prop["MolecularFormula"],"status":"nominal_parent_structure_verified","structure_type":"parent_normalized_not_experiment_stock_form_certified","experimental_stock_form":"unknown","experimental_stock_form_certified":False,"fingerprint_row":row,"fingerprint_sha256":hashlib.sha256(vector.tobytes()).hexdigest(),"source_response":str(response.relative_to(OUT)),"source_response_sha256":sha(response),"source_url":request["url"],"query":request["query"],"synonym_evidence":None if synonyms is None else str((raw / f"{name}_synonyms.json").relative_to(OUT)),"reuse_existing_entity":name == "venetoclax"})
    matrix = np.stack(vectors)
    np.save(OUT / "morgan2048_chiral.npy", matrix, allow_pickle=False)
    (OUT / "items.json").write_text(json.dumps(items, indent=2),encoding="utf-8")
    function = PACKAGE / "function/published_means.csv"
    frame = pd.read_csv(function)
    bindings = bind_singles(frame,items)
    if len(bindings) != 20 or bindings.chemical_id.nunique() != 4: raise ValueError("Expected 20 source single candidates / four entities")
    bindings.to_csv(OUT / "single_candidate_bindings.csv",index=False)
    pd.DataFrame(items).to_csv(OUT / "chemical_items.csv",index=False)
    saved = np.load(OUT / "morgan2048_chiral.npy",allow_pickle=False)
    if not np.array_equal(saved,matrix): raise ValueError("Fingerprint readback mismatch")
    for item in items:
        if hashlib.sha256(saved[item["fingerprint_row"]].tobytes()).hexdigest() != item["fingerprint_sha256"]: raise ValueError("FP row identity mismatch")
    article = ROOT / "experiments/20261011-data-alignment-round16/state_function_new_sources/PMC10035388.xml"
    text = " ".join(ET.parse(article).getroot().find(".//sec[@id='sec2-2']").itertext())
    if "All small-molecule inhibitors were synthesized according to published methods" not in text: raise ValueError("Source reagent statement changed")
    manifest={"status":"nominal_parent_inputs_materialized", "items_file":"items.json","bindings_file":"single_candidate_bindings.csv","fingerprints_file":"morgan2048_chiral.npy","n_chemicals":4,"n_single_candidates":20,"fingerprint":{"type":"Morgan_binary","radius":2,"bits":2048,"includeChirality":True,"dtype":"uint8","rdkit_version":rdBase.rdkitVersion,"convention_source":"scripts/build_reviewed_identity_v3.py"},"source_function_path":str(function.relative_to(ROOT)),"source_function_sha256":sha(function),"source_article_path":str(article.relative_to(ROOT)),"source_article_sha256":sha(article),"source_dose_caption_path":"experiments/20261011-data-alignment-round16/state_function_new_sources/Martin2022_supplement.md", "source_dose_caption_sha256":sha(ROOT/"experiments/20261011-data-alignment-round16/state_function_new_sources/Martin2022_supplement.md"), "source_dose_caption_location":"Supplement Figure S2 F/M", "source_reagent_statement":"All small-molecule inhibitors were synthesized according to published methods (Methods sec2-2)","experimental_stock_form_certified":False,"dose_basis":"source_nominal_nM; stock salt/active-moiety basis unverified","combinations":"remain source function auxiliary rows; no pseudo-single fingerprints","limits":["Parent nominal reference only; no laboratory batch/stock salt certification","Chemical identity does not prove source RNA-function same-well pairing","No model/training/forward/prediction","Not Annexin fraction or A549 data"],"files":{str(p.relative_to(OUT)):{"bytes":p.stat().st_size,"sha256":sha(p)} for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"chemicals":[{k:x[k] for k in ['source_drug_name','chemical_id','pubchem_inchikey','fingerprint_row','fingerprint_sha256']} for x in items],"single_candidates":len(bindings),"fp_shape":list(matrix.shape)},indent=2))

if __name__ == "__main__": main()
