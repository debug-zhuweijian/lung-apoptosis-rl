"""Register the 11 source-named E-nitrochalcones without guessing PubChem IDs."""

import json
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.endpoint_alignment import file_hash

DIRECTORY = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/mphahlele"


def build():
    source = DIRECTORY / "article.html"
    soup = BeautifulSoup(source.read_bytes(), "html.parser")
    names = {}
    for paragraph in soup.find_all("p"):
        text = paragraph.get_text(" ", strip=True)
        if "prop-2-en-1-one" not in text or len(text) > 240:
            continue
        for code in ("3" + letter for letter in "abcdefghijk"):
            if f"( {code} )" in text:
                names[code] = text
    if len(names) != 11:
        raise ValueError("The 11 systematic source names were not recovered")
    # Manual structural transcription from each systematic name. Slash bonds
    # preserve the explicitly reported E geometry; ortho nitro stays on ring A.
    rings = ["c1ccccc1", "c1cccc(F)c1", "c1ccc(F)cc1", "c1ccccc1Cl",
             "c1ccc(Cl)cc1", "c1ccc(C)cc1", "c1cccc(OC)c1", "c1ccc(OC)cc1",
             "c1cc(OC)cc(OC)c1", "c1c(OC)c(OC)c(OC)cc1", "c1ccc(OC(F)(F)C(F)F)cc1"]
    formulas = ["C15H11NO3", "C15H10FNO3", "C15H10FNO3", "C15H10ClNO3",
                "C15H10ClNO3", "C16H13NO3", "C16H13NO4", "C16H13NO4",
                "C17H15NO5", "C18H17NO6", "C17H11F4NO4"]
    rows = []
    for letter, ring, expected in zip("abcdefghijk", rings, formulas):
        code = "3" + letter
        mol = Chem.MolFromSmiles(f"O=C(/C=C/{ring})c1ccccc1[N+](=O)[O-]")
        formula = rdMolDescriptors.CalcMolFormula(mol)
        if formula != expected:
            raise ValueError(f"Transcription formula mismatch: {code}")
        alkene = [bond for bond in mol.GetBonds() if bond.GetStereo() == Chem.BondStereo.STEREOE]
        if len(alkene) != 1:
            raise ValueError(f"Source E geometry lost: {code}")
        warning = ""
        if code == "3k":
            warning = "source HRMS formula/mass lacks the four fluorines in systematic name; unresolved source inconsistency"
        elif code == "3e":
            warning = "source HRMS found 288.0699 differs from source calculated 288.0347; source analytical QC caveat"
        rows.append({
            "compound_id": "Mphahlele2023:" + code, "source_compound_id": code,
            "systematic_source_name": names[code],
            "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
            "inchi_key": Chem.MolToInchiKey(mol), "computed_neutral_formula": formula,
            "structure_basis": "manual_transcription_of_published_systematic_name",
            "geometry_verified": "one_E_alkene", "source_file": source.relative_to(ROOT).as_posix(),
            "source_sha256": file_hash(source), "source_locator": f"Experimental/synthesis/{code}",
            "source_quality_warning": warning,
            "structure_candidate_available": True,
            "identity_approved_for_features": not warning,
            "independent_structure_review": "pending",
        })
    output = DIRECTORY / "chemical_registry.csv"
    if output.exists():
        raise FileExistsError("Chemical transcription is versioned; inspect before replacing")
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(json.dumps({"structures": len(rows), "distinct_inchi_keys": len({r['inchi_key'] for r in rows}),
                      "source_quality_caveats": [r['source_compound_id'] for r in rows if r['source_quality_warning']],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    build()
