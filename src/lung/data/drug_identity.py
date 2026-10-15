"""Versioned full-form identities; parent fragments are review aids, not aliases."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def _identifier(namespace: str, smiles: str) -> str:
    return namespace + ":" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def describe_structure(smiles) -> dict:
    """Preserve salts, charge, isotopes and stereo; do not select an active moiety.

    ``parent_id`` means a unique largest carbon-containing fragment by heavy
    atom count. It is not PubChem's parent relation or evidence of equivalent
    material, potency, concentration basis, or experimental protocol.
    """
    unresolved = {
        "identity_id": None, "canonical_full_smiles": None, "rdkit_inchikey": None,
        "connectivity_smiles": None, "fragment_count": 0, "organic_fragment_count": 0,
        "parent_smiles": None, "parent_id": None, "parent_status": "unresolved",
        "formal_charge": None, "potential_stereo_count": 0,
        "unspecified_stereo_count": 0, "defined_stereo_count": 0,
    }
    if not isinstance(smiles, str) or not smiles.strip():
        return {**unresolved, "parse_status": "missing_structure"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        return {**unresolved, "parse_status": "unparseable_structure"}
    full = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    no_stereo = Chem.Mol(mol)
    Chem.RemoveStereochemistry(no_stereo)
    fragments = Chem.GetMolFrags(mol, asMols=True)
    organic = [frag for frag in fragments if any(a.GetAtomicNum() == 6 for a in frag.GetAtoms())]
    parent = None
    parent_status = "no_organic_fragment"
    if organic:
        largest = max(frag.GetNumHeavyAtoms() for frag in organic)
        candidates = {Chem.MolToSmiles(f, isomericSmiles=True, canonical=True)
                      for f in organic if f.GetNumHeavyAtoms() == largest}
        if len(candidates) == 1:
            parent = next(iter(candidates))
            parent_status = "single_fragment" if len(fragments) == 1 else "largest_organic_fragment_candidate"
        else:
            parent_status = "ambiguous_largest_organic_fragments"
    stereo = Chem.FindPotentialStereo(mol)
    unspecified = sum(str(s.specified) != "Specified" for s in stereo)
    return {
        "parse_status": "parsed", "canonical_full_smiles": full,
        "identity_id": _identifier("full_isomeric_smiles_v2", full),
        "rdkit_inchikey": Chem.MolToInchiKey(mol),
        "connectivity_smiles": Chem.MolToSmiles(no_stereo, isomericSmiles=True, canonical=True),
        "fragment_count": len(fragments), "organic_fragment_count": len(organic),
        "parent_smiles": parent, "parent_id": _identifier("largest_organic_fragment_v2", parent) if parent else None,
        "parent_status": parent_status, "formal_charge": Chem.GetFormalCharge(mol),
        "potential_stereo_count": len(stereo), "unspecified_stereo_count": unspecified,
        "defined_stereo_count": len(stereo) - unspecified,
    }


def compare_structures(left, right) -> str:
    """Return an evidence tier; even full structure equality is not sample pairing."""
    a, b = describe_structure(left), describe_structure(right)
    if a["parse_status"] != "parsed" or b["parse_status"] != "parsed":
        return "unresolved"
    if a["identity_id"] == b["identity_id"]:
        return "same_full_form"
    if a["connectivity_smiles"] == b["connectivity_smiles"]:
        return "same_connectivity_different_stereochemistry"
    if a["parent_id"] and a["parent_id"] == b["parent_id"]:
        return "same_largest_organic_fragment_only"
    return "different_structures"


def material_form_review_reason(name: str, description: dict) -> str | None:
    """Flag explicit source-form information missing from the CID reference.

    This checks limited observable contradictions, not all possible product
    formulations. A null result is not experimental material verification.
    """
    smiles = description.get("canonical_full_smiles")
    if not smiles:
        return None
    fragments = smiles.split(".")
    lower = name.casefold()
    if "hydrochloride" in lower and not any(s in ("Cl", "[Cl-]") for s in fragments):
        return "source_hydrochloride_label_without_chloride_in_reference"
    if "hydrate" in lower and "O" not in fragments:
        return "source_hydrate_label_without_water_in_reference"
    if "solvate" in lower and description["fragment_count"] == 1:
        return "source_solvate_label_without_solvent_component_in_reference"
    if "%" in lower and description["fragment_count"] == 1:
        return "source_composition_label_not_represented_by_single_reference"
    return None


def build_identity_catalog(catalog: pd.DataFrame, properties: dict[int, dict]):
    """Resolve each original source label through its original CID only.

    Missing CIDs, different connectivity and CID/SMILES inconsistencies remain
    explicit. No name search supplies silent substitutes for missing values.
    """
    if catalog.drug_id.duplicated().any():
        raise ValueError("Source drug_id must be unique")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    records, vectors = [], []
    for item in catalog.sort_values("drug_id", kind="stable").to_dict("records"):
        source = describe_structure(item.get("canonical_smiles"))
        cid_value = item.get("pubchem_cid")
        cid = int(cid_value) if pd.notna(cid_value) and float(cid_value).is_integer() and cid_value > 0 else None
        reference = properties.get(cid, {})
        reference_smiles = reference.get("SMILES", reference.get("IsomericSMILES"))
        corrected = describe_structure(reference_smiles)
        vehicle = item.get("component_role") in ("vehicle", "vehicle_control")
        relation = compare_structures(item.get("canonical_smiles"), reference_smiles)
        key_matches = bool(reference) and corrected.get("rdkit_inchikey") == reference.get("InChIKey")
        form_review = material_form_review_reason(item["drug_name"], corrected)
        if vehicle:
            status = "vehicle_control"
        elif cid is None:
            status = "source_cid_missing_or_invalid"
        elif not reference:
            status = "pubchem_reference_unavailable"
        elif reference.get("CID") != cid:
            status = "returned_cid_mismatch"
        elif corrected["parse_status"] != "parsed":
            status = "pubchem_structure_unparseable"
        elif not key_matches:
            status = "pubchem_smiles_inchikey_mismatch"
        elif relation not in ("same_full_form", "same_connectivity_different_stereochemistry"):
            status = "source_connectivity_conflict_review_required"
        elif form_review:
            status = "source_material_form_review_required"
        else:
            status = "original_cid_structure_verified"
        usable = status == "original_cid_structure_verified"
        record = {
            "drug_id": item["drug_id"], "drug_name": item["drug_name"],
            "source_drug_name": item.get("drug_name_original", item["drug_name"]),
            "source_pubchem_cid": cid, "returned_pubchem_cid": reference.get("CID"),
            "source_smiles": item.get("canonical_smiles"),
            "source_canonical_full_smiles": source.get("canonical_full_smiles"),
            "source_identity_id": source.get("identity_id"),
            "source_inchikey": source.get("rdkit_inchikey"),
            "source_defined_stereo_count": source.get("defined_stereo_count"),
            "source_unspecified_stereo_count": source.get("unspecified_stereo_count"),
            "pubchem_smiles": reference_smiles, "pubchem_inchikey": reference.get("InChIKey"),
            "pubchem_inchi": reference.get("InChI"), "pubchem_title": reference.get("Title"),
            "pubchem_molecular_formula": reference.get("MolecularFormula"),
            "pubchem_molecular_weight": reference.get("MolecularWeight"),
            "pubchem_undefined_atom_stereo_count": reference.get("UndefinedAtomStereoCount"),
            "pubchem_undefined_bond_stereo_count": reference.get("UndefinedBondStereoCount"),
            "source_to_reference_relation": relation, "reference_inchikey_agrees": key_matches,
            "source_material_id": item["drug_id"], "material_form_review_reason": form_review,
            "identity_status": status, "eligible_for_structural_features": usable,
            "experimental_material_verified": False, "cross_source_pair_verified": False,
            "dose_basis": "not_applicable_vehicle" if vehicle else "nominal_source_uM_salt_or_active_moiety_basis_unverified",
            "is_vehicle": vehicle, "fingerprint_row": len(vectors),
            **corrected,
        }
        if vehicle:
            record["identity_id"] = "vehicle:DMSO"
        vector = np.zeros(2048, dtype=np.uint8)
        if usable:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(reference_smiles)), vector)
        records.append(record)
        vectors.append(vector)
    return pd.DataFrame(records), np.stack(vectors)


def map_conditions(conditions: pd.DataFrame, identities: pd.DataFrame) -> pd.DataFrame:
    """Add new references without changing condition IDs, dose/time, or old split."""
    if conditions.condition_id.duplicated().any():
        raise ValueError("Source condition_id must be unique")
    columns = ["drug_id", "identity_id", "parent_id", "identity_status", "fingerprint_row",
               "eligible_for_structural_features", "dose_basis", "experimental_material_verified"]
    linked = conditions.merge(identities[columns], on="drug_id", how="left", validate="many_to_one", indicator=True)
    if not linked._merge.eq("both").all():
        raise ValueError("Unmapped condition drug_id")
    return linked.drop(columns="_merge").assign(identity_version="identity-v2", source_condition_id=linked.condition_id)


def review_stereo_omission(original, reference):
    """Review a derived nominal registry completion, never experimental isomer proof.

    The caller must independently establish that both records name the SAME CID
    and source material. This structural check alone does not resolve identity.
    Existing specified stereo, salts or connectivity are never overwritten.
    Unspecified stereo remaining in the reference is retained explicitly.
    """
    a, b = describe_structure(original), describe_structure(reference)
    result = {"eligible": False, "reason": "unresolved_structure",
              "reference_unspecified_stereo_count": b.get("unspecified_stereo_count"),
              "experimental_stereochemistry_verified": False}
    if a['parse_status'] != 'parsed' or b['parse_status'] != 'parsed':
        return result
    if a['identity_id'] == b['identity_id']:
        return {**result, "reason": "already_same_full_form"}
    if a['connectivity_smiles'] != b['connectivity_smiles']:
        return {**result, "reason": "connectivity_or_material_form_differs"}
    def specified(mol):
        return (sum(x.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for x in mol.GetAtoms())
                + sum(x.GetStereo() != Chem.BondStereo.STEREONONE for x in mol.GetBonds()))
    if specified(Chem.MolFromSmiles(original)):
        return {**result, "reason": "original_has_specified_stereo_requires_review"}
    if not specified(Chem.MolFromSmiles(reference)):
        return {**result, "reason": "reference_adds_no_specified_stereo"}
    return {**result, "eligible": True,
            "reason": "complete_omission_with_same_CID_reference_only",
            "derived_reference_smiles": b['canonical_full_smiles']}
