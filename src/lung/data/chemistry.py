"""Full-form chemical identity and the architecture's Morgan fingerprint."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def build_chemical_catalog(catalog: pd.DataFrame):
    """Canonicalize atom order, preserving salts, isotopes and stereochemistry.

    No largest-fragment, charge, tautomer or salt standardization is applied.
    A successful parse establishes a supplied full-structure grouping, not an
    independent verification of the vendor's material identity.
    """
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    if catalog["drug_id"].duplicated().any():
        raise ValueError("Duplicate source drug_id")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True,
        useBondTypes=True, includeRedundantEnvironments=False,
    )
    records, fingerprints = [], []
    for item in catalog.sort_values("drug_id", kind="stable").to_dict("records"):
        vehicle = item.get("component_role") == "vehicle"
        smiles = item.get("canonical_smiles")
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        identity = None
        canonical = None
        vector = np.zeros(2048, dtype=np.uint8)
        if vehicle:
            identity, status = "vehicle:DMSO", "vehicle"
        elif mol is None or mol.GetNumAtoms() == 0:
            status = "DRUG_IDENTITY_UNRESOLVED"
        else:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            identity = "full_smiles:" + hashlib.sha256(canonical.encode()).hexdigest()
            status = "parsed_full_form"
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
        records.append({
            "drug_id": item["drug_id"], "drug_name": item["drug_name"],
            "source_smiles": smiles, "canonical_full_smiles": canonical,
            "chemical_or_experiment_group_id": identity, "identity_status": status,
            "fingerprint_row": len(fingerprints), "is_vehicle": vehicle,
        })
        fingerprints.append(vector)
    return pd.DataFrame(records), np.stack(fingerprints), {
        "rdkit": rdBase.rdkitVersion, "canonicalization": "isomeric_full_form_no_desalting",
        "identity_limit": "source_structure_parsed_not_independently_vendor_verified",
        "fingerprint": {"type": "Morgan_binary", "radius": 2, "bits": 2048,
                        "includeChirality": True, "useBondTypes": True,
                        "includeRedundantEnvironments": False},
    }
