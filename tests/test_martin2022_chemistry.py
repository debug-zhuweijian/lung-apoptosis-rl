"""Nominal structure, source binding and units, without model use."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("martin_chem",ROOT/"scripts/build_martin2022_chemistry.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize("value,expected",[(160,"0.16"),(300,"0.3"),(100,"0.1")])
def test_exact_nominal_units(value,expected): assert m.dose_um(value,"nM")==expected

@pytest.mark.parametrize("value,unit",[(0,"nM"),(-1,"nM"),("NaN","nM"),(1,"uM")])
def test_invalid_dose(value,unit):
    with pytest.raises(ValueError): m.dose_um(value,unit)

def test_structure_and_row_hashes():
    items=json.loads((m.OUT/"items.json").read_text());fp=np.load(m.OUT/"morgan2048_chiral.npy")
    assert fp.shape==(4,2048) and fp.dtype==np.uint8
    assert set(np.unique(fp))=={0,1}
    assert len({x["chemical_id"] for x in items})==4
    for item in items:
        _, recomputed=m.fingerprint(item["pubchem_smiles"],item["pubchem_inchikey"])
        assert np.array_equal(recomputed,fp[item["fingerprint_row"]])
        assert hashlib.sha256(recomputed.tobytes()).hexdigest()==item["fingerprint_sha256"]
        assert not item["experimental_stock_form_certified"]
        assert m.sha(m.OUT/item["source_response"])==item["source_response_sha256"]

@pytest.mark.parametrize("smiles,key",[("bad","x"),("CC.Cl","x"),("CC","wrong")])
def test_invalid_structure_or_key(smiles,key):
    with pytest.raises(ValueError):m.fingerprint(smiles,key)

def test_source_bindings_no_combinations_and_no_labels():
    items=json.loads((m.OUT/"items.json").read_text())
    source=pd.read_csv(m.PACKAGE/"function/published_means.csv")
    got=m.bind_singles(source,items)
    saved=pd.read_csv(m.OUT/"single_candidate_bindings.csv",dtype={"nominal_dose_uM":str,"source_nominal_dose":str})
    assert got.observation_id.tolist()==saved.observation_id.tolist()
    assert len(got)==20
    assert not any("mean" in c or "y"==c.lower() for c in got.columns)
    lookup={x["chemical_id"]:x["fingerprint_row"] for x in items}
    assert all(lookup[x.chemical_id]==x.fingerprint_row for x in got.itertuples())
    combination=set(source.loc[source.candidate_type.ne("single"),"observation_id"])
    assert not combination.intersection(got.observation_id)
    assert got.nominal_dose_uM.tolist()==saved.nominal_dose_uM.tolist()

def test_malformed_single_component_is_rejected():
    source=pd.read_csv(m.PACKAGE/"function/published_means.csv")
    single=source.loc[source.candidate_type.eq("single")].head(1).copy()
    single["components_json"]='[]'
    with pytest.raises(ValueError):m.bind_singles(single,json.loads((m.OUT/"items.json").read_text()))


def test_source_panel_dose_mismatch():
    source=pd.read_csv(m.PACKAGE/"function/published_means.csv")
    single=source.loc[source.candidate_type.eq("single")].head(1).copy()
    single["components_json"]='[{"drug":"osimertinib","concentration":500,"unit":"nM"}]'
    with pytest.raises(ValueError):m.bind_singles(single,json.loads((m.OUT/"items.json").read_text()))
