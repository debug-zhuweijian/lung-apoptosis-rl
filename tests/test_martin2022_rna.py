"""Source schema/clock/numeric boundaries; no training or functional Y."""
import copy
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("martin", ROOT / "scripts/materialize_martin2022_rna.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

@pytest.mark.parametrize("value,counts,raises", [(0.2,True,False),(-0.1,True,True),(-6.64,False,False),(float("nan"),False,True),(float("inf"),True,True)])
def test_numeric_boundary(value, counts, raises):
    frame = pd.DataFrame([[value]], index=["A1BG"], columns=["s"])
    if raises:
        with pytest.raises(ValueError): m.validate_matrix(frame, counts)
    else:
        assert m.validate_matrix(frame, counts)[0, 0] == value

@pytest.mark.parametrize("axis", ["genes", "samples"])
def test_duplicate_axis(axis):
    frame = pd.DataFrame([[1,2],[3,4]], index=["a","a"] if axis == "genes" else ["a","b"], columns=["s","s"] if axis == "samples" else ["s","t"])
    with pytest.raises(ValueError): m.validate_matrix(frame, True)

def load_samples():
    return json.loads((m.SOURCE / "GSE193258_60samples.json").read_text())

def columns():
    return pd.read_csv(m.OUT / "rna/rna_samples.csv").matrix_column.tolist()

def test_metadata_structure_and_order():
    rows = m.sample_table(load_samples(), columns())
    assert len(rows) == 60
    assert len([x for x in rows if x["parallel_reference"]]) == 12
    assert [x["matrix_column"] for x in rows] == columns()
    assert all(x["functional_y"] == "not_attached" for x in rows)
    for cell, hours in m.LONG_HOURS.items():
        assert {x["washout_hours"] for x in rows if x["cell_line"] == cell and x["state"] == "long_washout"} == {hours}

@pytest.mark.parametrize("mutation", ["missing", "duplicate", "clock", "cell"])
def test_metadata_corruption(mutation):
    samples = copy.deepcopy(load_samples())
    if mutation == "missing": samples.pop()
    elif mutation == "duplicate": samples[-1] = samples[0]
    elif mutation == "clock": samples[3]["!Sample_characteristics_ch1"][-1] = "treatment: osimertinib DTPs long washout 999 hrs."
    elif mutation == "cell": samples[0]["!Sample_characteristics_ch1"][0] = "cell line: A549"
    with pytest.raises(ValueError): m.sample_table(samples, columns())

def test_real_source_readback_and_missing_mask():
    manifest = json.loads((m.OUT / "rna_manifest.json").read_text())
    for key in m.NAMES:
        source = pd.read_csv(m.OUT / "raw" / m.NAMES[key], sep="\t", index_col=0, float_precision="round_trip")
        assert np.array_equal(source.to_numpy(), np.load(m.OUT / "rna" / f"{key}.npy"))
    coverage = pd.read_csv(m.OUT / "rna/common_axis_coverage.csv")
    assert coverage.loc[~coverage.source_observed, "source_row_index"].isna().all()
    assert int(coverage.source_observed.sum()) == manifest["common_axis"]["covered"]
    assert manifest["common_axis"]["comparison_only"]


def test_axis_ambiguity_and_missing_are_absent():
    rows = m.axis_rows(["A", "B"], ["e1", "e2", "e3", "e4"], ["A", "A", "B", "C"])
    assert rows.source_observed.tolist() == [False, False, True, False]
    assert rows.source_row_index.isna().tolist() == [True, True, False, True]
    assert rows.mapping_status.tolist() == ["ambiguous_joint_symbol", "ambiguous_joint_symbol", "unique_exact_symbol", "missing_source_symbol"]
    with pytest.raises(ValueError): m.axis_rows(["A", "A"], ["e"], ["A"])
