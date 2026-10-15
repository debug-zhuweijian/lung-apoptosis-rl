"""Synthetic contract fixtures; actual-source results are separate evidence."""
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.recover_cpa_combination_subset import bind_metadata, recover


def metadata():
    return pd.DataFrame({"Cell": ["c1", "c2"], "Drug1": ["DMSO", "panobinostat"],
        "Drug2": ["DMSO", "crizotinib"], "Well": ["E10", "E4"],
        "RT_well": ["RT58", "RT52"], "sample": ["source", "source"],
        "n.umi": [3, 5], "cell_type": ["DMSO_DMSO", "panobinostat_crizotinib"],
        "drugname_drugconc": ["DMSO placeholder", "DMSO placeholder"]})


def test_binding_uses_identity_not_position_and_preserves_dose_conflict():
    observed = metadata()
    result = bind_metadata(observed, observed.iloc[::-1])
    assert result.condition_id.tolist() == ["DMSO_DMSO", "panobinostat_crizotinib"]
    assert result.resolved_dose_uM.isna().all()
    assert result.cell_line.eq("A549").all()
    assert not result.measured_apoptosis_label.any()
    assert result.author_interface_drug_placeholder.eq("DMSO placeholder").all()


@pytest.mark.parametrize("key,value", [("Drug1", "DMSO"), ("Well", "wrong"), ("n.umi", 99)])
def test_identity_conflicts_fail(key, value):
    observed = metadata()
    original = observed.copy()
    original.loc[1, key] = value
    with pytest.raises(ValueError, match="GEO identity conflict"):
        bind_metadata(observed, original)


def test_missing_or_duplicate_cells_fail():
    observed = metadata()
    with pytest.raises(ValueError, match="absent"):
        bind_metadata(observed, observed.iloc[:1])
    with pytest.raises(ValueError, match="duplicate"):
        bind_metadata(observed, pd.concat([observed, observed]))


def fixture_files(tmp_path, fractional=False):
    cell_path, gene_path, axis_path, h5_path = [tmp_path / n for n in ["cells.tsv", "genes.tsv", "axis.csv", "source.h5ad"]]
    frame = metadata()
    frame.to_csv(cell_path, sep="\t", index=False)
    pd.DataFrame({"id": ["ENSG1.2", "ENSG2.1"], "gene_short_name": ["G1", "G2"]}).to_csv(gene_path, sep="\t", index=False)
    pd.DataFrame({"ensembl_id": ["ENSG2", "ENSG3", "ENSG1"], "common_gene_index": [0, 1, 2]}).to_csv(axis_path, index=False)
    mat = sparse.csr_matrix([[1., 2.], [2., 3.]])
    if fractional:
        mat.data[0] = 1.5
    with h5py.File(h5_path, "w") as f:
        obs = f.create_group("obs")
        for key in frame:
            values = frame[key].to_numpy()
            obs.create_dataset(key, data=values, dtype=h5py.string_dtype() if values.dtype == object else None)
        obs.create_dataset("_index", data=frame.Cell.to_numpy(), dtype=h5py.string_dtype())
        var = f.create_group("var")
        var.create_dataset("id", data=["ENSG1.2", "ENSG2.1"], dtype=h5py.string_dtype())
        var.create_dataset("gene_short_name", data=["G1", "G2"], dtype=h5py.string_dtype())
        for name in ["layers/counts", "X"]:
            group = f.create_group(name)
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = mat.shape
            data = mat.data if name.startswith("layers") else np.log1p(mat.data * np.repeat(10000 / np.asarray(mat.sum(axis=1)).ravel(), np.diff(mat.indptr)))
            group.create_dataset("data", data=data)
            group.create_dataset("indices", data=mat.indices)
            group.create_dataset("indptr", data=mat.indptr)
    return h5_path, cell_path, gene_path, axis_path


def test_recovery_preserves_missing_genes_and_no_overwrite(tmp_path):
    paths = fixture_files(tmp_path)
    out = tmp_path / "out"
    manifest = recover(*paths, out)
    assert manifest["n_cells"] == 2 and manifest["n_wells"] == 2
    assert manifest["n_common_axis_observed"] == 2
    values = np.load(out / "well_pseudobulk_common_axis.npz")
    assert np.isnan(values["values"][:, 1]).all()
    assert not values["observed_mask"][:, 1].any()
    assert not manifest["training_eligible"] and not manifest["true_sequence_evaluation_eligible"]
    with pytest.raises(ValueError, match="not overwritten"):
        recover(*paths, out)


def test_fractional_raw_counts_rejected(tmp_path):
    paths = fixture_files(tmp_path, fractional=True)
    with pytest.raises(ValueError, match="integer counts"):
        recover(*paths, tmp_path / "out")


@pytest.mark.parametrize("kind", ["normalized", "gene"])
def test_source_semantic_conflict_rejected(tmp_path, kind):
    paths = fixture_files(tmp_path)
    with h5py.File(paths[0], "r+") as f:
        if kind == "normalized":
            f["X/data"][0] = 999
        else:
            f["var/id"][0] = "ENSG999"
    with pytest.raises(ValueError, match="documented log1p|gene identity"):
        recover(*paths, tmp_path / "out")
