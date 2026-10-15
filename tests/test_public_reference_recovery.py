"""Source-backed reference recovery; fixtures test implementation, not biology."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.recover_public_reference import build_reference, recover_reference


def source_frames():
    # Different column orders must be aligned by identity, never by position.
    genes = pd.DataFrame({"ensembl_id": ["ENSG00000000003", "ENSG00000000001", "ENSG00000000002"],
                          "gene_symbol": ["C", "A", "B"], "token_id": [5, 3, 4]})
    counts = pd.DataFrame([[7, 0, 2], [0, 4, 1], [3, 1, 8]], index=["b", "a", "unknown"],
                          columns=["ENSG00000000003", "ENSG00000000001", "ENSG00000000004"])
    metadata = pd.DataFrame({"Cell_line": ["A549", "A549", None], "Plate": [8, 8, 8],
                             "Well": ["A1", "A2", "A3"], "cell_state": ["PI-APC-", "PI-APC+", "PI+APC+"],
                             "total_counts": counts.sum(axis=1), "n_genes_by_counts": (counts > 0).sum(axis=1)},
                            index=counts.index)
    return genes, counts, metadata


def test_identity_intersection_order_and_unknown_cell_exclusion():
    axis, cells, values, exclusions = build_reference(*source_frames())
    assert axis.ensembl_id.tolist() == ["ENSG00000000001", "ENSG00000000003"]
    assert axis.common_gene_index.tolist() == [0, 1]
    assert axis.tahoe_column.tolist() == [0, 2]
    assert axis.tidyi_column.tolist() == [1, 0]
    assert cells.source_cell_id.tolist() == ["b", "a"]
    np.testing.assert_array_equal(values, [[0, 7], [4, 0]])
    assert values.dtype == np.int32
    assert cells.state_label_original.tolist() == ["PI-APC-", "PI-APC+"]
    assert cells.data_role.eq("unassigned_external_source").all()
    assert exclusions.source_cell_id.tolist() == ["unknown"]
    assert exclusions.exclusion_reason.tolist() == ["unconfirmed_A549_identity"]


@pytest.mark.parametrize("defect", ["duplicate_gene", "duplicate_cell", "metadata_order", "fractional_count", "negative_count", "infinite_count", "wrong_total", "wrong_detected", "negative_token", "duplicate_token", "blank_gene"])
def test_reference_rejects_untrusted_inputs(defect):
    genes, counts, metadata = source_frames()
    if defect == "duplicate_gene": genes.loc[1, "ensembl_id"] = genes.loc[0, "ensembl_id"]
    elif defect == "duplicate_cell": counts.index = ["a", "a", "unknown"]
    elif defect == "metadata_order": metadata = metadata.iloc[::-1]
    elif defect == "fractional_count": counts = counts.astype(float); counts.iloc[0, 0] = .5
    elif defect == "negative_count": counts.iloc[0, 0] = -1
    elif defect == "infinite_count": counts = counts.astype(float); counts.iloc[0, 0] = np.inf
    elif defect == "wrong_total": metadata.iloc[0, metadata.columns.get_loc("total_counts")] += 1
    elif defect == "wrong_detected": metadata.iloc[0, metadata.columns.get_loc("n_genes_by_counts")] += 1
    elif defect == "negative_token": genes.loc[0, "token_id"] = -1
    elif defect == "duplicate_token": genes.loc[0, "token_id"] = genes.loc[1, "token_id"]
    elif defect == "blank_gene": genes.loc[0, "ensembl_id"] = None
    with pytest.raises(ValueError):
        build_reference(genes, counts, metadata)


def test_recovery_roundtrip_is_consumable_without_frozen_identity_claim(tmp_path):
    from src.lung.data.external_gene_axis import align_gene_axis
    genes, counts, metadata = source_frames()
    g, c, m = [tmp_path / name for name in ("genes.parquet", "counts.csv", "metadata.csv")]
    genes.to_parquet(g, index=False); counts.to_csv(c); metadata.to_csv(m)
    out = tmp_path / "recovered"
    report = recover_reference(g, c, m, out)
    axis = pd.read_parquet(out / "genes.parquet")
    assert report["shared_genes"] == 2 and report["confirmed_a549_cells"] == 2
    assert report["frozen_asset_identity_verified"] is False
    assert report["training_performed"] is False
    frame = pd.DataFrame({"ensembl_id": ["ENSG00000000001.7"], "sample1": [0.0]})
    projected, mask, _ = align_gene_axis(frame, axis, identifier="ensembl_id", samples=["sample1"], scale="bulk_raw_counts")
    assert projected[0, 0] == 0 and mask[0, 0]
    assert np.isnan(projected[0, 1]) and not mask[0, 1]
    with np.load(out / "tidyi_common_counts.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["expression"], [[0, 7], [4, 0]])
        assert data["observed_mask"].all()
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["sources"]) == 3
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError): recover_reference(g, c, m, out)
    assert (out / "manifest.json").read_bytes() == before


def test_deposited_qc_difference_requires_exact_explicit_source_signature():
    from scripts.recover_public_reference import qc_difference_signature
    genes, counts, metadata = source_frames()
    metadata.loc["b", "total_counts"] += 2
    metadata.loc["b", "n_genes_by_counts"] += 1
    with pytest.raises(ValueError, match="QC"):
        build_reference(genes, counts, metadata)
    signature = qc_difference_signature(counts, metadata)
    axis, cells, values, _ = build_reference(genes, counts, metadata, expected_qc_difference_sha256=signature)
    assert cells.loc[0, "total_counts"] == 11
    assert cells.loc[0, "export_total_counts"] == 9
    assert cells.loc[0, "source_total_minus_export"] == 2
    metadata.loc["b", "total_counts"] += 1
    with pytest.raises(ValueError, match="QC"):
        build_reference(genes, counts, metadata, expected_qc_difference_sha256=signature)
