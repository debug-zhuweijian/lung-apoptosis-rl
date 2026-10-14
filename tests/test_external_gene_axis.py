import numpy as np
import pandas as pd
import pytest

from lung.data.external_gene_axis import align_gene_axis


def test_mapping_uses_gene_identity_not_source_row_and_preserves_missingness():
    axis = pd.DataFrame({"common_gene_index": [0, 1, 2], "ensembl_id": ["E1", "E2", "E3"],
                         "gene_symbol": ["A", "B", "C"]})
    frame = pd.DataFrame({"ensembl_id": ["E3.5", "E1.2"], "sample": [-2.5, 0.0]})
    values, mask, ledger = align_gene_axis(frame, axis, identifier="ensembl_id", samples=["sample"], scale="bulk_log2CPM")
    np.testing.assert_allclose(values, [[0, np.nan, -2.5]], equal_nan=True)
    assert mask.tolist() == [[True, False, True]]
    assert ledger.mapping_status.tolist() == ["matched", "matched"]


def test_ambiguous_symbols_are_not_silently_assigned():
    axis = pd.DataFrame({"common_gene_index": [0, 1], "ensembl_id": ["E1", "E2"],
                         "gene_symbol": ["A", "A"]})
    frame = pd.DataFrame({"gene_symbol": ["A"], "sample": [12]})
    _, mask, ledger = align_gene_axis(frame, axis, identifier="gene_symbol", samples=["sample"], scale="bulk_raw_counts")
    assert not mask.any()
    assert ledger.mapping_status.tolist() == ["ambiguous_target_symbol"]


def test_duplicate_versioned_ids_are_not_double_counted():
    axis = pd.DataFrame({"common_gene_index": [0], "ensembl_id": ["E1"], "gene_symbol": ["A"]})
    frame = pd.DataFrame({"ensembl_id": ["E1.1", "E1.2"], "sample": [4, 9]})
    _, mask, ledger = align_gene_axis(frame, axis, identifier="ensembl_id", samples=["sample"], scale="bulk_raw_counts")
    assert not mask.any()
    assert ledger.mapping_status.tolist() == ["duplicate_source_identifier"] * 2


def test_bulk_scales_cannot_be_silently_reinterpreted():
    axis = pd.DataFrame({"common_gene_index": [0], "ensembl_id": ["E1"], "gene_symbol": ["A"]})
    frame = pd.DataFrame({"gene_symbol": ["A"], "sample": [-2.1]})
    with pytest.raises(ValueError, match="Raw count"):
        align_gene_axis(frame, axis, identifier="gene_symbol", samples=["sample"], scale="bulk_raw_counts")
