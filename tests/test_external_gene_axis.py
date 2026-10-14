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


@pytest.mark.parametrize('scale', ['bulk_TPM', 'bulk_estimated_counts'])
def test_fractional_nonnegative_domains_preserve_source_values(scale):
    axis = pd.DataFrame({'common_gene_index': [0, 1], 'ensembl_id': ['E1', 'E2']})
    frame = pd.DataFrame({'ensembl_id': ['E1.2'], 'sample': [1.125]})
    values, mask, _ = align_gene_axis(frame, axis, identifier='ensembl_id', samples=['sample'], scale=scale)
    np.testing.assert_array_equal(values[:, :1], [[1.125]])
    assert np.isnan(values[0, 1]) and mask.tolist() == [[True, False]]


@pytest.mark.parametrize('scale', ['bulk_TPM', 'bulk_estimated_counts'])
def test_nonnegative_domains_reject_negative_values(scale):
    axis = pd.DataFrame({'common_gene_index': [0], 'ensembl_id': ['E1']})
    frame = pd.DataFrame({'ensembl_id': ['E1'], 'sample': [-0.01]})
    with pytest.raises(ValueError, match='negative'):
        align_gene_axis(frame, axis, identifier='ensembl_id', samples=['sample'], scale=scale)


def test_microarray_processed_domain_does_not_invent_a_log_base():
    axis = pd.DataFrame({'common_gene_index': [0], 'ensembl_id': ['E1']})
    frame = pd.DataFrame({'ensembl_id': ['E1'], 'sample': [-3.125]})
    values, _, _ = align_gene_axis(frame, axis, identifier='ensembl_id', samples=['sample'], scale='bulk_microarray_processed')
    assert values[0, 0] == -3.125


def test_duplicate_sample_selection_cannot_create_extra_replicates():
    axis = pd.DataFrame({'common_gene_index': [0], 'ensembl_id': ['E1']})
    frame = pd.DataFrame({'ensembl_id': ['E1'], 'sample': [3]})
    with pytest.raises(ValueError, match='sample'):
        align_gene_axis(frame, axis, identifier='ensembl_id', samples=['sample', 'sample'], scale='bulk_raw_counts')


def test_duplicate_source_columns_are_rejected_before_alignment():
    axis = pd.DataFrame({'common_gene_index': [0], 'ensembl_id': ['E1']})
    frame = pd.DataFrame([['E1', 2, 3]], columns=['ensembl_id', 'sample', 'sample'])
    with pytest.raises(ValueError, match='columns'):
        align_gene_axis(frame, axis, identifier='ensembl_id', samples=['sample'], scale='bulk_raw_counts')


def test_par_y_version_is_removed_without_collapsing_x_and_y_identity():
    axis = pd.DataFrame({'common_gene_index': [0, 1], 'ensembl_id': ['ENSG000001', 'ENSG000001_PAR_Y']})
    frame = pd.DataFrame({'ensembl_id': ['ENSG000001.7', 'ENSG000001.7_PAR_Y'], 'sample': [2, 5]})
    values, mask, _ = align_gene_axis(frame, axis, identifier='ensembl_id', samples=['sample'], scale='bulk_raw_counts')
    np.testing.assert_array_equal(values, [[2, 5]])
    assert mask.all()


@pytest.mark.parametrize('bad_id', [None, '', '   '])
def test_missing_common_axis_identity_is_rejected(bad_id):
    axis = pd.DataFrame({'common_gene_index': [0], 'ensembl_id': [bad_id]})
    frame = pd.DataFrame({'ensembl_id': ['E1'], 'sample': [3]})
    with pytest.raises(ValueError, match='identity'):
        align_gene_axis(frame, axis, identifier='ensembl_id', samples=['sample'], scale='bulk_raw_counts')


def test_single_cell_native_counts_are_not_labeled_as_bulk_or_tpm():
    axis = pd.DataFrame({'common_gene_index': [0], 'ensembl_id': ['E1']})
    frame = pd.DataFrame({'ensembl_id': ['E1'], 'cell': [0]})
    x, mask, _ = align_gene_axis(frame, axis, identifier='ensembl_id', samples=['cell'], scale='single_cell_raw_counts')
    assert x[0, 0] == 0 and mask[0, 0]
    with pytest.raises(ValueError, match='count'):
        align_gene_axis(frame.assign(cell=[0.5]), axis, identifier='ensembl_id', samples=['cell'], scale='single_cell_raw_counts')
