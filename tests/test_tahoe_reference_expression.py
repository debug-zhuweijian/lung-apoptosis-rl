"""Synthetic role/identity and sparse-summary checks; no scientific predictions."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.audit_tahoe_reference_expression import accumulate, averages, difference, validate_control_rows, verify_obs_identity


def metadata():
    controls = pd.DataFrame({"condition_id": ["tahoe:a"], "source_sample_id": ["a"], "plate": ["plate1"],
                             "condition_type": ["control"], "is_vehicle": [True]})
    rows = pd.DataFrame({"cell_id": ["tahoe:x"], "matrix_path": ["tiny.npz"], "matrix_row": [1],
                         "condition_id": ["tahoe:a"], "source_sample_id": ["a"], "plate": ["plate1"],
                         "role": ["fit"], "outer_split": ["train"], "observation_role": ["contemporaneous_control"],
                         "source_id": ["Tahoe-100M"], "background_id": ["CVCL_0023"]})
    obs = pd.DataFrame({"BARCODE_SUB_LIB_ID": ["other", "x"], "sample": ["a", "a"], "plate": ["plate1", "plate1"],
                        "drug": ["other_drug", "DMSO_TF"], "cell_line_id": ["CVCL_0023", "CVCL_0023"]})
    return rows, controls, obs


def test_fit_vehicle_identity_is_checked_against_actual_csr_row():
    rows, controls, obs = metadata()
    validate_control_rows(rows, controls)
    verify_obs_identity(rows, obs)
    rows.loc[0, "matrix_row"] = 0
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_obs_identity(rows, obs)


@pytest.mark.parametrize(("column", "value"), [("role", "selection"), ("outer_split", "test"),
                                                ("observation_role", "treated"), ("source_sample_id", "wrong")])
def test_forbidden_roles_and_mismatched_sources_rejected(column, value):
    rows, controls, _ = metadata()
    rows.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_control_rows(rows, controls)


def test_duplicate_cell_or_physical_row_cannot_be_counted_twice():
    rows, controls, _ = metadata()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_control_rows(pd.concat([rows, rows]), controls)
    changed = rows.copy()
    changed.loc[0, "cell_id"] = "tahoe:different"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_control_rows(pd.concat([rows, changed]), controls)


def test_streaming_raw_means_detection_and_cell_weighting_do_not_impute_missing():
    values = sparse.csr_matrix(np.array([[1, 0, 3], [0, 2, 1], [9, 0, 0]], dtype=np.int32))
    sums = np.zeros((3, 3), dtype=np.int64)
    detected = np.zeros_like(sums)
    n = np.zeros(3, dtype=np.int64)
    accumulate(values[:1], np.array([0]), sums, detected, n)
    accumulate(values[1:], np.array([0, 1]), sums, detected, n)
    mean, rate, available = averages(sums, detected, n)
    np.testing.assert_array_equal(mean[0], [0.5, 1, 2])
    np.testing.assert_array_equal(rate[0], [0.5, 0.5, 1])
    assert not available[2] and np.isnan(mean[2]).all() and np.isnan(rate[2]).all()
    pooled, _, _ = averages(sums[:2].sum(0, keepdims=True), detected[:2].sum(0, keepdims=True), np.array([3]))
    np.testing.assert_allclose(pooled[0], [10 / 3, 2 / 3, 4 / 3])
    assert not np.allclose(pooled[0], mean[:2].mean(0))
    missing = difference(0, 2, mean, rate, n, ["a", "b", "missing"], "synthetic")
    assert missing["status"] == "missing_fit_reference" and missing["raw_mean_mae"] is None


def test_stored_zeros_and_duplicate_sparse_coordinates_are_not_extra_detected_cells():
    values = sparse.csr_matrix((np.array([2, 3, 0]), np.array([0, 0, 1]), np.array([0, 3])), shape=(1, 2))
    sums, detected, n = np.zeros((1, 2), np.int64), np.zeros((1, 2), np.int64), np.zeros(1, np.int64)
    accumulate(values, np.array([0]), sums, detected, n)
    assert sums.tolist() == [[5, 0]] and detected.tolist() == [[1, 0]]
    assert values.nnz == 3  # No mutation of the caller's/source-derived sparse arrays.


@pytest.mark.parametrize("bad", [np.nan, -1, 0.5])
def test_noncounts_rejected(bad):
    with pytest.raises(ValueError, match="raw counts"):
        accumulate(sparse.csr_matrix([[bad]]), np.array([0]), np.zeros((1, 1), np.int64),
                   np.zeros((1, 1), np.int64), np.zeros(1, np.int64))
