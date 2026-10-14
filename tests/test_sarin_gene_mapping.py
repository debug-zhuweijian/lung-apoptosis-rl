import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location("sarin_alignment", Path(__file__).resolve().parents[1] / "scripts/align_sarin_state_expression.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixtures(rows):
    annotation = pd.DataFrame([{"probe_id": f"probe{i}", "REFSEQ": "", "LOCUSLINK_ID": entrez,
                                "GENE_SYMBOL": "same_name_is_not_evidence", "ENSEMBL_ID": tx,
                                "ACCESSION_STRING": other} for i, (entrez, tx, other) in enumerate(rows)])
    axis = pd.DataFrame({"common_gene_index": [0, 1, 2], "ensembl_id": ["ENSG1", "ENSG2", "ENSG3"],
                         "gene_symbol": ["A", "B", "same_name_is_not_evidence"]})
    hgnc = pd.DataFrame({"entrez_id": ["1", "2"], "ensembl_gene_id": ["ENSG1", "ENSG2"],
                         "status": ["Approved", "Approved"], "hgnc_id": ["HGNC:1", "HGNC:2"]})
    tx = pd.DataFrame({"transcript_id": ["ENST1", "ENST2"], "ensembl_gene_id": ["ENSG1", "ENSG2"]})
    return annotation, axis, hgnc, tx


def test_id_conflict_multiple_transcripts_and_name_only_are_quarantined():
    ledger = MODULE.mapping_ledger(*fixtures([("1", "ENST2", ""), ("1", "ENST1", "ens|ENST2"), ("", "", "")]))
    assert ledger.mapping_status.tolist() == ["entrez_transcript_annotation_conflict", "ambiguous_gene_annotation", "no_verified_gene_bridge"]
    assert not ledger.symbol_used_to_choose_gene.any()
    matrix, mask = MODULE.aligned_matrix(np.array([[4.0], [5.0], [6.0]]), ledger, 3)
    assert np.isnan(matrix).all() and not mask.any()


def test_multiple_probes_are_preserved_without_averaging_and_missing_remains_nan():
    ledger = MODULE.mapping_ledger(*fixtures([("1", "ENST1", ""), ("1", "ENST1", ""), ("2", "ENST2", "")]))
    assert ledger.gene_matrix_status.tolist()[:2] == ["multiple_resolved_probes_preserved_not_aggregated"] * 2
    original = np.array([[1.25, 9.9], [3.7, 11.2], [-0.3, 0.0]], dtype=np.float64)
    matrix, mask = MODULE.aligned_matrix(original, ledger, 3)
    np.testing.assert_array_equal(matrix[:, 1], original[2])
    assert np.isnan(matrix[:, [0, 2]]).all()
    assert mask.tolist() == [[False, True, False], [False, True, False]]


def test_partial_or_version_unknown_transcript_cannot_supply_a_gene_without_bridge():
    ledger = MODULE.mapping_ledger(*fixtures([("", "ENST1", "ens|ENST999"), ("", "ENST1.7", ""), ("1", "ENST999", "")]))
    assert ledger.mapping_status.tolist() == ["incomplete_transcript_mapping_without_gene_bridge", "no_verified_gene_bridge", "mapped_one_gene"]
    assert ledger.mapping_basis.iloc[2] == "platform_Entrez_to_HGNC_ENSG"
    assert not ledger.transcript_crosscheck_complete.any()


def test_nonfinite_values_rejected_and_axis_coordinates_checked():
    annotation, axis, hgnc, tx = fixtures([("1", "ENST1", "")])
    ledger = MODULE.mapping_ledger(annotation, axis, hgnc, tx)
    with pytest.raises(ValueError, match="finite"):
        MODULE.aligned_matrix([[np.nan]], ledger, 3)
    axis.loc[1, "common_gene_index"] = 0
    with pytest.raises(ValueError, match="unique"):
        MODULE.mapping_ledger(annotation, axis, hgnc, tx)
