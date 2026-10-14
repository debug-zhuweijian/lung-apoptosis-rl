"""Synthetic identity/missingness fixtures, never RNA--apoptosis observations."""

import numpy as np
import pandas as pd
import pytest

from scripts.align_gse279184_expression import SAMPLES, align, sample_record


def annotation():
    axis = pd.DataFrame({"common_gene_index": [0, 1, 2], "ensembl_id": ["ENSG1", "ENSG2", "ENSG3"], "gene_symbol": ["A", "B", "C"]})
    hgnc = pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2", "HGNC:3"], "symbol": ["A", "B", "C"],
                         "status": ["Approved"] * 3, "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3"],
                         "prev_symbol": ["OLD_A", "", ""], "alias_symbol": ["shared", "shared", ""]})
    return axis, hgnc


def data(symbols):
    return pd.DataFrame({"GeneSymbol": symbols, **{column: np.arange(len(symbols)) for column in SAMPLES}})


def test_unique_previous_name_maps_while_ambiguous_and_placeholder_remain_missing():
    matrix, mask, ledger = align(data(["OLD_A", "shared", "NO_GENE_NAME"]), *annotation())
    assert ledger.mapping_status.tolist() == ["matched", "ambiguous_or_conflicting_gene_identity", "missing_or_placeholder_source_symbol"]
    assert mask[:, 0].all() and not mask[:, 1:].any()
    assert (matrix[:, 0] == 0).all() and np.isnan(matrix[:, 1:]).all()


def test_different_symbols_resolving_same_gene_cannot_be_averaged_or_selected():
    matrix, mask, ledger = align(data(["A", "OLD_A"]), *annotation())
    assert ledger.mapping_status.eq("duplicate_resolved_gene_not_aggregated").all()
    assert not mask.any() and np.isnan(matrix).all()


def test_exact_axis_hgnc_conflict_is_not_overridden():
    axis, hgnc = annotation()
    hgnc.loc[0, "ensembl_gene_id"] = "ENSG2"
    _, mask, ledger = align(data(["A"]), axis, hgnc)
    assert ledger.mapping_status.iloc[0] == "ambiguous_or_conflicting_gene_identity"
    assert not mask.any()


def test_statistics_not_used_as_counts_and_negative_count_rejected():
    frame = data(["A"]).assign(log2FoldChange=-999, pvalue=0.001)
    matrix, mask, _ = align(frame, *annotation())
    assert (matrix[mask] == 0).all()
    frame.loc[0, "ut_r1"] = -1
    with pytest.raises(ValueError, match="Raw count"):
        align(frame, *annotation())


def test_sample_uses_final_biological_replicate_label_not_middle_R1():
    text = "\n".join(["!Sample_geo_accession = GSM8563735", "!Sample_series_id = GSE279184",
                       "!Sample_title = Saliomycin_Sen_A549_R1_biological replicate 3", "!Sample_source_name_ch1 = A549",
                       "!Sample_data_processing = The de-duplicated gene counts used Ensembl release 102"])
    record = sample_record(text, "GSM8563735")
    assert record["deposited_count_column"] == "tr_r3"
    assert record["salinomycin_treatment_h"] == 48 and not record["RNA_apoptosis_pairing"]
    assert record["measurement_scale"] == "author_deposited_processed_count_table"
    assert record["raw_unmodified_count_matrix_verified"] is False and record["transform_applied"] == "none"
    text = text.replace("Saliomycin", "Untreated")
    reference = sample_record(text, "GSM8563735")
    assert reference["deposited_count_column"] == "ut_r3" and reference["salinomycin_treatment_h"] is None
    assert not reference["observed_t0"]
