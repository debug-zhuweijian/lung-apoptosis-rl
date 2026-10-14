"""Small synthetic parser fixtures; real-source executions are reported separately."""
import gzip
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from lung.data.external_gene_axis import align_gene_axis

SPEC = importlib.util.spec_from_file_location("recovery", Path(__file__).resolve().parents[1] / "scripts/recover_public_expression.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_fractional_estimates_preserved_without_weakening_raw_counts():
    frame = pd.DataFrame({"ensembl_id": ["ENSG1.2", "ENSG2"], "a": [1.001, 0.0]})
    axis = pd.DataFrame({"ensembl_id": ["ENSG2", "ENSG3", "ENSG1"], "common_gene_index": [0, 1, 2]})
    values, mask, _ = align_gene_axis(frame, axis, identifier="ensembl_id", samples=["a"], scale="bulk_estimated_counts")
    np.testing.assert_array_equal(mask, [[True, False, True]])
    np.testing.assert_array_equal(values, [[0.0, np.nan, 1.001]])
    with pytest.raises(ValueError, match="noninteger"):
        align_gene_axis(frame, axis, identifier="ensembl_id", samples=["a"], scale="bulk_raw_counts")
    frame.loc[0, "a"] = -0.01
    with pytest.raises(ValueError, match="negative"):
        align_gene_axis(frame, axis, identifier="ensembl_id", samples=["a"], scale="bulk_estimated_counts")


def test_duplicate_gene_identity_not_silently_averaged():
    frame = pd.DataFrame({"ensembl_id": ["ENSG1.1", "ENSG1.2"], "a": [1.1, 2.1]})
    axis = pd.DataFrame({"ensembl_id": ["ENSG1"], "common_gene_index": [0]})
    values, mask, ledger = align_gene_axis(frame, axis, identifier="ensembl_id", samples=["a"], scale="bulk_estimated_counts")
    assert not mask.any() and np.isnan(values).all()
    assert ledger.mapping_status.tolist() == ["duplicate_source_identifier"] * 2


def test_soft_repeated_metadata_preserved_and_duplicate_samples_rejected(tmp_path):
    path = tmp_path / "source.gz"
    block = "^SAMPLE = GSM1\n!Sample_title = title\n!Sample_characteristics_ch1 = state: sensitive\n!Sample_characteristics_ch1 = time: 24h\n"
    with gzip.open(path, "wt") as f:
        f.write(block)
    _, records = MOD.read_soft(path)
    assert records[0][1]["characteristics_ch1"] == ["state: sensitive", "time: 24h"]
    with gzip.open(path, "wt") as f:
        f.write(block + block)
    with pytest.raises(ValueError, match="duplicate"):
        MOD.read_soft(path)


def test_annexin_blanks_do_not_become_zero_or_replica_pairs(tmp_path):
    rows = [["Cell line", "A549", "A549rCDDP2000"],
            ["Treatment concentration", "11 µM", "11 µM", "34 µM"],
            ["FITC-Annexin [fold change to untreated control]"],
            ["", "6.922", "2.141", "4.956"], ["", "", "2.516", ""],
            ["Mean (SEM)", "ignored", "ignored", "ignored"],
            ["SubG1-phase"], ["", "99", "99", "99"]]
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl>'
    xml += ''.join('<w:tr>' + ''.join('<w:tc><w:p><w:r><w:t>' + c + '</w:t></w:r></w:p></w:tc>' for c in r) + '</w:tr>' for r in rows)
    xml += '</w:tbl></w:body></w:document>'
    path = tmp_path / "source.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)
    data = MOD.sarin_annexin(path)
    assert data.value.tolist() == [6.922, 2.141, 4.956, 2.516]
    assert not data.paired_to_RNA_sample.any()
    assert data.cross_condition_replicate_pairing.eq("not_established").all()
    assert data.unit.eq("fold_change_to_respective_untreated_control").all()


def test_native_matrix_retains_samples_and_values(tmp_path):
    frame = pd.DataFrame({"gene_id": ["ENSG1", "ENSG2"], "colB": [0, 1.25], "colA": [5.5, 0]})
    samples = pd.DataFrame({"matrix_column_id": ["colA", "colB"], "geo_accession": ["GSM2", "GSM1"]})
    report = MOD.save_native(tmp_path, frame, samples, "gene_id", "bulk_estimated_counts")
    with np.load(tmp_path / "source_expression.npz") as a:
        np.testing.assert_array_equal(a["expression"], [[5.5, 0], [0, 1.25]])
        assert a["sample_ids"].tolist() == ["GSM2", "GSM1"]
    assert report["noninteger_values"] == 2


def test_history_preserves_cotreatment_and_duplicate_symbols(tmp_path):
    matrix = tmp_path / "counts.tsv"
    pd.DataFrame({"gene_id": ["A", "A"], "locus": ["chr1:1-2", "chr1:3-4"],
                  "JQ1_1_FPKM": [1.25, 2.5]}).to_csv(matrix, sep="\t", index=False)
    soft = tmp_path / "meta.gz"
    text = ("^SAMPLE = GSM1\n!Sample_title = MSR-A549 cells with JQ1 treatment 1\n"
            "!Sample_characteristics_ch1 = drugs treated: 5uM momelotinib, 1uM selumetinib and 200nM JQ1\n"
            "!Sample_treatment_protocol_ch1 = 200nM JQ1 for 24 hours; withdrawal for 10 days\n")
    with gzip.open(soft, "wt") as f:
        f.write(text)
    frame, samples, _ = MOD.history_expression(matrix, soft)
    assert len(frame) == 2 and frame.source_feature_id.is_unique
    assert samples.co_treatment.iloc[0] == "5uM momelotinib, 1uM selumetinib and 200nM JQ1"
    assert samples.JQ1_exposure_h.iloc[0] == 24
    with gzip.open(soft, "wt") as f:
        f.write(text.replace("5uM momelotinib, 1uM selumetinib and 200nM JQ1", "200nM JQ1"))
    with pytest.raises(ValueError, match="co-treatment"):
        MOD.history_expression(matrix, soft)


def test_pulse_joins_by_gene_and_preserves_unit_conflict_and_source_replicate(tmp_path):
    a, b, soft, paper = [tmp_path / n for n in ("a.csv", "b.csv", "meta.gz", "paper.xml")]
    pd.DataFrame({"gene.id": ["ENSG1", "ENSG2"], "AGR_1_1": [1, 2]}).to_csv(a, index=False)
    pd.DataFrame({"gene.id": ["ENSG2", "ENSG1"], "AGR_8_4": [40, 30]}).to_csv(b, index=False)
    text = ""
    for gsm, title, column in [("GSM1", "A549 Control#1", "AGR_1_1"), ("GSM2", "A549 Cisplatin Post-Sort Proliferative#3", "AGR_8_4")]:
        text += (f"^SAMPLE = {gsm}\n!Sample_title = {title}\n!Sample_description = {column}\n"
                 "!Sample_characteristics_ch1 = cell line: A549\n"
                 "!Sample_characteristics_ch1 = phenotype: Proliferative\n"
                 "!Sample_characteristics_ch1 = facs sorting: Post-sorted\n"
                 "!Sample_treatment_protocol_ch1 = Cells were treated with or without cisplatin (5 mg/mL) for 2 hr\n")
    with gzip.open(soft, "wt") as f:
        f.write(text)
    paper.write_text("<article>For cell sorting and RNAseq analysis, A549 cells were treated with or without cisplatin (5 μg/ml) for 2 hr</article>")
    frame, samples, _ = MOD.pulse_expression(a, b, soft, paper)
    assert frame.AGR_8_4.tolist() == [30, 40]
    assert samples.matrix_column_id.tolist() == ["AGR_1_1", "AGR_8_4"]
    assert samples.paper_dose_ug_per_ml.tolist() == [0, 5]
    assert samples.GEO_protocol_dose_text.eq("5 mg/mL").all()
    paper.write_text("<article>No corroborating protocol</article>")
    with pytest.raises(ValueError, match="dose resolution"):
        MOD.pulse_expression(a, b, soft, paper)


def test_raw_native_counts_cannot_be_fractional(tmp_path):
    frame = pd.DataFrame({"id": ["ENSG1"], "GSM1": [1.25]})
    samples = pd.DataFrame({"geo_accession": ["GSM1"]})
    with pytest.raises(ValueError, match="integers"):
        MOD.save_native(tmp_path, frame, samples, "id", "bulk_raw_counts")
