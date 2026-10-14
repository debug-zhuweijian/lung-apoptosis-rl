"""Synthetic parsing tests; real public-data execution is recorded separately."""
import gzip
from pathlib import Path
import importlib.util
import pytest

SPEC = importlib.util.spec_from_file_location("damtc", Path(__file__).resolve().parents[1] / "scripts/recover_damtc_reference.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


@pytest.fixture
def inputs(tmp_path):
    soft, paper = tmp_path / "meta.gz", tmp_path / "paper.xml"
    text = "!Series_summary = 160 microgram/ml for 24 h\n"
    for i in range(6):
        treatment = "control" if i < 3 else "treated with DAMTC"
        text += (f"^SAMPLE = GSM{i}\n!Sample_title = A549-{treatment} rep{i % 3 + 1}\n"
                 "!Sample_platform_id = GPL6884\n!Sample_data_processing = Average Normalization using Illumina Beadstudio\n"
                 f"!Sample_description = array_{i}\n!sample_table_begin\nID_REF\tVALUE\np1\t-1\np2\t2.5\n!sample_table_end\n")
    with gzip.open(soft, "wt") as f:
        f.write(text)
    paper.write_text('''<article><article-id pub-id-type="doi">10.1038/cddis.2012.141</article-id><body>
    <p>10 mg/ml in DMSO. DMSO-treated cells (0.1%)</p>
    <p>DAMTC (80 <italic>μ</italic>g/ml) alone resulted in 32.37% apoptotic cells;
    DAMTC (160 μg/ml) alone resulted in 50.57%; etoposide (10 μM) alone resulted in 5%;
    DAMTC (80 μg/ml) in combination with etoposide resulted in 36.57%;
    DAMTC (160 μg/ml) in combination with etoposide resulted in 57.1% apoptotic cells</p>
    <fig id="fig8"><caption>mean from four independent experiments</caption></fig></body></article>''')
    return soft, paper


def test_real_units_summary_not_pseudoreplicates(inputs):
    expression, samples, endpoints = MOD.recover(*inputs)
    assert expression.shape == (2, 7)
    assert expression.iloc[0, 1:].eq(-1).all()
    assert samples.DAMTC_ug_ml.tolist() == [0, 0, 0, 160, 160, 160]
    assert endpoints.value.tolist() == [32.37, 50.57, 5, 36.57, 57.1]
    assert len(endpoints) == 5  # Four reported experiments are NOT expanded to fabricated raw data.
    assert not endpoints.paired_to_RNA_replicate.any()
    assert not endpoints.raw_replicates_available.any()
    assert endpoints.combination_order.tolist()[-2:] == ["simultaneous"] * 2


@pytest.mark.parametrize("old,new,match", [
    ("0.1%", "1.6%", "vehicle"),
    ("10.1038/cddis.2012.141", "wrong", "article"),
    ("four independent", "two independent", "count"),
    ("50.57%", "not reported", "sentence"),
])
def test_source_change_requires_re_review(inputs, old, new, match):
    soft, paper = inputs
    paper.write_text(paper.read_text().replace(old, new))
    with pytest.raises(ValueError, match=match):
        MOD.recover(soft, paper)


def test_incomplete_rna_not_paired_to_all_endpoints(inputs):
    soft, paper = inputs
    with gzip.open(soft, "rt") as f:
        text = f.read()
    text = text.rsplit("^SAMPLE", 1)[0]
    with gzip.open(soft, "wt") as f:
        f.write(text)
    with pytest.raises(ValueError, match="three biological"):
        MOD.recover(soft, paper)
