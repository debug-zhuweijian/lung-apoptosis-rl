import csv
import json
from pathlib import Path

import pytest

from lung.data.bothe_count_matrix import (annotation_axis, build_audit_mock,
    build_real, load_matrix, parse_featurecounts, sample_axis, sha)


def write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path):
    annotation = tmp_path / "annotation"
    annotation.mkdir()
    write(annotation / "unique_introns.saf", ["GeneID", "Chr", "Start", "End", "Strand"],
          [["f1", "chr1", 11, 20, "+"], ["f2", "chr1", 31, 40, "+"], ["f3", "chr2", 1, 9, "-"]])
    write(annotation / "feature_gene_mapping.tsv", ["feature_id", "ncbi_gene_id", "chrom", "strand", "start_0based", "end_exclusive", "length_bp"],
          [["f1", "1", "chr1", "+", 10, 20, 10], ["f2", "1", "chr1", "+", 30, 40, 10], ["f3", "2", "chr2", "-", 0, 9, 9]])
    write(annotation / "gene_table.tsv", ["ncbi_gene_id", "retained_interval_count", "retained_intron_bp", "status"],
          [["1", 2, 20, "retained"], ["2", 1, 9, "retained"], ["3", 0, 0, "no_source_intron"], ["4", 0, 0, "all_intronic_bases_excluded"]])
    (annotation / "manifest.json").write_text(json.dumps({"output_files": {p.name: {"sha256": sha(p)} for p in annotation.iterdir()}}))
    path = tmp_path / "counts.txt"
    rows = [["f1", "chr1", 11, 20, "+", 10, 2], ["f2", "chr1", 31, 40, "+", 10, 3], ["f3", "chr2", 1, 9, "-", 9, 0]]
    header = ["Geneid", "Chr", "Start", "End", "Strand", "Length", "sample.bam"]
    write(path, header, rows)
    columns = {"sample.bam": sample_axis()[0]["sample"]}
    return annotation, path, rows, header, columns


def test_fixed_axes():
    genes, features = annotation_axis()
    assert (len(genes), len(features)) == (29634, 207555)
    assert len(sample_axis()) == 12


def test_partial_zero_unmeasurable_missing(fixture, tmp_path):
    annotation, path, _, _, columns = fixture
    output = tmp_path / "matrix"
    result = build_audit_mock(output, [{"path": path, "column_samples": columns}], annotation=annotation)
    assert result["status"] == "prepared_partial"
    assert len(result["missing_samples"]) == 11
    assert result["readyRNA"] is False
    with pytest.raises(ValueError, match="opt-in"):
        load_matrix(output)
    _, rows = load_matrix(output, allow_audit_mock=True)
    assert rows[0][2:4] == ("5", "")
    assert rows[1][2:4] == ("0", "")
    assert rows[2][2:] == ("",) * 12
    assert rows[3][2:] == ("",) * 12
    with pytest.raises(ValueError, match="overwrite"):
        build_audit_mock(output, [], annotation=annotation)


@pytest.mark.parametrize("field,value", [(1,"chr3"),(2,"10"),(3,"21"),(4,"-"),(5,"11"),(6,"-1"),(6,"1.0"),(6,"nan"),(6,"True"),(6,"01"),(6,str(2**63))])
def test_corrupt_metadata_or_count(fixture, field, value):
    annotation, path, rows, header, columns = fixture
    rows[0][field] = value
    write(path, header, rows)
    with pytest.raises(ValueError):
        parse_featurecounts(path, columns, annotation_axis(annotation, frozen=False)[1])


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown", "ragged", "header", "unbound"])
def test_bad_axis(fixture, mutation):
    annotation, path, rows, header, columns = fixture
    if mutation == "duplicate": rows.append(rows[0])
    if mutation == "missing": rows.pop()
    if mutation == "unknown": rows[0][0] = "unknown"
    if mutation == "ragged": rows[0].pop()
    if mutation == "header": header[0] = "GeneID"
    if mutation == "unbound": header[-1] = "other.bam"
    write(path, header, rows)
    with pytest.raises(ValueError):
        parse_featurecounts(path, columns, annotation_axis(annotation, frozen=False)[1])


@pytest.mark.parametrize("mutation", ["eligibility", "counts", "bindings", "samples", "source"])
def test_loader_rechecks(fixture, tmp_path, mutation):
    annotation, path, _, _, columns = fixture
    output = tmp_path / "matrix"
    build_audit_mock(output, [{"path": path, "column_samples": columns}], annotation=annotation)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "eligibility": manifest["readyRNA"] = True
    if mutation == "counts":
        p = output / "counts.tsv"
        p.write_text(p.read_text().replace("\t5\t", "\t9\t"))
        manifest["counts_sha256"] = sha(p)
    if mutation == "bindings": manifest["bindings"] = {}
    if mutation == "samples": manifest["available_samples"] = []
    if mutation == "source": path.write_text(path.read_text().replace("\t2\n", "\t8\n"))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_matrix(output, allow_audit_mock=True)


def test_real_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="pending_verified_count_executor"):
        build_real(tmp_path / "real", receipt={"real": True}, annotation_sha256="caller")
    assert not (tmp_path / "real").exists()


def test_duplicate_and_unknown_samples(fixture, tmp_path):
    annotation, path, _, _, columns = fixture
    entry = {"path": path, "column_samples": columns}
    with pytest.raises(ValueError, match="more than once"):
        build_audit_mock(tmp_path / "duplicate", [entry, entry], annotation=annotation)
    with pytest.raises(ValueError, match="unknown sample"):
        build_audit_mock(tmp_path / "unknown", [{"path": path, "column_samples": {"sample.bam": "invented"}}], annotation=annotation)


def test_all_twelve_still_mock(fixture, tmp_path):
    annotation, path, rows, header, _ = fixture
    samples = sample_axis()
    columns = {s["sample"] + ".bam": s["sample"] for s in samples}
    write(path, header[:6] + list(columns), [r[:6] + [r[-1]] * 12 for r in rows])
    result = build_audit_mock(tmp_path / "all", [{"path": path, "column_samples": columns}], annotation=annotation)
    assert result["status"] == "audit_complete"
    assert result["readyRNA"] is False
    load_matrix(tmp_path / "all", allow_audit_mock=True)


def test_non_utf8_native_comment(fixture):
    annotation, path, _, _, columns = fixture
    path.write_bytes(b'# Program: "C:\\' + "科研".encode("cp936") + b'\\featureCounts.exe"\n' + path.read_bytes())
    result = parse_featurecounts(path, columns, annotation_axis(annotation, frozen=False)[1])
    assert result[sample_axis()[0]["sample"]] == {"1": 5, "2": 0}


def test_actual_native_tiny_bytes():
    # Round18 actual native execution on synthetic SAM, never real RNA.
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/20261011-data-alignment-round18/bothe_portable_runtime/tiny_counts.tsv"
    original = sha(path)
    assert any(line.startswith(b"#") for line in path.read_bytes().splitlines())
    result = parse_featurecounts(path, {"tiny.sam": "audit_synthetic"},
                                 {"I1": ("audit_gene", ("chr1", "101", "200", "+", "100"))})
    assert result == {"audit_synthetic": {"audit_gene": 2}}
    assert sha(path) == original


@pytest.mark.parametrize("position", ["header", "data"])
def test_non_utf8_table_rejected(fixture, position):
    annotation, path, _, _, columns = fixture
    data = path.read_bytes()
    if position == "header":
        data = b"\xff" + data
    else:
        data = data.replace(b"f1\t", b"f1\xff\t")
    path.write_bytes(data)
    with pytest.raises(UnicodeDecodeError):
        parse_featurecounts(path, columns, annotation_axis(annotation, frozen=False)[1])
