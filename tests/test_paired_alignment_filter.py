import json
from pathlib import Path
import pytest
from src.lung.data.paired_alignment_filter import filter_sam

HEADER = "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:1000\n"


def read(q="pair", flag=99, pos=110, mate=150, quality=20, tlen=50, cigar="10M", nh="NH:i:1"):
    return "\t".join(map(str, [q, flag, "chr1", pos, quality, cigar, "=", mate, tlen, "A" * 10, "I" * 10, nh])) + "\n"


def pair(q="pair"):
    return [read(q), read(q, flag=147, pos=150, mate=110, tlen=-50)]


def run(tmp_path, lines):
    source = tmp_path / "source.sam"
    source.write_text(HEADER + "".join(lines), encoding="utf-8")
    receipt = filter_sam(source, tmp_path / "result")
    return receipt, (tmp_path / "result/filtered.sam").read_text()


def test_nonadjacent_preserves_header_and_input_order(tmp_path):
    a, b = pair("a"), pair("b")
    receipt, output = run(tmp_path, [a[0], b[0], a[1], b[1]])
    assert output == HEADER + a[0] + b[0] + a[1] + b[1]
    assert receipt["counters"]["templates_retained"] == 2
    assert len(receipt["source_sha256"]) == len(receipt["output_sha256"]) == 64


@pytest.mark.parametrize("lines,reason", [
    ([read()], "orphan"),
    (pair() + [read()], "duplicate_primary"),
    ([read(), read(flag=99, pos=150, mate=110, tlen=-50)], "invalid_R1_R2"),
    ([read(quality=9), pair()[1]], "MAPQ"),
    ([read(quality=255), pair()[1]], "MAPQ"),
    ([read(mate=151), pair()[1]], "nonreciprocal_coordinates"),
    ([read(flag=67), pair()[1]], "nonreciprocal_strand_flags"),
    ([read(tlen=51), pair()[1]], "nonreciprocal_TLEN"),
    ([read(flag=103), pair()[1]], "not_paired_both_mapped"),
    ([read(nh="NH:i:2"), pair()[1]], "NH_multimap"),
    ([read(cigar="9M"), pair()[1]], "malformed"),
    ([read(cigar="0M10M"), pair()[1]], "malformed"),
    ([read(cigar="5M1S4M"), pair()[1]], "malformed"),
    ([read(cigar="1S1H9M"), pair()[1]], "malformed"),
    ([read(cigar="9M1H1S"), pair()[1]], "malformed"),
    ([read(pos=999), pair()[1]], "malformed"),
    ([read(nh="NH:i:1\tNH:i:1"), pair()[1]], "malformed"),
])
def test_whole_template_rejection(tmp_path, lines, reason):
    receipt, output = run(tmp_path, lines)
    assert output == HEADER
    assert receipt["counters"]["rejected_" + reason] == 1


def test_nonprimary_excluded_unknown_nh_duplicates_retained(tmp_path):
    lines = [read(flag=1121, nh="NM:i:0"), read(flag=1171, pos=150, mate=110, tlen=-50, nh="NM:i:0"), read(flag=355), read(flag=2147)]
    receipt, output = run(tmp_path, lines)
    assert len(output.splitlines()) == 4
    assert receipt["counters"]["excluded_nonprimary_records"] == 2
    assert receipt["counters"]["retained_NH_unknown_templates"] == 1
    assert receipt["counters"]["retained_duplicate_flag_templates"] == 1


def test_secondary_cannot_stand_in_for_missing_primary(tmp_path):
    receipt, output = run(tmp_path, [read(), read(flag=403, pos=150, mate=110, tlen=-50)])
    assert output == HEADER
    assert receipt["counters"]["rejected_orphan"] == 1


def test_cross_header_fails_closed(tmp_path):
    source = tmp_path / "bad.sam"
    source.write_text(HEADER + pair()[0] + HEADER + pair()[1])
    with pytest.raises(ValueError, match="header after"):
        filter_sam(source, tmp_path / "out")
    assert not (tmp_path / "out/filtered.sam").exists()
    assert json.loads((tmp_path / "out/receipt.json").read_text())["status"] == "failed"


def test_refuses_existing_output(tmp_path):
    run(tmp_path, pair())
    with pytest.raises(FileExistsError):
        filter_sam(tmp_path / "source.sam", tmp_path / "result")


@pytest.mark.parametrize("cigar", ["1H1S8M1S1H", "1H10M", "10M1H", "1S9M", "9M1S"])
def test_valid_clipping_order(tmp_path, cigar):
    receipt, output = run(tmp_path, [read(cigar=cigar), pair()[1]])
    assert receipt["counters"]["templates_retained"] == 1
    assert len(output.splitlines()) == 4


def test_duplicate_sq_tag_fails_closed(tmp_path):
    source = tmp_path / "bad.sam"
    source.write_text(HEADER.replace("SN:chr1", "SN:wrong\tSN:chr1") + "".join(pair()))
    with pytest.raises(ValueError, match="duplicate header tag"):
        filter_sam(source, tmp_path / "out")
    assert not (tmp_path / "out/filtered.sam").exists()
    assert json.loads((tmp_path / "out/receipt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("header", ["@HD\tVN:1.6\tVN:1.6\n", "@RG\tID:a\tID:b\n", "@PG\tID:a\tID:b\n"])
def test_non_sq_duplicate_header_tags(tmp_path, header):
    source = tmp_path / "bad.sam"
    source.write_text(header + HEADER + "".join(pair()))
    with pytest.raises(ValueError, match="duplicate header tag"):
        filter_sam(source, tmp_path / "out")
    assert not (tmp_path / "out/filtered.sam").exists()


def test_co_arbitrary_comment_preserved(tmp_path):
    source = tmp_path / "comment.sam"
    source.write_text(HEADER + "@CO\tfree text\tID:a\tID:b\n" + "".join(pair()))
    receipt = filter_sam(source, tmp_path / "out")
    assert receipt["counters"]["templates_retained"] == 1
    assert (tmp_path / "out/filtered.sam").read_bytes() == source.read_bytes()


def test_write_failure_does_not_publish_partial_pair(tmp_path, monkeypatch):
    source = tmp_path / "source.sam"
    source.write_text(HEADER + "".join(pair()))
    original_open = Path.open

    class BrokenOutput:
        def __init__(self, stream):
            self.stream, self.calls = stream, 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.calls += 1
            if self.calls == 4:
                raise OSError("synthetic disk write failure")
            return self.stream.write(data)

    def patched_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.name == "unqualified_filtered.sam.part" and args and args[0] == "xb":
            return BrokenOutput(stream)
        return stream

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="synthetic disk"):
        filter_sam(source, tmp_path / "out")
    assert not (tmp_path / "out/filtered.sam").exists()
    assert len((tmp_path / "out/unqualified_filtered.sam.part").read_text().splitlines()) == 3
    assert json.loads((tmp_path / "out/receipt.json").read_text())["status"] == "failed"


def test_existing_round18_actual_synthetic_readback(tmp_path):
    root = Path(__file__).resolve().parents[1]
    smoke = root / "experiments/20261011-data-alignment-round18/bothe_portable_runtime"
    for index, (source, retained) in enumerate([(smoke / "tiny.sam", 1), (smoke / "tiny_native_smoke/absolute2.sam", 2), (smoke / "tiny_native_smoke/relative2.sam", 2)]):
        receipt = filter_sam(source, tmp_path / str(index))
        assert receipt["counters"]["templates_retained"] == retained
        assert receipt["counters"]["output_records"] == 2 * retained


def test_receipt_write_failure_demotes_published_sam(tmp_path, monkeypatch):
    source = tmp_path / "source.sam"
    source.write_text(HEADER + "".join(pair()))
    original_write = Path.write_text

    def fail_receipt(path, *args, **kwargs):
        if path.name == "unqualified_receipt.json.part":
            raise OSError("synthetic receipt write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_receipt)
    with pytest.raises(OSError, match="receipt write"):
        filter_sam(source, tmp_path / "out")
    assert not (tmp_path / "out/filtered.sam").exists()
    assert not (tmp_path / "out/receipt.json").exists()
    assert (tmp_path / "out/unqualified_filtered.sam.part").read_bytes() == source.read_bytes()
