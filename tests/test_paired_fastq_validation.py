import gzip
import json
from pathlib import Path
import pytest
from src.lung.data import paired_fastq_validation as module


def fastq(name="q/1", sequence="ACNT", quality="IIII", plus="+"):
    return f"@{name}\n{sequence}\n{plus}\n{quality}\n".encode()


def files(tmp_path, left=None, right=None):
    result = [tmp_path / "R1.fastq.gz", tmp_path / "R2.fastq.gz"]
    for path, content in zip(result, [left if left is not None else fastq(), right if right is not None else fastq("q/2")]):
        path.write_bytes(gzip.compress(content))
    return result


def test_valid_variable_lengths_charset_and_mock_boundary(tmp_path):
    paths = files(tmp_path, fastq() + fastq("b/1", "AC", "!~"), fastq("q/2") + fastq("b/2", "ACG", "ABC"))
    receipt = module.validate_audit_pair(*paths, tmp_path / "out")
    assert receipt["actual_pair_count"] == 2
    assert receipt["actual_sequencing_read_count"] == 4
    assert receipt["mode"] == "audit_mock" and receipt["actual_registered"] is False
    assert receipt["model_forward_training_or_PT_load"] is False
    assert receipt["ends"][0]["quality_offset"] == "unknown_not_inferred"
    assert receipt["ends"][0]["length_histogram"] == {2: 1, 4: 1}
    assert receipt["ends"][0]["N_bases"] == 1
    assert receipt["gzip_CRC_EOF_verified"] is True


@pytest.mark.parametrize("left,right,error", [
    (fastq(), fastq("wrong/2"), "QNAME"),
    (fastq(), b"", "UNEQUAL"),
    (fastq()[:-2], fastq("q/2"), "TRUNCATED"),
    (fastq(plus="+wrong"), fastq("q/2"), "PLUS"),
    (fastq(sequence="ACXG"), fastq("q/2"), "SEQUENCE"),
    (fastq(quality="III"), fastq("q/2"), "QUALITY"),
    (fastq(quality="II I"), fastq("q/2"), "QUALITY"),
    (fastq(quality="IIéI"), fastq("q/2"), "ASCII"),
    (fastq("q/2"), fastq("q/1"), "READ_END"),
    (fastq("q/1 2:N:0:AC"), fastq("q/2"), "CONFLICTING"),
    (fastq("q"), fastq("q/2"), "CONVENTION"),
    (b"", b"", "EMPTY"),
])
def test_invalid_pair_fails_with_receipt(tmp_path, left, right, error):
    paths = files(tmp_path, left, right)
    with pytest.raises(ValueError, match=error):
        module.validate_audit_pair(*paths, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["actual_registered"] is False


@pytest.mark.parametrize("damage", ["truncated", "crc", "trailing"])
def test_gzip_full_eof_crc_failure(tmp_path, damage):
    paths = files(tmp_path)
    data = bytearray(paths[0].read_bytes())
    if damage == "truncated":
        data = data[:-4]
    elif damage == "crc":
        data[-8] ^= 1
    else:
        data += b"garbage"
    paths[0].write_bytes(data)
    with pytest.raises((OSError, EOFError)):
        module.validate_audit_pair(*paths, tmp_path / "out")
    assert json.loads((tmp_path / "out/receipt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("name1,name2", [("q", "q"), ("q/1", "q/2"), ("q 1:N:0:AC", "q 2:N:0:AC")])
def test_supported_source_id_conventions(tmp_path, name1, name2):
    paths = files(tmp_path, fastq(name1, plus="+" + name1.split()[0]), fastq(name2))
    assert module.validate_audit_pair(*paths, tmp_path / "out")["actual_pair_count"] == 1


def test_partial_and_same_file_rejected(tmp_path):
    paths = files(tmp_path)
    part = paths[0].with_suffix(".gz.part")
    paths[0].rename(part)
    with pytest.raises(ValueError, match="NO_PART"):
        module.validate_audit_pair(part, paths[1], tmp_path / "out")
    with pytest.raises(ValueError, match="DISTINCT"):
        module.validate_audit_pair(paths[1], paths[1], tmp_path / "same")


def test_registered_unregistered_and_identity_spoof_refused(tmp_path):
    with pytest.raises(ValueError, match="UNREGISTERED"):
        module.validate_registered_run("madeup", tmp_path / "out")
    with pytest.raises(TypeError):
        module.validate_registered_run("ERR5000305", tmp_path / "out2", raw="caller")


def test_frozen_mapping_mismatch_no_raw_access(tmp_path, monkeypatch):
    changed = tmp_path / "mapping.json"
    changed.write_text("[]")
    monkeypatch.setattr(module, "MAPPING", changed)
    with pytest.raises(ValueError, match="PINNED"):
        module.validate_registered_run("ERR5000305", tmp_path / "out")


def test_exclusive_receipt_directory(tmp_path):
    paths = files(tmp_path)
    module.validate_audit_pair(*paths, tmp_path / "out")
    with pytest.raises(FileExistsError):
        module.validate_audit_pair(*paths, tmp_path / "out")


def test_actual_existing_reference_synthetic_gzip(tmp_path):
    source = module.ROOT / "experiments/20261011-data-alignment-round19/reference_index_probe"
    result = module.validate_audit_pair(source / "R1.fastq.gz", source / "R2.fastq.gz", tmp_path / "out")
    assert result["actual_pair_count"] == 10
    assert result["ends"][0]["length_histogram"] == {75: 10}
    assert result["mode"] == "audit_mock" and result["actual_registered"] is False


def test_changed_during_scan_rejected(tmp_path, monkeypatch):
    paths = files(tmp_path)
    original_digest = module.digest
    calls = 0

    def changed_digest(path):
        nonlocal calls
        calls += 1
        value = original_digest(path)
        if calls == 3:
            return {**value, "sha256": "0" * 64}
        return value

    monkeypatch.setattr(module, "digest", changed_digest)
    with pytest.raises(ValueError, match="CHANGED_DURING"):
        module.scan_pair(*paths)


def test_processing_line_bound(tmp_path):
    paths = files(tmp_path, fastq("q/1", "A" * module.MAX_LINE, "I" * module.MAX_LINE), fastq("q/2"))
    with pytest.raises(ValueError, match="BOUND"):
        module.scan_pair(*paths)


def test_late_concatenated_gzip_crc_failure(tmp_path):
    paths = files(tmp_path)
    late = bytearray(gzip.compress(fastq("b/1")))
    late[-8] ^= 1
    paths[0].write_bytes(paths[0].read_bytes() + late)
    paths[1].write_bytes(paths[1].read_bytes() + gzip.compress(fastq("b/2")))
    with pytest.raises(OSError):
        module.validate_audit_pair(*paths, tmp_path / "out")


def test_failed_receipt_write_never_publishes_final_receipt(tmp_path, monkeypatch):
    paths = files(tmp_path)
    original_open = Path.open

    def fail_stage(path, *args, **kwargs):
        if path.name == "unqualified_receipt.json.part" and args and args[0] == "x":
            raise OSError("synthetic receipt failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_stage)
    with pytest.raises(OSError, match="receipt failure"):
        module.validate_audit_pair(*paths, tmp_path / "out")
    assert not (tmp_path / "out/receipt.json").exists()


@pytest.mark.parametrize("end", [1, 2])
def test_ena_spot_name_end_enforced(end):
    opposite = 3 - end
    assert module.read_identity(f"@ERR5000305.1 spot/{end}".encode(), end, "ERR5000305") == (b"ERR5000305.1", "ena_spot_name_end")
    with pytest.raises(ValueError, match="WRONG_READ_END"):
        module.read_identity(f"@ERR5000305.1 spot/{opposite}".encode(), end, "ERR5000305")
    with pytest.raises(ValueError, match="REGISTERED_ENA"):
        module.read_identity(f"@ERR5000306.1 spot/{end}".encode(), end, "ERR5000305")


@pytest.mark.parametrize("header", [b"@ERR5000305.1 spot/2 extra", b"@ERR5000305.1 spot/1 extra"])
def test_registered_ena_extra_tokens_cannot_hide_end_marker(header):
    with pytest.raises(ValueError, match="UNSUPPORTED_REGISTERED_ENA_HEADER_FORMAT"):
        module.read_identity(header, 1, "ERR5000305")


def test_registered_scan_download_binding_direct_comparison(tmp_path, monkeypatch):
    from types import SimpleNamespace
    run = "ERR5000305"
    paths = files(tmp_path, fastq(run + ".1 spot/1"), fastq(run + ".1 spot/2"))
    original = [path.read_bytes() for path in paths]
    bound = {"files": [{"read_end": end + 1, "path": str(path), **module.digest(path)} for end, path in enumerate(paths)]}
    receipt_path = tmp_path / "download.json"
    receipt_path.write_text("{}")

    def begin(requested):
        assert requested == run
        for end, path in enumerate(paths, 1):
            path.write_bytes(gzip.compress(fastq(f"{run}.1 spot/{end}", "TTTT")))
        return {}, receipt_path, bound, SimpleNamespace(verify_pair=lambda *args, **kwargs: bound)

    original_scan = module.scan_pair

    def middle(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        for path, content in zip(paths, original):
            path.write_bytes(content)
        return result

    monkeypatch.setattr(module, "_registered_source", begin)
    monkeypatch.setattr(module, "scan_pair", middle)
    with pytest.raises(ValueError, match="SCANNED_DOWNLOAD_FILE_BINDING_MISMATCH"):
        module.validate_registered_run(run, tmp_path / "out")
    value = json.loads((tmp_path / "out/receipt.json").read_text())
    assert value["status"] == "failed" and value["actual_registered"] is False
