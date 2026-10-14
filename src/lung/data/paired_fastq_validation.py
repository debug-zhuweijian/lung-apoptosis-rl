"""Streaming paired FASTQ validation, with a fixed registered Bothe entry point."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
MAPPING = ROOT / "experiments/20261011-data-alignment-round17/bothe_protocol_source/raw_sample_mapping.json"
MAPPING_SHA = "eac6bced59a1c41e3ecea8311590edc89e6c288ba591cd9587e9bccc31e86ec0"
COMPLETION = ROOT / "scripts/verify_bothe_downloads.py"
COMPLETION_SHA = "8dd499c0f820fd6415ec7025085982d3c436f2ddd5fa201fb5e4c7b5729a7a37"
MAX_LINE = 65536
POLICY = {
    "version": "bothe-paired-fastq-v1", "format": "strict four-line ASCII FASTQ",
    "gzip": "full decompression through CRC and EOF; no prefix qualification",
    "IDs": "identical first token, /1-/2 suffix, Illumina end, or ENA run.spot spot-name/read-index; explicit end must match file",
    "sequence": "nonempty ACGTN/acgtn; variable lengths allowed; line <=65536 bytes",
    "quality": "printable ASCII 33..126; encoding offset not inferred from characters",
    "processing_quality_declaration": "new processing Subjunc -P 3 (Phred+33), separate from observed charset",
    "metadata": "2x75bp describes source protocol, not exact per-record constraint; read_count/base_count comparison reported, not enforced",
    "read_count_semantics": "ENA says number of reads; precise paired generated-FASTQ aggregation remains unknown",
}


def digest(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    return {"bytes": Path(path).stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def read_identity(header, expected_end, expected_run=None):
    if not header.startswith(b"@") or not header[1:] or header[1:2].isspace():
        raise ValueError("INVALID_FASTQ_ID")
    parts = header[1:].split()
    if expected_run is not None and len(parts) > 2:
        raise ValueError("UNSUPPORTED_REGISTERED_ENA_HEADER_FORMAT")
    token, convention, end = parts[0], "identical_token_no_end", None
    if token.endswith((b"/1", b"/2")):
        end, token, convention = int(chr(token[-1])), token[:-2], "slash_end"
    if len(parts) > 1 and re.fullmatch(rb"[12]:[YN]:[0-9]+:[!-~]*", parts[1]):
        illumina_end = int(chr(parts[1][0]))
        if end is not None and end != illumina_end:
            raise ValueError("CONFLICTING_READ_END_MARKERS")
        end, convention = illumina_end, "illumina_end"
    ena = re.fullmatch(rb"([EDS]RR[0-9]+)\.([1-9][0-9]*)", token)
    if expected_run is not None and (ena is None or ena[1].decode("ascii") != expected_run):
        raise ValueError("REGISTERED_ENA_RUN_SPOT_ID_REQUIRED")
    if ena and len(parts) == 2 and parts[1].endswith((b"/1", b"/2")):
        ena_end = int(chr(parts[1][-1]))
        if end is not None and end != ena_end:
            raise ValueError("CONFLICTING_READ_END_MARKERS")
        end, convention = ena_end, "ena_spot_name_end"
    if end is not None and end != expected_end:
        raise ValueError("WRONG_READ_END")
    if not token or re.search(rb"[^!-~]", token):
        raise ValueError("INVALID_FASTQ_QNAME")
    return token, convention


def read_record(stream, end, expected_run=None):
    header = stream.readline(MAX_LINE + 1)
    if not header:
        return None
    lines = [header] + [stream.readline(MAX_LINE + 1) for _ in range(3)]
    if any(not line or not line.endswith(b"\n") for line in lines):
        raise ValueError("TRUNCATED_FOUR_LINE_FASTQ")
    if any(len(line) > MAX_LINE for line in lines):
        raise ValueError("FASTQ_LINE_PROCESSING_BOUND_EXCEEDED")
    lines = [line[:-1].removesuffix(b"\r") for line in lines]
    if any(re.search(rb"[^ -~]", line) for line in lines):
        raise ValueError("NON_ASCII_OR_CONTROL_CHARACTER")
    header, sequence, plus, quality = lines
    qname, convention = read_identity(header, end, expected_run)
    if not plus.startswith(b"+") or (plus[1:] and plus[1:] not in (header[1:], header[1:].split()[0])):
        raise ValueError("PLUS_ID_MISMATCH")
    if not sequence or not re.fullmatch(rb"[ACGTNacgtn]+", sequence):
        raise ValueError("INVALID_SEQUENCE")
    if len(sequence) != len(quality) or re.search(rb"[^!-~]", quality):
        raise ValueError("INVALID_QUALITY_OR_LENGTH")
    return qname, convention, sequence, quality


def scan_pair(r1, r2, *, expected_run=None):
    paths = [Path(r1).resolve(), Path(r2).resolve()]
    if paths[0] == paths[1] or any(not p.name.endswith(".fastq.gz") or not p.is_file() for p in paths):
        raise ValueError("DISTINCT_COMPLETE_FASTQ_GZ_FILES_REQUIRED_NO_PART")
    before = [digest(path) for path in paths]
    lengths, qualities, bases, conventions = [Counter(), Counter()], [Counter(), Counter()], [Counter(), Counter()], [Counter(), Counter()]
    count = 0
    with gzip.open(paths[0], "rb") as left, gzip.open(paths[1], "rb") as right:
        while True:
            records = [read_record(left, 1, expected_run), read_record(right, 2, expected_run)]
            if records == [None, None]:
                break
            if any(record is None for record in records):
                raise ValueError("UNEQUAL_READ_END_RECORD_COUNTS")
            if records[0][0] != records[1][0]:
                raise ValueError("PAIRED_QNAME_OR_ORDER_MISMATCH")
            if records[0][1] != records[1][1]:
                raise ValueError("PAIRED_ID_CONVENTION_MISMATCH")
            count += 1
            for end, (_, convention, sequence, quality) in enumerate(records):
                lengths[end][len(sequence)] += 1
                qualities[end].update(quality)
                bases[end].update(sequence.upper())
                conventions[end][convention] += 1
    if count == 0:
        raise ValueError("EMPTY_FASTQ_PAIR")
    after = [digest(path) for path in paths]
    if before != after:
        raise ValueError("FASTQ_CHANGED_DURING_VALIDATION")
    return {"files": [{"read_end": i + 1, "path": str(path), **after[i]} for i, path in enumerate(paths)],
            "actual_pair_count": count, "actual_sequencing_read_count": 2 * count,
            "gzip_CRC_EOF_verified": True, "FASTQ_content_or_pair_sync_verified": True,
            "ends": [{"read_end": i + 1, "length_histogram": dict(sorted(lengths[i].items())),
                      "records_not_75bp": count - lengths[i][75],
                      "base_count": sum(n * number for n, number in lengths[i].items()),
                      "N_bases": bases[i][ord("N")], "quality_ascii_histogram": dict(sorted(qualities[i].items())),
                      "quality_offset": "unknown_not_inferred", "ID_conventions": dict(conventions[i])} for i in range(2)]}


def _write_receipt(output_dir, receipt):
    pending = output_dir / "unqualified_receipt.json.part"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    pending.rename(output_dir / "receipt.json")


def _validate(output_dir, mode, operation):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "bothe-paired-fastq-validation-v1", "mode": mode, "status": "failed",
               "policy": POLICY, "model_forward_training_or_PT_load": False, "RNA_counts_recovered": False,
               "biological_replicates_created": 0, "actual_registered": False}
    try:
        receipt["validator_source_sha256"] = digest(Path(__file__))["sha256"]
        receipt.update(operation())
        receipt.update(status="complete", actual_registered=mode == "actual_registered")
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _write_receipt(output_dir, receipt)
    return receipt


def validate_audit_pair(r1, r2, output_dir):
    """Arbitrary inputs can ONLY obtain audit_mock status, never registered RNA."""
    return _validate(output_dir, "audit_mock", lambda: scan_pair(r1, r2))


def _registered_source(run):
    if digest(MAPPING)["sha256"] != MAPPING_SHA or digest(COMPLETION)["sha256"] != COMPLETION_SHA:
        raise ValueError("PINNED_REGISTERED_SOURCE_OR_COMPLETION_CONSUMER_CHANGED")
    rows = json.loads(MAPPING.read_text(encoding="utf-8"))
    if len(rows) != 12 or len({r["run"] for r in rows}) != 12:
        raise ValueError("REGISTERED_12_RUN_AXIS_REQUIRED")
    row = next((r for r in rows if r["run"] == run), None)
    if row is None:
        raise ValueError("UNREGISTERED_BOTHE_RUN")
    spec = importlib.util.spec_from_file_location("_pinned_bothe_completion", COMPLETION)
    completion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(completion)
    receipt_path = completion.PILOT if run == "ERR5000305" else completion.REMAINING / (run + ".completed.json")
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    binding = completion.verify_pair(row, value, verify_files=True)
    return row, receipt_path, binding, completion


def validate_registered_run(run, output_dir):
    """Fixed source-only API: caller cannot nominate raw/hash/count/receipt identity."""
    def operation():
        row, receipt_path, binding, completion = _registered_source(run)
        receipt_hash = digest(receipt_path)["sha256"]
        paths = [ROOT / entry["path"] for entry in binding["files"]]
        actual = scan_pair(*paths, expected_run=run)
        scanned = {entry["read_end"]: entry for entry in actual["files"]}
        if len(scanned) != 2 or set(scanned) != {1, 2}:
            raise ValueError("SCANNED_READ_END_BINDING_MISMATCH")
        for entry in binding["files"]:
            observed = scanned[entry["read_end"]]
            if (Path(observed["path"]).resolve() != (ROOT / entry["path"]).resolve()
                    or any(observed[key] != entry[key] for key in ("bytes", "md5", "sha256"))):
                raise ValueError("SCANNED_DOWNLOAD_FILE_BINDING_MISMATCH")
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
        if completion.verify_pair(row, value, verify_files=True) != binding or digest(receipt_path)["sha256"] != receipt_hash:
            raise ValueError("REGISTERED_COMPLETION_CHANGED_DURING_VALIDATION")
        if digest(MAPPING)["sha256"] != MAPPING_SHA or digest(COMPLETION)["sha256"] != COMPLETION_SHA:
            raise ValueError("REGISTERED_SOURCE_CHANGED_DURING_VALIDATION")
        return {**actual, "run": run, "sample": row["sample"], "history": row["history"],
                "source_mapping_sha256": MAPPING_SHA, "completion_consumer_sha256": COMPLETION_SHA,
                "completion_receipt_path": str(receipt_path), "completion_receipt_sha256": receipt_hash,
                "download_binding": binding,
                "source_quality_declaration": {"offset": 33, "type": "Phred", "file_role": "ENA archive-generated FASTQ (fastq_ftp)",
                    "basis": "https://ena-docs.readthedocs.io/en/latest/retrieval/file-download/archive-generated-files.html#fastq-file-format",
                    "inferred_from_observed_characters": False},
                "metadata_comparison": {"declared_read_count": row["read_count"], "declared_base_count": row["base_count"],
                    "pair_count_equal": actual["actual_pair_count"] == row["read_count"],
                    "sequencing_read_count_equal": actual["actual_sequencing_read_count"] == row["read_count"],
                    "total_bases_equal": sum(e["base_count"] for e in actual["ends"]) == row["base_count"],
                    "read_count_aggregation_semantics": "unknown; diagnostic only, not a rejection criterion"}}
    return _validate(output_dir, "actual_registered", operation)
