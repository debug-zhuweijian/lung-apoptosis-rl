"""Controlled registered Bothe RNA execution; no model, checkpoint or download."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

from filelock import FileLock

from .bothe_count_matrix import (ROOT, ANNOTATION, SAMPLES, annotation_axis,
                                sample_axis, parse_featurecounts, sha, _json)
from .paired_alignment_filter import POLICY, filter_sam

DATA = ROOT / "08-datasets/Bothe2021/E-MTAB-9923"
PROCESSED = DATA / "processed"
INDEX = DATA / "reference/subread-hg19-index-v1"
TOOLS = DATA / "tools/subread-2.1.1/subread-2.1.1-Windows-x86_64/bin"
INDEX_RECEIPT = ROOT / "experiments/20261011-data-alignment-round18/index_completion_validation.json"
PILOT_RECEIPT = ROOT / "experiments/20261011-data-alignment-round18/bothe_download_pilot.json"
REMAINING = ROOT / "experiments/20261011-data-alignment-round19/raw_remaining_acquisition"
DOMAIN = "bulk_total_RNA_unique_intron_fragment_counts"
PINNED = {
    ROOT / "src/lung/data/bothe_count_matrix.py": "e59cae1286c551e86e074225adac6f1105612bb0b83ae11c3f45151f44c55dd1",
    ROOT / "src/lung/data/paired_alignment_filter.py": "a45b8807277319522c9f820b519bfe5ccc083301dcb0a06eaafd6abbbf5e3f87",
    ROOT / "scripts/verify_bothe_downloads.py": "8dd499c0f820fd6415ec7025085982d3c436f2ddd5fa201fb5e4c7b5729a7a37",
    TOOLS / "subjunc.exe": "b26da8b3f41ed706e687d503ac4fb4aeb79709b1857ed2c9871dffec02dc4743",
    TOOLS / "featureCounts.exe": "30cf41c2bac4707754b4d7faab96743a16957bdb21ecee82e64fd941905185d9",
    INDEX_RECEIPT: "cc9013c3522fdad2c761325ddcebb18102f338879ac6f90f7061ef28adfc6d43",
    ROOT / "src/lung/data/paired_fastq_validation.py": "c669c1cb8b3ef38dd8d3ea9d97551b00e7264b4f09d5ac9619962022faf330da",
}
# Frozen index completion fixes all six index hashes, separately from annotation.
_RUN_SEAL = object()


def utc():
    return datetime.now(timezone.utc).isoformat()


def version_path(version):
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", version):
        raise ValueError("INVALID_NEW_PROCESSING_VERSION")
    path = (PROCESSED / version).resolve()
    if path.parent != PROCESSED.resolve() or not path.is_relative_to(DATA.resolve()):
        raise ValueError("VERSION_PATH_ESCAPE")
    return path


def registered_rows(runs):
    if not isinstance(runs, (tuple, list)) or not runs or any(not isinstance(r, str) for r in runs) or len(runs) != len(set(runs)):
        raise ValueError("UNIQUE_REGISTERED_RUNS_REQUIRED")
    mapping = {row["run"]: row for row in sample_axis()}
    if not set(runs) <= mapping.keys():
        raise ValueError("UNREGISTERED_BOTHE_RUN")
    return [mapping[run] for run in runs]


def download_receipt(run):
    return PILOT_RECEIPT if run == "ERR5000305" else REMAINING / (run + ".completed.json")


def pinned_bindings(*, verify_index_files=True):
    bindings = {}
    for path, expected in PINNED.items():
        if sha(path) != expected:
            raise ValueError("PINNED_CODE_TOOL_OR_INDEX_RECEIPT_CHANGED: " + path.name)
        bindings[str(path.resolve())] = expected
    index_receipt = _json(INDEX_RECEIPT)
    run_path = INDEX / "run.json"
    run = _json(run_path)
    if (sha(run_path) != index_receipt["run_json_sha256"] or run.get("status") != "completed"
            or run.get("exit_code") != 0 or len(run["files"]) != 6
            or run["files"] != index_receipt["independently_verified_files"]):
        raise ValueError("COMPLETED_PINNED_HUMAN_INDEX_REQUIRED")
    bindings[str(run_path.resolve())] = sha(run_path)
    for name, record in run["files"].items():
        path = INDEX / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("INDEX_FILE_SIZE_MISMATCH")
        if verify_index_files and sha(path) != record["sha256"]:
            raise ValueError("INDEX_FILE_HASH_MISMATCH")
        bindings[str(path.resolve())] = record["sha256"]
    annotation_axis()
    for name in ("manifest.json", "unique_introns.saf", "feature_gene_mapping.tsv", "gene_table.tsv"):
        bindings[str((ANNOTATION / name).resolve())] = sha(ANNOTATION / name)
    bindings[str(SAMPLES.resolve())] = sha(SAMPLES)
    for path in (Path(__file__), ROOT / "src/lung/data/bothe_count_production.py", ROOT / "scripts/process_bothe_rna.py"):
        bindings[str(path.resolve())] = sha(path)
    return bindings


def estimate_storage(run):
    row = registered_rows([run])[0]
    # ENA declares number of reads; aggregation in generated FASTQ is unknown.
    records = row["read_count"]
    return {"run": run, "nature": "unvalidated planning estimate; not measured RAM or disk",
            "declared_ENA_read_count": records, "declared_bases": row["base_count"],
            "denominator_assumption": "read_count used as putative SAM record count; FASTQ aggregate semantics unconfirmed",
            "declared_mean_read_length": row["base_count"] / records,
            "assumed_SAM_bytes_per_record_range": [250, 600],
            "estimated_primary_only_SAM_bytes_range": [records * 250, records * 600],
            "SQLite_cost": "stores original SAM record BLOB + QNAME index and accepted keys; overhead unmeasured",
            "peak_disk": "raw gzip + SAM + SQLite + filtered SAM + count output coexist; secondary records add cost",
            "RAM": "Subjunc human-index peak unmeasured; SQLite cache policy 8MiB is not total process RAM"}


def preflight(run, *, verify_index_files=False):
    row = registered_rows([run])[0]
    receipt = download_receipt(run)
    raw_files = [DATA / "raw" / run / f"{run}_{end}.fastq.gz" for end in (1, 2)]
    missing = [str(p) for p in [receipt, *raw_files] if not p.is_file()]
    result = {"run": run, "sample": row["sample"], "history": row["history"],
              "status": "pending_raw_download" if missing else "prepared_requires_full_FASTQ_validation",
              "missing_paths": missing, "estimate": estimate_storage(run),
              "index_hashes_actually_recomputed": verify_index_files is True,
              "RNA_counts_recovered": False, "model_eligible": False}
    # Metadata preflight never decompresses/qualifies raw RNA.
    result["bindings"] = pinned_bindings(verify_index_files=verify_index_files)
    return result


def commands(run_directory, row):
    run_directory = Path(run_directory).resolve()
    relative = lambda path: os.path.relpath(path, run_directory).replace("\\", "/")
    subjunc = [str(TOOLS / "subjunc.exe"), "-T", "1", "-P", "3", "-n", "14", "-m", "1",
               "-p", "1", "-M", "3", "-I", "5", "-d", "50", "-D", "600", "-S", "fr",
               "-i", relative(INDEX / "hg19"),
               "-r", relative(DATA / "raw" / row["run"] / f"{row['run']}_1.fastq.gz"),
               "-R", relative(DATA / "raw" / row["run"] / f"{row['run']}_2.fastq.gz"),
               "--SAMoutput", "-o", "aligned.sam"]
    count = [str(TOOLS / "featureCounts.exe"), "-T", "1", "-F", "SAF", "-a",
             relative(ANNOTATION / "unique_introns.saf"), "-o", "counts.tsv", "-f", "-p",
             "--countReadPairs", "-B", "-C", "--primary", "-Q", "10", "-s", "0", "strict_pairs/filtered.sam"]
    return {"subjunc": subjunc, "featureCounts": count}


def _write(path, value):
    # Only the new executor-owned lifecycle file is updated, never source files.
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _stage(label, argv, directory):
    value = {"label": label, "argv": argv, "cwd": str(directory), "status": "running", "started": utc()}
    receipt_path = directory / (label + ".command.json")
    _write(receipt_path, value)
    try:
        with (directory / (label + ".log")).open("xb") as log:
            result = subprocess.run(argv, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
    except BaseException as error:
        value.update(status="failed", error_type=type(error).__name__, error=str(error), finished=utc())
        _write(receipt_path, value)
        raise
    value.update(exit_code=result.returncode, status="complete" if result.returncode == 0 else "failed", finished=utc())
    _write(receipt_path, value)
    if result.returncode:
        raise RuntimeError(label + "_NONZERO_EXIT: " + str(result.returncode))
    return value


@dataclass(frozen=True, init=False)
class _CompletedRun:
    payload_json: str

    def __init__(self, payload, *, _seal=None):
        if _seal is not _RUN_SEAL:
            raise ValueError("ACTUAL_CONTROLLED_EXECUTION_REQUIRED")
        object.__setattr__(self, "payload_json", json.dumps(payload, sort_keys=True, allow_nan=False))


def _execute_one(row, directory):
    from .paired_fastq_validation import validate_registered_run
    directory.mkdir()
    validation = validate_registered_run(row["run"], directory / "FASTQ_validation")
    from .bothe_count_production import _validation_bindings
    _validation_bindings(row, validation)
    recipes = commands(directory, row)
    subjunc = _stage("subjunc", recipes["subjunc"], directory)
    filtered = filter_sam(directory / "aligned.sam", directory / "strict_pairs")
    if (filtered.get("status") != "complete" or filtered.get("policy") != POLICY
            or filtered["source_sha256"] != sha(directory / "aligned.sam")
            or filtered["output_sha256"] != sha(directory / "strict_pairs/filtered.sam")):
        raise ValueError("STRICT_PAIR_FILTER_COMPLETION_REQUIRED")
    count = _stage("featureCounts", recipes["featureCounts"], directory)
    _, features = annotation_axis()
    parse_featurecounts(directory / "counts.tsv", {"strict_pairs/filtered.sam": row["sample"]}, features)
    # summary must exist, but assignment is not used to invent biological labels.
    if not (directory / "counts.tsv.summary").is_file():
        raise ValueError("COUNT_SUMMARY_REQUIRED")
    payload = {"run": row["run"], "sample": row["sample"], "history": row["history"],
               "directory": str(directory), "FASTQ_validation": validation,
               "commands": [subjunc, count], "filter_receipt": filtered,
               "status": "complete", "domain": DOMAIN}
    _write(directory / "execution.json", payload)
    return _CompletedRun(payload, _seal=_RUN_SEAL)


def _wait_for_registered_download(row, directory, lifecycle):
    """Wait for fixed committed files; qualification still occurs in _execute_one."""
    required = [download_receipt(row["run"])] + [
        DATA / "raw" / row["run"] / f"{row['run']}_{end}.fastq.gz" for end in (1, 2)]
    missing = [str(path) for path in required if not path.is_file()]
    if not missing:
        return
    lifecycle.update(waiting_run=row["run"], waiting_since=utc(),
                     waiting_for="fixed_download_commit_and_both_complete_FASTQ_files",
                     missing_download_paths=missing)
    _write(directory / "driver.json", lifecycle)
    print(json.dumps({"status": "waiting_for_registered_download", "run": row["run"],
                      "missing_paths": missing, "RNA_counts_recovered": False}), flush=True)
    while any(not path.is_file() for path in required):
        time.sleep(30)
    for key in ("waiting_run", "waiting_since", "waiting_for", "missing_download_paths"):
        lifecycle.pop(key, None)
    _write(directory / "driver.json", lifecycle)


def _execute_registered(runs, version, *, wait_for_downloads):
    """Only registered source inputs and a new version name are accepted."""
    from .bothe_count_production import _publish
    rows = registered_rows(runs)
    directory = version_path(version)
    if directory.exists():
        raise FileExistsError("NEW_PROCESSING_VERSION_REQUIRED_NO_OVERWRITE")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # OS-backed FileLock is released after process death. Its file is not liveness.
    with FileLock(str(PROCESSED / ".native-count.lock"), timeout=0):
        directory.mkdir()
        lifecycle = {"schema": "bothe-controlled-count-execution-v1", "status": "running",
                     "started": utc(), "runs": list(runs), "domain": DOMAIN,
                     "model_forward_training_or_PT_load": False, "completed_runs": [],
                     "wait_for_fixed_registered_downloads": wait_for_downloads}
        _write(directory / "driver.json", lifecycle)
        try:
            bindings = pinned_bindings(verify_index_files=True)
            from . import paired_fastq_validation
            validator_path = Path(paired_fastq_validation.__file__).resolve()
            bindings[str(validator_path)] = sha(validator_path)
            completed = []
            for row in rows:
                if wait_for_downloads:
                    _wait_for_registered_download(row, directory, lifecycle)
                completed.append(_execute_one(row, directory / row["run"]))
                lifecycle["completed_runs"].append(row["run"])
                _write(directory / "driver.json", lifecycle)
            if any(sha(Path(path)) != checksum for path, checksum in bindings.items()):
                raise ValueError("EXECUTION_INPUT_CODE_TOOL_INDEX_CHANGED")
            lifecycle.update(status="complete", finished=utc())
            _write(directory / "driver.json", lifecycle)
            return _publish(directory, completed, bindings)
        except BaseException as error:
            lifecycle.update(status="failed", error_type=type(error).__name__, error=str(error), finished=utc())
            _write(directory / "driver.json", lifecycle)
            raise


def execute_registered_runs(runs, version):
    return _execute_registered(runs, version, wait_for_downloads=False)


def execute_registered_when_ready(runs, version):
    """Serial real-data job that waits for each fixed download without a time cap."""
    return _execute_registered(runs, version, wait_for_downloads=True)


def execute_registered_run(run, version):
    return execute_registered_runs([run], version)
