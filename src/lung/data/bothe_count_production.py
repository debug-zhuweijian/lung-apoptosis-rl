"""Source-bound bulk count readback. No weight/function/model qualification."""
from dataclasses import dataclass, field
import csv
import importlib.util
import json
from pathlib import Path
import re

from .bothe_count_matrix import annotation_axis, sample_axis, parse_featurecounts, sha, _json
from . import bothe_count_executor as driver
from .paired_alignment_filter import POLICY
from .paired_fastq_validation import POLICY as FASTQ_POLICY


def _validation_bindings(row, validation):
    if (validation.get("status") != "complete" or validation.get("mode") != "actual_registered"
            or validation.get("actual_registered") is not True
            or validation.get("run") != row["run"] or validation.get("sample") != row["sample"]
            or validation.get("history") != row["history"] or validation.get("policy") != FASTQ_POLICY
            or validation.get("FASTQ_content_or_pair_sync_verified") is not True
            or validation.get("gzip_CRC_EOF_verified") is not True
            or validation.get("model_forward_training_or_PT_load") is not False
            or validation.get("source_mapping_sha256") != sha(driver.SAMPLES)
            or validation.get("completion_consumer_sha256") != driver.PINNED[driver.ROOT / "scripts/verify_bothe_downloads.py"]
            or validation.get("validator_source_sha256") != driver.PINNED[driver.ROOT / "src/lung/data/paired_fastq_validation.py"]):
        raise ValueError("ACTUAL_REGISTERED_FASTQ_VALIDATION_REQUIRED")
    pairs = validation.get("actual_pair_count")
    if type(pairs) is not int or pairs <= 0 or validation.get("actual_sequencing_read_count") != pairs * 2:
        raise ValueError("INVALID_ACTUAL_FASTQ_RECORD_COUNTS")
    receipt = driver.download_receipt(row["run"]).resolve()
    if Path(validation["completion_receipt_path"]).resolve() != receipt:
        raise ValueError("FIXED_DOWNLOAD_RECEIPT_REQUIRED")
    bindings = {str(receipt): validation["completion_receipt_sha256"]}
    # Validate the fixed download receipt fields again without rehashing raw here.
    source = driver.ROOT / "scripts/verify_bothe_downloads.py"
    if sha(source) != driver.PINNED[source]:
        raise ValueError("PINNED_COMPLETION_CONSUMER_CHANGED")
    spec = importlib.util.spec_from_file_location("_production_bothe_completion", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binding = module.verify_pair(row, _json(receipt), verify_files=False)
    original = dict(validation["download_binding"])
    original["actual_file_bytes_MD5_SHA256_verified"] = False
    original["verification_level"] = "receipt_fields_only"
    if original != binding:
        raise ValueError("DOWNLOAD_VALIDATION_BINDING_MISMATCH")
    files = validation["files"]
    if len(files) != 2 or any(type(r["read_end"]) is not int for r in files) or {r["read_end"] for r in files} != {1, 2}:
        raise ValueError("TWO_REGISTERED_FASTQ_ENDS_REQUIRED")
    for record in files:
        end = record["read_end"]
        path = (driver.DATA / "raw" / row["run"] / f"{row['run']}_{end}.fastq.gz").resolve()
        downloaded = next(entry for entry in binding["files"] if entry["read_end"] == end)
        if (Path(record["path"]).resolve() != path or record["bytes"] != row[f"R{end}_bytes"]
                or record["md5"] != row["md5"].split(";")[end - 1]
                or record["sha256"] != downloaded["sha256"]
                or (driver.ROOT / downloaded["path"]).resolve() != path
                or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])):
            raise ValueError("REGISTERED_FASTQ_SOURCE_BINDING_MISMATCH")
        bindings[str(path)] = record["sha256"]
    return bindings


def _run_bindings(row, payload, directory):
    if (payload.get("status") != "complete" or payload.get("domain") != driver.DOMAIN
            or payload.get("run") != row["run"] or payload.get("sample") != row["sample"]
            or payload.get("history") != row["history"] or Path(payload["directory"]).resolve() != directory):
        raise ValueError("CONTROLLED_EXECUTION_SOURCE_MISMATCH")
    bindings = _validation_bindings(row, payload["FASTQ_validation"])
    if _json(directory / "FASTQ_validation/receipt.json") != payload["FASTQ_validation"]:
        raise ValueError("FASTQ_EXECUTION_RECEIPT_MISMATCH")
    recipes = driver.commands(directory, row)
    if len(payload["commands"]) != 2:
        raise ValueError("EXACT_NATIVE_STAGES_REQUIRED")
    for item, label in zip(payload["commands"], ("subjunc", "featureCounts")):
        if (item.get("label") != label or item.get("argv") != recipes[label]
                or item.get("cwd") != str(directory) or item.get("exit_code") != 0
                or item.get("status") != "complete" or not item.get("started") or not item.get("finished")
                or _json(directory / (label + ".command.json")) != item):
            raise ValueError("FIXED_NATIVE_EXECUTION_REQUIRED")
    filtered = payload["filter_receipt"]
    if (filtered.get("status") != "complete" or filtered.get("policy") != POLICY
            or Path(filtered["source"]).resolve() != directory / "aligned.sam"
            or Path(filtered["output"]).resolve() != directory / "strict_pairs/filtered.sam"
            or filtered["source_sha256"] != sha(directory / "aligned.sam")
            or filtered["output_sha256"] != sha(directory / "strict_pairs/filtered.sam")
            or _json(directory / "strict_pairs/receipt.json") != filtered):
        raise ValueError("STRICT_PAIR_FILTER_EXECUTION_REQUIRED")
    for name in ("execution.json", "FASTQ_validation/receipt.json", "subjunc.command.json", "subjunc.log",
                 "featureCounts.command.json", "featureCounts.log", "aligned.sam", "strict_pairs/filtered.sam",
                 "strict_pairs/receipt.json", "counts.tsv", "counts.tsv.summary"):
        path = directory / name
        bindings[str(path)] = sha(path)
    return bindings


def _matrix(records, directory):
    genes, features = annotation_axis()
    samples = sample_axis()
    totals = {}
    for row in records:
        counts = parse_featurecounts(directory / row["run"] / "counts.tsv",
                                     {"strict_pairs/filtered.sam": row["sample"]}, features)
        totals.update(counts)
    rows = [[gene["ncbi_gene_id"], gene["status"]] +
            [str(totals[s["sample"]][gene["ncbi_gene_id"]])
             if gene["status"] == "retained" and s["sample"] in totals else "" for s in samples]
            for gene in genes]
    return samples, rows


def _publish(directory, completed, bindings):
    """Private publication requires sealed results from actual controlled driver."""
    if not completed or any(type(item) is not driver._CompletedRun for item in completed):
        raise ValueError("ACTUAL_CONTROLLED_EXECUTION_REQUIRED")
    payloads = [json.loads(item.payload_json) for item in completed]
    records = driver.registered_rows([item["run"] for item in payloads])
    expected_directory = driver.version_path(directory.name)
    if directory != expected_directory:
        raise ValueError("FIXED_PRODUCTION_DIRECTORY_REQUIRED")
    if bindings != driver.pinned_bindings(verify_index_files=False):
        raise ValueError("PINNED_EXECUTION_BINDING_SET_REQUIRED")
    all_bindings = dict(bindings)
    for row, payload in zip(records, payloads):
        all_bindings.update(_run_bindings(row, payload, directory / row["run"]))
    all_bindings[str(directory / "driver.json")] = sha(directory / "driver.json")
    if any(sha(path) != checksum for path, checksum in all_bindings.items()):
        raise ValueError("EXECUTION_SOURCE_OR_OUTPUT_CHANGED")
    samples, rows = _matrix(records, directory)
    with (directory / "gene_counts.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["ncbi_gene_id", "annotation_status"] + [s["sample"] for s in samples])
        writer.writerows(rows)
    available = [r["sample"] for r in records]
    manifest = {"schema": "bothe-source-bound-counts-v1", "evidence": "actual_registered_RNA_native_execution",
                "version": directory.name, "domain": driver.DOMAIN, "runs": [r["run"] for r in records],
                "status": "counts_complete" if len(records) == 12 else "prepared_partial",
                "counts_complete": len(records) == 12, "samples": samples, "available_samples": available,
                "missing_samples": [s["sample"] for s in samples if s["sample"] not in available],
                "bindings": all_bindings, "gene_counts_sha256": sha(directory / "gene_counts.tsv"),
                "readyRNA": False, "model_eligible": False, "function_eligible": False,
                "qualification": "registered RNA fragment counts only; no weights, model, function or clinical validation"}
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return manifest


@dataclass(frozen=True)
class BulkCountReadback:
    version: str
    sample_ids: tuple
    gene_ids: tuple
    annotation_status: tuple
    values: tuple
    observed_samples: tuple
    counts_complete: bool
    domain: str = field(default=driver.DOMAIN, init=False)
    model_eligible: bool = field(default=False, init=False)
    function_eligible: bool = field(default=False, init=False)


def load_production(version):
    """One full source/content check per load, not per gene or sample access."""
    directory = driver.version_path(version)
    manifest = _json(directory / "manifest.json")
    records = driver.registered_rows(manifest["runs"])
    lifecycle = _json(directory / "driver.json")
    if (manifest.get("schema") != "bothe-source-bound-counts-v1" or manifest.get("version") != version
            or manifest.get("evidence") != "actual_registered_RNA_native_execution"
            or manifest.get("domain") != driver.DOMAIN
            or any(manifest.get(key) is not False for key in ("readyRNA", "model_eligible", "function_eligible"))
            or lifecycle.get("status") != "complete" or lifecycle.get("runs") != manifest["runs"]
            or lifecycle.get("completed_runs") != manifest["runs"]
            or lifecycle.get("model_forward_training_or_PT_load") is not False):
        raise ValueError("CONTROLLED_RNA_COUNT_LINEAGE_REQUIRED")
    expected = driver.pinned_bindings(verify_index_files=False)
    for row in records:
        run_directory = directory / row["run"]
        expected.update(_run_bindings(row, _json(run_directory / "execution.json"), run_directory))
    expected[str(directory / "driver.json")] = sha(directory / "driver.json")
    if set(expected) != set(manifest["bindings"]) or expected != manifest["bindings"]:
        raise ValueError("REQUIRED_SOURCE_EXECUTION_BINDINGS_MISMATCH")
    # Each large index/raw file is hashed once in this load operation.
    if any(sha(path) != checksum for path, checksum in expected.items()):
        raise ValueError("SOURCE_OR_EXECUTION_CONTENT_CHANGED")
    samples, rows = _matrix(records, directory)
    with (directory / "gene_counts.tsv").open(encoding="utf-8", newline="") as stream:
        actual = list(csv.reader(stream, delimiter="\t"))
    available = [r["sample"] for r in records]
    if (actual != [["ncbi_gene_id", "annotation_status"] + [s["sample"] for s in samples]] + rows
            or sha(directory / "gene_counts.tsv") != manifest["gene_counts_sha256"]
            or manifest["samples"] != samples or manifest["available_samples"] != available
            or manifest["missing_samples"] != [s["sample"] for s in samples if s["sample"] not in available]
            or manifest["counts_complete"] is not (len(records) == 12)
            or manifest["status"] != ("counts_complete" if len(records) == 12 else "prepared_partial")):
        raise ValueError("SOURCE_GENE_MATRIX_CACHE_MISMATCH")
    return BulkCountReadback(version, tuple(s["sample"] for s in samples), tuple(r[0] for r in rows),
                             tuple(r[1] for r in rows), tuple(tuple(int(v) if v else None for v in r[2:]) for r in rows),
                             tuple(available), len(records) == 12)
