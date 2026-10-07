"""Download pinned Tahoe shards and extract every published A549 expression row.

Source shards remain on disk. Atomic per-shard checkpoints support restart.
No experiment, normalization, feature selection or training is performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from scipy import sparse

REPO = "tahoebio/Tahoe-100M"
REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
A549 = "CVCL_0023"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets" / "Tahoe-100M" / REVISION
OUT = BASE / "full_expression"
SOURCE_ROWS = 95624334
DECODE_SLOTS = threading.Semaphore(2)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc):
    # requests exception text can contain a redirected signed download URL.
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return f"{type(exc).__name__}" + (f" HTTP {status}" if status is not None else "")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    st = path.stat()
    return {"path": str(path.relative_to(BASE)).replace("\\", "/"),
            "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest(path)}


def get_manifest():
    dest = OUT / "source_manifest.json"
    if dest.exists():
        data = json.loads(dest.read_text(encoding="utf-8"))
        if data["revision"] != REVISION or len(data["files"]) != 3388:
            raise ValueError("Unexpected cached source manifest")
        return data
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data?limit=1000"
    files = []
    while url:
        r = requests.get(url, timeout=(30, 120))
        r.raise_for_status()
        files.extend(x for x in r.json() if x["path"].endswith(".parquet"))
        url = r.links.get("next", {}).get("url")
    files.sort(key=lambda x: x["path"])
    if len(files) != 3388 or len({x["path"] for x in files}) != 3388:
        raise ValueError("Incomplete source shard inventory")
    if any(x["lfs"]["size"] != x["size"] for x in files):
        raise ValueError("Source size disagreement")
    data = {"repository": REPO, "revision": REVISION, "queried_utc": now(),
            "source_bytes": sum(x["size"] for x in files), "files": files}
    write_json(dest, data)
    return data


def fetch(entry, attempts):
    dest = BASE / entry["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size != entry["size"] or digest(dest) != entry["lfs"]["oid"]:
            raise ValueError(f"Existing source failed SHA256: {dest}")
        return dest
    part = dest.with_suffix(dest.suffix + ".partial")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}"
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > entry["size"]:
                raise ValueError("Partial file larger than source")
            while offset < entry["size"]:
                end = min(offset + 8 * 1024 * 1024, entry["size"]) - 1
                headers = {"Range": f"bytes={offset}-{end}"}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    if r.status_code == 206:
                        if not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Incorrect HTTP range response")
                    else:
                        raise ValueError("Server ignored bounded HTTP range")
                    with part.open("ab") as f:
                        for chunk in r.iter_content(256 * 1024):
                            if chunk:
                                f.write(chunk)
                offset = part.stat().st_size
                if offset != end + 1:
                    raise ValueError("Incomplete bounded HTTP range")
            if part.stat().st_size != entry["size"]:
                raise ValueError("Incomplete source download")
            if digest(part) != entry["lfs"]["oid"]:
                # Preserve the suspect source for audit instead of overwriting it.
                part.rename(part.with_name(part.name + f".bad-{time.time_ns()}"))
                raise ValueError("Downloaded source failed SHA256")
            part.replace(dest)
            return dest
        except (requests.RequestException, ValueError, OSError) as exc:
            print(json.dumps({"event": "download_retry", "source": entry["path"],
                              "attempt": attempt + 1, "error": safe_error(exc)}), flush=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Download loop exhausted")


def make_csr(table, lookup, n_genes):
    """Map gene tokens to a fixed vocabulary and remove exactly one CLS per row."""
    genes = table["genes"].combine_chunks()
    counts = table["expressions"].combine_chunks()
    glen = pc.list_value_length(genes).to_numpy()
    xlen = pc.list_value_length(counts).to_numpy()
    if not np.array_equal(glen, xlen) or np.any(glen < 1):
        raise ValueError("Gene/count length mismatch or missing CLS")
    g = pc.list_flatten(genes).to_numpy()
    x = pc.list_flatten(counts).to_numpy()
    starts = np.r_[0, np.cumsum(glen[:-1], dtype=np.int64)]
    if not np.all(g[starts] == 1) or not np.all(x[starts] == -2):
        raise ValueError("Unexpected CLS sentinel")
    keep = np.ones(len(g), dtype=bool)
    keep[starts] = False
    g, x = g[keep], x[keep]
    if np.any(g < 0) or np.any(g >= len(lookup)) or np.any(lookup[g] < 0):
        raise ValueError("Expression gene token absent from vocabulary")
    if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x != np.floor(x)):
        raise ValueError("Biological counts are not nonnegative finite integers")
    if len(x) and x.max() > np.iinfo(np.int32).max:
        raise ValueError("Count does not fit int32")
    indptr = np.r_[0, np.cumsum(glen - 1, dtype=np.int64)]
    matrix = sparse.csr_matrix((x.astype(np.int32), lookup[g], indptr),
                               shape=(len(table), n_genes))
    if matrix.has_canonical_format is False:
        before = matrix.nnz
        matrix.sum_duplicates()
        if matrix.nnz != before:
            raise ValueError("Duplicate gene tokens within a cell")
    matrix.sort_indices()
    matrix.eliminate_zeros()
    return matrix


def checkpoint_valid(record, entry, verify_hashes=False):
    if record.get("source_sha256") != entry["lfs"]["oid"]:
        return False
    for item in record["files"]:
        path = BASE / item["path"]
        if not path.exists():
            return False
        stat = path.stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            return False
        if verify_hashes and digest(path) != item["sha256"]:
            return False
    return True


def process(entry, lookup, n_genes, attempts):
    stem = Path(entry["path"]).stem
    report = OUT / "checkpoints" / f"{stem}.json"
    if report.exists():
        record = json.loads(report.read_text(encoding="utf-8"))
        if checkpoint_valid(record, entry):
            return record
    source = fetch(entry, attempts)
    with DECODE_SLOTS:
        return extract(entry, source, report, lookup, n_genes)


def extract(entry, source, report, lookup, n_genes):
    stem = Path(entry["path"]).stem
    pf = pq.ParquetFile(source)
    selected = []
    for batch in pf.iter_batches(batch_size=1024, use_threads=False):
        table = pa.Table.from_batches([batch])
        sub = table.filter(pc.equal(table["cell_line_id"], A549))
        if len(sub):
            selected.append(sub)
    if selected:
        data = pa.concat_tables(selected)
    else:
        data = pa.Table.from_batches([], schema=pf.schema_arrow)
    data = data.append_column("source_shard", pa.array([entry["path"]] * len(data), type=pa.string()))
    matrix = make_csr(data, lookup, n_genes) if len(data) else sparse.csr_matrix((0, n_genes), dtype=np.int32)
    paths = [OUT / "raw_a549" / f"{stem}.parquet", OUT / "csr" / f"{stem}.npz",
             OUT / "obs" / f"{stem}.parquet"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    for table, path in ((data, paths[0]), (data.drop(["genes", "expressions"]), paths[2])):
        temp = path.with_suffix(".tmp.parquet")
        pq.write_table(table, temp, compression="zstd")
        temp.replace(path)
    temp = paths[1].with_suffix(".tmp.npz")
    sparse.save_npz(temp, matrix, compressed=True)
    temp.replace(paths[1])
    record = {"completed_utc": now(), "source": entry["path"],
              "source_sha256": entry["lfs"]["oid"], "source_rows": pf.metadata.num_rows,
              "a549_rows": len(data), "nonzero_counts": matrix.nnz,
              "files": [file_record(path) for path in [source, *paths]]}
    write_json(report, record)
    return record


def audit(records, expected_shards, verify_hashes=False):
    obs_path = BASE / "derived/a549_observed_cells.parquet"
    observed = pd.read_parquet(obs_path)
    index = pd.Index(observed["BARCODE_SUB_LIB_ID"])
    if not index.is_unique:
        raise ValueError("Reference obs barcodes are not unique")
    seen = np.zeros(len(observed), dtype=np.uint8)
    for record in records:
        if verify_hashes:
            for item in record["files"]:
                if digest(BASE / item["path"]) != item["sha256"]:
                    raise ValueError(f"Fresh SHA256 mismatch: {item['path']}")
        path = OUT / "obs" / (Path(record["source"]).stem + ".parquet")
        frame = pd.read_parquet(path)
        pos = index.get_indexer(frame["BARCODE_SUB_LIB_ID"])
        if (pos < 0).any() or len(np.unique(pos)) != len(pos) or seen[pos].any():
            raise ValueError("Unmatched or duplicate expression barcodes")
        ref = observed.iloc[pos]
        for col in ["sample", "plate"]:
            if not np.array_equal(frame[col].astype(str).values, ref[col].astype(str).values):
                raise ValueError(f"Barcode-linked {col} mismatch")
        if not frame["cell_line_id"].eq(A549).all():
            raise ValueError("Non-A549 expression row")
        seen[pos] = 1
    observed = observed.assign(expression_downloaded=seen.astype(bool))
    observed.to_parquet(OUT / "a549_expression_coverage.parquet", index=False)
    grouped = observed.groupby(["sample", "plate", "pass_filter"], observed=True).agg(
        observed_cells=("expression_downloaded", "size"),
        expression_cells=("expression_downloaded", "sum")).reset_index()
    grouped.to_csv(OUT / "a549_expression_sample_coverage.csv", index=False, encoding="utf-8-sig")
    report = {"checked_utc": now(), "revision": REVISION, "processed_shards": len(records),
              "expected_shards": expected_shards, "published_source_rows": sum(r["source_rows"] for r in records),
              "a549_expression_rows": int(seen.sum()), "observed_a549_rows": len(observed),
              "covered_filter_counts": observed.loc[seen.astype(bool), "pass_filter"].value_counts().to_dict(),
              "missing_filter_counts": observed.loc[~seen.astype(bool), "pass_filter"].value_counts().to_dict(),
              "observed_reference_sha256": digest(obs_path),
              "barcode_sample_plate_checks": "passed",
              "source_hashes": ("fresh_all_processed_source_and_derived_hashes_verified" if verify_hashes else
                                "verified_at_acquisition; resumed_checkpoints_use_size_and_mtime"),
              "scope": "complete_published_expression" if len(records) == expected_shards else "partial_expression"}
    if len(records) == expected_shards:
        problems = []
        if report["published_source_rows"] != SOURCE_ROWS:
            problems.append("published_total_row_count_mismatch")
        if report["missing_filter_counts"].get("full", 0):
            problems.append("published_expression_missing_reference_full_A549_barcodes")
        if problems:
            report["scope"] = "coverage_mismatch"
            report["coverage_problems"] = problems
    write_json(OUT / "coverage_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Smoke testing only; 0 means all shards")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true", help="Rehash every completed source and output during final audit")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "download.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"Existing download lock {lock}; check its PID before removing stale lock")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        manifest = get_manifest()
        entries = manifest["files"][:args.limit or None]
        genes = pd.read_parquet(BASE / "metadata/gene_metadata.parquet")
        if not genes["token_id"].is_unique or not genes["ensembl_id"].is_unique:
            raise ValueError("Gene vocabulary is not unique")
        genes = genes.sort_values("token_id").reset_index(drop=True)
        genes.to_parquet(OUT / "var.parquet", index=False)
        lookup = np.full(int(genes.token_id.max()) + 1, -1, dtype=np.int32)
        lookup[genes.token_id.values] = np.arange(len(genes), dtype=np.int32)
        records, failures = [], []
        start = time.monotonic()
        print(json.dumps({"event": "start", "utc": now(), "shards": len(entries),
                          "total_source_bytes": manifest["source_bytes"], "workers": args.workers}), flush=True)
        if args.audit_only:
            for entry in entries:
                path = OUT / "checkpoints" / (Path(entry["path"]).stem + ".json")
                record = json.loads(path.read_text(encoding="utf-8"))
                if not checkpoint_valid(record, entry):
                    raise ValueError(f"Invalid checkpoint {path}")
                records.append(record)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(process, entry, lookup, len(genes), args.retries): entry for entry in entries}
                for future in as_completed(pending):
                    entry = pending[future]
                    try:
                        records.append(future.result())
                    except Exception as exc:
                        failures.append({"source": entry["path"], "error": safe_error(exc)})
                    status = {"updated_utc": now(), "state": "running", "pid": os.getpid(),
                              "elapsed_seconds": round(time.monotonic() - start, 1),
                              "completed_shards": len(records), "requested_shards": len(entries),
                              "total_shards": len(manifest["files"]), "failed_shards": failures,
                              "downloaded_verified_source_bytes": sum(r["files"][0]["bytes"] for r in records),
                              "a549_rows": sum(r["a549_rows"] for r in records)}
                    write_json(OUT / "download_status.json", status)
                    print(json.dumps({k: v for k, v in status.items() if k != "failed_shards"}), flush=True)
        if failures:
            write_json(OUT / "download_status.json", {**status, "state": "incomplete_retry_required"})
            raise SystemExit(f"{len(failures)} shards failed; rerun to resume")
        report = audit(records, len(manifest["files"]), args.verify_hashes)
        final_state = {"complete_published_expression": "complete", "partial_expression": "smoke_complete",
                       "coverage_mismatch": "coverage_mismatch"}[report["scope"]]
        write_json(OUT / "download_status.json", {**report, "state": final_state})
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if final_state == "coverage_mismatch":
            raise SystemExit(2)
    except Exception as exc:
        status_path = OUT / "download_status.json"
        previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        write_json(status_path, {**previous, "state": "failed", "updated_utc": now(),
                                 "error": safe_error(exc), "pid": os.getpid()})
        print(json.dumps({"event": "failed", "error": safe_error(exc)}), flush=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        write_json(OUT / "download_status.json", {"state": "interrupted", "updated_utc": now(),
                                                  "pid": os.getpid(), "resume_supported": True})
        raise SystemExit(130) from None
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
