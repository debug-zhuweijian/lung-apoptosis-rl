"""Fetch official sci-Plex3 metadata and a bounded expression prefix for audit.

This does not prepare a training dataset or download the full expression matrix.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08-datasets/sciPlex3/GSM4150378"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/"
PREFIX = "GSM4150378_sciPlex3_"
FILES = [
    "hashSampleSheet.txt.gz",
    "A549_MCF7_K562_hashTable_metadata.txt.gz",
    "A549_MCF7_K562_screen_gene.annotations.txt.gz",
    "A549_MCF7_K562_screen_cell.annotations.txt.gz",
    "pData.txt.gz",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_manifest(records: dict) -> None:
    dest = OUT / "source_inventory.json"
    temp = dest.with_suffix(".json.tmp")
    temp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(dest)


def main() -> None:
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    (OUT / "audit").mkdir(exist_ok=True)
    manifest = OUT / "source_inventory.json"
    records = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    session = requests.Session()
    session.trust_env = False
    for suffix in FILES:
        name = PREFIX + suffix
        path = OUT / "sources" / name
        previous = records.get(name, {})
        if path.exists():
            if previous.get("sha256") != sha256(path):
                raise RuntimeError(f"Existing file lacks matching manifest: {name}")
            print(f"Verified cached {name}", flush=True)
            continue
        part = path.with_suffix(path.suffix + ".part")
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with session.get(BASE + name, headers=headers, stream=True, timeout=(15, 40)) as res:
            res.raise_for_status()
            if offset and (res.status_code != 206 or not res.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise RuntimeError("Server did not honor resume range; partial retained")
            expected = int(res.headers["Content-Length"]) + offset
            last = time.monotonic()
            with part.open("ab" if offset else "wb") as dst:
                for block in res.iter_content(1 << 20):
                    dst.write(block)
                    offset += len(block)
                    if time.monotonic() - last > 10:
                        print(f"Metadata {name}: {offset}/{expected} bytes", flush=True)
                        last = time.monotonic()
            if part.stat().st_size != expected:
                raise RuntimeError(f"Size mismatch: {name}")
            with gzip.open(part, "rb") as src:
                for _ in iter(lambda: src.read(1 << 20), b""):
                    pass  # Full gzip checksum/end-of-stream validation.
            part.rename(path)
            records[name] = {
                "url": BASE + name, "bytes": expected, "sha256": sha256(path),
                "last_modified": res.headers.get("Last-Modified"), "status": "complete_metadata_gzip_verified",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(records)
            print(f"Saved {name}: {expected} bytes", flush=True)
    # First 200,000 sparse triples. This is a file-order prefix, NOT a random sample.
    name = PREFIX + "A549_MCF7_K562_screen_UMI.count.matrix.gz"
    preview = OUT / "audit/expression_prefix_200000.tsv"
    if not preview.exists():
        with session.get(BASE + name, stream=True, timeout=(15, 40)) as res:
            res.raise_for_status()
            with gzip.GzipFile(fileobj=res.raw) as src, preview.open("wb") as dst:
                for i in range(200000):
                    line = src.readline()
                    if not line:
                        raise RuntimeError("Unexpected matrix EOF")
                    dst.write(line)
            records["expression_prefix"] = {
                "url": BASE + name, "source_compressed_bytes": int(res.headers["Content-Length"]),
                "status": "partial_prefix_only_not_gzip_complete", "rows": i + 1,
                "path": str(preview.relative_to(OUT)), "sha256": sha256(preview),
                "source_last_modified": res.headers.get("Last-Modified"),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(records)
    print("Metadata complete; expression prefix only. No training started.", flush=True)


if __name__ == "__main__":
    main()
