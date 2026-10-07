"""Resume the official sci-Plex3 count download; content audit is a separate step."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "08-datasets/sciPlex3/GSM4150378"
NAME = "GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz"
URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/" + NAME


def write_json(path, value):
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    part.replace(path)


def main():
    destination = DATA / "sources" / NAME
    partial = destination.with_suffix(destination.suffix + ".part")
    status_path = DATA / "expression_download_status.json"
    lock = DATA / "expression_download.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    session = requests.Session()
    session.trust_env = False
    status = {}
    try:
        head = session.head(URL, timeout=(15, 40))
        head.raise_for_status()
        size = int(head.headers["Content-Length"])
        modified = head.headers.get("Last-Modified")
        prior = json.loads(status_path.read_text(encoding="utf8")) if status_path.exists() else {}
        if prior and (prior.get("source_bytes") != size or prior.get("last_modified") != modified):
            raise RuntimeError("Remote source identity changed; do not reuse partial bytes")
        if destination.exists():
            digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
            if destination.stat().st_size != size or prior.get("sha256") != digest:
                raise RuntimeError("Existing expression file lacks matching download record")
            print("Existing download verified against local record; content audit remains separate.")
            return
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError("Partial file exceeds expected length")
        if shutil.disk_usage(DATA).free < size - offset + 8_000_000_000:
            raise RuntimeError("Insufficient free disk space")
        status = {"url": URL, "source_bytes": size, "last_modified": modified,
                  "downloaded_bytes": offset, "status": "downloading", "pid": os.getpid(),
                  "content_audit_passed": False}
        write_json(status_path, status)
        for attempt in range(3):
            if offset == size:
                break
            headers = {"If-Unmodified-Since": modified} if modified else {}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with session.get(URL, headers=headers, stream=True, timeout=(15, 45)) as res:
                    res.raise_for_status()
                    if offset and (res.status_code != 206 or not res.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError("Server did not honor requested resume range")
                    if res.headers.get("Last-Modified") != modified:
                        raise RuntimeError("Remote modification stamp changed during download")
                    last = time.monotonic()
                    with partial.open("ab" if offset else "wb") as dst:
                        for block in res.iter_content(4 << 20):
                            dst.write(block)
                            offset += len(block)
                            if time.monotonic() - last >= 10:
                                dst.flush()
                                status.update(downloaded_bytes=offset, updated_utc=datetime.now(timezone.utc).isoformat())
                                write_json(status_path, status)
                                print(f"Expression download: {100*offset/size:.2f}% ({offset}/{size} bytes); content not yet audited", flush=True)
                                last = time.monotonic()
                if offset != size:
                    raise RuntimeError("Downloaded file length mismatch")
            except requests.RequestException as exc:
                offset = partial.stat().st_size if partial.exists() else 0
                status.update(downloaded_bytes=offset, error_type=type(exc).__name__, attempt=attempt + 1)
                write_json(status_path, status)
                if attempt == 2:
                    raise
                time.sleep(2)
        if partial.stat().st_size != size:
            raise RuntimeError("File size mismatch before hash")
        status.update(status="hashing", downloaded_bytes=size)
        write_json(status_path, status)
        with partial.open("rb") as src:
            digest = hashlib.file_digest(src, "sha256").hexdigest()
        partial.rename(destination)
        status.update(status="downloaded_pending_content_audit", sha256=digest,
                      updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        print(f"Downloaded {size} bytes, SHA256={digest}; gzip and matrix audit still required.", flush=True)
    except Exception as error:
        if status:
            status.update(status="failed_resumable", error_type=type(error).__name__,
                          updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(status_path, status)
        raise
    finally:
        session.close()
        lock.unlink()  # Only the lock acquired by this invocation.


if __name__ == "__main__":
    main()
