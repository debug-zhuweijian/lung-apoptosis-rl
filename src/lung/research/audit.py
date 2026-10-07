"""New-directory-only audit output with input hashes and an atomic final report."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import uuid

from .sciplex_diagnostics import run_sciplex
from .tidyi_diagnostics import run_tidyi


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InputTracker:
    def __init__(self):
        self.records = {}

    def __call__(self, path, expected_sha256=None):
        path = path.resolve(strict=True)
        key = str(path)
        if key not in self.records:
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("Input changed while hashing")
            self.records[key] = {"path": key, "bytes": after.st_size, "sha256": digest,
                                 "mtime_ns": after.st_mtime_ns}
        if expected_sha256 is not None and self.records[key]["sha256"] != expected_sha256:
            raise ValueError("Input SHA256 differs from extraction manifest: " + path.name)
        return path

    def verify_unchanged(self):
        for key, record in self.records.items():
            current = Path(key).stat()
            if (current.st_size, current.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise RuntimeError("Input changed during audit: " + key)


def run_audit(root, output, only="all"):
    if only not in {"all", "tidyi", "sciplex"}:
        raise ValueError("Unknown diagnostic selection")
    root, output = Path(root).resolve(), Path(output).resolve()
    allowed = root / "experiments/research-diagnostics"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a new run directory under experiments/research-diagnostics")
    output.mkdir(parents=True, exist_ok=False)
    tracker, results, tables = InputTracker(), {}, {}
    code = sorted(Path(__file__).parent.glob("*.py")) + [root / "scripts/audit_research_progress.py"]
    identity = [{"path": str(path), "sha256": sha256(path)} for path in code]
    # Build every requested result before writing any completed report.
    for name, function in [("tidyi", run_tidyi), ("sciplex", run_sciplex)]:
        if only in {"all", name}:
            results[name], new_tables = function(root, tracker)
            tables.update(new_tables)
    tracker.verify_unchanged()
    for record in identity:
        if sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Diagnostic code changed during execution")
    artifacts = []
    for name, table in tables.items():
        path = output / (name + ".csv")
        table.to_csv(path, index=False)
        artifacts.append({"file": path.name, "rows": len(table), "sha256": sha256(path)})
    report = {"schema_version": "research-diagnostics-v1", "status": "complete", "no_model_fit": True,
              "formal_training_started": False, "exploratory": True,
              "created_utc": datetime.now(timezone.utc).isoformat(), "only": only,
              "seed": None, "seed_reason": "deterministic exact empirical statistics; no randomized sampling",
              "experiment_definition": "recompute 2026-09-26 checkpoint sections 4.3 and 6.2-6.4; no new biological labels",
              "runtime": {"python": platform.python_version(), "executable": sys.executable,
                          "packages": {name: version(name) for name in ["numpy", "scipy", "pandas", "pyarrow"]}},
              "code_identity": identity, "inputs": list(tracker.records.values()),
              "results": results, "artifacts": artifacts,
              "interpretation": "Exploratory diagnostics of historical data/predictions, not new training results, natural apoptosis rates or treatment benefit"}
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    return output / "report.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only exploratory diagnostics; no training, downloads or dependency installation.")
    parser.add_argument("--only", choices=["tidyi", "sciplex", "all"], default="all")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, help="New directory inside PROJECT/experiments/research-diagnostics; existing directory rejected")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or args.project_root / "experiments/research-diagnostics" / (stamp + "-" + uuid.uuid4().hex[:8])
    print(run_audit(args.project_root, output, args.only))
