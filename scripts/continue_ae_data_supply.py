"""Register a verified, byte-preserving AE continuation; never start training."""

# pyright: reportMissingImports=false

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.checkpoint import atomic_json, load_checkpoint
from lung.engine.config import config_hash, file_hash, implementation_identity, read_yaml
from lung.engine.latent_pipeline import _model_config, latent_settings
from lung.engine.modular_config import load_modular_config, science_identity, stage_config
from lung.engine.numerics import PrecisionPolicy

ALLOWED = {
    "src/lung/data/reader.py", "src/lung/data/sparse_cache.py",
    "src/lung/data/ae_expression_cache.py", "src/lung/engine/ae_data.py",
    "src/lung/engine/latent_pipeline.py",
    "src/lung/engine/world.py", "src/lung/engine/diagnostics.py",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_path(base, name):
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    path = (ROOT / base / name).resolve()
    if not path.is_relative_to((ROOT / base).resolve()):
        raise ValueError("Continuation path escapes project")
    return path


def source_versions(baseline):
    import hashlib
    old = {}
    with tarfile.open(baseline / "baseline-src.tar") as archive:
        for member in archive.getmembers():
            name = member.name.replace("\\", "/").removeprefix("./")
            if member.isfile() and name.startswith("src/lung/") and name.endswith(".py"):
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing source file in baseline archive")
                old[name] = hashlib.sha256(stream.read()).hexdigest()
    for name in ("train.py", "test.py"):
        old[name] = file_hash(baseline / name)
    current = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in (
        ROOT / "train.py", ROOT / "test.py", *sorted((ROOT / "src/lung").rglob("*.py")))}
    changed = {name for name in old.keys() | current.keys() if old.get(name) != current.get(name)}
    if not changed or not changed.issubset(ALLOWED):
        raise ValueError(f"Unexpected source changes: {sorted(changed)}")
    return config_hash(old), config_hash(current), sorted(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--new-run")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    if not baseline.is_relative_to(ROOT.resolve()):
        raise ValueError("Baseline must stay inside this project")
    old_parent = run_path("experiments", args.source_run)
    latest = ROOT / "experiments/modular_latest.json"
    latest_hash = file_hash(latest)
    if read(latest).get("run_id") != args.source_run:
        raise ValueError("Source run is no longer the current continuation target")
    old_state = read(old_parent / "modular_state.json")
    if set(old_state["stages"]) != {"ae"} or old_state.get("status") != "interrupted":
        raise ValueError("Only an interrupted, AE-only run can use this migration")
    old_child_id = old_state["stages"]["ae"]["run_id"]
    old_child = run_path("experiments", old_child_id)
    metadata_child = old_child
    if (old_child / "run_manifest.json").is_file():
        if read(old_child / "run_manifest.json")["status"] != "interrupted":
            raise ValueError("Source AE must be stopped before migration")
    else:
        previous = old_state.get("continuation", {})
        if (previous.get("new_child") != old_child_id
                or previous.get("new_run") != args.source_run
                or previous.get("formal_training_started") is not False):
            raise ValueError("Missing manifest without a verified, unstarted continuation")
        metadata_child = run_path("experiments", previous.get("data_metadata_source_child", previous["source_child"]))
        if read(metadata_child / "run_manifest.json")["status"] != "interrupted":
            raise ValueError("Original continuation metadata is not interrupted")
    old_source, new_source, changed = source_versions(baseline)
    if old_source != old_state["implementation_identity"] or new_source != implementation_identity():
        raise ValueError("Source snapshot does not match recorded training identity")
    config = load_modular_config()
    old_config = read_yaml(old_parent / "active_config.yaml")
    stage = stage_config(config, "ae", {})
    if (science_identity(config) != old_state["identity"]
            or science_identity(old_config) != science_identity(config)
            or config_hash(stage) != config_hash(read_yaml(old_child / "active_config.yaml"))):
        raise ValueError("Scientific or AE stage configuration changed")
    last_path = baseline / "baseline-checkpoints" / "last.pt"
    best_path = baseline / "baseline-checkpoints" / "best.pt"
    original_weights = run_path("checkpoints", old_child_id) / "latent_ae"
    digests = {name: file_hash(baseline / "baseline-checkpoints" / name)
               for name in ("last.pt", "best.pt")}
    if any(file_hash(original_weights / name) != digest for name, digest in digests.items()):
        raise ValueError("Source checkpoint advanced after the fixed baseline snapshot")
    saved = load_checkpoint(last_path)
    position = saved["extra"].get("resume")
    if not position or position["epoch"] != saved["epoch"]:
        raise ValueError("Expected the recorded in-epoch AE group checkpoint")
    cursor = position["next_group_start"]
    if (type(cursor) is not int or not 0 <= cursor <= len(position["keys"])
            or (cursor != len(position["keys"]) and cursor % stage["response"]["conditions_per_update"])):
        raise ValueError("Invalid AE continuation group cursor")
    data_record = read(metadata_child / "data_manifest.json")
    if saved["metadata"]["data_identity"] != data_record["identity"]:
        raise ValueError("Checkpoint and recorded data identity differ")
    assets = ROOT / data_record["manifest"]["config"]["assets"]
    current_manifest = read(assets / "manifest.json")
    current_data = config_hash({"signature": current_manifest["signature"], "files": current_manifest["files"]})
    if current_data != data_record["identity"]:
        raise ValueError("Frozen training manifest differs from the baseline")
    for name, item in current_manifest["files"].items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets.resolve()) or file_hash(path) != item["sha256"]:
            raise ValueError(f"Frozen training asset changed: {name}")
    shape = _model_config(stage, latent_settings(stage), saved["metadata"]["model_config"]["n_genes"])
    ae = latent_settings(stage)["ae"]
    expected = config_hash({"data": current_data, "stage": "latent_ae", "model": asdict(shape),
                           "lr": float(ae.get("lr", stage["response"]["lr"])),
                           "batch": int(ae.get("batch", stage["response"]["ot_samples"])),
                           "epochs": stage["epochs"], "groups": stage["response"]["conditions_per_update"],
                           "precision": PrecisionPolicy.from_config(stage).identity()})
    if saved["identity"] != expected:
        raise ValueError("AE checkpoint identity is incompatible with optimized execution")
    now = datetime.now(timezone(timedelta(hours=8)))
    new_id = args.new_run or now.strftime("%Y%m%d-%H%M%S-ae-loader-continuation")
    new_child_id = new_id + "-ae"
    new_parent, new_child = run_path("experiments", new_id), run_path("experiments", new_child_id)
    new_weights = run_path("checkpoints", new_child_id) / "latent_ae"
    receipt = {"schema_version": 1, "scope": "AE_scientific_configuration_and_checkpoint_unchanged",
               "source_run": args.source_run, "source_child": old_child_id,
               "new_run": new_id, "new_child": new_child_id,
               "old_implementation_identity": old_source, "new_implementation_identity": new_source,
               "science_identity": science_identity(config), "data_identity": current_data,
               "changed_sources": changed, "checkpoint_sha256": digests,
               "data_metadata_source_child": metadata_child.name,
               "checkpoint_transfer": "byte_for_byte_copy", "formal_training_started": False,
               "resume": {"epoch_completed": saved["epoch"],
                          "next_condition_draw": position["next_group_start"] + 1,
                          "conditions_total": len(position["keys"]),
                          "optimizer_step": saved["counters"]["optimizer_step"]},
               "validation": "Requires accompanying CPU equivalence, resume and real-data GPU update evidence"}
    if args.publish:
        if file_hash(latest) != latest_hash:
            raise ValueError("Latest run changed during continuation verification")
        if any(path.exists() for path in (new_parent, new_child, new_weights.parent)):
            raise FileExistsError("Continuation destinations already exist")
        new_parent.mkdir()
        new_child.mkdir()
        new_weights.mkdir(parents=True)
        for name in ("last.pt", "best.pt"):
            shutil.copy2(baseline / "baseline-checkpoints" / name, new_weights / name)
            if file_hash(new_weights / name) != digests[name]:
                raise ValueError("Checkpoint copy differs")
        shutil.copy2(old_parent / "active_config.yaml", new_parent / "active_config.yaml")
        shutil.copy2(old_child / "active_config.yaml", new_child / "active_config.yaml")
        shutil.copy2(metadata_child / "data_manifest.json", new_child / "data_manifest.json")
        record = {"schema_version": 1, "identity": science_identity(config),
                  "implementation_identity": new_source, "status": "interrupted",
                  "stages": {"ae": {"run_id": new_child_id, "status": "interrupted", "reused": False}},
                  "continuation": receipt}
        atomic_json(new_parent / "modular_state.json", record)
        atomic_json(new_parent / "ae_loader_continuation.json", receipt)
        if file_hash(latest) != latest_hash:
            raise ValueError("Latest run changed before continuation publication")
        shutil.copy2(latest, new_parent / "modular_latest_before.json")
        atomic_json(latest, {"identity": science_identity(config), "run_id": new_id,
                             "implementation_identity": new_source})
    print(json.dumps({"publish": args.publish, **receipt}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
