"""Small immutable data-contract helpers for training assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

DEFAULT_BUNDLE = "08-datasets/LungCancer_A549_Integration/prepared/v1"
DEFAULT_ASSETS = "08-datasets/LungCancer_A549_Integration/training-v1"
ROLES = ("fit", "selection", "residual_fit", "aux_selection", "calibration", "final_test")
ROLE_LETTERS = dict(zip(ROLES, "TVRGCE"))
OUTER = dict(zip(ROLES, ("train", "val", "train", "val", "val", "test")))
STATE_LABELS = ("PI-APC-", "PI-APC+", "PI+APC+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def inside(root: Path, value: str | Path) -> Path:
    root = Path(root).resolve()
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Data path must remain inside the project")
    return resolved


def dataset_config(root: Path, config_path=None) -> dict:
    root = Path(root).resolve()
    path = inside(root, config_path or "configs/dataset.yaml")
    if path.exists():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("dataset.yaml must contain a mapping")
    else:
        config = {}
    config.setdefault("bundle", DEFAULT_BUNDLE)
    config.setdefault("assets", DEFAULT_ASSETS)
    config.setdefault("gene_count", 32461)
    config.setdefault("graph_neighbors", 32)
    config.setdefault("graph_block_size", 128)
    config.setdefault("stream_batch_size", 512)
    for key in ("bundle", "assets"):
        inside(root, config[key])
    if config["bundle"] == config["assets"]:
        raise ValueError("Training assets may not overwrite the prepared bundle")
    if config["gene_count"] < 2 or not 0 < config["graph_neighbors"] < config["gene_count"]:
        raise ValueError("Invalid gene_count / graph_neighbors")
    if config["graph_block_size"] < 1 or config["stream_batch_size"] < 1:
        raise ValueError("Block and batch sizes must be positive")
    return config


def source_shard_registry(prepared):
    """Reuse the recorded strict audit and check cheap source-file attributes.

    No 337 GB re-hash is implied. The recorded source checksum remains in the
    registry, while every load checks the size/mtime recorded by that audit.
    """
    if "tahoe_shards" not in prepared.manifest["tables"]:
        return {}  # Tiny standalone fixtures have no prior full-source audit.
    name = prepared.manifest["tables"]["tahoe_shards"]
    records = {item["path"]: item for item in prepared.manifest.get("outputs", [])}
    path = inside(prepared.bundle_path, name)
    if name not in records or sha256(path) != records[name]["sha256"]:
        raise ValueError("Source shard audit registry is missing or modified")
    registry = prepared.table("tahoe_shards")
    if registry["matrix_path"].duplicated().any():
        raise ValueError("Duplicate source shard registry path")
    return registry.set_index("matrix_path").to_dict("index")


def check_shard(path: Path, record):
    if record is not None:
        stat = path.stat()
        if stat.st_size != record["csr_bytes"] or stat.st_mtime_ns != record["csr_mtime_ns"]:
            raise ValueError(f"Source shard changed since its strict audit: {path.name}")
