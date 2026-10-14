"""Small immutable data-contract helpers for training assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

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
    identity_keys = ("drug_identity_manifest", "drug_identity_manifest_sha256")
    if any(key in config for key in identity_keys):
        if not all(config.get(key) for key in identity_keys):
            raise ValueError("Drug identity requires both a manifest path and its expected SHA256")
        inside(root, config["drug_identity_manifest"])
        if not re.fullmatch(r"[0-9a-f]{64}", str(config["drug_identity_manifest_sha256"])):
            raise ValueError("Drug identity expected SHA256 must be 64 lowercase hexadecimal characters")
        if inside(root, config["assets"]) == inside(root, DEFAULT_ASSETS):
            raise ValueError("Drug identity changes require a new training assets version")
    if "sciplex_source" in config:
        source = config["sciplex_source"]
        for key in ("extraction_manifest", "chemical_manifest", "base_training_manifest"):
            inside(root, source[key])
            if not re.fullmatch(r"[0-9a-f]{64}", str(source.get(key + "_sha256", ""))):
                raise ValueError(f"sci-Plex requires a pinned {key}")
        if inside(root, config["assets"]) == inside(root, source["base_training_manifest"]).parent:
            raise ValueError("Additional response sources require new training assets")
    return config


def identity_dependencies(root: Path, config: dict):
    """Verify the explicitly pinned identity manifest and every data dependency.

    This check is mandatory even when a reader skips ordinary asset hashing.
    A path alone cannot establish which chemical identities produced a model.
    """
    if "drug_identity_manifest" not in config:
        return None
    path = inside(root, config["drug_identity_manifest"])
    manifest_hash = sha256(path)
    if manifest_hash != config["drug_identity_manifest_sha256"]:
        raise ValueError("Drug identity manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("version") not in ("identity-v2", "identity-v3")
            or manifest.get("status") != "reference_mapping_complete_with_explicit_exclusions"):
        raise ValueError("Drug identity manifest is not a completed supported mapping")
    required = {"drug_identity.parquet", "condition_identity.parquet", "morgan2048_chiral.npy", "sources.json"}
    if not required.issubset(manifest.get("outputs", {})) or not manifest.get("inputs_sha256"):
        raise ValueError("Drug identity manifest lacks required dependencies")
    outputs, inputs, responses = {}, {}, {}
    for name, record in manifest["outputs"].items():
        dependency = inside(path.parent, name)
        digest = sha256(dependency)
        if digest != record["sha256"] or dependency.stat().st_size != record["bytes"]:
            raise ValueError(f"Drug identity output hash mismatch: {name}")
        outputs[name] = digest
    for name, expected in manifest["inputs_sha256"].items():
        dependency = inside(root, name)
        digest = sha256(dependency)
        if digest != expected:
            raise ValueError(f"Drug identity input hash mismatch: {name}")
        inputs[dependency.relative_to(root).as_posix()] = digest
    sources = json.loads((path.parent / "sources.json").read_text(encoding="utf-8"))
    if not isinstance(sources, list) or len(sources) != manifest.get("raw_response_count"):
        raise ValueError("Drug identity source response registry is incomplete")
    for record in sources:
        dependency = inside(path.parent, record["path"])
        digest = sha256(dependency)
        if digest != record["sha256"] or dependency.stat().st_size != record["bytes"]:
            raise ValueError(f"Drug identity raw response hash mismatch: {record['path']}")
        responses[dependency.relative_to(path.parent).as_posix()] = digest
    if len(responses) != len(sources):
        raise ValueError("Duplicate drug identity source response")
    return {"version": manifest["version"], "manifest": path.relative_to(root).as_posix(),
            "manifest_sha256": manifest_hash, "outputs": outputs,
            "inputs": inputs, "raw_responses": responses}


def response_source_dependencies(root: Path, config: dict):
    if "sciplex_source" not in config:
        return None
    source, bindings = config["sciplex_source"], {}
    for key in ("extraction_manifest", "chemical_manifest", "base_training_manifest"):
        path = inside(root, source[key])
        digest = sha256(path)
        if digest != source[key + "_sha256"]:
            raise ValueError(f"sci-Plex dependency hash mismatch: {key}")
        bindings[key] = {"path": path.relative_to(root).as_posix(), "sha256": digest}
    extraction = inside(root, source["extraction_manifest"])
    status = json.loads(extraction.read_text(encoding="utf-8"))
    if status["status"] != "complete_expression_extraction_not_training_ready":
        raise ValueError("sci-Plex extraction is incomplete")
    bindings["tables"] = {name: sha256(extraction.parent / name)
                           for name in ("a549_cells.parquet", "common_genes.parquet")}
    path = inside(root, source["chemical_manifest"])
    chemical = json.loads(path.read_text(encoding="utf-8"))
    outputs = {}
    if chemical.get("input_path_base") != "project_root" or chemical.get("input_base") != ".":
        raise ValueError("sci-Plex chemical input path base must be explicit project_root")
    for name, expected in chemical["outputs_sha256"].items():
        file = inside(path.parent, name)
        digest = sha256(file)
        if digest != expected:
            raise ValueError(f"sci-Plex chemical output hash mismatch: {name}")
        outputs[name] = digest
    bindings["chemical_outputs"] = outputs
    for name, expected in chemical.get("inputs_sha256", {}).items():
        if sha256(inside(root, name)) != expected:
            raise ValueError(f"sci-Plex chemical input hash mismatch: {name}")
    bindings["protocol_files"] = {}
    for name, expected in source["protocol_files"].items():
        path = inside(root, name)
        if sha256(path) != expected:
            raise ValueError(f"sci-Plex protocol dependency hash mismatch: {name}")
        bindings["protocol_files"][name] = expected
    if (source["planned_wells"] not in source["protocol_files"] or source["time_contract"] not in source["protocol_files"]
            or source["planned_dose_unit"] != "nM"):
        raise ValueError("sci-Plex planned wells require their pinned nM source table")
    return bindings


def training_contract(root: Path, config: dict):
    """Preserve the exact schema-1 signature unless a reviewed identity is requested."""
    source_hash = sha256(inside(root, config["bundle"]) / "manifest.json")
    identity = identity_dependencies(root, config)
    source = response_source_dependencies(root, config)
    schema = 3 if source is not None else 2 if identity is not None else 1
    value = {"config": config, "source_manifest": source_hash, "training_schema": schema}
    if identity is not None:
        value["drug_identity"] = identity
    if source is not None:
        value["sciplex_source"] = source
    return digest_json(value), source_hash, identity, schema


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
