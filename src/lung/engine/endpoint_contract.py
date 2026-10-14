"""Schema 2: predicted condition features supervised by measured Annexin units.

``make_manifest_v2`` accepts already-written partition files and metadata-only
``unit_index`` entries from ``endpoint_unit_index``.  It never fabricates a
feature vector, an expression sample ID, or an experimental measurement.  The
preparation step may read all labels; fitting validates their file hashes and
the complete metadata index, then deserializes T/V values only.
"""

from pathlib import Path
import json
import math

import numpy as np

from .checkpoint import load_checkpoint
from .config import StageUnavailable, config_hash, file_hash


SUPERVISION_KIND = "predicted_features_measured_endpoint"
PARTITION_ROLES = {"T": "fit", "V": "selection", "C": "calibration", "E": "final_test"}
FEATURE_KEYS = ("representation", "response", "pool", "state", "backward",
                "correction", "correction_gate")
REQUIRED_ASSETS = {"gene_axis", "preprocessing", "chemical_identity",
                   "feature_generator", "condition_inputs", "initial_state"}
INDEX_FIELDS = ("role", "experimental_condition_id", "endpoint_sample_id",
                "observation_ids", "experiment_id", "source_id", "group_id",
                "chemical_group_id", "protocol_id", "label_type", "label_kind",
                "denominator", "sampling_gate", "initial_state",
                "measurement_provenance")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _strings(value):
    return (isinstance(value, list) and bool(value) and all(_text(x) for x in value)
            and len(value) == len(set(value)))


def _source(base, record):
    if not isinstance(record, dict) or not _text(record.get("path")):
        raise StageUnavailable("ENDPOINT_ASSET_UNRECORDED")
    path = (Path(base) / record["path"]).resolve()
    if (not path.is_file() or not _text(record.get("sha256"))
            or file_hash(path) != record["sha256"]):
        raise StageUnavailable("ENDPOINT_ASSET_HASH_MISMATCH: " + str(path))
    return path


def endpoint_unit_index(row):
    """Only supervision metadata, never endpoint values or model features."""
    return {key: row[key] for key in INDEX_FIELDS if key in row}


def feature_dependencies_v2(pipeline, feature_assets, *, base):
    """Bind every present feature-producing stage and actual asset bytes."""
    if any(not pipeline.get(key) for key in ("response", "pool", "state")):
        raise StageUnavailable("ENDPOINT_FEATURE_CHECKPOINTS_REQUIRED: response/pool/state")
    if pipeline.get("correction") and (not pipeline.get("backward") or not pipeline.get("correction_gate")):
        raise StageUnavailable("ENDPOINT_CORRECTION_DEPENDENCIES_REQUIRED")
    if pipeline.get("correction_gate") and not pipeline.get("correction"):
        raise StageUnavailable("ENDPOINT_CORRECTION_DEPENDENCIES_REQUIRED")
    checkpoints = {}
    for key in FEATURE_KEYS:
        if pipeline.get(key):
            paths = pipeline[key] if isinstance(pipeline[key], (list, tuple)) else [pipeline[key]]
            if not all(Path(value).is_file() for value in paths):
                raise StageUnavailable("ENDPOINT_FEATURE_CHECKPOINT_MISSING: " + key)
            checkpoints[key] = [file_hash(value) for value in paths]
    if not isinstance(feature_assets, dict) or not REQUIRED_ASSETS <= feature_assets.keys():
        raise StageUnavailable("ENDPOINT_FEATURE_ASSETS_REQUIRED")
    assets = {}
    for key, records in feature_assets.items():
        if not isinstance(records, list) or not records:
            raise StageUnavailable("ENDPOINT_FEATURE_ASSETS_REQUIRED: " + key)
        assets[key] = [{"path": str(_source(base, item)), "sha256": item["sha256"]}
                       for item in records]
    return {"pipeline": checkpoints, "assets": assets}


def frozen_feature_identity_v2(data_identity, dependencies, feature_generation):
    """Compute before writing rows; each row records this exact frozen identity."""
    if not isinstance(feature_generation, dict) or not feature_generation:
        raise StageUnavailable("ENDPOINT_FEATURE_GENERATION_UNRECORDED")
    return config_hash({"data": data_identity, "dependencies": dependencies,
                        "generation": feature_generation})


def _feature_identity(manifest):
    return frozen_feature_identity_v2(manifest["data_identity"], manifest["feature_dependencies"],
                                      manifest.get("feature_generation"))


def make_manifest_v2(data_identity, pipeline, feature_assets, protocols, partitions,
                     split_policy, feature_generation, *, base):
    """Return a manifest; callers write it only after validating complete rows.

    Partitions: role -> {path, sha256, unit_index: [endpoint_unit_index(row)]}.
    Assets: purpose -> [{path, sha256}], including REQUIRED_ASSETS.  Generation
    records the frozen sampling/inference configuration and algorithm version.
    The builder is not a biological feasibility certificate.
    """
    if not isinstance(feature_generation, dict) or not feature_generation:
        raise StageUnavailable("ENDPOINT_FEATURE_GENERATION_UNRECORDED")
    manifest = {"schema_version": 2, "status": "ready",
                "supervision_kind": SUPERVISION_KIND,
                "feature_kind": "predicted", "label_kind": "measured",
                "label_type": "Annexin_positive_fraction", "data_identity": data_identity,
                "protocols": protocols, "partitions": partitions,
                "partition_roles": PARTITION_ROLES.copy(), "split_policy": split_policy,
                "feature_generation": feature_generation,
                "feature_dependencies": feature_dependencies_v2(pipeline, feature_assets, base=base)}
    manifest["frozen_feature_identity"] = _feature_identity(manifest)
    # Checkpoints, full split index and physical source identities are checked
    # even when a caller constructs this contract outside the training entry.
    validate_manifest_v2(Path(base) / "manifest.json", manifest, data_identity, pipeline)
    # Preparation is allowed to read all four partitions. Fail here rather than
    # publish a ready manifest for missing/placeholder C/E features. Fitting
    # later checks these immutable hashes without deserializing their values.
    from lung.models import ModelConfig
    responses = pipeline["response"] if isinstance(pipeline["response"], (list, tuple)) else [pipeline["response"]]
    shape = ModelConfig(**load_checkpoint(responses[0])["metadata"]["model_config"])
    for role in PARTITION_ROLES:
        load_units_v2(Path(base) / "manifest.json", manifest, role, shape)
    return manifest


def _check_index_row(row, role, protocols):
    if (not isinstance(row, dict) or set(row) - set(INDEX_FIELDS)
            or row.get("role") != role or row.get("label_kind") != "measured"
            or row.get("label_type") != "Annexin_positive_fraction"):
        raise StageUnavailable("ENDPOINT_UNIT_INDEX_INVALID")
    required = ("experimental_condition_id", "experiment_id", "source_id", "group_id",
                "chemical_group_id", "protocol_id", "denominator", "sampling_gate")
    if any(not _text(row.get(key)) for key in required) or not _strings(row.get("observation_ids")):
        raise StageUnavailable("ENDPOINT_UNIT_IDENTITY_REQUIRED")
    # Aggregates need not invent a sample accession when source observation IDs
    # already identify the measurements. A provided sample ID must be genuine.
    if "endpoint_sample_id" in row and not _text(row["endpoint_sample_id"]):
        raise StageUnavailable("ENDPOINT_UNIT_IDENTITY_REQUIRED")
    protocol = protocols.get(row["protocol_id"], {})
    if any(protocol.get(key) != row[key] for key in ("label_type", "denominator", "sampling_gate")):
        raise StageUnavailable("ENDPOINT_PROTOCOL_MISMATCH")
    state = row.get("initial_state", {})
    if (not isinstance(state, dict) or state.get("kind") not in
            {"reference", "observed_population", "observed_cell"}
            or set(state) - {"kind", "reference", "evidence", "expression_sample_ids",
                             "pairing_evidence", "source_files"}
            or any(not _text(state.get(key)) for key in ("reference", "evidence"))):
        raise StageUnavailable("ENDPOINT_INITIAL_STATE_UNVERIFIED")
    if state["kind"] != "reference" and (not _strings(state.get("expression_sample_ids"))
            or not _text(state.get("pairing_evidence")) or not state.get("source_files")):
        raise StageUnavailable("ENDPOINT_OBSERVED_INITIAL_STATE_UNVERIFIED")
    provenance = row.get("measurement_provenance")
    if (not isinstance(provenance, list) or len(provenance) != len(row["observation_ids"])
            or any(not isinstance(p, dict) or not _text(p.get("locator"))
                   or set(p) != {"observation_id", "path", "sha256", "locator"} for p in provenance)
            or {p.get("observation_id") for p in provenance} != set(row["observation_ids"])):
        raise StageUnavailable("ENDPOINT_MEASUREMENT_PROVENANCE_REQUIRED")
    if any(not isinstance(p, dict) or set(p) != {"path", "sha256"}
           for p in state.get("source_files", [])):
        raise StageUnavailable("ENDPOINT_INITIAL_STATE_SOURCE_INVALID")


def validate_manifest_v2(path, manifest, data_identity, pipeline):
    """Validate all-role metadata and hashes without parsing C/E value files."""
    if (manifest.get("schema_version") != 2 or manifest.get("status") != "ready"
            or manifest.get("supervision_kind") != SUPERVISION_KIND
            or manifest.get("feature_kind") != "predicted"
            or manifest.get("label_kind") != "measured"
            or manifest.get("label_type") != "Annexin_positive_fraction"
            or manifest.get("data_identity") != data_identity
            or manifest.get("partition_roles") != PARTITION_ROLES):
        raise StageUnavailable("ENDPOINT_SCHEMA2_MANIFEST_UNVERIFIED")
    recorded = manifest.get("feature_dependencies", {})
    current = feature_dependencies_v2(pipeline, recorded.get("assets"), base=path.parent)
    if recorded != current or manifest.get("frozen_feature_identity") != _feature_identity(manifest):
        raise StageUnavailable("ENDPOINT_FROZEN_FEATURE_IDENTITY_MISMATCH")
    for key in recorded["pipeline"]:
        if key == "correction_gate":
            continue
        paths = pipeline[key] if isinstance(pipeline[key], (list, tuple)) else [pipeline[key]]
        for checkpoint in paths:
            saved = load_checkpoint(checkpoint)
            if (saved.get("epoch", 0) < 1 or not saved.get("model")
                    or saved.get("metadata", {}).get("data_identity") != data_identity):
                raise StageUnavailable("ENDPOINT_TRAINED_FEATURE_CHECKPOINT_REQUIRED: " + key)
    protocols = manifest.get("protocols")
    if not isinstance(protocols, dict) or not protocols:
        raise StageUnavailable("ENDPOINT_PROTOCOL_REGISTRY_REQUIRED")
    for protocol in protocols.values():
        if (not isinstance(protocol, dict) or protocol.get("label_type") != manifest["label_type"]
                or any(not _text(protocol.get(key)) for key in
                       ("denominator", "sampling_gate", "protocol_identity"))):
            raise StageUnavailable("ENDPOINT_PROTOCOL_REGISTRY_INVALID")
    definitions = {(p["label_type"], p["denominator"], p["sampling_gate"]) for p in protocols.values()}
    if len(definitions) != 1:
        raise StageUnavailable("ENDPOINT_INCOMPATIBLE_OBSERVATION_DEFINITIONS")
    policy = manifest.get("split_policy", {})
    if (policy.get("chemical_holdout") not in {"identity_group", "condition"}
            or type(policy.get("source_holdout")) is not bool):
        raise StageUnavailable("ENDPOINT_SPLIT_POLICY_REQUIRED")
    partitions = manifest.get("partitions", {})
    if set(partitions) != set(PARTITION_ROLES):
        raise StageUnavailable("ENDPOINT_SCHEMA2_PARTITIONS_REQUIRED: T/V/C/E")
    unique = {key: set() for key in ("condition", "sample", "observation", "location")}
    group_roles, paths, used_protocols = {}, set(), set()
    group_fields = ["group_id", "experiment_id"]
    if policy["chemical_holdout"] == "identity_group":
        group_fields.append("chemical_group_id")
    if policy["source_holdout"]:
        group_fields.append("source_id")
    for role, entry in partitions.items():
        source = _source(path.parent, entry)
        if source in paths:
            raise ValueError("ENDPOINT_PARTITION_REUSED")
        paths.add(source)
        index = entry.get("unit_index")
        if not isinstance(index, list) or not index:
            raise StageUnavailable("ENDPOINT_UNIT_INDEX_REQUIRED: " + role)
        for row in index:
            _check_index_row(row, role, protocols)
            used_protocols.add(row["protocol_id"])
            identifiers = {"condition": [row["experimental_condition_id"]],
                           "observation": row["observation_ids"]}
            if row.get("endpoint_sample_id"):
                identifiers["sample"] = [(row["source_id"], row["endpoint_sample_id"])]
            locations = []
            for record in row["measurement_provenance"]:
                _source(path.parent, record)
                locations.append((record["sha256"], record["locator"], row["label_type"]))
            identifiers["location"] = locations
            for key, values in identifiers.items():
                for value in values:
                    if value in unique[key]:
                        raise ValueError("DUPLICATE_ENDPOINT_UNIT: " + key)
                    unique[key].add(value)
            for record in row["initial_state"].get("source_files", []):
                _source(path.parent, record)
            for field in group_fields:
                key = (field, row[field])
                if key in group_roles and group_roles[key] != role:
                    raise ValueError("ENDPOINT_GROUP_LEAKAGE: " + field)
                group_roles[key] = role
    if used_protocols != set(protocols):
        raise StageUnavailable("ENDPOINT_UNUSED_PROTOCOL_DECLARATION")
    return True


def load_units_v2(path, manifest, role, shape):
    if role not in PARTITION_ROLES:
        raise StageUnavailable("ENDPOINT_SCHEMA2_ROLE_UNAVAILABLE: " + str(role))
    entry = manifest["partitions"][role]
    source = _source(path.parent, entry)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows or len(rows) != len(entry["unit_index"]):
        raise StageUnavailable("ENDPOINT_PARTITION_INDEX_MISMATCH")
    expected = {item["experimental_condition_id"]: item for item in entry["unit_index"]}
    seen = set()
    for row in rows:
        condition = row.get("experimental_condition_id")
        if condition in seen or endpoint_unit_index(row) != expected.get(condition):
            raise StageUnavailable("ENDPOINT_PARTITION_INDEX_MISMATCH")
        seen.add(condition)
        if (row.get("feature_kind") != "predicted" or row.get("feature_status") != "complete"
                or row.get("frozen_feature_identity") != manifest["frozen_feature_identity"]
                or "expression_sample_id" in row or "pairing_verified" in row):
            raise StageUnavailable("ENDPOINT_PREDICTED_FEATURES_REQUIRED")
        y, elapsed, horizon = (row.get(key) for key in ("label_fraction", "elapsed", "horizon"))
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (y, elapsed, horizon)) or not 0 <= y <= 1 or not 0 <= elapsed <= horizon or horizon <= 0):
            raise ValueError("ENDPOINT_VALUE_OR_TIME_INVALID")
        sizes = {"population": (shape.d_model,), "bio": (27,), "context": (3 * shape.context_dim,)}
        for key, size in sizes.items():
            value = np.asarray(row.get(key, []), dtype=float)
            if value.shape != size or not np.isfinite(value).all():
                raise ValueError("ENDPOINT_FEATURE_SHAPE_OR_VALUE_MISMATCH: " + key)
        history = np.asarray(row.get("history", []), dtype=float)
        if (history.ndim != 2 or not len(history) or history.shape[1] != shape.history_input_dim
                or not np.isfinite(history).all()):
            raise ValueError("ENDPOINT_HISTORY_SHAPE_OR_VALUE_MISMATCH")
    return rows


def dependency_files_v2(path, manifest):
    """Files whose bytes must stay frozen when loading/calibrating G."""
    files = {path.resolve()}
    for records in manifest["feature_dependencies"]["assets"].values():
        files.update((path.parent / record["path"]).resolve() for record in records)
    for entry in manifest["partitions"].values():
        files.add((path.parent / entry["path"]).resolve())
        for row in entry["unit_index"]:
            records = row["measurement_provenance"] + row["initial_state"].get("source_files", [])
            files.update((path.parent / record["path"]).resolve() for record in records)
    return sorted(files)
