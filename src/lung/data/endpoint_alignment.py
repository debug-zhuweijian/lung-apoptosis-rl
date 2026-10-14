"""Condition-level measured endpoints; never invent matched expression samples."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


FRACTION_ENDPOINTS = {"Annexin_positive_fraction", "SYTOX_lethal_fraction"}


def stable_id(namespace: str, *values) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return namespace + ":" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_alignment(conditions, interventions, observations, protocols):
    """Validate experimental units, not a claim that every condition is model-ready."""
    required = {
        "conditions": {"condition_id", "source_id", "protocol_id", "cell_line",
                       "initial_state_id", "initial_state_kind", "treatment_hours",
                       "exposure_design", "split_group"},
        "interventions": {"condition_id", "compound_id", "source_compound_name",
                          "dose_value", "dose_unit", "identity_status"},
        "observations": {"observation_id", "condition_id", "endpoint", "value",
                         "unit", "label_kind", "replicate_id", "source_file",
                         "source_sha256", "source_locator", "measurement_status",
                         "exclusion_reason"},
    }
    frames = {"conditions": conditions, "interventions": interventions,
              "observations": observations}
    for name, fields in required.items():
        missing = fields - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name}: missing fields {sorted(missing)}")
    for frame, field in ((conditions, "condition_id"), (observations, "observation_id")):
        if frame[field].isna().any() or frame[field].duplicated().any():
            raise ValueError(f"nonunique {field}")
    if observations.duplicated(["condition_id", "endpoint", "replicate_id"]).any():
        raise ValueError("duplicate experimental replicate; technical readings are not independent repeats")
    condition_ids = set(conditions.condition_id)
    for frame in (observations, interventions):
        if not set(frame.condition_id) <= condition_ids:
            raise ValueError("orphan experimental condition")
    if interventions.duplicated(["condition_id", "compound_id"]).any():
        raise ValueError("duplicate intervention within condition")
    for row in conditions.to_dict("records"):
        if row["protocol_id"] not in protocols:
            raise ValueError("missing endpoint protocol")
        protocol = protocols[row["protocol_id"]]
        if not all(protocol.get(k) for k in ("denominator", "sampling_gate", "source")):
            raise ValueError("undefined denominator or sampling gate")
        if not math.isfinite(float(row["treatment_hours"])) or row["treatment_hours"] <= 0:
            raise ValueError("invalid exposure time")
        if row["initial_state_kind"] not in {"reference", "observed_population", "observed_cell"}:
            raise ValueError("unknown initial state provenance")
        if row["initial_state_kind"] != "reference":
            for field in ("initial_expression_sample_id", "initial_state_evidence"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("observed initial expression requires sample identity and evidence")
        if row.get("same_sample_expression_endpoint") is True:
            evidence = row.get("same_sample_pairing_evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError("same-sample pairing requires independent pairing evidence")
    if not interventions.dose_unit.isin(["uM", "ng/mL"]).all():
        raise ValueError("dose units require explicit conversion")
    doses = pd.to_numeric(interventions.dose_value, errors="raise")
    if not doses.map(math.isfinite).all() or (doses <= 0).any():
        raise ValueError("invalid intervention dose")
    seen_locations = set()
    condition_protocol = conditions.set_index("condition_id").protocol_id.to_dict()
    for row in observations.to_dict("records"):
        if row["label_kind"] != "measured":
            raise ValueError("generated scores cannot become measured endpoints")
        if row["endpoint"] not in FRACTION_ENDPOINTS or row["unit"] != "fraction":
            raise ValueError("endpoint definition is not interchangeable")
        if protocols[condition_protocol[row["condition_id"]]].get("endpoint") != row["endpoint"]:
            raise ValueError("observation endpoint disagrees with experimental protocol")
        if not all(row[k] for k in ("source_file", "source_sha256", "source_locator", "replicate_id")):
            raise ValueError("missing observation provenance")
        if len(row["source_sha256"]) != 64:
            raise ValueError("invalid source hash")
        locator = (row["source_sha256"], row["source_locator"], row["endpoint"])
        if locator in seen_locations:
            raise ValueError("one measured endpoint duplicated into multiple examples")
        seen_locations.add(locator)
        if row["measurement_status"] == "missing":
            if not pd.isna(row["value"]) or not row["exclusion_reason"]:
                raise ValueError("missing observation must remain missing with a reason")
        elif row["measurement_status"] == "observed":
            if not math.isfinite(float(row["value"])) or not 0 <= row["value"] <= 1:
                raise ValueError("measured fraction outside [0,1]")
        else:
            raise ValueError("unknown measurement status")
    return {"conditions": len(conditions), "observations": len(observations),
            "observed": int((observations.measurement_status == "observed").sum()),
            "excluded": int(observations.exclusion_reason.fillna("").ne("").sum())}


def validate_source_links(observations, source_files):
    declared = {}
    for source in source_files:
        name, digest = source["file"], source["sha256"]
        if name in declared and declared[name] != digest:
            raise ValueError("conflicting source registry")
        declared[name] = digest
    for row in observations.to_dict("records"):
        if declared.get(row["source_file"]) != row["source_sha256"]:
            raise ValueError("observation source is not bound to verified source registry")


def aggregate_units(conditions, observations):
    """One row per experimental condition/endpoint; repeats remain in raw table."""
    usable = observations[(observations.measurement_status == "observed") &
                          observations.exclusion_reason.fillna("").eq("")].copy()
    kinds = usable.get("replicate_kind", pd.Series("unspecified", index=usable.index))
    usable["independent_repeat_available"] = kinds.eq("independent_experiment")
    summary = usable.groupby(["condition_id", "endpoint"], sort=True).agg(
        mean_fraction=("value", "mean"), sd_fraction=("value", "std"),
        available_numeric_records=("observation_id", "size"),
        measured_replicates=("independent_repeat_available", "sum"),
    ).reset_index()
    # A single reported flow panel has no estimate of biological replicate variance.
    return summary.merge(conditions, on="condition_id", validate="many_to_one")


def assert_split_isolation(conditions, assignments):
    """Replicates, chemical series and connected source blocks cannot leak roles."""
    if assignments.condition_id.duplicated().any():
        raise ValueError("condition appears in more than one partition")
    merged = assignments.merge(conditions[["condition_id", "split_group"]],
                               on="condition_id", how="left", validate="one_to_one")
    if merged.split_group.isna().any():
        raise ValueError("partition references unknown condition")
    if (merged.groupby("split_group").partition.nunique() > 1).any():
        raise ValueError("experimental/source group crosses partitions")


def load_alignment(directory: Path, project_root: Path, verify_sources=True):
    """Load the versioned endpoint tables and verify actual source bytes."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported alignment table schema")
    for item in manifest["files"]:
        if file_hash(directory / item["file"]) != item["sha256"]:
            raise ValueError(f"alignment table changed: {item['file']}")
    if verify_sources:
        for item in manifest["source_files"]:
            if file_hash(project_root / item["file"]) != item["sha256"]:
                raise ValueError(f"experimental source changed: {item['file']}")
    frames = {k: pd.read_parquet(directory / f"{k}.parquet")
              for k in ("conditions", "interventions", "observations")}
    protocols = json.loads((directory / "protocols.json").read_text(encoding="utf-8"))
    validate_source_links(frames["observations"], manifest["source_files"])
    validate_alignment(**frames, protocols=protocols)
    return {**frames, "protocols": protocols, "manifest": manifest}
