"""Build versioned training indices without modifying any prepared source."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.lung_data import PreparedLungData

from .chemistry import build_chemical_catalog
from .common import OUTER, ROLES, atomic_json, dataset_config, digest_json, inside, sha256
from .preprocess import build_preprocessing as _build_preprocessing
from .splits import condition_roles, control_roles, tidyi_splits


def _verify_source_tables(data, names):
    recorded = {item["path"]: item for item in data.manifest.get("outputs", [])}
    verified = {}
    for name in names:
        value = data.manifest["tables"][name]
        path = inside(data.bundle_path, value)
        digest = sha256(path)
        if value in recorded and digest != recorded[value]["sha256"]:
            raise ValueError(f"Prepared table hash mismatch: {name}")
        verified[name] = {"path": str(path.relative_to(data.project_root).as_posix()), "sha256": digest}
    return verified


def _build_metadata(root, assets, config, progress):
    data = PreparedLungData(root, config["bundle"])
    verified = _verify_source_tables(data, ("gene_map", "drug_catalog", "tahoe_conditions",
                                           "tahoe_cells", "tahoe_control_links", "tidyi_cells"))
    genes = data.genes().sort_values("common_gene_index", kind="stable")
    if len(genes) != config["gene_count"]:
        raise ValueError(f"Expected {config['gene_count']} common genes, found {len(genes)}")
    catalog, fp, chemistry = build_chemical_catalog(data.table("drug_catalog"))
    conditions = condition_roles(data.table("tahoe_conditions"), catalog)
    treated_roles = conditions.loc[conditions["condition_type"].eq("treated") & conditions["eligible"], "role"]
    if set(treated_roles) != set(ROLES):
        raise ValueError("CHEMICAL_SPLIT_INSUFFICIENT: one or more required roles are empty")
    genes.to_parquet(assets / "genes.parquet", index=False)
    catalog.to_parquet(assets / "chemicals.parquet", index=False)
    np.save(assets / "fingerprints.npy", fp, allow_pickle=False)
    conditions.to_parquet(assets / "conditions.parquet", index=False)
    conditions.loc[~conditions["eligible"]].to_parquet(assets / "unresolved_conditions.parquet", index=False)
    links = data.table("tahoe_control_links")
    if not links["individual_cell_pairing"].eq("not_available").all():
        raise ValueError("Controls must not claim individual biological pairing")
    if not links["relationship_type"].eq("same_plate_control_candidate").all():
        raise ValueError("Unsupported control relationship")
    by_id = conditions.set_index("condition_id")
    for row in links.itertuples():
        target, control = by_id.loc[row.condition_id], by_id.loc[row.control_condition_id]
        if target.plate != control.plate or control.condition_type != "control":
            raise ValueError("Control links must point to controls on the same plate")
    links.to_parquet(assets / "control_links.parquet", index=False)
    cells_path = inside(data.bundle_path, data.manifest["tables"]["tahoe_cells"])
    control_ids = conditions.loc[conditions["condition_type"].eq("control"), "condition_id"].tolist()
    controls = pd.read_parquet(cells_path, filters=[("condition_id", "in", control_ids)],
                               columns=["cell_id", "condition_id", "plate"])
    controls["source_id"] = controls["condition_id"].map(by_id["source_id"])
    control_map = control_roles(controls)
    writers = {}
    counters = Counter()
    seen, excluded = 0, 0
    fields = ["condition_id", "role", "outer_split", "condition_type", "eligible", "source_id",
              "background_id", "protocol_id", "replicate_id", "replicate_scope",
              "chemical_or_experiment_group_id"]
    total = pq.ParquetFile(cells_path).metadata.num_rows
    try:
        for record in pq.ParquetFile(cells_path).iter_batches(batch_size=65536):
            rows = record.to_pandas()
            seen += len(rows)
            rows = rows.merge(conditions[fields], on="condition_id", how="left", validate="many_to_one")
            if rows["eligible"].isna().any():
                raise ValueError("Cells refer to an unknown condition")
            excluded += int((~rows["eligible"]).sum())
            rows = rows.loc[rows["eligible"]].copy()
            control = rows["condition_type"].eq("control")
            rows.loc[control, "role"] = rows.loc[control, "cell_id"].map(control_map)
            rows.loc[control, "outer_split"] = rows.loc[control, "role"].map(OUTER)
            rows.loc[control, "chemical_or_experiment_group_id"] = (
                "control|" + rows.loc[control, "source_id"] + "|" + rows.loc[control, "plate"]
                + "|" + rows.loc[control, "cell_id"])
            rows["observation_role"] = np.where(control, "contemporaneous_control", "posttreatment")
            rows["dataset"] = "Tahoe-100M"
            rows["source_version"] = "2dc57900b7981cfcf5e211527169a0b006546a95"
            rows = rows[["dataset", "source_id", "source_version", "cell_id", "source_sample_id",
                         "condition_id", "matrix_path", "matrix_row", "plate", "background_id",
                         "protocol_id", "replicate_id", "replicate_scope", "observation_role",
                         "chemical_or_experiment_group_id", "role", "outer_split"]]
            if rows["role"].isna().any():
                raise ValueError("Unassigned training role")
            # Path confinement applies even to rows not opened during this run.
            for path in rows["matrix_path"].unique():
                inside(root, path)
            for (role, condition), count in rows.groupby(["role", "condition_id"]).size().items():
                counters[(role, condition)] += int(count)
            for outer, part in rows.groupby("outer_split", sort=True):
                table = pa.Table.from_pandas(part, preserve_index=False)
                if outer not in writers:
                    writers[outer] = pq.ParquetWriter(assets / f"{outer}.parquet", table.schema,
                                                     compression="zstd")
                writers[outer].write_table(table)
            if progress:
                progress("split_indices", seen, total)
    finally:
        for writer in writers.values():
            writer.close()
    if set(writers) != {"train", "val", "test"}:
        raise ValueError("Empty outer split")
    counts = pd.DataFrame([{"role": role, "condition_id": condition, "n_cells": count}
                           for (role, condition), count in sorted(counters.items())])
    counts.to_parquet(assets / "role_condition_counts.parquet", index=False)
    # Every treated condition must have nonempty same-role, same-plate controls.
    for condition in conditions.loc[conditions["eligible"] & conditions["condition_type"].eq("treated")].itertuples():
        ids = links.loc[links["condition_id"].eq(condition.condition_id), "control_condition_id"]
        if sum(counters[(condition.role, item)] for item in ids) == 0:
            raise ValueError(f"NO_SAME_ROLE_CONTROL: {condition.condition_id}")
    state, state_report = tidyi_splits(data.table("tidyi_cells"))
    state.to_parquet(assets / "tidyi_splits.parquet", index=False)
    atomic_json(assets / "tidyi_split_report.json", state_report)
    return {"chemistry": chemistry, "source_tables": verified,
            "cells_in_indices": int(counts["n_cells"].sum()), "excluded_unresolved_cells": excluded,
            "role_cells": {str(k): int(v) for k, v in counts.groupby("role")["n_cells"].sum().items()},
            "chemical_groups": int(conditions.loc[conditions["condition_type"].eq("treated") & conditions["eligible"],
                                                  "chemical_or_experiment_group_id"].nunique()),
            "tidyi_folds": state_report,
            "endpoint_status": "ENDPOINT_UNAVAILABLE_no_verified_expression_endpoint_pairs",
            "history_support": "single_exposure_only",
            "coupling_type": "minibatch_entropic_ot_prior_not_experimental_pairing"}


def _asset_hashes(assets):
    names = ["genes.parquet", "chemicals.parquet", "fingerprints.npy", "conditions.parquet",
             "unresolved_conditions.parquet", "control_links.parquet", "train.parquet", "val.parquet",
             "test.parquet", "role_condition_counts.parquet", "tidyi_splits.parquet",
             "tidyi_split_report.json", "normalization.npz", "neighbors.npz", "bandwidths.npz",
             "bandwidth_pairs.parquet"]
    return {name: {"sha256": sha256(assets / name), "bytes": (assets / name).stat().st_size}
            for name in names if (assets / name).exists()}


def prepare_training(project_root, config_path=None, *, build_preprocessing=True, progress=None, device=None):
    """Return training-v1 assets; lightweight mode creates genuine indices only.

    A complete bundle is immutable and hash checked. Incomplete preprocessing
    resumes the same frozen configuration; it never silently changes splits.
    """
    root = Path(project_root).resolve()
    config = dataset_config(root, config_path)
    assets = inside(root, config["assets"])
    assets.mkdir(parents=True, exist_ok=True)
    lock = assets / ".prepare.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Preparation lock exists: {lock}. Verify no preparation is running before removing it.") from error
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        source_hash = sha256(inside(root, config["bundle"]) / "manifest.json")
        signature = digest_json({"config": config, "source_manifest": source_hash,
                                 "training_schema": 1})
        path = assets / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if manifest is not None and manifest["signature"] != signature:
            raise ValueError("Training asset configuration/source changed; create a new version, do not overwrite")
        if manifest is not None:
            for name, item in manifest.get("files", {}).items():
                if sha256(inside(assets, name)) != item["sha256"]:
                    raise ValueError(f"Frozen training asset changed: {name}")
        else:
            manifest = {"schema_version": 1, "signature": signature, "config": config,
                        "source_manifest_sha256": source_hash, "status": "building_metadata",
                        "created_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
            atomic_json(path, manifest)
        if manifest["status"] == "building_metadata":
            manifest["metadata"] = _build_metadata(root, assets, config, progress)
            manifest["status"] = "metadata_ready"
            manifest["files"] = _asset_hashes(assets)
            atomic_json(path, manifest)
        if build_preprocessing and manifest["status"] != "complete":
            manifest["preprocessing"] = _build_preprocessing(root, assets, config, progress, device=device)
            from .bandwidth import build_bandwidths
            manifest["bandwidth"] = build_bandwidths(root, assets, config, progress, device=device)
            manifest["status"] = "complete"
            manifest["files"] = _asset_hashes(assets)
            atomic_json(path, manifest)
        return assets
    finally:
        lock.unlink(missing_ok=True)
