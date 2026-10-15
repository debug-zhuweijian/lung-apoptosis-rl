"""Source-reported sci-Plex endpoints in the actual shared response data bundle."""

from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from src.lung_data import PreparedLungData
from .common import OUTER, ROLES, atomic_json, digest_json, inside, sha256, source_shard_registry
from .drug_identity import describe_structure
from .splits import control_roles, grouped_roles

SOURCE = "sciPlex3:GSM4150378"
PROTOCOL = "sciplex_author_reported_endpoint_v1"
SPLIT_POLICY = "joint_full_and_parent_connectivity_components_v1"
KEYS = ["catalog_number", "time_point", "dose_uM", "replicate", "culture_plate", "well_oligo"]


def split_components(chemical):
    """Conservative leakage groups, separate from full chemical identity."""
    rows = chemical.loc[chemical.eligible_for_structural_features.eq(True) & ~chemical.is_vehicle].to_dict("records")
    parent = {row["drug_id"]: row["drug_id"] for row in rows}

    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key

    seen = {}
    for row in rows:
        keys = []
        value = row.get("connectivity_smiles")
        if isinstance(value, str) and value:
            keys.append("full:" + value)
        value = row.get("parent_smiles")
        if isinstance(value, str) and value:
            graph = describe_structure(value).get("connectivity_smiles")
            if isinstance(graph, str) and graph:
                keys.append("parent:" + graph)
        for key in keys:
            if key in seen:
                left, right = find(row["drug_id"]), find(seen[key])
                parent[max(left, right)] = min(left, right)
            else:
                seen[key] = row["drug_id"]
    members = {}
    for key in parent:
        members.setdefault(find(key), []).append(key)
    groups = {drug: "leakage:" + digest_json(sorted(items))
              for items in members.values() for drug in items}
    roles = grouped_roles(groups.values(), prefix=SPLIT_POLICY + "|")
    return pd.DataFrame([{"drug_id": drug, "split_group_id": group, "role": roles[group]}
                         for drug, group in sorted(groups.items())])


def _catalog(directory, old, old_fp):
    source = pd.read_parquet(directory / "chemical_identity.parquet")
    fp = np.load(directory / "fingerprints.npy", allow_pickle=False)
    usable = source.eligible_for_structural_features
    if (source.drug_id.duplicated().any() or usable.isna().any() or not pd.api.types.is_bool_dtype(usable)
            or not source.loc[usable, "parse_status"].eq("parsed").all()
            or source.experimental_material_verified.any() or source.cross_source_pair_verified.any()
            or fp.shape != (int(usable.sum()), 2048) or fp.dtype != np.uint8
            or not np.isin(fp, [0, 1]).all()
            or sorted(source.loc[usable, "fingerprint_row"].tolist()) != list(range(len(fp)))
            or not source.loc[~usable, "fingerprint_row"].eq(-1).all()):
        raise ValueError("Invalid sci-Plex source-reported chemical feature contract")
    if not fp.any(axis=1).all():
        raise ValueError("sci-Plex usable fingerprints may not contain fake zero rows")
    for value in source.loc[usable, "canonical_full_smiles"]:
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        if mol is None or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            raise ValueError("sci-Plex usable structure contains unknown or dummy atoms")
    source = source.assign(drug_name=source.source_name,
                           fingerprint_row=(source.fingerprint_row + len(old_fp)).where(usable, -1))
    vehicle = {"drug_id": "sciplex:S0000", "drug_name": "DMSO", "catalog_number": "S0000",
               "identity_id": "vehicle:DMSO", "chemical_or_experiment_group_id": "vehicle:DMSO",
               "identity_status": "vehicle_control", "is_vehicle": True, "eligible_for_structural_features": False,
               "identity_version": source.identity_version.iloc[0],
               "experimental_material_verified": False, "cross_source_pair_verified": False,
               "fingerprint_row": len(old_fp) + len(fp)}
    return (pd.concat([old, source, pd.DataFrame([vehicle])], ignore_index=True, sort=False),
            np.concatenate([old_fp, fp, np.zeros((1, 2048), dtype=np.uint8)]))


def _source_rows(root, source, genes, chemical, planned_path, time_contract):
    columns = ["cell", "sample", "cell_type", "source_cell_index", "catalog_number", "vehicle", "dose_uM", "time_point",
               "replicate", "culture_plate", "well_oligo", "plate_oligo", "author_hash_qc", "shard_id", "row_in_shard"]
    cells = pd.read_parquet(source / "a549_cells.parquet", columns=columns)
    axis = pd.read_parquet(source / "common_genes.parquet")
    if axis.ensembl_id.tolist() != genes.ensembl_id.tolist() or axis.common_gene_index.tolist() != list(range(len(genes))):
        raise ValueError("sci-Plex common gene axis differs from training genes")
    if (cells.cell.duplicated().any() or not cells.cell_type.eq("A549").all() or cells[KEYS].isna().any().any()
            or not cells.time_point.isin([24., 72.]).all()
            or not cells.author_hash_qc.isin([True, False]).all()
            or not np.isfinite(cells.dose_uM).all() or cells.dose_uM.lt(0).any()
            or not cells.vehicle.eq(cells.catalog_number.eq("S0000")).all()):
        raise ValueError("Invalid sci-Plex source cell/condition metadata")
    planned = pd.read_csv(planned_path)
    planned["culture_plate"] = planned.plate_oligo
    planned["dose_uM"] = planned.dose.astype(float) / 1000.
    if planned[KEYS].duplicated().any() or planned[KEYS].isna().any().any():
        raise ValueError("Invalid planned sci-Plex culture wells")
    observed = {key: group for key, group in cells.groupby(KEYS, sort=True, dropna=False)}
    records, mapping = [], {}
    lookup = chemical.set_index("drug_id", drop=False)
    for item in planned.to_dict("records"):
        key = tuple(item[name] for name in KEYS)
        group = observed.pop(key, cells.iloc[:0])
        drug, time, dose, replicate, plate, well = key
        cid = "sciplex:" + digest_json(list(key))[:24]
        mapping.update({i: cid for i in group.index})
        entry = lookup.loc["sciplex:" + drug]
        vehicle = drug == "S0000"
        qc = int(group.author_hash_qc.sum())
        if qc != int(item["hash_pass_cells"]):
            raise ValueError("Planned well QC cell counts differ from extracted cells")
        records.append({"condition_id": cid, "drug_id": entry.drug_id, "condition_type": "control" if vehicle else "treated",
            "plate": plate, "source": SOURCE, "source_id": SOURCE, "cell_line_id": "CVCL_0023", "background_id": "CVCL_0023",
            "source_sample_id": cid, "source_sample_label": group["sample"].iloc[0] if len(group) else None, "dose_value": float(dose),
            "dose_unit": "uM", "dose_um": float(dose),
            "source_dose_value": float(item["dose"]), "source_dose_unit": "nM",
            "dose_semantics": ("vehicle_source_placeholder_not_solvent_molarity" if vehicle else
                               "source_reported_nominal_compound_concentration"),
            "treatment_hours": float(time), "duration_h": float(time),
            "source_time_h": float(time), "reported_treatment_hours": float(time), "exposure_duration_h": None,
            "withdrawal_protocol": None, "time_semantics": time_contract["time_semantics"],
            "is_sequential": False, "protocol_id": PROTOCOL, "replicate_id": str(replicate),
            "replicate_scope": "author_reported_replicate_identifier", "culture_plate": plate, "well_oligo": well,
            "plate_oligo": item["plate_oligo"], "chemical_or_experiment_group_id": entry.chemical_or_experiment_group_id,
            "identity_status": entry.identity_status, "fingerprint_row": int(entry.fingerprint_row),
            "identity_version": entry.identity_version, "eligible_for_structural_features": bool(entry.eligible_for_structural_features),
            "is_vehicle": vehicle, "eligible": bool(qc > 0 and (vehicle or entry.eligible_for_structural_features)),
            "decision_eligible": False, "n_source_cells": len(group), "n_qc_cells": qc,
            "missing_reason": ("planned_well_no_qc_cells" if not qc else
                               "source_structure_not_encodable" if not vehicle and not entry.eligible_for_structural_features else None)})
    if observed or len(mapping) != len(cells):
        raise ValueError("Observed sci-Plex wells are absent from the planned source table")
    cells = cells.assign(condition_id=pd.Series(mapping)).loc[cells.author_hash_qc].copy()
    conditions = pd.DataFrame(records)
    controls = conditions[conditions.condition_type.eq("control") & conditions.eligible]
    links = []
    context = ["source_id", "background_id", "protocol_id", "source_time_h", "replicate_id", "culture_plate"]
    for record in conditions[conditions.condition_type.eq("treated") & conditions.eligible].to_dict("records"):
        matching = controls.loc[np.logical_and.reduce([controls[name].eq(record[name]) for name in context])]
        if matching.empty:
            raise ValueError(f"sci-Plex condition lacks matching vehicle: {record['condition_id']}")
        links.extend({"condition_id": record["condition_id"], "control_condition_id": cid,
                      "relationship_type": "same_source_time_replicate_culture_plate_vehicle",
                      "individual_cell_pairing": "not_available"} for cid in matching.condition_id)
    extraction = json.loads((source / "extraction_status.json").read_text(encoding="utf-8"))
    registry = {}
    for shard in extraction["shards"]:
        item, sid = shard["common"], shard["shard_id"]
        path = inside(source, item["file"])
        if path.stat().st_size != item["bytes"] or item["columns"] != len(genes):
            raise ValueError("sci-Plex source shard attributes differ")
        registry[sid] = {"matrix_path": path.relative_to(root).as_posix(), "gene_axis": "common",
                         "csr_bytes": item["bytes"], "csr_mtime_ns": path.stat().st_mtime_ns,
                         "csr_sha256": item["sha256"], "source_rows": item["rows"], "source_columns": item["columns"]}
    cells["matrix_path"] = cells.shard_id.map({k: v["matrix_path"] for k, v in registry.items()})
    if cells.matrix_path.isna().any():
        raise ValueError("sci-Plex unknown shard ID")
    cells["matrix_row"] = cells.row_in_shard
    cells["cell_id"] = SOURCE + ":" + cells.cell
    return cells, conditions, pd.DataFrame(links), list(registry.values()), int((~pd.read_parquet(source / "a549_cells.parquet", columns=["author_hash_qc"]).author_hash_qc).sum())


def _write_joint_indices(assets, source_cells, conditions):
    by_id = conditions.set_index("condition_id")
    base_schema = pq.read_schema(assets / "train.parquet")
    schema = base_schema.append(pa.field("split_group_id", pa.string())).append(pa.field("legacy_v3_role", pa.string()))
    counts, writers = Counter(), {}

    def write(frame, legacy):
        frame = frame.copy()
        frame["legacy_v3_role"] = frame.role if legacy else None
        treated = frame.observation_role.eq("posttreatment")
        frame.loc[treated, "role"] = frame.loc[treated, "condition_id"].map(by_id.role)
        frame["outer_split"] = frame.role.map(OUTER)
        frame["split_group_id"] = frame.condition_id.map(by_id.split_group_id)
        control = ~treated
        frame.loc[control, "split_group_id"] = frame.loc[control, "chemical_or_experiment_group_id"]
        if frame.role.isna().any() or frame.split_group_id.isna().any():
            raise ValueError("Unassigned joint source role or leakage group")
        for (role, cid), n in frame.groupby(["role", "condition_id"]).size().items():
            counts[(role, cid)] += int(n)
        for split, rows in frame.groupby("outer_split", sort=True):
            table = pa.Table.from_pandas(rows[schema.names], preserve_index=False).cast(schema)
            if split not in writers:
                writers[split] = pq.ParquetWriter(assets / f"{split}.joint.tmp.parquet", schema, compression="zstd")
            writers[split].write_table(table)

    try:
        for split in ("train", "val", "test"):
            for batch in pq.ParquetFile(assets / f"{split}.parquet").iter_batches(batch_size=65536):
                write(batch.to_pandas(), True)
        write(source_cells, False)
    finally:
        for writer in writers.values():
            writer.close()
    if set(writers) != {"train", "val", "test"}:
        raise ValueError("Empty joint outer split")
    for split in writers:
        (assets / f"{split}.joint.tmp.parquet").replace(assets / f"{split}.parquet")
    result = pd.DataFrame([{"role": role, "condition_id": cid, "n_cells": n} for (role, cid), n in sorted(counts.items())])
    result.to_parquet(assets / "role_condition_counts.parquet", index=False)
    return counts, result


def extend_with_sciplex(root, assets, config, metadata):
    source_config = config["sciplex_source"]
    if source_config["split_policy"] != SPLIT_POLICY:
        raise ValueError("Unregistered joint sci-Plex split policy")
    source = inside(root, source_config["extraction_manifest"]).parent
    chemical_dir = inside(root, source_config["chemical_manifest"]).parent
    genes = pd.read_parquet(assets / "genes.parquet")
    old_conditions = pd.read_parquet(assets / "conditions.parquet")
    baseline = inside(root, source_config["base_training_manifest"]).parent
    previous = pd.read_parquet(baseline / "conditions.parquet")
    pd.testing.assert_frame_equal(old_conditions, previous, check_dtype=False)
    old_conditions = old_conditions.assign(legacy_v3_role=old_conditions.role, decision_eligible=old_conditions.eligible)
    chemical, fp = _catalog(chemical_dir, pd.read_parquet(assets / "chemicals.parquet"),
                             np.load(assets / "fingerprints.npy", allow_pickle=False))
    cells, new_conditions, links, source_registry, qc_excluded = _source_rows(
        root, source, genes, chemical, inside(root, source_config["planned_wells"]),
        json.loads(inside(root, source_config["time_contract"]).read_text(encoding="utf-8")))
    source_qc_cells = len(cells)
    conditions = pd.concat([old_conditions, new_conditions], ignore_index=True, sort=False)
    split = split_components(chemical)
    mapping = split.set_index("drug_id")
    conditions["split_group_id"] = conditions.drug_id.map(mapping.split_group_id)
    treated = conditions.condition_type.eq("treated") & conditions.eligible
    conditions.loc[treated, "role"] = conditions.loc[treated, "drug_id"].map(mapping.role)
    conditions["outer_split"] = conditions.role.map(OUTER)
    conditions.loc[conditions.source_id.eq(SOURCE) & ~treated, "role"] = None
    cells = cells.merge(conditions[["condition_id", "eligible", "condition_type", "source_id", "background_id", "protocol_id",
        "replicate_id", "replicate_scope", "chemical_or_experiment_group_id", "role", "outer_split", "source_sample_id"]], on="condition_id", validate="many_to_one")
    cells = cells[cells.eligible].copy()
    cells["plate"] = cells.culture_plate
    cells["dataset"], cells["source_version"] = "sciPlex3", "GSM4150378/extracted/v1"
    control = cells.condition_type.eq("control")
    control_map = control_roles(cells.loc[control, ["source_id", "plate", "cell_id"]])
    cells.loc[control, "role"] = cells.loc[control, "cell_id"].map(control_map)
    cells.loc[control, "chemical_or_experiment_group_id"] = "control|" + SOURCE + "|" + cells.loc[control, "cell_id"]
    cells["observation_role"] = np.where(control, "contemporaneous_control", "posttreatment")
    counts, role_counts = _write_joint_indices(assets, cells, conditions)
    all_links = pd.concat([pd.read_parquet(assets / "control_links.parquet"), links], ignore_index=True)
    for row in conditions.loc[treated].itertuples():
        controls = all_links.loc[all_links.condition_id.eq(row.condition_id), "control_condition_id"]
        if not sum(counts[(row.role, cid)] for cid in controls):
            raise ValueError(f"NO_SAME_ROLE_CONTROL: {row.condition_id}")
    prepared = PreparedLungData(root, config["bundle"])
    old_registry = source_shard_registry(prepared)
    tahoe_width = len(prepared.table("tahoe_genes"))
    registry = [{"matrix_path": path, "gene_axis": "tahoe", "csr_bytes": item["csr_bytes"],
        "csr_mtime_ns": item["csr_mtime_ns"], "csr_sha256": item.get("csr_sha256_from_checkpoint", item.get("csr_sha256")),
        "source_rows": item["source_rows"], "source_columns": tahoe_width}
        for path, item in old_registry.items()]
    if not registry:  # Standalone tiny test bundles have no archived source audit.
        for path in pd.read_parquet(baseline / "train.parquet", columns=["matrix_path"]).matrix_path.unique():
            file = inside(root, path)
            with np.load(file, allow_pickle=False) as archive:
                shape = archive["shape"]
            registry.append({"matrix_path": path, "gene_axis": "tahoe", "csr_bytes": file.stat().st_size,
                "csr_mtime_ns": file.stat().st_mtime_ns, "csr_sha256": sha256(file), "source_rows": int(shape[0]), "source_columns": int(shape[1])})
    pd.DataFrame(registry + source_registry).to_parquet(assets / "source_shards.parquet", index=False)
    for name, frame in (("conditions", conditions), ("chemicals", chemical), ("control_links", all_links), ("split_groups", split),
                        ("unresolved_conditions", conditions[~conditions.eligible])):
        frame.to_parquet(assets / f"{name}.parquet", index=False)
    np.save(assets / "fingerprints.npy", fp, allow_pickle=False)
    summary = {"source": SOURCE, "source_qc_excluded_cells": qc_excluded, "source_qc_cells": source_qc_cells,
        "source_usable_cells": len(cells),
        "source_treated_conditions": int((new_conditions.condition_type.eq("treated") & new_conditions.eligible).sum()),
        "source_planned_treated_conditions": int(new_conditions.condition_type.eq("treated").sum()),
        "planned_wells_without_qc_cells": int(new_conditions.n_qc_cells.eq(0).sum()),
        "source_control_wells": int(new_conditions.condition_type.eq("control").sum()), "split_policy": SPLIT_POLICY,
        "split_groups": split.split_group_id.nunique(), "source_time_h": [24, 72], "decision_eligible": False,
        "feature_scope": "source_reported_structures_not_independently_material_verified", "expression_normalization_fitted": False}
    atomic_json(assets / "sciplex_source_summary.json", summary)
    return {**metadata, "sciplex_source": summary, "cells_in_indices": int(role_counts.n_cells.sum()),
        "role_cells": {k: int(v) for k, v in role_counts.groupby("role").n_cells.sum().items()},
        "chemical_groups": int(conditions.loc[treated, "chemical_or_experiment_group_id"].nunique()),
        "split_policy": SPLIT_POLICY, "time_support": "source_specific_24h_and_72h_snapshots_not_trajectories"}
