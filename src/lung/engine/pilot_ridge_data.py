"""Full unique T/V population means and frozen-condition Ridge features."""

import hashlib
import json
from contextlib import closing

import numpy as np
import pyarrow.dataset as ds
import torch

from .config import config_hash
from .ridge_stream_cuda import CudaPopulationAccumulator, pack_counts


CONTEXT_FIELDS = ("source_id", "background_id", "protocol_id")
ROLE_NAMES = {"T": "fit", "V": "selection"}


def require_ridge_data(data):
    if data.reader.allowed_roles != frozenset(ROLE_NAMES.values()) or data.reader.allow_test:
        raise PermissionError("RIDGE_REQUIRES_T_V_ONLY_READER")
    if data.manifest["status"] != "complete" or data.bandwidth is None:
        raise ValueError("PREPROCESSING_REQUIRED: Ridge never prepares assets")
    if (np.shape(data.mu) != (data.n_genes,) or np.shape(data.scale) != (data.n_genes,)
            or not np.isfinite(data.mu).all() or not np.isfinite(data.scale).all()
            or (np.asarray(data.scale) <= 0).any()):
        raise ValueError("INVALID_FROZEN_NORMALIZATION")


def mean_plan(data, role):
    """Resolve all eligible conditions and matched pools without decoding expression."""
    require_ridge_data(data)
    if role not in ROLE_NAMES:
        raise PermissionError("RIDGE_ONLY_T_V; no final-test response access")
    reader, selected = data.reader, data.condition_ids(role)
    counts = reader.role_counts.loc[reader.role_counts.role.eq(ROLE_NAMES[role])]
    counts = counts.set_index("condition_id").n_cells.to_dict()
    conditions, needed = [], set()
    for key in selected:
        row = reader.condition_table.loc[key]
        if not bool(row["eligible"]) or row["condition_type"] != "treated":
            raise ValueError("RIDGE_REQUIRES_ALL_ELIGIBLE_TREATED_CONDITIONS")
        if float(row["duration_h"]) != 24.:
            raise ValueError("PILOT_V1_REQUIRES_24_H")
        links = reader.controls.loc[reader.controls.condition_id.eq(key), "control_condition_id"]
        controls = sorted(set(links) & set(counts))
        if not controls or counts.get(key, 0) < 1 or any(counts[item] < 1 for item in controls):
            raise ValueError(f"NO_SAME_ROLE_POPULATION: {role}/{key}")
        for item in controls:
            control = reader.condition_table.loc[item]
            if (control["condition_type"] != "control" or any(
                    control[field] != row[field] for field in (*CONTEXT_FIELDS, "plate"))):
                raise ValueError("RIDGE_CONTROL_BACKGROUND_MISMATCH")
        conditions.append({"condition_id": key, "group_id": str(row["chemical_or_experiment_group_id"]),
                           "control_condition_ids": controls})
        needed.update([key, *controls])
    if not conditions:
        raise ValueError(f"NO_ELIGIBLE_{role}_CONDITIONS")
    return {"role": role, "conditions": conditions,
            "pool_counts": {key: int(counts[key]) for key in sorted(needed)},
            "sampling": "all_unique_cells_without_sampling_or_replacement",
            "control_weighting": "equal_cells_across_linked_same_role_control_pools"}


def stream_means(data, plan, *, batch_size=512, device="cpu", progress=None,
                 seen_cells=None, seen_physical=None, prefetch=True):
    """Read each planned physical cell once; bound expression memory by batch_size.

    The shared reader freezes float32 standardized coordinates; their sums and
    means are computed in float64. No new normalization is fitted here.
    """
    require_ridge_data(data)
    role = plan.get("role")
    if role not in ROLE_NAMES:
        raise PermissionError("RIDGE_ONLY_T_V")
    if plan != mean_plan(data, role):
        raise ValueError("RIDGE_MEAN_PLAN_CHANGED")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    names = list(plan["pool_counts"])
    positions = {key: i for i, key in enumerate(names)}
    use_cuda = torch.device(device).type == "cuda"
    accumulator = None
    sums = None
    if use_cuda:
        accumulator = CudaPopulationAccumulator(len(names), data.mu, data.scale, device)
    else:
        sums = torch.zeros((len(names), data.n_genes), dtype=torch.float64, device=device)
    counts = np.zeros(len(names), dtype=np.int64)
    seen = set() if seen_cells is None else seen_cells
    physical_seen = set() if seen_physical is None else seen_physical
    digest, processed = hashlib.sha256(), 0
    role_name = ROLE_NAMES[role]
    source = data.reader.assets_path / ("train.parquet" if role == "T" else "val.parquet")
    scanner = ds.dataset(source, format="parquet").scanner(
        columns=["cell_id", "condition_id", "matrix_path", "matrix_row", "role"],
        filter=(ds.field("role") == role_name) & ds.field("condition_id").isin(names),
        batch_size=batch_size, use_threads=False)
    total = sum(plan["pool_counts"].values())
    phase = f"{role} full unique cells" + (" [CUDA sparse FP64]" if use_cuda else " [CPU]")
    if progress:
        progress(phase, 0, total)

    def batches():
        for batch in scanner.to_batches():
            rows = batch.to_pandas()
            if rows.empty:
                continue
            if not rows.role.eq(role_name).all():
                raise PermissionError("RIDGE_EXPRESSION_ROLE_VIOLATION")
            for record in rows.itertuples(index=False):
                physical = (str(record.matrix_path), int(record.matrix_row))
                if record.cell_id in seen or physical in physical_seen:
                    raise ValueError("DUPLICATE_PHYSICAL_CELL: no duplicated mean observations")
                seen.add(record.cell_id)
                physical_seen.add(physical)
                digest.update((json.dumps(list(record), ensure_ascii=False) + "\n").encode("utf-8"))
            yield rows, np.asarray([positions[key] for key in rows.condition_id], dtype=np.int64)

    def load(item):
        rows, indices = item
        raw = data.reader.rows.read(rows)
        return indices, pack_counts(raw, indices) if use_cuda else raw

    if use_cuda and prefetch:
        from lung.data.stream_cache import prefetched
        # A single producer owns SparseRows' mutable LRU; it never races another reader.
        loaded = prefetched(batches(), load, workers=1)
    else:
        loaded = (load(item) for item in batches())
    with closing(loaded):
        for indices, raw in loaded:
            if accumulator is not None:
                accumulator.add(raw)
            else:
                assert sums is not None
                normalized = data.reader.normalize(raw)
                if normalized.shape != (len(indices), data.n_genes) or not np.isfinite(normalized).all():
                    raise ValueError("INVALID_FULL_GENE_OBSERVATION")
                sums.index_add_(0, torch.as_tensor(indices, device=device),
                                torch.as_tensor(normalized, dtype=torch.float64, device=device))
            counts += np.bincount(indices, minlength=len(names))
            processed += len(indices)
            if progress:
                progress(phase, processed, total)
    expected = np.asarray(list(plan["pool_counts"].values()), dtype=np.int64)
    if not np.array_equal(counts, expected):
        raise ValueError("RIDGE_FULL_POPULATION_COUNT_MISMATCH")
    if accumulator is not None:
        pool_means = accumulator.means(counts).cpu().numpy()
    else:
        assert sums is not None
        pool_means = (sums / torch.as_tensor(counts[:, None], device=device)).cpu().numpy()
    controls, targets = [], []
    for condition in plan["conditions"]:
        control_indices = [positions[key] for key in condition["control_condition_ids"]]
        # Pool means are weighted by actual unique cell counts, never by pool count.
        controls.append(np.average(pool_means[control_indices], axis=0, weights=counts[control_indices]))
        targets.append(pool_means[positions[condition["condition_id"]]])
    return {"condition_ids": np.asarray([row["condition_id"] for row in plan["conditions"]], dtype=str),
            "group_ids": np.asarray([row["group_id"] for row in plan["conditions"]], dtype=str),
            "control_mean": np.stack(controls), "target_mean": np.stack(targets),
            "pool_condition_ids": np.asarray(names, dtype=str), "pool_mean": pool_means,
            "pool_n": counts, "metadata_json": np.asarray(json.dumps({
                "role": role, "data_identity": data.identity, "plan_sha256": config_hash(plan),
                "ordered_cell_read_sha256": digest.hexdigest(), "unique_cells": processed,
                "standardized_input_dtype": "float32_frozen_shared_reader",
                "accumulation_dtype": "float64", "control_summary_scope": f"full_unique_{role}_control_pool_mean",
                "numeric_backend": "cuda_sparse_frozen_coordinates" if use_cuda else "cpu_dense_frozen_coordinates",
                "prefetch_batches": int(use_cuda and prefetch)}))}


def features(data, condition_ids, control_mean):
    """No target expression enters X; categorical registry matches the main model."""
    if np.shape(control_mean) != (len(condition_ids), data.n_genes):
        raise ValueError("FULL_CONTROL_GENE_AXIS_REQUIRED")
    vocabulary = data.reader.context_vocabulary
    blocks = [np.asarray(control_mean, dtype=np.float64)]
    names = [f"control:{gene}" for gene in data.reader.gene_ids]
    fingerprints, doses, times, contexts = [], [], [], []
    for key in condition_ids:
        action = data.action_features(str(key))
        fingerprint = np.asarray(action["fingerprint"], dtype=np.float64)
        if fingerprint.shape != (2048,) or not np.isin(fingerprint, [0., 1.]).all():
            raise ValueError("FROZEN_2048_BIT_MORGAN_REQUIRED")
        if action["is_vehicle"] or not np.isfinite(action["dose"]) or action["dose"] < 0:
            raise ValueError("VALID_TREATED_DOSE_REQUIRED")
        if action["duration"] != 24.:
            raise ValueError("PILOT_V1_REQUIRES_24_H")
        fingerprints.append(fingerprint)
        doses.append(np.log1p(action["dose"]))
        # Same fixed single-exposure setting as condition_kwargs(elapsed=0,horizon=24).
        times.append([np.log1p(action["duration"]), 0., 1., 0.])
        encoded = []
        for field in CONTEXT_FIELDS:
            values = np.zeros(len(vocabulary[field]), dtype=np.float64)
            values[action[field]] = 1.
            encoded.extend(values)
        contexts.append(encoded)
    fingerprint = np.stack(fingerprints)
    dose = np.asarray(doses)[:, None]
    blocks.extend([fingerprint, dose, fingerprint * dose, np.asarray(times), np.asarray(contexts)])
    names += [f"morgan:{i}" for i in range(2048)] + ["log1p_dose_um"]
    names += [f"morgan_log1p_dose:{i}" for i in range(2048)]
    names += ["log1p_duration_h", "elapsed_fraction", "duration_fraction", "is_vehicle"]
    names += [f"{field}:{value}" for field in CONTEXT_FIELDS for value in vocabulary[field]]
    matrix = np.concatenate(blocks, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("NONFINITE_RIDGE_INPUT")
    return matrix, names
