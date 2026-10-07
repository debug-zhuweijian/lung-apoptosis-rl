"""V-only baseline reporting; no fitting, preprocessing, or final-test access."""

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from lung.models import physical_projection, rbf_mmd
from .config import config_hash, file_hash


METRICS = ("mean_rmse", "mmd_u", "delta_correlation", "direction_agreement",
           "variance_rmse", "output_valid_rate")


def require_v_only(data):
    if data.reader.allowed_roles != frozenset({"selection"}) or data.reader.allow_test:
        raise PermissionError("PILOT_REQUIRES_V_ONLY_READER")
    if data.manifest["status"] != "complete" or data.bandwidth is None:
        raise ValueError("PREPROCESSING_REQUIRED: pilot never fits or prepares assets")


def condition_pools(data, condition_id):
    require_v_only(data)
    reader = data.reader
    target = reader.cell_metadata(condition_id, role="selection")
    links = reader.controls.loc[reader.controls.condition_id.eq(condition_id), "control_condition_id"]
    present = set(reader.role_counts.loc[reader.role_counts.role.eq("selection"), "condition_id"])
    pools = [reader.cell_metadata(key, role="selection") for key in sorted(set(links) & present)]
    control = pd.concat(pools, ignore_index=True) if pools else target.iloc[:0].copy()
    for name, frame in (("target", target), ("control", control)):
        if frame.cell_id.duplicated().any():
            raise ValueError(f"DUPLICATE_{name.upper()}_CELL_ID")
    if set(target.cell_id) & set(control.cell_id):
        raise ValueError("CONTROL_TARGET_CELL_OVERLAP")
    return control.sort_values("cell_id"), target.sort_values("cell_id")


def make_plan(data, seeds, max_cells=512):
    """Plan all eligible V conditions and unique cells before reading expression."""
    require_v_only(data)
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError("Distinct nonnegative integer seeds are required")
    if not isinstance(max_cells, int) or not 1 <= max_cells <= 512:
        raise ValueError("max_cells must be in 1..512")
    rows, excluded = [], []
    for key in data.condition_ids("V"):
        condition = data.reader.condition_table.loc[key]
        if not bool(condition["eligible"]):
            excluded.append({"condition_id": key, "reason": "ineligible_chemical_identity"})
            continue
        if float(condition["duration_h"]) != 24.:
            raise ValueError("PILOT_V1_REQUIRES_24_H")
        control, target = condition_pools(data, key)
        for seed in seeds:
            row = {"condition_id": key, "group_id": str(condition["chemical_or_experiment_group_id"]),
                   "seed": seed, "role": "V", "replacement": False,
                   "control_sampling": "uniform_cells_across_same_role_linked_control_pools"}
            for name, frame in (("target", target), ("control", control)):
                stream = int(config_hash([seed, key, name])[:16], 16)
                indices = np.random.default_rng(stream).choice(len(frame), min(max_cells, len(frame)), replace=False)
                row[f"{name}_cell_ids"] = frame.iloc[indices].cell_id.astype(str).tolist()
                row[f"{name}_pool_size"] = len(frame)
                row[f"{name}_unique_n"] = len(indices)
            rows.append(row)
    if not rows:
        raise ValueError("NO_ELIGIBLE_V_CONDITIONS")
    return {"role": "V", "seeds": list(seeds), "max_cells": max_cells,
            "conditions": sorted({row["condition_id"] for row in rows}),
            "excluded_conditions": excluded, "samples": rows}


def load_shifts(path, data):
    """Check a declared T-fitted P1 prediction artifact; never fit using V labels."""
    require_v_only(data)
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata_json"].item()))
        conditions = artifact["condition_ids"].astype(str).tolist()
        genes = artifact["gene_ids"].astype(str)
        delta = np.asarray(artifact["delta"], dtype=np.float64)
    if metadata.get("schema_version") != 1 or metadata.get("model_family") != "full_gene_ridge":
        raise ValueError("P1_RIDGE_ARTIFACT_REQUIRED")
    if (metadata.get("fit_role") != "T" or metadata.get("selection_role") != "V"
            or metadata.get("evaluation_role") != "V"
            or metadata.get("prediction_inputs") != ["V_controls", "condition_metadata"]):
        raise PermissionError("SHIFT_ARTIFACT_ROLE_VIOLATION")
    if metadata.get("data_identity") != data.identity:
        raise ValueError("SHIFT_DATA_IDENTITY_MISMATCH")
    if not np.array_equal(genes, data.reader.gene_ids):
        raise ValueError("SHIFT_GENE_ORDER_MISMATCH")
    if (len(set(conditions)) != len(conditions) or not conditions
            or not set(conditions).issubset(data.condition_ids("V"))):
        raise PermissionError("SHIFT_CONDITIONS_MUST_BE_UNIQUE_V_CONDITIONS")
    fit_ids = metadata.get("fit_condition_ids", [])
    allowed = set(data.reader.role_counts.loc[data.reader.role_counts.role.eq("fit"), "condition_id"])
    treated = set(data.reader.condition_table.loc[
        data.reader.condition_table.condition_type.eq("treated"), "condition_id"])
    if not fit_ids or not set(fit_ids).issubset(allowed & treated):
        raise PermissionError("SHIFT_FIT_CONDITIONS_MUST_BE_T_TREATED")
    for name in ("fit_artifact_sha256", "fit_code_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(metadata.get(name, ""))):
            raise ValueError(f"SHIFT_PROVENANCE_REQUIRED: {name}")
    if delta.shape != (len(conditions), data.n_genes) or not np.isfinite(delta).all():
        raise ValueError("SHIFT_FULL_GENE_FINITE_MATRIX_REQUIRED")
    return dict(zip(conditions, delta)), {"path": str(path), "sha256": file_hash(path),
            "metadata": metadata, "verification": "identity_and_declared_provenance_only_not_training_audit"}


@torch.no_grad()
def distribution_metrics(raw_prediction, control, target, mean, scale, bandwidth, *, device="cpu"):
    """Distribution statistics only; no arbitrary cell-to-cell matching."""
    result = {**dict.fromkeys(METRICS), "delta_reference": "physically_projected_controls",
              "output_validity_definition": "per_cell_physical_projection"}
    reasons = {}
    x, y = (torch.as_tensor(value, dtype=torch.float64, device=device) for value in (control, target))
    if not len(x) or not len(y):
        return {**result, "status": "missing_population", "na_reasons": {name: "missing_population" for name in METRICS}}
    if not torch.isfinite(y).all():
        return {**result, "status": "nonfinite_observation", "na_reasons": {name: "nonfinite_observation" for name in METRICS}}
    mean = torch.as_tensor(mean, dtype=x.dtype, device=device)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=device)
    reference_projection = physical_projection(x, mean, scale)
    projected = (reference_projection if raw_prediction is None else physical_projection(
        torch.as_tensor(raw_prediction, dtype=x.dtype, device=device), mean, scale))
    result["output_valid_rate"] = float(projected.valid.double().mean())
    if not projected.valid.all():
        return {**result, "status": "invalid_prediction", "na_reasons": {
            name: "invalid_prediction_no_row_dropping" for name in METRICS if name != "output_valid_rate"}}
    prediction = projected.x
    pred_mean, obs_mean = prediction.mean(0), y.mean(0)
    result["mean_rmse"] = float((pred_mean - obs_mean).square().mean().sqrt())
    if min(len(prediction), len(y)) >= 2:
        result["mmd_u"] = float(rbf_mmd(prediction, y, bandwidth=bandwidth, unbiased=True))
        result["variance_rmse"] = float((prediction.var(0) - y.var(0)).square().mean().sqrt())
    else:
        reasons.update({name: "fewer_than_two_unique_cells" for name in ("mmd_u", "variance_rmse")})
    if reference_projection.valid.all():
        # Reuse the same physical reference for every method. P0 has exactly zero
        # predicted delta, even when float32 normalized controls are reprojected.
        reference_mean = reference_projection.x.mean(0)
        pred_delta, obs_delta = pred_mean - reference_mean, obs_mean - reference_mean
        left, right = pred_delta - pred_delta.mean(), obs_delta - obs_delta.mean()
        denominator = left.norm() * right.norm()
        if denominator > 0:
            result["delta_correlation"] = float((left @ right) / denominator)
        else:
            reasons["delta_correlation"] = "constant_delta_vector"
        result["direction_agreement"] = float((pred_delta.sign() == obs_delta.sign()).double().mean())
    else:
        reasons.update({name: "invalid_control_reference" for name in ("delta_correlation", "direction_agreement")})
    return {**result, "status": "ok" if not reasons else "partial_metrics", "na_reasons": reasons}


def baseline_metrics(control, target, mean, scale, bandwidth, *, delta=None, device="cpu"):
    raw = (None if delta is None else torch.as_tensor(control, dtype=torch.float64)
           + torch.as_tensor(delta, dtype=torch.float64))
    return distribution_metrics(raw, control, target, mean, scale, bandwidth, device=device)


def iter_plan_batches(data, plan):
    """Shared immutable cell identities for every method; read only V expressions."""
    require_v_only(data)
    if plan.get("role") != "V":
        raise PermissionError("PILOT_PLAN_MUST_BE_V")
    for condition_id in plan["conditions"]:
        control, target = condition_pools(data, condition_id)
        pools = {"control": control.set_index("cell_id", drop=False), "target": target.set_index("cell_id", drop=False)}
        for sample in (row for row in plan["samples"] if row["condition_id"] == condition_id):
            if sample["role"] != "V":
                raise PermissionError("PILOT_SAMPLE_MUST_BE_V")
            for name in pools:
                ids = sample[f"{name}_cell_ids"]
                if len(ids) != len(set(ids)) or len(ids) != sample[f"{name}_unique_n"]:
                    raise ValueError("PILOT_SAMPLE_IDENTITY_MISMATCH")
            arrays = {name: data.reader.normalize(data.reader.rows.read(
                pools[name].loc[sample[f"{name}_cell_ids"]])) for name in pools}
            yield sample, arrays["control"], arrays["target"]


def point_metrics(point, target):
    """P1 is a raw standardized population mean, never a projected single cell."""
    result = {**dict.fromkeys(METRICS), "not_applicable": ["mmd_u", "variance_rmse",
              "delta_correlation", "direction_agreement"], "na_reasons": {},
              "output_validity_definition": "finite_mean_vector_only_not_cell_physical_validity"}
    if point is None or not len(target):
        return {**result, "status": "missing_mean_point_or_observation",
                "na_reasons": {"mean_rmse": "missing_mean_point_or_observation"}}
    point, target = np.asarray(point, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if point.shape != target.shape[1:]:
        raise ValueError("MEAN_POINT_GENE_AXIS_MISMATCH")
    result["output_valid_rate"] = float(np.isfinite(point).all())
    if not np.isfinite(point).all() or not np.isfinite(target).all():
        return {**result, "status": "invalid_mean_point_or_observation",
                "na_reasons": {"mean_rmse": "invalid_mean_point_or_observation"}}
    return {**result, "mean_rmse": float(np.sqrt(np.mean((point - target.mean(0)) ** 2))), "status": "ok"}


def evaluate_plan(data, plan, shifts=None, device="cpu", *, mean_predictions=None, shift_control_scope=None, progress=None):
    rows = []
    batches = iter_plan_batches(data, plan)
    bar = progress(batches, total=len(plan["samples"]), desc="V P0/P1/P2") if progress else batches
    for sample, control, target in bar:
        condition_id = sample["condition_id"]
        if progress:
            bar.set_postfix_str(f"{condition_id} seed={sample['seed']}", refresh=True)
        base = {key: sample[key] for key in ("condition_id", "group_id", "seed", "role",
                                            "target_unique_n", "control_unique_n", "replacement")}
        for method in (["P0_no_change", "P2_mean_shift"] if shifts is not None else ["P0_no_change"]):
            if method == "P2_mean_shift" and condition_id not in shifts:
                result = {**dict.fromkeys(METRICS), "status": "missing_shift",
                          "na_reasons": {name: "missing_T_fitted_prediction" for name in METRICS}}
            else:
                result = baseline_metrics(control, target, data.mu, data.scale,
                    data.bandwidth, delta=shifts[condition_id] if method == "P2_mean_shift" else None, device=device)
            generated_n = (None if result["status"] in {"missing_shift", "missing_population", "nonfinite_observation"}
                           else sample["control_unique_n"])
            rows.append({**base, "method": method, "generated_n": generated_n,
                         "output_validity_definition": "per_cell_physical_projection",
                         "control_summary_scope": ("sampled_V_controls" if method == "P0_no_change"
                                                   else "sampled_V_controls_plus_external_delta"),
                         "delta_control_summary_scope": shift_control_scope if method == "P2_mean_shift" else None,
                         **result})
        if mean_predictions is not None:
            rows.append({**base, "method": "P1_ridge_mean", "generated_n": None,
                "prediction_point_n": 1, "distribution_defined": False,
                "control_summary_scope": "full_unique_V_control_pool_mean",
                **point_metrics(mean_predictions.get(condition_id), target)})
    if progress:
        bar.close()
    return rows


def summarize(rows, bootstrap_repeats=10000, bootstrap_seed=20260924):
    """Seed -> condition -> chemical entity; incomplete groups stay in coverage."""
    if not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be a positive integer")
    expected = defaultdict(set)
    for row in rows:
        expected[row["group_id"]].add((row["condition_id"], row["seed"]))
    methods = sorted({row["method"] for row in rows})
    summaries, complete = {}, {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summaries[method], complete[method] = {}, {}
        for metric in METRICS:
            groups = {}
            for group, required in expected.items():
                available = [row for row in selected if row["group_id"] == group and row[metric] is not None]
                if {(row["condition_id"], row["seed"]) for row in available} != required:
                    continue
                conditions = defaultdict(list)
                for row in available:
                    conditions[row["condition_id"]].append(row[metric])
                groups[group] = float(np.mean([np.mean(values) for values in conditions.values()]))
            complete[method][metric] = groups
            not_applicable = sum(metric in row.get("not_applicable", []) for row in selected)
            computed = sum(row[metric] is not None for row in selected)
            summaries[method][metric] = {
                "estimate": float(np.mean(list(groups.values()))) if groups else None,
                "status": "not_applicable" if not_applicable == len(selected) else "applicable",
                "computed_rows": computed, "total_rows": len(selected), "not_applicable_rows": not_applicable,
                "applicable_rows": len(selected) - not_applicable,
                "missing_or_failed_rows": len(selected) - not_applicable - computed,
                "complete_chemical_groups": len(groups), "total_chemical_groups": len(expected),
                "aggregation": "seeds_then_conditions_then_equal_chemical_groups_complete_only"}
            if metric == "output_valid_rate":
                summaries[method][metric]["definitions"] = sorted({
                    row.get("output_validity_definition", "not_declared") for row in selected})
    comparisons = {}
    pairs = [(method, "P0_no_change", f"{method}_minus_P0") for method in methods if method != "P0_no_change"]
    if {"P3_cfm_mmd", "P4_cfm_only"}.issubset(methods):
        pairs.append(("P3_cfm_mmd", "P4_cfm_only", "P3_cfm_mmd_minus_P4_cfm_only"))
    for method, reference, label in pairs:
        for metric in ("mean_rmse", "mmd_u"):
            common = sorted(set(complete[method][metric]) & set(complete[reference][metric]))
            differences = np.asarray([complete[method][metric][key] - complete[reference][metric][key]
                                      for key in common])
            interval = None
            if len(common) >= 2:
                rng = np.random.default_rng(bootstrap_seed)
                draws = np.asarray([rng.choice(differences, size=len(common), replace=True).mean()
                                    for _ in range(bootstrap_repeats)])
                interval = np.quantile(draws, [.025, .975]).tolist()
            not_applicable = summaries[method][metric]["status"] == "not_applicable"
            comparisons[f"{label}/{metric}"] = {
                "status": "not_applicable" if not_applicable else "applicable",
                "difference": float(differences.mean()) if len(common) else None, "ci95": interval,
                "paired_chemical_groups": len(common), "total_chemical_groups": len(expected),
                "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": bootstrap_seed,
                "na_reason": ("not_applicable" if not_applicable else
                              "fewer_than_two_complete_chemical_groups" if interval is None else None)}
    return {"methods": summaries, "paired_comparisons": comparisons,
            "interpretation": "development_validation_only_not_biological_replicates_or_final_test"}
