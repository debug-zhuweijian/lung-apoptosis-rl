"""T-fitted full-gene Ridge, V-only selection, and explicit P1/P2 artifacts."""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import torch

from .checkpoint import atomic_json
from .config import config_hash, file_hash
from .pilot_ridge_data import features, mean_plan, require_ridge_data, stream_means


LAMBDAS = (1e-4, 1e-2, 1., 1e2, 1e4)
SELECTION_CRITERION = "equal_chemical_entity_mean_of_condition_full_V_population_mean_RMSE_raw_standardized_mean_no_projection"
POINT_SEMANTICS = "raw_standardized_full_control_mean_plus_delta_no_single_cell_projection"


def checked_device(device):
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("RIDGE_DEVICE_MUST_BE_CPU_OR_CUDA")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE: no automatic device or precision fallback")
    return selected


def finite_matrix(value, device):
    result = torch.as_tensor(value, dtype=torch.float64, device=device)
    if result.ndim != 2 or min(result.shape) < 1 or not torch.isfinite(result).all():
        raise ValueError("FINITE_NONEMPTY_MATRIX_REQUIRED")
    return result


@dataclass(frozen=True)
class RidgeModel:
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    train_scaled: torch.Tensor
    dual: torch.Tensor
    regularization: float

    def predict(self, inputs):
        x = finite_matrix(inputs, self.x_mean.device)
        if x.shape[1] != len(self.x_mean):
            raise ValueError("RIDGE_FEATURE_AXIS_MISMATCH")
        return ((x - self.x_mean) / self.x_scale @ self.train_scaled.T @ self.dual
                + self.y_mean).cpu().numpy()

    def arrays(self):
        return {name: getattr(self, name).cpu().numpy() for name in (
            "x_mean", "x_scale", "y_mean", "train_scaled", "dual")}


def ridge_design(inputs, targets, device="cpu"):
    """Only T is supplied here. Loss is SUM squared error + lambda ||B||_F^2."""
    device = checked_device(device)
    x, y = finite_matrix(inputs, device), finite_matrix(targets, device)
    if len(x) != len(y):
        raise ValueError("RIDGE_TRAINING_ROWS_MISMATCH")
    constant = x.amin(0).eq(x.amax(0))
    x_mean = torch.where(constant, x[0], x.mean(0))
    centered = x - x_mean
    x_scale = torch.where(constant, torch.ones_like(x_mean), centered.square().mean(0).sqrt())
    scaled, y_mean = centered / x_scale, y.mean(0)
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean,
            "train_scaled": scaled, "target_centered": y - y_mean,
            "kernel": scaled @ scaled.T}


def solve_ridge(design, regularization):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("RIDGE_LAMBDA_MUST_BE_POSITIVE")
    kernel = design["kernel"]
    penalized = kernel + float(regularization) * torch.eye(len(kernel), dtype=kernel.dtype, device=kernel.device)
    # Centering removes the unpenalized intercept; no G-by-G matrix is allocated.
    cholesky = torch.linalg.cholesky(penalized)
    dual = torch.cholesky_solve(design["target_centered"], cholesky)
    if not torch.isfinite(dual).all():
        raise FloatingPointError("NONFINITE_RIDGE_SOLUTION")
    return RidgeModel(*(design[name] for name in ("x_mean", "x_scale", "y_mean", "train_scaled")),
                      dual, float(regularization))


def fit_ridge(inputs, targets, regularization, device="cpu"):
    """Pure numerical helper; the real workflow enforces T/V permissions separately."""
    return solve_ridge(ridge_design(inputs, targets, device), regularization)


def raw_mean_points(control_mean, delta, device="cpu"):
    """A population mean is not a cell: do not impose single-cell library mass."""
    device = checked_device(device)
    control = finite_matrix(control_mean, device)
    changes = torch.as_tensor(delta, dtype=torch.float64, device=device)
    if control.shape != changes.shape:
        raise ValueError("RIDGE_MEAN_DELTA_SHAPE_MISMATCH")
    prediction = control + changes
    valid = torch.isfinite(prediction).all(-1)
    points = torch.where(valid[:, None], prediction, torch.full_like(prediction, float("nan")))
    return points.cpu().numpy(), valid.cpu().numpy()


def chemical_mean_rmse(prediction, target, group_ids):
    prediction, target = np.asarray(prediction), np.asarray(target)
    groups = np.asarray(group_ids, dtype=str)
    if prediction.ndim != 2 or prediction.shape != target.shape or len(groups) != len(target):
        raise ValueError("RIDGE_SELECTION_SHAPE_MISMATCH")
    if not len(groups) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("RIDGE_SELECTION_REQUIRES_COMPLETE_FINITE_CONDITIONS")
    per_condition = np.sqrt(np.mean((prediction - target) ** 2, axis=1))
    per_entity = {str(group): float(per_condition[groups == group].mean()) for group in sorted(set(groups))}
    return float(np.mean(list(per_entity.values()))), per_condition, per_entity


def select_ridge(inputs, targets, validation_inputs, control_mean, target_mean, group_ids,
                 *, device="cpu", lambdas=LAMBDAS, progress=None):
    if tuple(lambdas) != LAMBDAS:
        raise ValueError("PILOT_V1_FIXED_LAMBDAS_REQUIRED")
    design = ridge_design(inputs, targets, device)
    candidates, best = [], None
    for number, regularization in enumerate(lambdas, 1):
        if progress:
            progress("V lambda selection", number - 1, len(lambdas))
        model = solve_ridge(design, regularization)
        delta = model.predict(validation_inputs)
        prediction, valid = raw_mean_points(control_mean, delta, device)
        candidate = {"lambda": regularization, "valid_conditions": int(valid.sum()),
                     "total_conditions": len(valid), "score": None, "status": "invalid_prediction"}
        if valid.all():
            score, condition_scores, entity_scores = chemical_mean_rmse(prediction, target_mean, group_ids)
            candidate.update(score=score, status="ok", condition_rmse=condition_scores.tolist(),
                             chemical_entity_rmse=entity_scores)
            if best is None or score < best["score"]:
                best = {"score": score, "model": model, "delta": delta,
                        "mean_prediction": prediction, "point_valid": valid}
        candidates.append(candidate)
    if progress:
        progress("V lambda selection", len(lambdas), len(lambdas))
    report = {"selection_role": "V", "criterion": SELECTION_CRITERION,
              "control_summary_scope": "full_unique_V_control_pool_mean",
              "target_summary_scope": "full_unique_V_target_pool_mean",
              "prediction_semantics": POINT_SEMANTICS, "tie_break": "first_in_ascending_lambda_grid",
              "invalid_candidate_policy": "exclude_whole_lambda_no_condition_dropping",
              "candidates": candidates,
              "selected_lambda": None if best is None else best["model"].regularization}
    return best, report


def save_npz(path, **arrays):
    """Lossless arrays, never pickle; each artifact is a new file."""
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
        raise FileExistsError(f"RIDGE_ARTIFACT_EXISTS: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("PICKLE_OBJECT_ARRAYS_FORBIDDEN")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_ridge(data, config, output, identity, *, progress=None):
    """Called only by the manual CLI or synthetic tests; output is reserved by caller."""
    require_ridge_data(data)
    device = str(checked_device(config["device"]))
    output = Path(output)
    if any(path.name not in {"status.json", "input_and_method_identity.json"} for path in output.iterdir()):
        raise FileExistsError("RIDGE_OUTPUT_NOT_FRESH: use a new experiment identity")
    plans = {role: mean_plan(data, role) for role in ("T", "V")}
    atomic_json(output / "population_plan.json", plans)
    seen, physical_seen = set(), set()
    populations = {}
    for role in ("T", "V"):
        population = stream_means(data, plans[role], batch_size=config["stream_batch_size"],
                                  device=device, progress=progress, seen_cells=seen, seen_physical=physical_seen)
        save_npz(output / f"{role.lower()}_condition_means.npz", **population, gene_ids=data.reader.gene_ids)
        populations[role] = population
    train, validation = populations["T"], populations["V"]
    x, feature_names = features(data, train["condition_ids"], train["control_mean"])
    vx, v_names = features(data, validation["condition_ids"], validation["control_mean"])
    if feature_names != v_names:
        raise ValueError("RIDGE_FEATURE_AXIS_CHANGED")
    best, selection = select_ridge(x, train["target_mean"] - train["control_mean"], vx,
        validation["control_mean"], validation["target_mean"], validation["group_ids"],
        device=device, lambdas=config["lambdas"], progress=progress)
    selection["condition_ids"] = validation["condition_ids"].tolist()
    atomic_json(output / "lambda_selection.json", selection)
    if best is None:
        raise FloatingPointError("NO_VALID_RIDGE_LAMBDA; all candidates retained in lambda_selection.json")
    provenance = {"schema_version": 1, "model_family": "full_gene_ridge", "fit_role": "T",
        "selection_role": "V", "evaluation_role": "V", "prediction_inputs": ["V_controls", "condition_metadata"],
        "data_identity": data.identity, "fit_condition_ids": train["condition_ids"].tolist(),
        "fit_code_sha256": config_hash(identity["code_sha256"]), "fit_source_code": identity["code_sha256"],
        "fit_data_sha256": file_hash(output / "t_condition_means.npz"),
        "selection_data_sha256": file_hash(output / "v_condition_means.npz"),
        "selection_artifact_sha256": file_hash(output / "lambda_selection.json"),
        "population_plan_sha256": file_hash(output / "population_plan.json"),
        "normalization_and_split_identity": data.identity,
        "gene_order_sha256": config_hash(data.reader.gene_ids.tolist()),
        "feature_order_sha256": config_hash(feature_names), "selected_lambda": best["model"].regularization,
        "selection_criterion": SELECTION_CRITERION, "intercept": "unpenalized_T_target_mean",
        "input_scaling": "T_condition_population_std_ddof0_constant_scale1",
        "loss": "sum_condition_gene_squared_error_plus_lambda_frobenius_squared",
        "numeric_dtype": "float64", "device": device,
        "fit_control_summary_scope": "full_unique_T_control_pool_mean",
        "control_summary_scope": "full_unique_V_control_pool_mean",
        "delta_semantics": "raw_standardized_mean_delta_for_P2_cellwise_projection",
        "prediction_semantics": POINT_SEMANTICS,
        "point_validity": "finite_mean_only_not_single_cell_mass_constraint",
        "distribution_metrics": "not_applicable_P1_is_mean_only",
        "P1_P2_mean_equivalence": "not_assumed_P1_raw_group_mean_P2_mean_of_cellwise_projected_samples"}
    model_path = output / "ridge_model.npz"
    save_npz(model_path, **best["model"].arrays(), gene_ids=data.reader.gene_ids,
             feature_names=np.asarray(feature_names), regularization=np.asarray(best["model"].regularization),
             metadata_json=np.asarray(json.dumps(provenance, ensure_ascii=False)))
    provenance = {**provenance, "fit_artifact_sha256": file_hash(model_path), "fit_artifact_file": model_path.name}
    save_npz(output / "v_delta.npz", condition_ids=validation["condition_ids"], gene_ids=data.reader.gene_ids,
        delta=best["delta"], mean_prediction=best["mean_prediction"], point_valid=best["point_valid"],
        metadata_json=np.asarray(json.dumps(provenance, ensure_ascii=False)))
    score, errors, _ = chemical_mean_rmse(best["mean_prediction"], validation["target_mean"], validation["group_ids"])
    rows = [{"condition_id": str(key), "group_id": str(group), "method": "P1_full_gene_ridge_mean",
             "role": "V", "mean_rmse": float(error), "mmd_u": None, "variance_rmse": None,
             "status": "ok", "na_reasons": {"mmd_u": "not_applicable_mean_only",
                 "variance_rmse": "not_applicable_mean_only"}, "control_summary_scope": provenance["control_summary_scope"],
             "target_summary_scope": "full_unique_V_target_pool_mean"}
            for key, group, error in zip(validation["condition_ids"], validation["group_ids"], errors)]
    atomic_json(output / "p1_full_population_metrics.json", {"conditions": rows, "criterion": SELECTION_CRITERION,
        "prediction_semantics": POINT_SEMANTICS,
        "chemical_entity_equal_mean_rmse": score, "selected_lambda": best["model"].regularization,
        "result_type": "development_validation_used_for_selection_not_independent_final_test"})
    return provenance


def load_mean_predictions(path, data):
    """V report adapter: invalid points stay all-NaN, never a fake population."""
    from .pilot import load_shifts
    shifts, provenance = load_shifts(path, data)
    with np.load(path, allow_pickle=False) as artifact:
        points = np.asarray(artifact["mean_prediction"], dtype=np.float64)
        valid = artifact["point_valid"]
        conditions = artifact["condition_ids"].astype(str).tolist()
    metadata = provenance["metadata"]
    if (metadata.get("prediction_semantics") != POINT_SEMANTICS
            or metadata.get("control_summary_scope") != "full_unique_V_control_pool_mean"):
        raise ValueError("P1_RAW_GROUP_MEAN_SEMANTICS_REQUIRED: projected-cell-mean artifacts are not compatible")
    if (points.shape != (len(shifts), data.n_genes) or valid.shape != (len(shifts),)
            or valid.dtype != np.dtype(bool) or not np.isfinite(points[valid]).all()
            or not np.isnan(points[~valid]).all()):
        raise ValueError("P1_POINT_VALIDITY_OR_GENE_AXIS_MISMATCH")
    return dict(zip(conditions, points)), {**provenance,
        "control_summary_scope": metadata["control_summary_scope"], "prediction_semantics": POINT_SEMANTICS,
        "distribution_metrics": "not_applicable_mean_only"}
