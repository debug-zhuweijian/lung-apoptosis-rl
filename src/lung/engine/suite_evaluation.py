"""Frozen held-out comparisons added by the state-conditioning review, protocol v2."""
import json
from pathlib import Path

import numpy as np
import torch

from lung.models import rbf_mmd
from .checkpoint import atomic_json, load_checkpoint
from .config import file_hash
from .data import open_data
from .response import load_response, projected, tensor


@torch.no_grad()
def state_final_test(result):
    from .auxiliary import load_auxiliary, encode_vectors, state_metrics, verify_auxiliary_dependencies
    from .evaluate import state_evaluation_config
    pipeline = result["pipeline_artifacts"]
    saved = load_checkpoint(pipeline["state"])
    config = state_evaluation_config(result["original_config"], saved["metadata"])
    paths = [pipeline["state"], pipeline["response"][0]]
    before = {p: file_hash(p) for p in paths}
    data = open_data(config, "state", prepare=False, test=True)
    verify_auxiliary_dependencies(pipeline["state"], pipeline)
    reference = load_response(pipeline["response"][0], data, config["device"])
    head = load_auxiliary(pipeline["state"], data, config["device"])
    batch = data.state_arrays(config["state_fold"], "test" if data.synthetic else "final_test")
    logits = head(encode_vectors(reference, batch["x"], config["device"]).to(config["device"])) / head.temperature
    labels = tensor(batch["labels"], config["device"], torch.long)
    metrics = state_metrics(logits, labels)
    probability = torch.softmax(logits, -1)
    metrics["brier_multiclass"] = float((probability - torch.nn.functional.one_hot(
        labels, probability.shape[-1])).square().sum(-1).mean())
    prediction = logits.argmax(-1)
    per_class = [{"class_index": c, "support": int((labels == c).sum()),
                  "recall": float((prediction[labels == c] == c).float().mean()) if (labels == c).any() else None}
                 for c in range(logits.shape[-1])]
    if before != {p: file_hash(p) for p in paths}:
        raise RuntimeError("FROZEN_STATE_TEST_MUTATED_WEIGHTS")
    return {"evaluation": "opposite_plate_frozen_diagnostic", "metrics": metrics,
            "per_class": per_class, "state_fold": config["state_fold"],
            "frozen_hashes": before, "model_updates": 0, "used_for_selection": False,
            "labels": "original_TIDYI_sorting_labels_not_apoptosis_outcomes",
            "synthetic": data.synthetic, "new_blind_test": False}


def _ridge_model(path, data, device):
    from .pilot_ridge import RidgeModel
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"])) if "metadata_json" in saved else {}
        if not data.synthetic and (metadata.get("data_identity") != data.identity
                or metadata.get("fit_role") != "T" or metadata.get("selection_role") != "V"):
            raise ValueError("RIDGE_FROZEN_T_V_PROVENANCE_REQUIRED")
        names = ("x_mean", "x_scale", "y_mean", "train_scaled", "dual")
        parameters = [torch.as_tensor(saved[n].copy(), dtype=torch.float64, device=device) for n in names]
        model = RidgeModel(*parameters, regularization=float(saved["regularization"]))
        feature_names = saved["feature_names"].astype(str).tolist() if "feature_names" in saved else None
    return model, feature_names


@torch.no_grad()
def baseline_final_test(data, config, weights):
    """Same frozen E control/target sampling plan as neural models; no E fitting."""
    from .pilot_ridge_data import features
    model, expected_features = _ridge_model(weights, data, config["device"])
    rng = np.random.default_rng(config["seed"] + 901)
    n = config["response"]["mmd_samples"] if data.synthetic else config["population"]
    rows = []
    for cid in data.condition_ids("E"):
        batch = data.sample_response(cid, n, rng, "E")
        x, y = tensor(batch["control"], config["device"]), tensor(batch["target"], config["device"])
        if data.synthetic:
            inputs = np.concatenate((batch["fingerprint"], [batch["dose"], batch["duration"]]))[None]
        else:
            inputs, names = features(data, [cid], batch["control"].mean(0, dtype=np.float64)[None])
            if names != expected_features:
                raise ValueError("FROZEN_RIDGE_FEATURE_ORDER_MISMATCH")
        delta = tensor(model.predict(inputs), config["device"])
        populations = {"no_change": projected(x, data), "ridge_shift": projected(x + delta, data)}
        for method, population in populations.items():
            rows.append({"method": method, "condition_id": cid, "group_id": batch["group_id"],
                         "population_mmd_v": float(rbf_mmd(population, y, data.bandwidth, unbiased=False)),
                         "mean_mse": float((population.mean(0) - y.mean(0)).square().mean())})
        rows.append({"method": "ridge_mean", "condition_id": cid, "group_id": batch["group_id"],
                     "population_mmd_v": None,
                     "mean_mse": float((x.mean(0) + delta[0] - y.mean(0)).square().mean())})
    return {"rows": rows, "role": "E", "model_updates": 0, "used_for_selection": False,
            "control_summary_scope": "same_fixed_E_sample_as_neural_comparison",
            "point_distribution_metric": "not_applicable_for_ridge_mean",
            "mmd_estimator": "empirical_population_V_not_unbiased_population_estimate"}


def response_final_test(context):
    from .evaluate import evaluate_expression
    folder, deps = Path(context["job_dir"]), context["dependencies"]
    config = context["config"]
    data = open_data(config, "response", prepare=False, test=True)
    paths = [deps["ridge"]["weights"]]
    branches = {k: v for k, v in deps.items() if k.startswith("response_")}
    for result in branches.values():
        paths.extend(result["pipeline_artifacts"]["response"])
    before = {p: file_hash(p) for p in paths}
    reports = {}
    for name, result in branches.items():
        target = folder / name
        target.mkdir(exist_ok=True)
        branch_config = result["original_config"]
        models = [load_response(p, data, config["device"]) for p in result["pipeline_artifacts"]["response"]]
        reports[name] = evaluate_expression(models, data, branch_config, target)
        reports[name]["reference_mode"] = models[0].config.reference_mode
        reports[name]["coupling"] = branch_config["response"]["coupling"]
        del models
    reports["baselines"] = baseline_final_test(data, config, deps["ridge"]["weights"])
    if before != {p: file_hash(p) for p in paths}:
        raise RuntimeError("FROZEN_RESPONSE_TEST_MUTATED_WEIGHTS")
    report = folder / "response-final-test.json"
    atomic_json(report, {"protocol": "state-conditioning-evaluation-v2", "role": "E",
        "reports": reports, "synthetic": data.synthetic, "frozen_hashes": before,
        "primary_response": "predeclared_cfm_mmd_not_selected_on_E", "model_updates": 0,
        "apoptosis_rate": None, "same_cell_fate_validated": False})
    return {"artifacts": [str(p) for p in folder.rglob("*") if p.is_file()], "report": str(report)}
