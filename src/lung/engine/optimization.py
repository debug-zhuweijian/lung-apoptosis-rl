"""Optional, predeclared response comparisons; no automatic promotion on test data."""
from pathlib import Path

from .config import file_hash
from .suite_state import atomic_json


RESPONSE_CANDIDATES = ("cfm_then_full", "one_step_then_full", "latent_flow")


def candidate_names(config):
    names = config.get("suite", {}).get("response_candidates", [])
    if not isinstance(names, list) or any(name not in RESPONSE_CANDIDATES for name in names):
        raise ValueError("UNKNOWN_RESPONSE_OPTIMIZATION_CANDIDATE")
    if len(set(names)) != len(names):
        raise ValueError("DUPLICATE_RESPONSE_OPTIMIZATION_CANDIDATE")
    return list(names)


def candidate_overrides(name, config):
    if name not in RESPONSE_CANDIDATES:
        raise ValueError("UNKNOWN_RESPONSE_OPTIMIZATION_CANDIDATE")
    schedule = {"mode": "full"}
    if name != "latent_flow":
        schedule = {"mode": name, "transition": "validation_plateau",
                    "warmup_epochs": 2, "patience": 5}
    model = {"reference_mode": "observed"}
    if name == "latent_flow":
        model["backbone"] = "latent_flow"
    overrides = config.get("suite", {}).get("candidate_settings", {}).get(name, {})
    allowed = {"training_schedule", "model", "reconstruction_weight"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("INVALID_RESPONSE_CANDIDATE_SETTINGS")
    if name == "latent_flow" and overrides.get("model", {}).get("backbone", "latent_flow") != "latent_flow":
        raise ValueError("LATENT_CANDIDATE_REQUIRES_LATENT_BACKBONE")
    return {"task": "RESPONSE-A549-V1", "response": {
        "mmd_weight": 1., "coupling": "entropic_ot_prior",
        "training_schedule": {**schedule, **overrides.get("training_schedule", {})},
        "model": {**model, **overrides.get("model", {})},
        **({"reconstruction_weight": overrides["reconstruction_weight"]}
           if "reconstruction_weight" in overrides else {})}}


def compare_responses(context):
    from .data import open_data
    from .evaluate import evaluate_expression
    from .response import load_response

    folder = Path(context["job_dir"])
    config, deps = context["config"], context["dependencies"]
    data = open_data(config, "response", prepare=False)
    files = [path for result in deps.values() for path in result["pipeline_artifacts"]["response"]]
    before = {path: file_hash(path) for path in files}
    reports = {}
    for name, result in deps.items():
        target = folder / name
        target.mkdir(exist_ok=True)
        branch = result["original_config"]
        models = [load_response(path, data, config["device"])
                  for path in result["pipeline_artifacts"]["response"]]
        reports[name] = {"metrics": evaluate_expression(models, data, branch, target, role="V"),
                         "response_config": branch["response"],
                         "training_completion": result.get("training_completion", {}),
                         "members": len(models)}
        del models
    if before != {path: file_hash(path) for path in files}:
        raise RuntimeError("FROZEN_COMPARISON_CHANGED_WEIGHTS")
    report = folder / "optimization-comparison.json"
    atomic_json(report, {"protocol": "response-optimization-v4", "role": "V",
        "reports": reports, "frozen_hashes": before, "model_updates": 0,
        "primary_response": "predeclared_cfm_mmd", "automatic_promotion": False,
        "synthetic": bool(context["synthetic"]),
        "scope": "development_expression_comparison_not_apoptosis_efficacy"})
    return {"report": str(report), "artifacts": [str(p) for p in folder.rglob("*") if p.is_file()]}
