"""Stable child runs using the existing single-stage training implementation."""
import copy
import json
from pathlib import Path

import yaml

from .config import file_hash, implementation_identity, merge, read_yaml
from .suite_state import atomic_json


def pipeline_result(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    paths = [str(path)]
    for value in record["artifacts"].values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and Path(item).is_file():
                paths.append(str(Path(item).resolve()))
    return {"pipeline": str(path), "pipeline_artifacts": record["artifacts"],
            "training_completion": record["artifacts"].get("_completion", {}),
            "data_identity": record["data_identity"], "original_config": record["config"],
            "synthetic": record.get("synthetic", False), "proxy": True,
            "artifacts": sorted(set(paths))}


def child(context, stage, pipeline=None, overrides=None):
    from .train import main
    root, folder = Path(context["root"]), Path(context["job_dir"])
    run_id = context["run_id"] + "-" + stage
    config = merge(context["config"], overrides or {})
    config.update(workflow="single", stage=stage, task=(overrides or {}).get("task", "PROXY-A549-V1"),
                  pipeline=str(pipeline) if pipeline else None, auto_prepare=False)
    config.pop("suite_data_identity", None)
    config["dashboard"] = False  # Suite owns the dashboard lifecycle.
    config["open_browser"] = False
    config_path = folder / (stage + "-config.yaml")
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    experiment = root / "experiments" / run_id
    resume = bool(context["resume"] and experiment.exists())
    arguments = ["--config", str(config_path), "--resume" if resume else "--run-id", run_id]
    code = main(arguments)
    if code == 130:
        raise KeyboardInterrupt
    if code != 0:
        raise RuntimeError(f"Child stage {stage} failed with exit {code}; run={run_id}")
    result = pipeline_result(experiment / "pipeline.json")
    result["artifacts"].append(str(config_path.resolve()))
    metrics = experiment / "metrics.csv"
    if metrics.is_file():
        result["metrics"] = str(metrics)
        result["artifacts"].append(str(metrics))
    return result


def merge_primary(context):
    dependencies = context["dependencies"]
    response = dependencies["response_cfm_mmd"]
    state = dependencies["state_" + context["config"].get("suite", {}).get("primary_state", "weighted_ce")]
    p = json.loads(Path(response["pipeline"]).read_text(encoding="utf-8"))
    c = json.loads(Path(state["pipeline"]).read_text(encoding="utf-8"))
    if p["data_identity"] != c["data_identity"]:
        raise ValueError("PRIMARY_PIPELINE_DATA_IDENTITY_MISMATCH")
    if p["artifacts"]["response"] != c["artifacts"]["response"]:
        raise ValueError("PRIMARY_STATE_DID_NOT_USE_PRIMARY_RESPONSE")
    artifacts = {**p["artifacts"], "state": c["artifacts"]["state"]}
    artifacts["_completed_epochs"] = {**p["artifacts"].get("_completed_epochs", {}),
                                      "state": c["artifacts"].get("_completed_epochs", {}).get("state", 0)}
    artifacts["_completion"] = {**p["artifacts"].get("_completion", {}),
        **{key: value for key, value in c["artifacts"].get("_completion", {}).items() if key == "state"}}
    artifact_paths = [item for value in artifacts.values()
                      for item in (value if isinstance(value, list) else [value])
                      if isinstance(item, str) and Path(item).is_file()]
    target = Path(context["job_dir"]) / "primary-pipeline.json"
    record = {**p, "artifacts": artifacts, "config": copy.deepcopy(c["config"]),
              "implementation_identity": implementation_identity(),
              "artifact_hashes": {item: file_hash(item) for item in artifact_paths}}
    atomic_json(target, record)
    return target


def sequence(context, stages, pipeline):
    results = []
    for stage in stages:
        result = child(context, stage, pipeline)
        pipeline = result["pipeline"]
        results.append(result)
    return {**results[-1], "artifacts": sorted(set(path for item in results for path in item["artifacts"]))}


def time_config(context):
    root = Path(context["root"])
    config = copy.deepcopy(context["config"])
    config["timecourse"] = merge(read_yaml(root / "configs/timecourse.yaml"), config.get("timecourse", {}))
    for section in ("time_auxiliary", "time_decision"):
        defaults = read_yaml(root / "configs" / (section + ".yaml"))
        config[section] = merge(defaults.get(section, defaults), config.get(section, {}))
    for section in ("timecourse", "time_auxiliary", "time_decision"):
        config[section]["epochs"] = context["config"].get(section, {}).get("epochs", config["epochs"])
        for key in ("device", "progress", "seed"):
            config[section][key] = config[key]
    config["log_root"] = str(root / "logs" / Path(context["suite_dir"]).name / Path(context["job_dir"]).parent.name)
    config["log_anchor"] = str(context["job_dir"])
    if config.get("synthetic"):
        config["timecourse"].update(synthetic_assets=str(Path(context["suite_dir"]) / "synthetic-time-assets"),
                                  synthetic_development_drugs=4,
                                  response={"members": config["response"]["members"]})
        config["time_auxiliary"].update(outer_folds=2, inner_folds=2, hidden=8, population=3, batch_size=4)
        config["time_decision"].update(policy_seeds=config["policy_seeds"], population=3,
                                       projection_width=4, scoring_repeats=2, audit_repeats=2,
                                       evaluation_candidates=4, daac=config["daac"], dlpa=config["dlpa"],
                                       population_audit={"repeats": 2, "tolerance": 0.5})
    return config
