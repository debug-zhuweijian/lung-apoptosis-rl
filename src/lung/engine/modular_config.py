"""Configuration and dependency graph for the independent modular entry."""

from pathlib import Path

from .config import ROOT, config_hash, load_config, merge, project_path

STAGE_DEPENDENCIES = {
    "ae": (), "cache": ("representation",), "response": ("representation", "representation_cache"),
    "pool": ("representation",), "state": ("representation",),
    "ode_audit": ("response",),
    "backward": ("representation", "response", "pool", "ode_audit"),
    "correction": ("response", "pool", "backward", "ode_audit"),
    "uncertainty": ("response", "pool", "backward", "correction"),
    "assemble": ("representation", "response", "pool", "state", "ode_audit"),
    "potential": ("freeze_certificate",), "daac": ("freeze_certificate", "potential"),
}
DEFAULT_STAGES = ("ae", "cache", "response", "pool", "state", "ode_audit", "assemble", "potential", "daac")
OUTPUT_KEYS = {"ae": "representation", "cache": "representation_cache", "assemble": "freeze_certificate"}
RESPONSE_STAGES = {"ae", "cache", "response"}


def load_modular_config(path="configs/modular_research.yaml"):
    config = load_config(path)
    if config.get("workflow") != "modular_research" or "suite" in config:
        raise ValueError("MODULAR_WORKFLOW_REQUIRED: independent config without suite")
    if config.get("task") != "PROXY-A549-V1" or config.get("algorithm") != "daac":
        raise ValueError("MODULAR_V1_REQUIRES_PROXY_DAAC")
    settings = config.get("modular", {})
    if settings.get("run_ablations", False) is not False:
        raise ValueError("MODULAR_ABLATIONS_NOT_REGISTERED")
    if settings.get("mode", "modular_pipeline") not in {"modular_pipeline", *STAGE_DEPENDENCIES}:
        raise ValueError("MODULAR_MODE_UNKNOWN")
    if config["response"]["members"] != 1:
        raise ValueError("MODULAR_V1_SINGLE_MEMBER_REQUIRED")
    for name, value in settings.get("stages", {}).items():
        if name not in STAGE_DEPENDENCIES or not isinstance(value, dict):
            raise ValueError(f"MODULAR_STAGE_UNKNOWN: {name}")
        if "epochs" in value and (type(value["epochs"]) is not int or value["epochs"] < 1):
            raise ValueError(f"MODULAR_EPOCHS_INVALID: {name}")
    return config


def science_identity(config):
    """Ignore UI and retired time limits when identifying compatible modules."""
    settings = config.get("modular", {})
    stages = {name: {k: v for k, v in value.items() if k != "budget_seconds"}
              for name, value in settings.get("stages", {}).items()}
    excluded = {"progress", "dashboard", "open_browser", "modular"}
    record = {k: v for k, v in config.items() if k not in excluded}
    record["daac"] = {k: v for k, v in config["daac"].items()
                      if k not in {"max_seconds", "resume_budget_seconds"}}
    record["modular"] = {"stages": stages, "artifacts": settings.get("artifacts", {}),
                         "run_ablations": False}
    return config_hash(record)


def stage_config(config, stage, pipeline):
    value = merge(config, config.get("modular", {}).get("stages", {}).get(stage, {}))
    value.pop("budget_seconds", None)
    value["stage"] = "response" if stage in RESPONSE_STAGES else (
        "all" if stage in {"ode_audit", "assemble"} else stage)
    value["workflow"] = "modular_stage"
    value["modular_stage"] = stage
    value["dashboard"] = value["open_browser"] = False
    value["task"] = "RESPONSE-A549-V1" if stage in RESPONSE_STAGES else config["task"]
    if pipeline.get("representation"):
        value["research"]["latent"]["ae_checkpoint"] = pipeline["representation"]
    if pipeline.get("representation_cache"):
        value["research"]["latent"]["cache_dir"] = pipeline["representation_cache"]
    if stage == "response":
        value["research"]["mode"] = "latent_cfm"
    return value


def dependency_plan(config, pipeline, records=None):
    records = records or {}
    available = set(pipeline)
    planned = set()
    rows = []
    for stage in DEFAULT_STAGES:
        key = OUTPUT_KEYS.get(stage, stage)
        present = bool(pipeline.get(key))
        prior = records.get(stage, {})
        rows.append({"stage": stage, "depends_on": list(STAGE_DEPENDENCIES[stage]),
                     "missing_now": [k for k in STAGE_DEPENDENCIES[stage] if k not in available],
                     "provided_by_earlier_stages": [k for k in STAGE_DEPENDENCIES[stage] if k in planned and k not in available],
                     "action": "verify_reuse" if present else ("resume" if prior else "train"),
                     "run_id": prior.get("run_id")})
        planned.add(key)
    return rows
