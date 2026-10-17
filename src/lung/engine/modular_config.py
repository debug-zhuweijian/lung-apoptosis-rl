"""Configuration for explicitly selected training stages."""

from .config import load_config, merge

STAGE_DEPENDENCIES = {
    "ae": (), "cache": ("representation",),
    "response": ("representation", "representation_cache"),
    "pool": ("representation",), "state": ("representation",),
    "backward": ("representation", "response", "pool"),
    "correction": ("representation", "response", "pool", "backward"),
    "endpoint": ("representation", "response", "pool", "state"),
    "daac": ("representation", "response", "pool", "state"),
}
OUTPUT_KEYS = {"ae": "representation", "cache": "representation_cache"}
RESPONSE_STAGES = {"ae", "cache", "response"}


def load_modular_config(path="configs/modular_research.yaml"):
    config = load_config(path)
    if config.get("workflow") != "modular_research":
        raise ValueError("MODULAR_WORKFLOW_REQUIRED")
    if config["response"]["members"] != 1:
        raise ValueError("MODULAR_SINGLE_MEMBER_REQUIRED")
    for name, value in config.get("modular", {}).get("stages", {}).items():
        if name not in STAGE_DEPENDENCIES or not isinstance(value, dict):
            raise ValueError(f"MODULAR_STAGE_UNKNOWN: {name}")
        if "epochs" in value and (type(value["epochs"]) is not int or value["epochs"] < 1):
            raise ValueError(f"MODULAR_EPOCHS_INVALID: {name}")
    return config


def stage_config(config, stage, pipeline):
    value = merge(config, config.get("modular", {}).get("stages", {}).get(stage, {}))
    value["stage"] = "response" if stage in RESPONSE_STAGES else stage
    value["workflow"] = "modular_stage"
    value["modular_stage"] = stage
    value["task"] = "RESPONSE-A549-V1" if stage in RESPONSE_STAGES else config["task"]
    if pipeline.get("representation"):
        value["research"]["latent"]["ae_checkpoint"] = pipeline["representation"]
    if pipeline.get("representation_cache"):
        value["research"]["latent"]["cache_dir"] = pipeline["representation_cache"]
    if stage == "response":
        value["research"]["mode"] = "latent_cfm"
    return value
