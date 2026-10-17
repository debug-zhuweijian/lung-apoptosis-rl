"""Read and write references to completed training artifacts."""

import json

from .checkpoint import atomic_json
from .config import project_path


def read_pipeline(config):
    path = project_path(config["pipeline"]) if config.get("pipeline") else None
    if path is None:
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("PIPELINE_SCHEMA_MISMATCH")
    return {**record["artifacts"], "_data_identity": record["data_identity"]}


def write_pipeline(monitor, data, pipeline):
    atomic_json(monitor.experiment / "pipeline.json", {
        "schema_version": 1, "data_identity": data.identity,
        "synthetic": monitor.config.get("synthetic", False), "artifacts": pipeline,
        "task": monitor.config["task"], "config": monitor.config})
