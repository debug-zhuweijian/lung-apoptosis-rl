"""Dispatch the root no-argument entry to the registered research suite."""

import json
from pathlib import Path

import torch

from .config import (ROOT, StageUnavailable, file_hash, load_config, project_path,
                     read_yaml, validate_config, verify_interpreter)


def suite_data_identity(config):
    """Identify source versions without inspecting held-out expression outcomes."""
    if config.get("synthetic"):
        return {"scope": "synthetic_engineering_only", "seed": config["seed"]}
    dataset = read_yaml(project_path(config["data"]))
    paths = [project_path(config["data"]), project_path(dataset["split_manifest"])]
    source = ROOT / "08-datasets/sciPlex3/GSM4150378/extracted/v1"
    paths.extend(source / name for name in
                 ("extraction_status.json", "a549_cells.parquet", "common_genes.parquet"))
    paths.extend(ROOT / "configs" / name for name in
                 ("timecourse.yaml", "time_auxiliary.yaml", "time_decision.yaml",
                  "pilot_ridge.yaml", "pilot_v.yaml"))
    # These reports read their own configurations at execution time. Bind both
    # their settings and referenced protocols before selecting a resumable suite.
    for name in ("pilot_ridge.yaml", "pilot_v.yaml"):
        settings = read_yaml(ROOT / "configs" / name)
        if settings.get("protocol"):
            paths.append(project_path(settings["protocol"]))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise StageUnavailable("SUITE_INPUT_MISSING: " + ", ".join(missing))
    return {"scope": "frozen_Tahoe_and_registered_sciPlex_time_sources",
            "files": {str(path.resolve()): file_hash(path) for path in paths}}


def dispatch_suite(arguments):
    # Explicit stage commands retain the existing single-stage interface. Child
    # resumes must work even though the user's root default is now research_all.
    if arguments.stage not in (None, "all") or arguments.prepare_only:
        return None
    if arguments.resume:
        if Path(arguments.resume).name != arguments.resume or arguments.resume in {".", ".."}:
            raise ValueError("--resume expects a safe run or suite ID")
        old = ROOT / "experiments" / arguments.resume
        if (old / "active_config.yaml").exists() or (old / "resolved_config.yaml").exists():
            return None
    config = load_config(arguments.config)
    if config.get("workflow", "single") != "research_all":
        return None
    if arguments.pipeline:
        raise ValueError("A research suite resolves its own dependencies; use --stage for an external pipeline")
    if arguments.resume:
        path = ROOT / "experiments/research-suites" / arguments.resume / "manifest.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            config = saved["config"]
    if arguments.smoke and not arguments.resume:
        from .synthetic import smoke_config
        config = smoke_config(config)
    if arguments.smoke and arguments.resume and not config.get("synthetic"):
        raise ValueError("Cannot resume a real suite as synthetic")
    for key in ("epochs", "device", "data", "task", "algorithm"):
        value = getattr(arguments, key)
        if value is not None:
            config = {**config, key: value}
    for option, key in (("no_dashboard", "dashboard"), ("no_browser", "open_browser"),
                         ("no_progress", "progress")):
        if getattr(arguments, option):
            config = {**config, key: False}
    if config.get("task") != "PROXY-A549-V1" or config.get("algorithm") != "both":
        raise ValueError("research_all requires task=PROXY-A549-V1 and algorithm=both; "
                         "use an explicit --stage for a different single-stage task")
    validate_config(config)
    verify_interpreter()
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise StageUnavailable("CUDA_UNAVAILABLE: no silent CPU fallback")
    config = {**config, "suite_data_identity": suite_data_identity(config)}
    from .research_suite import run_suite
    return run_suite(config, arguments)
