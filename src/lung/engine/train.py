"""Single entry orchestration: preparation, explicit phases, manual-run training UI."""

import argparse
import json
import sys
import shutil
import time
from pathlib import Path

import torch
import yaml

from .checkpoint import atomic_json, seed_all, load_checkpoint
from .config import ROOT, STAGES, StageUnavailable, load_config, read_yaml, validate_config, verify_interpreter, project_path, file_hash, implementation_identity
from .data import EngineData, open_data
from .synthetic import smoke_config
from lung.monitoring import RunMonitor


def parser():
    command = argparse.ArgumentParser(description="肺癌细胞响应 / DA-AC / DLPA 统一训练入口")
    command.add_argument("--config", default="configs/train.yaml")
    command.add_argument("--stage", choices=STAGES)
    command.add_argument("--task", choices=("RESPONSE-A549-V1", "PROXY-A549-V1", "ANNEXIN-A549-V1"))
    command.add_argument("--algorithm", choices=("daac", "dlpa", "both"))
    command.add_argument("--epochs", type=int)
    command.add_argument("--device")
    command.add_argument("--data")
    command.add_argument("--pipeline", help="A previously trained experiments/<run>/pipeline.json")
    command.add_argument("--resume", metavar="RUN_ID", help="Continue the saved target epoch, not another 1000 epochs")
    command.add_argument("--run-id")
    command.add_argument("--dry-run", action="store_true", help="Read-only configuration/data readiness report")
    command.add_argument("--prepare-only", action="store_true", help="Create training-only preprocessing assets; do not train")
    command.add_argument("--smoke", action="store_true", help="Labelled synthetic tiny engineering test, not a research run")
    command.add_argument("--no-dashboard", action="store_true")
    command.add_argument("--no-browser", action="store_true")
    command.add_argument("--no-progress", action="store_true")
    return command


def resolve(arguments):
    config = load_config(arguments.config)
    if arguments.resume:
        if Path(arguments.resume).name != arguments.resume:
            raise ValueError("--resume expects a run_id, not a filesystem path")
        directory = ROOT / "experiments" / arguments.resume
        active = directory / "active_config.yaml"
        config = read_yaml(active if active.exists() else directory / "resolved_config.yaml")
        for key in ("stage", "task", "data", "algorithm", "pipeline"):
            if getattr(arguments, key) is not None and getattr(arguments, key) != config.get(key):
                raise ValueError(f"Cannot change {key} when resuming; create a new run")
    if arguments.resume and arguments.smoke and not config.get("synthetic", False):
        raise ValueError("Cannot change a real research run to synthetic during resume")
    if arguments.smoke and not arguments.resume:
        config = smoke_config(config)
    for key in ("stage", "task", "epochs", "device", "data", "algorithm", "pipeline"):
        value = getattr(arguments, key)
        if value is not None:
            config[key] = value
    for argument, key in (("no_dashboard", "dashboard"), ("no_browser", "open_browser"), ("no_progress", "progress")):
        if getattr(arguments, argument):
            config[key] = False
    validate_config(config)
    return config


def read_pipeline(config, run_id=None):
    path = ROOT / "experiments" / run_id / "pipeline.json" if run_id else (
        project_path(config["pipeline"]) if config.get("pipeline") else None)
    if run_id and (path is None or not path.exists()) and config.get("pipeline"):
        # A child may be interrupted before publishing its first stage. Its
        # recorded input pipeline remains the required frozen dependency.
        path = project_path(config["pipeline"])
    if path is None or not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("PIPELINE_SCHEMA_MISMATCH")
    if run_id and record.get("implementation_identity") not in (None, implementation_identity()):
        raise StageUnavailable("IMPLEMENTATION_CHANGED: changed source requires a new run, not mixed-version replay")
    for artifact, expected in record.get("artifact_hashes", {}).items():
        if file_hash(artifact) != expected:
            raise ValueError(f"FROZEN_PIPELINE_ARTIFACT_CHANGED: {artifact}")
    return {**record["artifacts"], "_data_identity": record["data_identity"]}


def write_pipeline(monitor, data, pipeline):
    paths = [path for value in pipeline.values() for path in (value if isinstance(value, list) else [value])
             if isinstance(path, str) and Path(path).is_file()]
    atomic_json(monitor.experiment / "pipeline.json", {
        "schema_version": 1, "data_identity": data.identity,
        "synthetic": monitor.config.get("synthetic", False), "artifacts": pipeline,
        "task": monitor.config["task"], "config": monitor.config,
        "implementation_identity": implementation_identity(),
        "artifact_hashes": {path: file_hash(path) for path in paths},
        "trained": {key: bool(pipeline.get(key)) for key in STAGES if key != "all"}})


def publish_stage(stage, pipeline):
    """The published inference checkpoint never aliases a mutable best.pt.

    An interrupted extended training run may update best/last without invalidating
    the previous completed pipeline or mixing its downstream dependencies.
    """
    value = pipeline[stage]
    paths = value if isinstance(value, list) else [value]
    frozen = []
    for item in paths:
        source = Path(item)
        digest = file_hash(source)
        target = source.with_name(f"selected-{digest[:20]}.pt")
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            shutil.copy2(source, temporary)
            if file_hash(temporary) != digest:
                raise ValueError("PUBLISHED_CHECKPOINT_COPY_MISMATCH")
            temporary.replace(target)
        if file_hash(target) != digest:
            raise ValueError("PUBLISHED_CHECKPOINT_HASH_MISMATCH")
        frozen.append(str(target))
    return {**pipeline, stage: frozen if isinstance(value, list) else frozen[0]}


class PreparationProgress:
    def __init__(self, monitor):
        self.monitor = monitor

    def __call__(self, phase, current, total):
        self.monitor.preparation_progress(phase, current, total)

    def close(self, status='complete'):
        self.monitor.finish_preparation(status)


def run_stage(stage, data, config, monitor, pipeline, resume):
    if stage == "response":
        from .response import train_response
        return {**pipeline, "response": train_response(data, config, monitor, resume)}
    if stage in {"pool", "state", "backward", "correction", "uncertainty", "endpoint"}:
        if not pipeline.get("response"):
            raise StageUnavailable("TRAINED_RESPONSE_REQUIRED: train response first, then select its pipeline.json")
        if stage == "pool":
            from .diagnostics import resolve_ode_steps
            pipeline = resolve_ode_steps(data, config, monitor, pipeline)
        from .auxiliary import train_auxiliary
        return train_auxiliary(stage, data, config, monitor, pipeline, resume=resume)
    if stage == "potential":
        from .potential import train_potential
        return train_potential(data, config, monitor, pipeline, resume=resume)
    from .decision import train_decision
    return train_decision(stage, data, config, monitor, pipeline, resume=resume)


def verify_completed_auxiliary(stage, paths, data, config, pipeline):
    from .auxiliary import verify_auxiliary_dependencies, pipeline_identity
    for path in paths:
        verify_auxiliary_dependencies(path, pipeline)
        if stage == "state":
            # Completed-stage skipping must enforce the same loss/fold identity
            # as an interrupted fit, rather than silently reusing another pilot.
            load_checkpoint(path, pipeline_identity("state", data, config, pipeline))


def main(argv=None):
    arguments = parser().parse_args(argv)
    monitor, preparation = None, None
    try:
        from .suite_entry import dispatch_suite
        suite_result = dispatch_suite(arguments)
        if suite_result is not None:
            return suite_result
        config = resolve(arguments)
        verify_interpreter()
        if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
            raise StageUnavailable("CUDA_UNAVAILABLE: no silent CPU fallback")
        if arguments.dry_run:
            from .preflight import stage_readiness
            result = {"status": "preflight_only_no_training", "interpreter": sys.executable,
                      "stage": config["stage"], "task": config["task"], "epochs": config["epochs"],
                      "device": config["device"], "genes": config["response"]["model"]["n_genes"]}
            data = None if config.get("synthetic") else EngineData(config, require_preprocessing=False)
            result.update(stage_readiness(config, data, read_pipeline(config, arguments.resume)))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if config["stage"] in {"potential", "daac", "dlpa", "all"} and config["task"] == "RESPONSE-A549-V1":
            raise StageUnavailable("TASK_REQUIRED: select PROXY-A549-V1 explicitly for model-only decisions")
        pipeline = read_pipeline(config, arguments.resume)
        if (config["task"] == "ANNEXIN-A549-V1" and
                config["stage"] in {"endpoint", "potential", "daac", "dlpa", "all"} and
                not config.get("endpoint_manifest") and not pipeline.get("endpoint")):
            raise StageUnavailable("ENDPOINT_UNAVAILABLE: 当前没有经核验的表达—Annexin 配对监督，不能启动正式终点流水线")
        if config["stage"] != "response" and config["stage"] != "all" and not pipeline.get("response"):
            raise StageUnavailable("PIPELINE_REQUIRED: configs/train.yaml 的 pipeline 设为已完成响应训练的 pipeline.json")
        seed_all(config["seed"])
        monitor = RunMonitor(config, run_id=arguments.resume or arguments.run_id, resume=bool(arguments.resume))
        preparation = PreparationProgress(monitor)
        data = open_data(config, config["stage"], prepare=config.get("auto_prepare", True), progress=preparation)
        preparation.close()
        if pipeline.get("_data_identity") not in (None, data.identity):
            raise StageUnavailable("PIPELINE_DATA_IDENTITY_MISMATCH: use a new experiment for changed data")
        atomic_json(monitor.experiment / "data_manifest.json", {"identity": data.identity, "manifest": data.manifest})
        if pipeline and not (monitor.experiment / "pipeline.json").exists():
            write_pipeline(monitor, data, pipeline)
        if arguments.prepare_only:
            monitor.say("训练数据资产准备完成；未启动模型训练。")
            monitor.close("prepared_only")
            return 0
        stages = [config["stage"]]
        if config["stage"] == "all":
            stages = ["response", "pool", "state", "backward", "correction", "uncertainty"]
            if config["task"] == "ANNEXIN-A549-V1":
                stages.append("endpoint")
            stages.append("potential")
            stages.extend(["daac", "dlpa"] if config["algorithm"] == "both" else [config["algorithm"]])
        for stage in stages:
            if arguments.resume and pipeline.get(stage):
                prior = pipeline[stage] if isinstance(pipeline[stage], list) else [pipeline[stage]]
                completed = [Path(path).parent / "last.pt" for path in prior]
                completed_epoch = pipeline.get("_completed_epochs", {}).get(stage)
                reached = completed_epoch is not None and completed_epoch >= config["epochs"]
                from .training_control import checkpoint_completion
                try:
                    verified_completion = [checkpoint_completion(load_checkpoint(path), config["epochs"]) for path in completed if path.exists()]
                    reached = len(verified_completion) == len(completed)
                except ValueError:
                    reached = False
                if reached:
                    if stage in {"pool", "state", "backward", "correction", "uncertainty", "endpoint"}:
                        verify_completed_auxiliary(stage, prior, data, config, pipeline)
                    monitor.say(f"阶段 {stage} 已满足已登记的停止条件，保留已冻结结果。")
                    continue
            # Each stage gets a new reader with only its scientific roles.
            selected = open_data(config, stage, prepare=False)
            monitor.say(f"开始阶段 {stage}；目标 {config['epochs']} epoch。")
            pipeline = run_stage(stage, selected, config, monitor, pipeline, bool(arguments.resume))
            from .training_control import checkpoint_completion
            selected_paths = pipeline[stage] if isinstance(pipeline[stage], list) else [pipeline[stage]]
            completions = [checkpoint_completion(load_checkpoint(Path(path).parent / "last.pt" if (Path(path).parent / "last.pt").exists() else path), config["epochs"]) for path in selected_paths]
            pipeline = publish_stage(stage, pipeline)
            pipeline = {**pipeline, "_completed_epochs": {**pipeline.get("_completed_epochs", {}), stage: min(item["actual_epochs"] for item in completions)},
                        "_completion": {**pipeline.get("_completion", {}), stage: completions}}
            write_pipeline(monitor, data, pipeline)
        monitor.say(f"所选阶段完成。权重与依赖：{monitor.experiment / 'pipeline.json'}")
        monitor.close("complete")
        return 0
    except KeyboardInterrupt:
        if preparation is not None:
            preparation.close('interrupted')
        if monitor is not None:
            monitor.say("已中断。响应训练从最近已保存的参数更新边界恢复；其他阶段按已保存的 epoch / 回合 / 规划根边界恢复。")
            monitor.close("interrupted")
        return 130
    except Exception as error:
        if preparation is not None:
            preparation.close('failed')
        if monitor is not None:
            monitor.say(f"停止：{type(error).__name__}: {error}")
            monitor.close("failed", error_type=type(error).__name__, reason=str(error))
        else:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
