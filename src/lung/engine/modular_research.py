"""Train one explicitly selected stage using existing upstream models."""

import argparse
import json

from .checkpoint import seed_all
from .config import StageUnavailable, project_path
from .modular_config import OUTPUT_KEYS, RESPONSE_STAGES, STAGE_DEPENDENCIES, load_modular_config, stage_config


def parser():
    command = argparse.ArgumentParser(description="分阶段训练；先准备数据，再明确选择一个阶段")
    command.add_argument("--config", default="configs/modular_research.yaml")
    command.add_argument("--stage", choices=tuple(STAGE_DEPENDENCIES))
    command.add_argument("--run-id")
    command.add_argument("--dry-run", action="store_true", help="只显示所选阶段的输入依赖")
    command.add_argument("--no-progress", action="store_true")
    return command


def input_pipeline(config):
    from .train import read_pipeline
    pipeline = read_pipeline(config)
    for key, value in config["modular"].get("artifacts", {}).items():
        if key not in {*OUTPUT_KEYS.values(), *STAGE_DEPENDENCIES, "correction_gate", "inference_ode_steps"}:
            raise ValueError(f"MODULAR_ARTIFACT_UNKNOWN: {key}")
        pipeline[key] = (value if key == "inference_ode_steps" else
                         [str(project_path(p)) for p in value] if isinstance(value, list) else
                         str(project_path(value)))
    return pipeline


def execute_stage(stage, data, config, monitor, pipeline):
    if stage in RESPONSE_STAGES:
        from .latent_pipeline import run_latent_stage
        result = run_latent_stage({"ae": "latent_ae", "cache": "latent_cache", "response": "latent_cfm"}[stage],
                                  config, monitor, data)
        artifact = result["directory"] if stage == "cache" else result["weights"]
        return {**pipeline, OUTPUT_KEYS.get(stage, stage): [artifact] if stage == "response" else artifact}
    if stage == "endpoint":
        from .endpoints import train_endpoint, finalize_calibration
        pipeline = train_endpoint(data, config, monitor, pipeline)
        return finalize_calibration(data, config, monitor, pipeline)
    if stage in {"pool", "state", "backward", "correction"}:
        from .auxiliary import train_auxiliary
        return train_auxiliary(stage, data, config, monitor, pipeline)
    if stage == "daac":
        from .decision import train_decision
        return train_decision(stage, data, config, monitor, pipeline)
    raise ValueError(f"Unknown training stage: {stage}")


def run_pipeline(config, monitor, pipeline, *, stage):
    from .data import open_data
    from .train import write_pipeline

    missing = [key for key in STAGE_DEPENDENCIES[stage] if not pipeline.get(key)]
    if missing:
        raise StageUnavailable(f"MODULAR_DEPENDENCIES_REQUIRED: {stage}: {missing}")
    data = open_data(config, config["stage"], prepare=False)
    if pipeline.get("_data_identity") not in (None, data.identity):
        raise ValueError("MODULAR_DATA_IDENTITY_MISMATCH")
    pipeline = execute_stage(stage, data, config, monitor, pipeline)
    pipeline = {**pipeline, "_data_identity": data.identity}
    write_pipeline(monitor, data, pipeline)
    return pipeline


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.stage is None:
        command.print_help()
        return 0
    monitor = None
    try:
        config = load_modular_config(args.config)
        pipeline = input_pipeline(config)
        if args.dry_run:
            dependencies = STAGE_DEPENDENCIES[args.stage]
            print(json.dumps({"stage": args.stage, "depends_on": dependencies,
                              "missing": [key for key in dependencies if not pipeline.get(key)]},
                             ensure_ascii=False, indent=2))
            return 0
        config = stage_config(config, args.stage, pipeline)
        if args.no_progress:
            config["progress"] = False
        import torch
        if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
            raise StageUnavailable("CUDA_UNAVAILABLE")
        from .numerics import PrecisionPolicy
        from lung.monitoring import RunMonitor
        seed_all(config["seed"])
        PrecisionPolicy.from_config(config).apply()
        monitor = RunMonitor(config, run_id=args.run_id)
        run_pipeline(config, monitor, pipeline, stage=args.stage)
        monitor.close("complete")
        return 0
    except KeyboardInterrupt:
        if monitor is not None:
            monitor.close("interrupted")
        return 130
    except Exception as error:
        if monitor is not None:
            monitor.close("failed", reason=str(error))
        else:
            print(f"{type(error).__name__}: {error}")
        return 1
