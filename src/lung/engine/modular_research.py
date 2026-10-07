"""Independent, resumable modular pretraining and a single staged Actor."""

import argparse
import gc
import json
import math
import time
from pathlib import Path

from .checkpoint import atomic_json, load_checkpoint, seed_all
from .config import ROOT, StageUnavailable, file_hash, implementation_identity, project_path, verify_interpreter
from .modular_config import (DEFAULT_STAGES, OUTPUT_KEYS, RESPONSE_STAGES, STAGE_DEPENDENCIES,
                             dependency_plan, load_modular_config, science_identity, stage_config)
from .response_research import ResearchBudget


def parser():
    command = argparse.ArgumentParser(description="模块预训练 → 固定环境 → drug/dose/joint")
    command.add_argument("--config", default="configs/modular_research.yaml")
    command.add_argument("--stage", choices=tuple(STAGE_DEPENDENCIES))
    command.add_argument("--resume", metavar="RUN_ID")
    command.add_argument("--run-id")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--budget-seconds", type=float, help="Explicit new cumulative cap for --stage")
    command.add_argument("--no-dashboard", action="store_true")
    command.add_argument("--no-browser", action="store_true")
    command.add_argument("--no-progress", action="store_true")
    return command


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_pipeline(config):
    from .train import read_pipeline
    pipeline = read_pipeline(config)
    for key, value in config["modular"].get("artifacts", {}).items():
        if key not in {*OUTPUT_KEYS.values(), *STAGE_DEPENDENCIES, "correction_gate",
                       "uncertainty_calibration", "numerical_audit", "precalibration_identity",
                       "inference_ode_steps"}:
            raise ValueError(f"MODULAR_ARTIFACT_UNKNOWN: {key}")
        if key in {"precalibration_identity", "inference_ode_steps"}:
            pipeline[key] = value
        else:
            pipeline[key] = ([str(project_path(p)) for p in value] if isinstance(value, list)
                             else str(project_path(value)))
    return pipeline


def verify_artifact(stage, pipeline, data, config):
    """A path in a JSON document is not evidence of a trained compatible model."""
    key = OUTPUT_KEYS.get(stage, stage)
    value = pipeline[key]
    device = config["device"]
    if stage in {"ae", "response", "pool", "state", "backward", "correction", "uncertainty", "potential"}:
        from .modular_completion import completion_record
        from .representation import require_finite_state
        for path in value if isinstance(value, list) else [value]:
            completion_record(path, config)
            require_finite_state(load_checkpoint(path)["model"])
        if stage == "state":
            saved = load_checkpoint(value)
            temperature = float(saved.get("extra", {}).get("temperature", 0.))
            if (saved["metadata"].get("calibration_role") != "calibration"
                    or not math.isfinite(temperature) or temperature <= 0
                    or float(saved["model"].get("temperature", 0.)) != temperature):
                raise StageUnavailable("STATE_CALIBRATION_INCOMPLETE")
    if stage == "ae":
        from .representation import load_representation
        model = load_representation(value, data, device)
        saved = load_checkpoint(value)
        if saved.get("extra", {}).get("budget_exhausted"):
            raise StageUnavailable("INCOMPLETE_REPRESENTATION: resume its original stage")
        del model
    elif stage == "cache":
        from .representation import load_representation
        from .representation_cache import encoder_identity, load_representation_cache
        model = load_representation(pipeline["representation"], data, device)
        cache = load_representation_cache(value, "cpu", expected_data_identity=data.identity,
                                         expected_encoder=encoder_identity(model.shared, model.config))
        del cache, model
    elif stage == "response":
        from .response_loader import load_response
        from .representation import load_representation, verify_response_representation
        representation = load_representation(pipeline["representation"], data, device)
        for path in value:
            model = load_response(path, data, device)
            saved = load_checkpoint(path)
            if saved.get("extra", {}).get("budget_exhausted"):
                raise StageUnavailable("INCOMPLETE_RESPONSE: resume its original stage")
            verify_response_representation(model, representation)
            del model
        del representation
    elif stage in {"pool", "state", "backward", "correction", "uncertainty"}:
        from .auxiliary import load_auxiliary, verify_auxiliary_dependencies
        verify_auxiliary_dependencies(value, pipeline)
        model = load_auxiliary(value, data, device)
        del model
    elif stage == "ode_audit":
        from .config import config_hash
        report = _read_json(value)
        expected = config_hash({"response": [file_hash(p) for p in pipeline["response"]], "data": data.identity})
        if report.get("identity") != expected or not report.get("passed") or (
                report.get("synthetic_test_only") and not config.get("synthetic")):
            raise StageUnavailable("MODULAR_ODE_AUDIT_MISMATCH")
    elif stage in {"assemble", "potential", "daac"}:
        from .world import BiologicalWorld
        world = BiologicalWorld(data, config, pipeline)
        world.adapter.validate_ready(require_potential=stage != "assemble")
        if stage == "potential":
            from .representation import require_trained
            saved = load_checkpoint(value)
            require_trained(saved, "TRAINED_POTENTIAL_REQUIRED")
            if saved["metadata"].get("stage") != "potential" or saved["metadata"].get("data_identity") != data.identity:
                raise ValueError("MODULAR_POTENTIAL_IDENTITY_MISMATCH")
        if stage == "daac":
            from .decision import daac_settings
            from lung.rl import DAACTrainer
            for path in value:
                saved = load_checkpoint(path)
                metadata = saved.get("metadata", {})
                if (metadata.get("stage") != "daac" or metadata.get("data_identity") != data.identity
                        or metadata.get("biological_identity") != world.biological_identity
                        or metadata.get("versions") != world.adapter.versions):
                    raise ValueError("MODULAR_DAAC_IDENTITY_MISMATCH")
                learner = DAACTrainer(world.registry, world.adapter.versions, daac_settings(config),
                                      device, actor_schedule=config["daac"]["actor_schedule"],
                                      legal_mask=world.protocol.legal_mask(world.registry, ()))
                learner.load_state_dict(saved["learner"])
                runtime = saved.get("extra", {}).get("decision_runtime", {})
                if runtime.get("status") != "complete" or not learner.actor_schedule.complete:
                    raise StageUnavailable("INCOMPLETE_DAAC: resume its original stage")
                del learner
        del world


class ModuleBudget:
    def __init__(self, local, shared=None):
        self.local, self.shared = local, shared

    @property
    def exhausted(self):
        return self.local.exhausted or bool(self.shared and self.shared.exhausted)

    def check(self, step, epoch):
        a = self.local.check(step, epoch)
        b = self.shared.check(step, epoch) if self.shared else False
        # A hard IDE stop can lose only the current atomic unit, not the whole segment.
        self.local.pause()
        if self.shared:
            self.shared.pause()
        return a or b

    def summary(self):
        return {"module": self.local.summary(),
                "response_chain": self.shared.summary() if self.shared else None,
                "exhausted": self.exhausted}


def execute_stage(stage, data, config, monitor, pipeline, budget, resume):
    if stage in RESPONSE_STAGES:
        from .latent_pipeline import run_latent_stage
        result = run_latent_stage({"ae": "latent_ae", "cache": "latent_cache", "response": "latent_cfm"}[stage],
                                  config, monitor, data, budget)
        if budget.exhausted:
            from .modular_budget import StagePaused
            raise StagePaused("budget_exhausted", result.get("weights"), stage=stage)
        key = OUTPUT_KEYS.get(stage, stage)
        artifact = result["directory"] if stage == "cache" else result["weights"]
        return {**pipeline, key: [artifact] if stage == "response" else artifact}
    if stage in {"pool", "state", "backward", "correction", "uncertainty"}:
        from .auxiliary import train_auxiliary
        return train_auxiliary(stage, data, config, monitor, pipeline, resume=resume)
    if stage == "ode_audit":
        from .diagnostics import resolve_ode_steps
        return resolve_ode_steps(data, config, monitor, pipeline)
    if stage == "assemble":
        from .diagnostics import prepare_frozen_environment
        world, pipeline = prepare_frozen_environment(data, config, monitor, pipeline)
        world.adapter.validate_ready(require_potential=False)
        del world
        return pipeline
    if stage == "potential":
        from .potential import train_potential
        return train_potential(data, config, monitor, pipeline, resume=resume)
    from .decision import train_decision
    result = train_decision("daac", data, config, monitor, pipeline, resume=resume)
    for path in result["daac"]:
        runtime = load_checkpoint(path).get("extra", {}).get("decision_runtime", {})
        if runtime.get("status") != "complete":
            from .modular_budget import StagePaused
            raise StagePaused(runtime.get("reason", "incomplete"), path, stage=stage)
    return result


def run_pipeline(config, monitor, pipeline, *, stage=None, state=None):
    from lung.monitoring import RunMonitor
    from .data import open_data
    from .modular_budget import StagePaused
    from .train import publish_stage, write_pipeline
    from .modular_completion import seal_completed
    state_path = monitor.experiment / "modular_state.json"
    state = state or {"schema_version": 1, "identity": science_identity(config),
                      "implementation_identity": implementation_identity(), "stages": {}}
    if state["identity"] != science_identity(config):
        raise ValueError("MODULAR_RESUME_CONFIG_MISMATCH")
    chosen = (stage,) if stage else DEFAULT_STAGES
    for name in chosen:
        cfg = stage_config(config, name, pipeline)
        key = OUTPUT_KEYS.get(name, name)
        missing = [dep for dep in STAGE_DEPENDENCIES[name] if not pipeline.get(dep)]
        if missing:
            raise StageUnavailable(f"MODULAR_DEPENDENCIES_REQUIRED: {name}: {missing}")
        prior = state["stages"].get(name, {})
        start = time.monotonic()
        if pipeline.get(key):
            data = open_data(cfg, cfg["stage"], prepare=False)
            verify_artifact(name, pipeline, data, cfg)
            if name in {"ae", "response", "pool", "state", "backward", "correction", "uncertainty", "potential"}:
                current = pipeline[key]
                pipeline = {**pipeline, key: ([seal_completed(p, cfg) for p in current]
                                             if isinstance(current, list) else seal_completed(current, cfg))}
            pipeline = {**pipeline, "_data_identity": data.identity}
            write_pipeline(monitor, data, pipeline)
            state["stages"][name] = {**prior, "status": "complete", "reused": True,
                "reuse_seconds": prior.get("reuse_seconds", 0.) + time.monotonic() - start,
                "artifact": pipeline[key]}
            atomic_json(state_path, state)
            monitor.say(f"{name}：兼容产物已校验，复用。")
            del data
            continue
        run_id = prior.get("run_id", f"{monitor.run_id}-{name}")
        resumable = (ROOT / "experiments" / run_id / "active_config.yaml").exists()
        training_resume = resumable
        if name == "daac":
            cap = config["modular"]["stages"]["daac"].get("budget_seconds")
            cfg["daac"]["max_seconds"] = cap
            if resumable:
                last = ROOT / "checkpoints" / run_id / "daac" / f"seed_{config.get('policy_seeds', [0])[0]}" / "last.pt"
                training_resume = last.is_file()
                if training_resume:
                    cfg["daac"].pop("init_from", None)
                if last.is_file() and cap is not None:
                    runtime = load_checkpoint(last).get("extra", {}).get("decision_runtime", {})
                    extension = cap - float(runtime.get("budget_seconds", cap))
                    if extension > 0:
                        cfg["daac"]["resume_budget_seconds"] = extension
                    else:
                        cfg["daac"].pop("resume_budget_seconds", None)
        child = RunMonitor(cfg, run_id=run_id, resume=resumable)
        local = ResearchBudget(config["modular"]["stages"].get(name, {}).get("budget_seconds"),
                               child.experiment / "module_budget.json")
        shared = (ResearchBudget(config["modular"].get("response_budget_seconds"),
                                 monitor.experiment / "response_budget.json") if name in RESPONSE_STAGES else None)
        budget = ModuleBudget(local, shared)
        setattr(child, "stop_check", budget.check)
        status, paused = "failed", None
        data = None
        state["stages"][name] = {**prior, "run_id": run_id, "status": "running", "task": cfg["task"]}
        atomic_json(state_path, state)
        try:
            if budget.check(0, 0):
                raise StagePaused("budget_exhausted", stage=name)
            data = open_data(cfg, cfg["stage"], prepare=False)
            if pipeline.get("_data_identity") not in (None, data.identity):
                raise ValueError("MODULAR_DATA_IDENTITY_MISMATCH")
            atomic_json(child.experiment / "data_manifest.json", {"identity": data.identity, "manifest": data.manifest})
            if budget.check(0, 0):
                raise StagePaused("budget_exhausted", stage=name)
            pipeline = execute_stage(name, data, cfg, child, pipeline, budget, training_resume)
            verify_artifact(name, pipeline, data, cfg)
            if name in {"ae", "response", "pool", "state", "backward", "correction", "uncertainty", "potential"}:
                current = pipeline[key]
                pipeline = {**pipeline, key: ([seal_completed(p, cfg) for p in current]
                                             if isinstance(current, list) else seal_completed(current, cfg))}
            if name in {"ae", "response", "pool", "state", "backward", "correction", "uncertainty", "potential", "daac"}:
                pipeline = publish_stage(key, pipeline)
            pipeline = {**pipeline, "_data_identity": data.identity}
            write_pipeline(child, data, pipeline)
            write_pipeline(monitor, data, pipeline)
            status = "complete"
        except StagePaused as error:
            status, paused = "paused", error
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            local.pause()
            local.check(0, 0)
            if shared:
                shared.pause()
                shared.check(0, 0)
            record = {**state["stages"][name], "status": status,
                      "wall_seconds": prior.get("wall_seconds", 0.) + time.monotonic() - start,
                      "budget": local.summary(), "reused": False}
            if status == "complete":
                record["artifact"] = pipeline[key]
            if paused is not None:
                record.update({"reason": str(paused), "checkpoint": getattr(paused, "checkpoint_path", None)})
            state["stages"][name] = record
            state["status"] = status
            atomic_json(state_path, state)
            child.close(status)
            del data
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if paused is not None:
            monitor.say(f"{name} 已暂停；保留本模块进度和全部上游产物。调整该阶段累计预算后恢复。")
            return {"status": "paused", "stage": name, "pipeline": pipeline, "state": state}
    state["status"] = "complete" if not stage else "stage_complete"
    atomic_json(state_path, state)
    return {"status": state["status"], "pipeline": pipeline, "state": state}


def main(argv=None):
    args = parser().parse_args(argv)
    monitor = None
    try:
        config = load_modular_config(args.config)
        for flag, key in ((args.no_dashboard, "dashboard"), (args.no_browser, "open_browser"), (args.no_progress, "progress")):
            if flag:
                config[key] = False
        selected = args.stage or (config["modular"].get("mode") if config["modular"].get("mode") != "modular_pipeline" else None)
        if args.budget_seconds is not None:
            import math
            if not selected or not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
                raise ValueError("--budget-seconds requires --stage and a positive finite cumulative cap")
            config["modular"]["stages"].setdefault(selected, {})["budget_seconds"] = args.budget_seconds
        identity = science_identity(config)
        latest = ROOT / "experiments" / "modular_latest.json"
        run_id, state = args.resume or args.run_id, None
        if run_id is None and config["modular"].get("auto_resume", True) and latest.is_file():
            record = _read_json(latest)
            if record.get("identity") == identity and record.get("implementation_identity") == implementation_identity():
                run_id = record["run_id"]
        if run_id and (Path(run_id).name != run_id or run_id in {".", ".."}):
            raise ValueError("run_id must be a single safe directory name")
        pipeline = input_pipeline(config)
        path = ROOT / "experiments" / run_id / "modular_state.json" if run_id else None
        resuming = bool(path and path.exists() and not args.run_id)
        if args.resume and not resuming:
            raise StageUnavailable("MODULAR_RESUME_MISSING")
        if resuming:
            from .train import read_pipeline
            state = _read_json(path)
            if state.get("identity") != identity:
                raise ValueError("MODULAR_RESUME_CONFIG_MISMATCH")
            if state.get("implementation_identity") != implementation_identity():
                raise StageUnavailable("IMPLEMENTATION_CHANGED: create a new run and explicitly reuse compatible artifacts")
            pipeline = {**pipeline, **read_pipeline(config, run_id)}
        if args.dry_run:
            print(json.dumps({"workflow": "modular_pipeline", "run_id": run_id, "resume": resuming,
                              "run_ablations": False, "selected_stage": selected,
                              "stages": dependency_plan(config, pipeline, state.get("stages") if state else None)},
                             ensure_ascii=False, indent=2))
            return 0
        verify_interpreter()
        import torch
        if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
            raise StageUnavailable("CUDA_UNAVAILABLE")
        from .numerics import PrecisionPolicy
        from lung.monitoring import RunMonitor
        seed_all(config["seed"])
        PrecisionPolicy.from_config(config).apply()
        monitor = RunMonitor(config, run_id=run_id, resume=resuming)
        atomic_json(latest, {"identity": identity, "run_id": monitor.run_id,
                            "implementation_identity": implementation_identity()})
        result = run_pipeline(config, monitor, pipeline, stage=selected, state=state)
        atomic_json(monitor.experiment / "modular_report.json", result)
        monitor.close(result["status"])
        return 0
    except KeyboardInterrupt:
        if monitor:
            monitor.close("interrupted")
        return 130
    except Exception as error:
        if monitor:
            monitor.say(f"停止：{type(error).__name__}: {error}")
            monitor.close("failed", reason=str(error))
        else:
            print(f"{type(error).__name__}: {error}")
        return 1
