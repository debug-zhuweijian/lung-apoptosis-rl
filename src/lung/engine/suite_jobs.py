"""Actual registered suite tasks. Synthetic outputs are always explicitly marked."""
from pathlib import Path
from contextlib import contextmanager

import yaml

from .config import read_yaml
from .suite_state import atomic_json
from .suite_training import child, merge_primary, sequence, time_config


@contextmanager
def job_progress(config):
    """One console bar per active phase, released on success or interruption."""
    bar, phase = None, None

    def update(name, current, total):
        nonlocal bar, phase
        if not config.get("progress", True):
            return
        from tqdm.auto import tqdm
        if bar is None or phase != name or bar.total != total or current < bar.n:
            if bar is not None:
                bar.close()
            bar = tqdm(total=total, desc=f"研究进度 · {name}", dynamic_ncols=True, mininterval=1.)
            phase = name
        bar.update(current - bar.n)

    try:
        yield update
    finally:
        if bar is not None:
            bar.close()


def execute_job(name, context):
    from . import suite_reports as reports
    root, folder = Path(context["root"]), Path(context["job_dir"])
    config, deps = context["config"], context["dependencies"]
    if name == "prepare":
        from .data import open_data
        with job_progress(config) as progress:
            data = open_data(config, "response", prepare=True, progress=progress)
        output = folder / "prepared.json"
        atomic_json(output, {"identity": data.identity, "manifest": data.manifest,
                             "synthetic": data.synthetic, "proxy": True})
        files = [str(output)]
        if not data.synthetic:
            files.append(str((data.assets_dir / "manifest.json").resolve()))
        result = {"artifacts": files, "data_identity": data.identity}
    elif name == "ridge":
        if context["synthetic"]:
            result = reports.synthetic_ridge(context)
        else:
            from scripts.fit_response_ridge import main
            selected = read_yaml(root / "configs/pilot_ridge.yaml")
            selected.update({key: config[key] for key in ("device", "progress", "data")})
            path = folder / "ridge-config.yaml"
            path.write_text(yaml.safe_dump(selected, allow_unicode=True), encoding="utf-8")
            output = folder / "data"
            if main(["--config", str(path), "--output", str(output)]) != 0:
                raise RuntimeError("RIDGE_FAILED")
            result = {"artifacts": [str(path), *[str(p) for p in output.rglob("*") if p.is_file()]],
                       "shifts": str(output / "v_delta.npz"), "weights": str(output / "ridge_model.npz")}
    elif name in {"response_cfm_then_full", "response_one_step_then_full", "response_latent_flow"}:
        from .optimization import candidate_overrides
        result = child(context, "response", overrides=candidate_overrides(name.removeprefix("response_"), config))
    elif name == "optimization_compare":
        from .optimization import compare_responses
        result = compare_responses(context)
    elif name in {"response_cfm_mmd", "response_cfm_only", "response_condition_only",
                  "response_population_mean", "response_random_product"}:
        mode = {"response_condition_only": "condition_only",
                "response_population_mean": "population_mean"}.get(name, "observed")
        result = child(context, "response", overrides={"task": "RESPONSE-A549-V1",
                       "response": {"mmd_weight": 0.0 if name == "response_cfm_only" else 1.0,
                                    "training_schedule": {"mode": "full"},
                                    "model": {"reference_mode": mode},
                                    "coupling": "random_product_control" if name == "response_random_product"
                                                else "entropic_ot_prior"}})
    elif name == "response_compare":
        result = reports.response_compare(context)
    elif name == "response_final_test":
        from .suite_evaluation import response_final_test
        result = response_final_test(context)
    elif name in {"state_weighted_ce", "state_ce", "state_ce_aggregate"}:
        objective = name.removeprefix("state_")
        result = child(context, "state", deps["response_cfm_mmd"]["pipeline"],
                       {"state_loss": objective, "state_aggregate_lambda": 1.0 if objective == "ce_aggregate" else 0.0})
    elif name == "state_compare":
        result = reports.state_compare(context)
    elif name == "tahoe_auxiliary":
        result = sequence(context, ["pool", "backward", "correction", "uncertainty"], merge_primary(context))
    elif name == "tahoe_decision":
        result = sequence(context, ["potential", "daac", "dlpa"], deps["tahoe_auxiliary"]["pipeline"])
        report = reports.decision_compare(context, result)
        result = {**result, "report": report["report"], "artifacts": result["artifacts"] + report["artifacts"]}
    elif name == "time_prepare":
        from lung.timecourse.data import prepare_time_data
        with job_progress(config) as progress:
            path = Path(prepare_time_data(root, time_config(context), progress=progress)).resolve()
        result = {"manifest": str(path), "artifacts": [str(path)]}
    elif name == "time_nested":
        from lung.timecourse.nested import run_nested
        with job_progress(config) as progress:
            result = run_nested(root, time_config(context), folder / "nested", resume=context["resume"], progress=progress)
    elif name == "time_decision":
        from lung.timecourse.decisions import run_comparison
        state = deps["state_" + config.get("suite", {}).get("primary_state", "weighted_ce")]
        state_pipeline = {**state["pipeline_artifacts"], "original_config": state["original_config"]}
        with job_progress(config) as progress:
            result = run_comparison(root, time_config(context), deps["time_nested"], state_pipeline,
                                    folder / "decisions", resume=context["resume"], progress=progress)
    elif name == "summary":
        result = reports.summary(context)
    else:
        raise ValueError(f"Unregistered suite job: {name}")
    return {**result, "synthetic": bool(context["synthetic"]), "proxy": True,
            "biological_endpoint_validated": False}
