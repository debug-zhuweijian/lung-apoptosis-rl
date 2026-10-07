"""Manual V baseline report. This entry never trains or opens an E-role reader."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parser():
    command = argparse.ArgumentParser(description="V-only P0/P1/P2 and frozen five-member P3/P4 report; no fitting or E evaluation")
    command.add_argument("--config", default="configs/pilot_v.yaml")
    command.add_argument("--output", required=True, help="New directory under experiments/; existing paths are refused")
    command.add_argument("--shifts", help="Optional T-fitted full-gene Ridge delta NPZ with declared provenance")
    command.add_argument("--p3-checkpoints", nargs=5, metavar="PT", help="Five same-run CFM+MMD member checkpoints")
    command.add_argument("--p4-checkpoints", nargs=5, metavar="PT", help="Five same-run CFM-only member checkpoints")
    command.add_argument("--no-progress", action="store_true", help="Disable condition/member/ODE progress display")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    from lung.engine.checkpoint import atomic_json
    from lung.engine.config import project_path, read_yaml, config_hash, file_hash, implementation_identity
    from lung.engine.data import EngineData
    from lung.engine.pilot import make_plan, load_shifts, evaluate_plan, summarize
    from lung.engine.pilot_response import inspect_response_set, evaluate_response_sets, shared_inference_steps, noise_seed, NOISE_POLICY
    from functools import partial
    from tqdm.auto import tqdm
    import numpy as np
    import torch
    progress = partial(tqdm, disable=args.no_progress, dynamic_ncols=True, mininterval=1.)

    config_path = project_path(args.config)
    config = read_yaml(config_path)
    if config.get("role") != "V":
        raise PermissionError("PILOT_ONLY_SUPPORTS_V; use the frozen test workflow for E")
    if config.get("gene_count") != 32461 or config.get("synthetic", False):
        raise ValueError("REAL_FULL_GENE_V_PILOT_REQUIRED")
    device = str(config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE: no silent device substitution")
    output = project_path(args.output)
    experiments = ROOT / "experiments"
    if output == experiments or not output.is_relative_to(experiments):
        raise ValueError("Pilot output must be a new child directory of experiments/")
    if output.exists():
        raise FileExistsError("PILOT_OUTPUT_EXISTS: use a new experiment identity")
    data = EngineData(config, roles=("V",), allow_test=False, require_preprocessing=True)
    if data.n_genes != config["gene_count"]:
        raise ValueError("PILOT_GENE_COUNT_MISMATCH")
    shifts, provenance = (None, None) if args.shifts is None else load_shifts(project_path(args.shifts), data)
    mean_predictions, mean_provenance = None, None
    if args.shifts is not None:
        with np.load(project_path(args.shifts), allow_pickle=False) as artifact:
            mean_fields = {"mean_prediction", "point_valid"}.intersection(artifact.files)
        if mean_fields and len(mean_fields) != 2:
            raise ValueError("MEAN_PREDICTION_AND_VALIDITY_REQUIRED_TOGETHER")
        if mean_fields:
            from lung.engine.pilot_ridge import load_mean_predictions
            mean_predictions, mean_provenance = load_mean_predictions(project_path(args.shifts), data)
    response_sets = [inspect_response_set([project_path(path) for path in paths], data, method)
                     for method, paths in (("P3_cfm_mmd", args.p3_checkpoints), ("P4_cfm_only", args.p4_checkpoints)) if paths]
    inference_steps = shared_inference_steps(response_sets)
    plan = make_plan(data, config["sampling_seeds"], config["max_cells"])
    if response_sets:
        plan = {**plan, "response_noise_policy": NOISE_POLICY, "inference_steps": inference_steps,
                "samples": [{**sample, "response_noise_seed": noise_seed(sample)} for sample in plan["samples"]]}
    methods = ["P0_no_change"] + (["P2_mean_shift"] if shifts is not None else [])
    methods += (["P1_ridge_mean"] if mean_predictions is not None else []) + [item["method"] for item in response_sets]
    code = [Path(__file__), ROOT / "src/lung/engine/pilot.py", ROOT / "src/lung/models/objectives.py",
            ROOT / "src/lung/engine/data.py", ROOT / "src/lung/data/reader.py",
            ROOT / "src/lung/engine/pilot_response.py", ROOT / "src/lung/engine/response.py",
            ROOT / "src/lung/models/flow.py"]
    if mean_predictions is not None:
        code.append(ROOT / "src/lung/engine/pilot_ridge.py")
    identity = {"protocol": "A549-real-pilot-V1-20260924", "role": "V", "data_identity": data.identity,
                "config": config, "config_sha256": file_hash(config_path), "methods": methods,
                "implementation_identity": implementation_identity(),
                "runtime": {"python": sys.version, "torch": str(torch.__version__),
                            "numpy": np.__version__, "cuda": torch.version.cuda, "device": device},
                "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in code},
                "mean_shift_artifact": provenance, "mean_prediction_artifact": mean_provenance,
                "response_sets": response_sets, "inference_steps": inference_steps, "fitting_performed": False,
                "control_information_comparison": "same_available_V_controls_different_estimators_not_strict_sample_input_pairing",
                "pending": ["real_training_and_V_results"]}
    identity["experiment_identity"] = config_hash(identity)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "input_and_method_identity.json", identity)
    atomic_json(output / "sampling_plan.json", plan)
    atomic_json(output / "status.json", {"status": "planned", "role": "V"})
    try:
        scope = provenance["metadata"].get("control_summary_scope", "not_declared") if provenance else None
        rows = evaluate_plan(data, plan, shifts=shifts, device=device,
                             mean_predictions=mean_predictions, shift_control_scope=scope, progress=progress)
        if response_sets:
            mixtures, members = evaluate_response_sets(data, plan, response_sets, device=device, progress=progress)
            rows += mixtures
            atomic_json(output / "response_member_metrics.json", members)
        atomic_json(output / "condition_seed_metrics.json", rows)
        summary = summarize(rows, config["bootstrap_repeats"], config["bootstrap_seed"])
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "status.json", {"status": "complete", "role": "V",
                    "result_type": "development_validation", "rows": len(rows),
                    "non_ok_rows": sum(row["status"] != "ok" for row in rows),
                    "failure_rows": sum(row["status"] not in {"ok", "partial_metrics"} for row in rows),
                    "not_applicable_metric_entries": sum(len(row.get("not_applicable", [])) for row in rows)})
    except Exception as error:
        atomic_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__,
                                             "message": str(error), "role": "V"})
        raise
    print(f"V report saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
