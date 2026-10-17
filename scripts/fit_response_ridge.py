"""Manual T-fitted, V-selected Ridge baseline; never prepares data or reads E."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parser():
    command = argparse.ArgumentParser(description="Fit full-gene P1 Ridge on T, select lambda on V, export P2 shifts")
    command.add_argument("--config", default="configs/pilot_ridge.yaml")
    command.add_argument("--output", required=True, help="New child directory of experiments/; never overwrite")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    from lung.engine.checkpoint import atomic_json
    from lung.engine.config import project_path, read_yaml, config_hash, file_hash
    from lung.engine.data import EngineData
    from lung.engine.pilot_ridge import LAMBDAS, checked_device, run_ridge
    from tqdm.auto import tqdm

    config_path = project_path(args.config)
    config = read_yaml(config_path)
    if (config.get("fit_role"), config.get("selection_role"), config.get("evaluation_role")) != ("T", "V", "V"):
        raise PermissionError("RIDGE_ONLY_SUPPORTS_T_FIT_V_SELECTION; E is forbidden")
    if config.get("gene_count") != 32461 or config.get("synthetic", False):
        raise ValueError("REAL_FULL_32461_GENE_RIDGE_REQUIRED")
    if config.get("auto_prepare") is not False or config.get("dtype") != "float64":
        raise ValueError("RIDGE_REQUIRES_FROZEN_ASSETS_AND_FLOAT64")
    if tuple(config.get("lambdas", [])) != LAMBDAS:
        raise ValueError("PILOT_V1_FIXED_LAMBDAS_REQUIRED")
    if not isinstance(config.get("stream_batch_size"), int) or config["stream_batch_size"] < 1:
        raise ValueError("stream_batch_size must be a positive integer")
    output = project_path(args.output)
    experiments = (ROOT / "experiments").resolve()
    if output == experiments or not output.is_relative_to(experiments):
        raise ValueError("Ridge output must be a new child directory of experiments/")
    if output.exists():
        raise FileExistsError("RIDGE_OUTPUT_EXISTS: use a new experiment identity")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"status": "planned", "fit_role": "T", "selection_role": "V"})
    bars = {}

    def progress(stage, current, total):
        if not config.get("progress", True):
            return
        if stage not in bars:
            for bar in bars.values():
                bar.close()
            bars[stage] = tqdm(total=total, desc=stage, unit="cells" if "cells" in stage else "lambda")
        bar = bars[stage]
        bar.update(current - bar.n)

    try:
        checked_device(config["device"])
        data = EngineData(config, roles=("T", "V"), allow_test=False, require_preprocessing=True)
        if data.n_genes != config["gene_count"]:
            raise ValueError("RIDGE_FULL_GENE_COUNT_MISMATCH")
        identity = {"protocol": "A549-real-pilot-V1-20260924", "config": config,
            "config_sha256": file_hash(config_path), "data_identity": data.identity,
            "frozen_training_manifest": data.manifest,
            "source_audit_registry_sha256": config_hash(data.reader.rows.registry),
            "protocol_sha256": file_hash(project_path(config["protocol"])),
            "fit_role": "T", "selection_role": "V", "evaluation_role": "V",
            "test_response_access": False, "preprocessing_fitted": False}
        identity["experiment_identity"] = config_hash(identity)
        atomic_json(output / "input_and_method_identity.json", identity)
        atomic_json(output / "status.json", {"status": "running", "fit_role": "T", "selection_role": "V"})
        metadata = run_ridge(data, config, output, identity, progress=progress)
        atomic_json(output / "status.json", {"status": "complete", "fit_role": "T", "selection_role": "V",
            "evaluation_role": "V", "selected_lambda": metadata["selected_lambda"],
            "result_type": "development_validation_not_final_test",
            "artifact_sha256": {path.name: file_hash(path) for path in sorted(output.glob("*.npz"))}})
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__,
            "message": str(error), "fit_role": "T", "selection_role": "V", "partial_artifacts_preserved": True})
        raise
    finally:
        for bar in bars.values():
            bar.close()
    print(f"P1 means and P2 raw shifts saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
