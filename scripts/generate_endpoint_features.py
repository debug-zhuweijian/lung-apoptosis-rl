"""Preflight/generate frozen external Annexin condition features without training.

Preflight needs only --output; current metadata and missing artifacts are reported.
Generation additionally needs --generate, complete matching checkpoints, a
--reference JSON and --context-map JSON. No arbitrary source/plate ID fallback.

Reference JSON: kind=reference, reference_id, reference_plate, condition_id (a fit condition whose
controls define the reference), role=T, data_identity, source_id/background_id/
protocol_id (actual training vocabulary strings), evidence, source_files with
path/sha256 for the training manifest and control_links.parquet.

Context-map JSON: list of source_id/cell_line/protocol_id, target_source_id/
target_background_id/target_protocol_id, status=explicit_assumption_unvalidated,
reference_id/target_reference_plate matching the reference descriptor, evidence,
and source_files with path/sha256. A map is not biological validation.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lung.engine.checkpoint import atomic_json
from lung.engine.config import load_config, StageUnavailable
from lung.engine.data import EngineData
from lung.engine.endpoint_contract import FEATURE_KEYS
from lung.engine.endpoint_features import (load_condition_inputs, preflight,
                                           generate_to_directory, record, propose_reference_inputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/condition-inputs-v5"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_identity_v3.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/modular_research.yaml"))
    parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--context-map", type=Path)
    parser.add_argument("--propose-reference", action="store_true",
        help="Register a fit-only reference pool and explicit unvalidated source mappings, using metadata and hashes only")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args(argv)
    def resolve(path):
        return (ROOT / path).resolve()
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("Output exists; select a new version")
    output.mkdir(parents=True)
    config = load_config(resolve(args.config))
    config["data"] = str(resolve(args.data_config))
    supplied = json.loads(resolve(args.pipeline).read_text(encoding="utf-8")) if args.pipeline else config.get("modular", {}).get("artifacts", {})
    supplied = supplied.get("pipeline", supplied)
    pipeline = {k: ([str(resolve(Path(p))) for p in v] if isinstance(v, list) else str(resolve(Path(v))))
                for k, v in supplied.items() if k in FEATURE_KEYS and v}
    if "inference_ode_steps" in supplied:
        pipeline["inference_ode_steps"] = supplied["inference_ode_steps"]
    bundle = load_condition_inputs(resolve(args.inputs), ROOT)
    if args.propose_reference:
        if args.reference or args.context_map:
            raise ValueError("Choose explicit descriptors or --propose-reference, not both")
        args.reference, args.context_map = propose_reference_inputs(ROOT, args.data_config, bundle, output / "reference_inputs")
    reference = json.loads(resolve(args.reference).read_text(encoding="utf-8")) if args.reference else None
    mappings = json.loads(resolve(args.context_map).read_text(encoding="utf-8")) if args.context_map else None
    report = preflight(ROOT, args.data_config, pipeline, bundle, reference, mappings)
    report["input_manifest"] = record(resolve(args.inputs) / "manifest.json")
    report["runtime_config"] = record(resolve(args.config))
    report["pipeline_registration"] = record(resolve(args.pipeline)) if args.pipeline else None
    report["reference_descriptor"] = record(resolve(args.reference)) if args.reference else None
    report["source_context_map"] = record(resolve(args.context_map)) if args.context_map else None
    report["no_formal_training_started"] = True
    atomic_json(output / "preflight.json", report)
    if args.generate and not report["missing_or_invalid"]:
        try:
            data = EngineData(config, roles=("T",), require_preprocessing=True)
            result = generate_to_directory(ROOT, data, config, pipeline, bundle, reference, mappings, output / "features",
                reference_path=resolve(args.reference), mapping_path=resolve(args.context_map),
                data_config_path=resolve(args.data_config))
        except (StageUnavailable, ValueError, FileNotFoundError) as error:
            failure = {"status": "pending", "formal_training_ready": False, "reason": str(error),
                       "no_placeholder_features_written": True}
            atomic_json(output / "generation_failure.json", failure)
            print(json.dumps(failure, ensure_ascii=False))
            return 2
        print(json.dumps({"status": result["status"], "generated_conditions": result["generated_conditions"],
                          "manifest": str(output / "features/manifest.json")}, ensure_ascii=False))
    else:
        print(json.dumps({"status": report["status"], "eligible_condition_inputs": report["eligible_condition_inputs"],
                          "missing_or_invalid": report["missing_or_invalid"],
                          "report": str(output / "preflight.json")}, ensure_ascii=False))
    return 2 if report["missing_or_invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
