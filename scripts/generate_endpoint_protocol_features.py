"""Metadata preflight by default; --generate explicitly runs frozen inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from lung.engine.checkpoint import atomic_json
from lung.engine.config import load_config
from lung.engine.endpoint_protocol_features_v2 import EndpointProtocolFeatureInputsV2
from lung.engine.endpoint_protocol_producer import preflight_standard_requests, generate_standard_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1")
    preflight = ROOT / "experiments/20261010-endpoint-feature-preflight-joint"
    parser.add_argument("--config", type=Path, default=preflight / "generator-config-v2.yaml")
    parser.add_argument("--pipeline", type=Path, default=preflight / "pipeline.json")
    parser.add_argument("--reference", type=Path, default=preflight / "final-v3/reference_inputs/reference.json")
    parser.add_argument("--context-map", type=Path, default=ROOT / "experiments/20261011-data-alignment-round14/producer_context_map.json")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/data_joint_sciplex_v1.json")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new immutable output directory")
    config = load_config(args.config)
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))["pipeline"]
    bridge = EndpointProtocolFeatureInputsV2.load(args.inputs)
    options = dict(reference_path=args.reference.resolve(), mapping_path=args.context_map.resolve(),
                   data_config_path=args.data_config.resolve(), pipeline=pipeline, config=config)
    report = preflight_standard_requests(bridge, **options)
    atomic_json(output / "preflight.json", report)
    if args.generate and not report["missing_stages"]:
        generate_standard_cache(bridge, **options, output=output / "cache")
    print(json.dumps(report, ensure_ascii=False))
    return 2 if report["missing_stages"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
