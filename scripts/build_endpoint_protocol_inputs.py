"""Build or read verified endpoint inputs without training or model inference."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.endpoint_protocol_inputs import BASE, EndpointProtocolInputs, build, sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "endpoint-protocol-inputs-v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    bundle = EndpointProtocolInputs.load(args.output) if args.verify else build(args.output)
    inputs = [bundle.model_input(cid) for cid in bundle.candidate_ids]
    receipt = {"status": "input_ready_response_features_missing", "formal_training_ready": False,
        "candidate_count": len(inputs), "unique_active_observations": len(bundle.observations()),
        "p_request_count": bundle.manifest["p_request_count"],
        "protocol_feature_names": inputs[0]["feature_names"],
        "real_fingerprint_shape": [len(inputs), len(inputs[0]["chemical_fingerprint"])],
        "actual_P_descriptors": sum(r["p_descriptor"] is not None for r in inputs),
        "manifest_sha256": sha(args.output / "manifest.json"),
        "source_binding_count": len(bundle.manifest["inputs_sha256"]),
        "model_execution_performed": False}
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
