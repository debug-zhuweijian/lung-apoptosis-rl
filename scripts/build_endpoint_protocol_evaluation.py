"""Write/readback metadata-only endpoint evaluation role plan; no models."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.data.endpoint_protocol_evaluation import DEFAULT_PLAN, build_plan, EndpointProtocolEvaluationPlan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    plan = EndpointProtocolEvaluationPlan.load(args.output) if args.verify else build_plan(args.output)
    print(json.dumps({"fold_ids": plan.fold_ids, **plan.manifest}, ensure_ascii=False))
