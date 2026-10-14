"""Prepare real protocol/reference inputs only. Never loads or executes weights."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from lung.engine.full_protocol_response import prepare, preflight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = prepare(args.output) if args.output else preflight()
    print(json.dumps({k: report[k] for k in ("schema", "candidate_count", "numeric_ready",
        "PT_loads", "model_forwards", "P_outputs", "G_outputs", "Q_outputs")}, ensure_ascii=False, indent=2))
