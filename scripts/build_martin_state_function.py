"""Build or metadata-verify the source-preserving Martin2022 bridge."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.data.martin_state_function import build, MartinStateFunctionInputs

BASE = ROOT / "08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "bridge_v2")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not args.verify:
        build(BASE, output, project_root=ROOT)
    inputs = MartinStateFunctionInputs(output, project_root=ROOT)
    print(json.dumps({"counts": inputs.manifest["counts"], "status": inputs.metadata["status"],
                      "actual_model_execution": False, "Annexin_fraction_eligible": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
