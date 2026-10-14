"""Metadata preflight or explicit frozen-caller Martin consistency evaluation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.data.martin_state_function import MartinStateFunctionInputs
from lung.data.state_function_evaluation import preflight, freeze, evaluate

BRIDGE = ROOT / "08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/bridge_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "freeze", "evaluate"))
    parser.add_argument("--bridge", type=Path, default=BRIDGE)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--model-artifact", type=Path)
    parser.add_argument("--frozen-directory", type=Path)
    args = parser.parse_args()
    if args.mode == "evaluate":
        if args.frozen_directory is None:
            parser.error("evaluate requires --frozen-directory")
        result = evaluate(args.frozen_directory)
    else:
        source = MartinStateFunctionInputs(args.bridge, project_root=ROOT)
        if args.mode == "preflight":
            result = preflight(source)
        else:
            if any(v is None for v in (args.predictions, args.model_artifact, args.frozen_directory)):
                parser.error("freeze requires --predictions --model-artifact --frozen-directory")
            result = freeze(source, args.predictions, args.model_artifact, args.frozen_directory)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
