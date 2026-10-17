"""Build metadata or evaluate frozen G/simple-rank scores; never execute models."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(directory))
from lung.data.joint_selection_evaluation import (
    DEFAULT_VIEW, JointSelectionEvaluation, build_view, canonical, read, require)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "preflight", "freeze", "evaluate"))
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT.resolve()), "OUTPUT_OUTSIDE_EXECUTION_PROJECT")
    require(not output.exists(), "REFUSING_OUTPUT_OVERWRITE")
    if args.mode == "build":
        result = build_view(output).preflight()
        print(canonical(result))
        return
    view = JointSelectionEvaluation.load(args.view)
    if args.mode == "preflight":
        result = view.preflight()
    elif args.mode == "freeze":
        require(args.packet is not None, "EXPLICIT_FROZEN_SCORE_PACKET_REQUIRED")
        result = view.freeze_scores(read(args.packet))
    else:
        require(args.receipt is not None, "EXPLICIT_FROZEN_RECEIPT_REQUIRED_BEFORE_Y")
        result = view.evaluate_frozen(read(args.receipt))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical(result) + "\n", encoding="utf-8")
    print(canonical({"mode": args.mode, "output": str(output),
                     "model_execution_performed": False, "execution_certified": False}))


if __name__ == "__main__":
    main()
