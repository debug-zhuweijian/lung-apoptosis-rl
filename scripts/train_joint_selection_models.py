"""Default source-only preflight; all fit/scoring/evaluation modes are explicit.

Real closed-form fitting and model scoring remain separately authorized future
operations. This entry never starts a response model or loads a PT checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(directory))

from lung.engine import joint_selection_learning as learning


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="preflight", choices=("preflight", "build-index", "build-state-index", "fit", "freeze", "evaluate"))
    parser.add_argument("--view", type=Path, default=learning.DEFAULT_VIEW)
    parser.add_argument("--prediction-index", type=Path, default=learning.DEFAULT_INDEX)
    parser.add_argument("--prediction-directory", type=Path)
    parser.add_argument("--state-sidecar-index", type=Path)
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--role", choices=("selection", "final_test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    view = (ROOT / args.view).resolve()
    index = (ROOT / args.prediction_index).resolve()
    state_index = learning.project_path(args.state_sidecar_index) if args.state_sidecar_index else None
    if args.mode != "preflight" and args.output is None:
        parser.error("Explicit mode requires a new --output path")
    output = learning.project_path(args.output) if args.output else None
    if args.mode == "preflight":
        result = learning.preflight(index, view_directory=view, state_sidecar_index=state_index)
        if output:
            learning.write(output, result)
    elif args.mode == "build-index":
        if args.prediction_directory is None:
            parser.error("build-index requires --prediction-directory")
        result = learning.build_prediction_index(learning.project_path(args.prediction_directory), output, view_directory=view)
    elif args.mode == "build-state-index":
        if args.state_directory is None:
            parser.error("build-state-index requires --state-directory")
        result = learning.build_state_sidecar_index(learning.project_path(args.state_directory), output, view_directory=view)
    elif args.mode == "fit":
        result = learning.fit_models(index, output, view_directory=view, state_sidecar_index=state_index)
    elif args.mode == "freeze":
        if args.models is None or args.role is None:
            parser.error("freeze requires --models and explicit --role")
        result = learning.freeze_scores(learning.project_path(args.models), output, role=args.role,
            prediction_index=index, view_directory=view, state_sidecar_index=state_index)
    else:
        if args.receipt is None:
            parser.error("evaluate requires --receipt")
        result = learning.evaluate_frozen(learning.project_path(args.receipt), output)
    compact = {k: v for k, v in result.items() if k not in
        ("full_P_bindings", "input_identity", "predictions", "score_receipt", "dependencies")}
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
