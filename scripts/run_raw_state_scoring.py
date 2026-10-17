"""Default metadata-only original state C; explicit future --fit or --score."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(directory))

from lung.engine import raw_state_scoring as state
from lung.engine.checkpoint import atomic_json
from lung.engine.config import load_config
from lung.engine.data import EngineData


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modular_research.yaml")
    parser.add_argument("--ae-run", type=Path)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fit", action="store_true")
    modes.add_argument("--score", action="store_true")
    modes.add_argument("--final-preflight", action="store_true")
    modes.add_argument("--evaluate-final", action="store_true")
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--state-declaration", type=Path)
    parser.add_argument("--final-plan", type=Path)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if (args.fit or args.score or args.evaluate_final) and args.output is None:
        parser.error("Model operations require a new project experiment output")
    if args.fit and args.ae_run is None:
        parser.error("--fit needs the completed independent joint --ae-run")
    if args.score and (args.prediction is None or args.state_declaration is None):
        parser.error("--score needs frozen --prediction and --state-declaration")
    if args.evaluate_final and (args.state_declaration is None or args.final_plan is None):
        parser.error("--evaluate-final needs frozen --state-declaration and --final-plan")
    resolve = lambda p: (ROOT / p).resolve() if p else None
    output, ae_run = resolve(args.output), resolve(args.ae_run)
    if output and not output.is_relative_to(ROOT / "experiments"):
        parser.error("Output must remain in project experiments")
    if args.score:
        result = state.score_prediction(resolve(args.prediction), resolve(args.state_declaration), output,
            microbatch=args.microbatch, device=args.device)
    elif args.evaluate_final:
        result = state.evaluate_final(load_config(args.config), resolve(args.state_declaration),
            resolve(args.final_plan), output, device=args.device)
    else:
        config = load_config(args.config)
        result = (state.final_preflight(config, declaration_path=resolve(args.state_declaration), device=args.device)
            if args.final_preflight else state.preflight(config, ae_run=ae_run))
        if args.fit:
            state.require(result["fit_inputs_ready"], "MATCHING_AE_REQUIRED_BEFORE_REAL_FIT")
            candidate = state.state_config(config)
            data = EngineData(candidate, roles=("T", "V", "C"), allow_test=False)
            result = state.fit_state(data, candidate, ae_run, output)
        elif output:
            state.require(not output.exists(), "NEW_PREFLIGHT_OUTPUT_REQUIRED")
            output.mkdir(parents=True)
            atomic_json(output / ("final_plan.json" if args.final_preflight else "preflight.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
