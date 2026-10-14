"""Default label-free metadata; --evaluate needs real model-run authority.

This command never fits models, estimates normalization/bandwidth or selects a
model. --compare reads three completed result packs without PT or RNA values.
The explicit --evaluate branch reads E RNA and matching completed PT.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from lung.engine.full_protocol_native_evaluation import NativeFinalMetadata, compare, evaluate, write
from lung.engine.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/modular_research.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--response-run', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--evaluate', action='store_true')
    mode.add_argument('--compare', nargs=3, type=Path, metavar='EVALUATION_DIR')
    parser.add_argument('--reference-mean-sensitivity', action='store_true')
    args = parser.parse_args(argv)
    if args.reference_mean_sensitivity and not args.evaluate:
        parser.error('--reference-mean-sensitivity requires --evaluate')
    if args.evaluate and (args.response_run is None or args.output is None):
        parser.error('--evaluate requires completed --response-run and new --output')
    output = (ROOT / args.output).resolve() if args.output else None
    if output and (not output.is_relative_to(ROOT / 'experiments') or output.exists()):
        parser.error('--output must be a new project experiments directory')
    config = load_config(args.config)
    if args.evaluate:
        result = evaluate(config, (ROOT / args.response_run).resolve(), output,
                          reference_mean_sensitivity=args.reference_mean_sensitivity)
    elif args.compare:
        result = compare(config, [(ROOT / p).resolve() for p in args.compare], output)
    else:
        view = NativeFinalMetadata(config)
        result = view.report()
        if output:
            output.mkdir(parents=True)
            write(output / 'preflight.json', result)
            write(output / 'final_requests.json', view.records)
            view.cell_index.to_parquet(output / 'final_cell_index.parquet', index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
