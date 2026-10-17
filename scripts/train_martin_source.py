"""Default real data preflight; --fit explicitly runs the independent source E/D.

No legacy training entry, split, reward or shared latent model is changed.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'src'):
    sys.path.insert(0, str(directory))

from lung.engine.config import load_config
from lung.engine.martin_source_learning import fit_source, preflight, prepare_fold, settings_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/modular_research.yaml')
    parser.add_argument('--fold-id', required=True)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fit', action='store_true')
    args = parser.parse_args()
    if args.fit and not args.output:
        parser.error('--fit requires a new experiment output')
    output = (args.output if args.output.is_absolute() else ROOT / args.output).resolve() if args.output else None
    if output is not None and not output.is_relative_to(ROOT / 'experiments'):
        parser.error('Output must stay within the project experiments directory')
    config = load_config(args.config)
    settings = settings_from_config(config, width=args.width)
    fit, selection = prepare_fold(ROOT, system='cycle', fold_id=args.fold_id)
    if args.fit:
        result = fit_source(fit, selection, settings, output)
    else:
        result = preflight(fit, selection, settings)
        if output is not None:
            if output.exists():
                raise FileExistsError('NEW_SOURCE_PREFLIGHT_OUTPUT_REQUIRED')
            output.mkdir(parents=True)
            (output / 'preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
