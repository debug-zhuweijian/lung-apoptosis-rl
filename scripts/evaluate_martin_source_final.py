"""Default final metadata preflight; evaluate-final needs explicit frozen run.

Real final model evaluation still requires the user's separate authorization.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from lung.engine import martin_source_final as final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='preflight', choices=('preflight', 'evaluate-final'))
    parser.add_argument('--fold-id', required=True)
    parser.add_argument('--trained-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    trained = (ROOT / args.trained_run).resolve() if args.trained_run else None
    output = (ROOT / args.output).resolve() if args.output else None
    for path in (trained, output):
        if path is not None and not path.is_relative_to(ROOT / 'experiments'):
            parser.error('All assets/output must remain in project experiments')
    if args.mode == 'evaluate-final':
        if trained is None or output is None:
            parser.error('evaluate-final needs --trained-run and a new --output')
        result = final.evaluate_final(ROOT, fold_id=args.fold_id, trained_run=trained, output=output, device=args.device)
    else:
        result = final.final_preflight(ROOT, fold_id=args.fold_id, trained_run=trained)
        if output:
            if output.exists():
                raise FileExistsError('NEW_FINAL_PREFLIGHT_REQUIRED')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
