"""Prepare observed Martin RNA protocol metadata only; no ML or weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from lung.data.martin_observed_protocol import BASE, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / BASE / 'observed-protocol-inputs-v1')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(ROOT, args.output), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
