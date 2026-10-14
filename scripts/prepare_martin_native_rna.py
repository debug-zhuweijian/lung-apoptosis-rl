"""Prepare actual observed RNA inputs only; never creates or runs a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from lung.data.martin_native_rna import BASE, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / BASE / 'native-rna-inputs-v1')
    args = parser.parse_args(argv)
    reader = prepare(ROOT, args.output)
    print(json.dumps(reader.manifest, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
