"""Build native response metadata only; never load functional Y or execute models."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.native_response_units import build

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/20261011-data-alignment-round17/native-response-units-v1')
    args = parser.parse_args()
    print(build(ROOT, args.output))
