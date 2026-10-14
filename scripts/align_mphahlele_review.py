"""Build a source audit bundle, without downloads, model fitting or frozen-data writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from lung.data.mphahlele_alignment import prepare_alignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New, nonexistent audit directory')
    parser.add_argument('--supplement', type=Path, help='Optional original DOCX; checks pinned size/SHA only')
    args = parser.parse_args()
    receipt = prepare_alignment(
        ROOT / 'scripts/collect_a549_endpoint_review_sources_20261009.py',
        ROOT / 'configs/mphahlele2023_nominal_structures_v1.json',
        args.output, supplement=args.supplement,
    )
    print(json.dumps({'status': receipt['status'], 'rows': receipt['rows'],
                      'training_ready': receipt['training_ready'],
                      'whole_project_qualified': receipt['whole_project_qualified']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
