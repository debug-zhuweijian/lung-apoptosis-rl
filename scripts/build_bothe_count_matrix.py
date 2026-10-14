"""Audit mock count parser CLI; real executor qualification remains pending."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lung.data.bothe_count_matrix import ANNOTATION, build_audit_mock, build_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-mock-inputs", type=Path,
                        help="JSON list of path and column_samples; never real RNA qualification")
    args = parser.parse_args()
    if args.audit_mock_inputs is None:
        build_real(args.output)
    inputs = json.loads(args.audit_mock_inputs.read_text(encoding="utf-8"))
    result = build_audit_mock(args.output, inputs, annotation=ANNOTATION)
    print(json.dumps({k: result[k] for k in ("evidence", "status", "readyRNA", "missing_samples")}))


if __name__ == "__main__":
    main()
