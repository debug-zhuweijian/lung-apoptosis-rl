"""Audit current metadata or compare supplied frozen results; no model runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.data.science_evaluation_comparison import read, metadata_preflight, compare_frozen_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "compare"))
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "compare" and args.packet is None:
        parser.error("compare requires --packet with existing frozen results")
    result = metadata_preflight(ROOT) if args.mode == "preflight" else compare_frozen_metrics(read(args.packet), ROOT)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside the execution project")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"endpoint_blocks", "comparisons"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
