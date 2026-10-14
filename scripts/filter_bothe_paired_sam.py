"""Strict SAM pair filter; creates a new version directory, never overwrites."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lung.data.paired_alignment_filter import filter_sam

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = filter_sam(args.sam, args.output_dir)
    print(receipt["status"], receipt["counters"])
