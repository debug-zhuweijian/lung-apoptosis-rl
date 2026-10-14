"""Fixed registered Bothe validation or explicit synthetic audit; no model execution."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lung.data.paired_fastq_validation import validate_registered_run, validate_audit_pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--audit-r1", type=Path)
    parser.add_argument("--audit-r2", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run and not (args.audit_r1 or args.audit_r2):
        receipt = validate_registered_run(args.run, args.output_dir)
    elif not args.run and args.audit_r1 and args.audit_r2:
        receipt = validate_audit_pair(args.audit_r1, args.audit_r2, args.output_dir)
    else:
        parser.error("use registered --run OR explicit --audit-r1 and --audit-r2")
    print(receipt["mode"], receipt["status"], receipt["actual_pair_count"])
