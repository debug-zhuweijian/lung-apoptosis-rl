"""Fixed registered RNA data driver; native processes are serial, no ML/PT."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from lung.data.bothe_count_executor import preflight, execute_registered_runs, execute_registered_when_ready
from lung.data.bothe_count_production import load_production


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--version")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-downloads", action="store_true")
    parser.add_argument("--readback", action="store_true")
    args = parser.parse_args()
    if args.wait_for_downloads and not args.execute:
        parser.error("wait-for-downloads requires execute")
    if args.readback:
        if args.execute or args.run or not args.version:
            parser.error("readback needs only --version")
        result = load_production(args.version)
        print(json.dumps({"domain": result.domain, "observed_samples": result.observed_samples,
                          "counts_complete": result.counts_complete, "model_eligible": result.model_eligible}))
    elif args.execute:
        if not args.run or not args.version:
            parser.error("execute needs registered --run and new --version")
        operation = execute_registered_when_ready if args.wait_for_downloads else execute_registered_runs
        result = operation(args.run, args.version)
        print(json.dumps({k: result[k] for k in ("status", "runs", "counts_complete", "model_eligible")}))
    else:
        if not args.run or args.version:
            parser.error("metadata preflight needs only registered --run")
        for run in args.run:
            result = preflight(run)
            print(json.dumps({k: result[k] for k in ("status", "run", "sample", "missing_paths", "estimate")}))


if __name__ == "__main__":
    main()
