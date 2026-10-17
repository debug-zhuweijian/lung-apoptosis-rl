"""Default metadata preflight; --execute requires separate real-model authority."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.config import load_config
from lung.engine import full_protocol_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modular_research.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.execute:
        if args.metadata is None or args.output is None:
            parser.error("--execute requires reviewed --metadata and independent --output")
        config = load_config(args.config)
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        result = full_protocol_batch.produce_all(metadata, ROOT / args.output,
            seed=config["seed"], steps=config["response"]["inference_ode_steps"],
            microbatch=config["response"]["microbatch"], device=config["device"])
    else:
        result = full_protocol_batch.preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
