"""Build/verify measured label views; no models or training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.data.endpoint_protocol_supervision import BASE, build_views, load_views


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "endpoint-protocol-supervision-v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = load_views(output) if args.verify else build_views(output)
    print(json.dumps({k: v for k, v in manifest.items() if k not in
        {"input_directory", "files_sha256"}}, ensure_ascii=False))
