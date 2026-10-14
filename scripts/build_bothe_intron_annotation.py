"""Build frozen annotation only; never downloads reads, aligns or trains."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.intron_annotation import build, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    reference = ROOT / "08-datasets/Bothe2021/E-MTAB-9923/reference"
    parser.add_argument("--source", type=Path, default=reference / "current-ucsc-hg19-20261011")
    parser.add_argument("--output", type=Path, default=reference / "unique-introns-v1")
    parser.add_argument("--receipt", type=Path, default=ROOT / "experiments/20261011-data-alignment-round18/bothe_annotation")
    args = parser.parse_args()
    manifest = build(args.source, args.output, args.receipt)
    manifest["generator_files"][Path(__file__).name] = sha(Path(__file__))
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (args.output / "manifest.json").write_text(payload, encoding="utf-8")
    (args.receipt / "annotation_build.json").write_text(payload, encoding="utf-8")
    print(json.dumps({"axes": manifest["axes"], "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
