"""Prepare frozen training indices and T-only assets; never launch training."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lung.data import prepare_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only real split indices/chemical fingerprints; defer full T statistics and graph")
    parser.add_argument('--device', default='cuda:0', help='Statistics/graph device; cpu is explicitly available')
    args = parser.parse_args()
    previous = {}

    def progress(stage, done, total):
        percent = int(100 * done / max(total, 1))
        if percent != previous.get(stage):
            print(f"{stage}: {done}/{total} ({percent}%)", flush=True)
            previous[stage] = percent

    path = prepare_training(ROOT, args.data, build_preprocessing=not args.metadata_only,
                            progress=progress, device=args.device)
    print(f"Training assets: {path}")


if __name__ == "__main__":
    main()
