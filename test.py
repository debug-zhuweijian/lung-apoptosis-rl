"""Frozen evaluation only. This script never trains, chooses a gate or calibrates."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())
