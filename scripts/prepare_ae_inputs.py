"""Prepare reusable fit/selection expression inputs without training."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.ae_data import AEDataSupply
from lung.engine.data import open_data
from lung.engine.modular_config import load_modular_config, stage_config


def main():
    config = stage_config(load_modular_config(), "ae", {})
    data = open_data(config, "response", prepare=False)

    def progress(stage, current, total):
        print(f"{stage}: {current}/{total}", flush=True)

    with AEDataSupply(data, config["research"]["latent"]["ae"]["batch"]) as supply:
        count = supply.expressions.prepare(progress=progress)
        print(f"Prepared {count} conditions: {supply.expressions.directory}")


if __name__ == "__main__":
    main()
