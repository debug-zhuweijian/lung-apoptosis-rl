"""Build/reuse fixed AE inputs for fit and selection; no model training."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.ae_data import AEDataSupply
from lung.engine.checkpoint import atomic_json
from lung.engine.config import implementation_identity
from lung.engine.data import open_data
from lung.engine.modular_config import load_modular_config, stage_config


def main():
    now = datetime.now(timezone(timedelta(hours=8)))
    report_path = ROOT / "logs/ae-input-preparation" / (now.strftime("%Y%m%d-%H%M%S") + ".json")
    config = stage_config(load_modular_config(), "ae", {})
    if config.get("synthetic"):
        raise ValueError("This entry prepares only the existing real fit/selection inputs")
    data = open_data(config, "response", prepare=False)
    batch = config["research"]["latent"]["ae"]["batch"]
    started = perf_counter()
    identity = implementation_identity()
    state = {"status": "running", "formal_training": False, "data_identity": data.identity,
             "implementation_identity": identity, "roles": ["fit", "selection"],
             "n_genes": data.n_genes, "report": str(report_path)}
    def progress(stage, current, total):
        state.update(stage=stage, current=current, total=total,
                     elapsed_seconds=perf_counter() - started)
        atomic_json(report_path, state)
        print(f"AE fixed inputs {current}/{total} conditions, "
              f"{state['elapsed_seconds']:.1f}s", flush=True)
    try:
        with AEDataSupply(data, batch) as supply:
            state["cache_directory"] = str(supply.expressions.directory)
            count = supply.expressions.prepare(progress=progress)
        if implementation_identity() != identity:
            raise ValueError("Source changed during input preparation")
        state.update(status="complete", prepared_conditions=count,
                     elapsed_seconds=perf_counter() - started)
        atomic_json(report_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error_type=type(error).__name__, elapsed_seconds=perf_counter() - started)
        atomic_json(report_path, state)
        raise


if __name__ == "__main__":
    main()
