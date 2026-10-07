"""Short-lived synthetic scalar stream for manually inspecting the local training UI."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from lung.engine.config import load_config
from lung.engine.synthetic import smoke_config
from lung.monitoring import RunMonitor

config = smoke_config(load_config())
config.update(dashboard=True, open_browser=False, epochs=20, progress=True)
monitor = RunMonitor(config)
print(f"UI_CHECK_RUN={monitor.run_id}", flush=True)
print(f"UI_CHECK_STATUS={monitor.logdir / 'dashboard.json'}", flush=True)
try:
    for step in monitor.progress(range(1, 21), desc="SYNTHETIC UI CHECK"):
        monitor.record("SYNTHETIC_UI/daac", step, {"base_return": step / 21,
                                                  "critic_loss": 1 / step})
        if (monitor.logdir / "stop-ui-check").exists():
            break
        time.sleep(5)
finally:
    monitor.close("synthetic_ui_check_complete")
