"""Build the new source-bound physical protocol package; no model invocation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.full_protocol_inputs import build

if __name__ == "__main__":
    receipt = build()
    print("source candidates:", receipt["candidate_count"], "CPU tensor validate:", receipt["cpu_tensor_validate"])
