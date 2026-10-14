"""Build independent nominal Dex protocol inputs; no model invocation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung.data.full_protocol_inputs_v3 import build

if __name__ == "__main__":
    receipt = build()
    print("candidates:", receipt["candidate_count"], "fingerprint-known events:", receipt["fingerprint_known_events"])
