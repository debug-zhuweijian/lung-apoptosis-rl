"""Prepare only registered source-bound inputs; no P/G/Q execution."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung.data.full_protocol_request_inputs import build

if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
