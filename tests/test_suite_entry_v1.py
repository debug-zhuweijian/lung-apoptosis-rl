"""Explicit input weight integrity and condition clocks, without real training."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lung.engine.response import condition_kwargs




def test_condition_clock_uses_real_endpoint_and_explicit_world_override():
    batch = {"fingerprint": [0.] * 2048, "dose": 1., "duration": 72.,
             "background_id": 0, "source_id": 1, "protocol_id": 0}
    actual = condition_kwargs(batch, 2, "cpu")
    torch.testing.assert_close(actual["horizon"], torch.full((2,), 72.))
    torch.testing.assert_close(actual["duration"], torch.full((2,), 72.))
    override = condition_kwargs(batch, 2, "cpu", horizon=96.)
    torch.testing.assert_close(override["horizon"], torch.full((2,), 96.))
    old = condition_kwargs({**batch, "duration": 24.}, 1, "cpu")
    assert old["horizon"].item() == 24.
