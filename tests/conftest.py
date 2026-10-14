"""Small shared fixtures; CPU-only behavior is explicitly requested by modules."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


@pytest.fixture
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def ae_reader(tmp_path):
    # Reuse the existing constructed-data builder, never production datasets.
    from src.lung.data import TrainingData, prepare_training
    from tests.test_training_data_v1 import tiny_bundle

    tiny_bundle(tmp_path)
    prepare_training(tmp_path)
    return TrainingData(tmp_path)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_same(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for one, other in zip(left, right):
            assert_same(one, other)
    else:
        assert left == right
