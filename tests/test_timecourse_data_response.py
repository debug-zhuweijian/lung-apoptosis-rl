"""Constructed data checks for timecourse reading, coordinates and permissions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timecourse_data import ArrayTimeData as TimeData
from lung.engine.response import condition_kwargs


def fixture_data(heldout=1., missing=False):
    rows, counts = [], []
    rng = np.random.default_rng(42)
    for time in [24, 72]:
        for drug in ["S0000", "d", "c", "e"]:
            for cell in range(3):
                if missing and time == 72 and drug == "S0000":
                    continue
                rows.append(dict(catalog_number=drug, time_point=time, dose_uM=0 if drug == "S0000" else 1,
                    replicate="rep1", culture_plate=f"plate{time}", author_hash_qc=True))
                # Near-uniform positive populations give a well-conditioned OT
                # kernel while retaining nonzero expression variation.
                value = 10000000 + rng.integers(1, 20, size=4).astype(float)
                if time == 72 and drug in ["c", "e"]:
                    value[0] *= heldout
                counts.append(value)
    return TimeData.from_arrays(counts, pd.DataFrame(rows), ["g1", "g2", "g3", "g4"],
        {drug: np.zeros(2048) for drug in ["S0000", "d", "c", "e"]},
        fit_drugs=["d"], allowed_target_drugs=["d"], synthetic=True,
        drug_groups={"D": ["d"], "C": ["c"], "E": ["e"]})


def test_heldout_cannot_change_coordinates_and_permissions():
    data, changed = fixture_data(), fixture_data(heldout=1000)
    assert np.array_equal(data.mu, changed.mu)
    assert np.array_equal(data.scale, changed.scale)
    assert np.array_equal(data.neighbors, changed.neighbors)
    assert data.coordinate_identity == changed.coordinate_identity
    assert data.identity != changed.identity
    cid = data.condition_ids(drugs=["e"])[0]
    with pytest.raises(PermissionError):
        data.sample_target(cid)
    view = data.view([], ["e"], "final_test")
    assert view.sample_target(cid).shape == (3, 4)
    with pytest.raises(PermissionError):
        view.sample_response(cid, 2, np.random.default_rng(0), role="T")
    assert data.genes.tolist() == ["g1", "g2", "g3", "g4"]
    assert len(data.canonical_features(view.sample_target(cid))) == 4


def test_actual_duration_and_strict_reference():
    data = fixture_data()
    for time in [24, 72]:
        cid = data.condition_ids(drugs=["d"], time=time)[0]
        batch = data.sample_controls(cid, 2, np.random.default_rng(1))
        condition = condition_kwargs(batch, 2, "cpu")
        assert condition["duration"].tolist() == [time, time]
        assert condition["horizon"].tolist() == [time, time]
        assert batch["context"].split(":")[1:] == ["rep1", f"plate{time}"]
    missing = fixture_data(missing=True)
    cid = missing.condition_ids(drugs=["d"])[0]
    with pytest.raises(ValueError, match="MISSING_MATCHED_VEHICLE"):
        missing.sample_controls(cid, 2, np.random.default_rng(0))


def test_kernel_bandwidth_units_and_gene_replication(tmp_path):
    from lung.timecourse.preparation import kernel_bandwidth
    from lung.models.objectives import rbf_kernel, rbf_mmd
    x = np.array([[0., 1.], [2., 1.], [0., 3.]])
    y = x + np.array([1., -2.])
    # Independent manual squared distances are 2, 2, 4: median = 2.
    assert kernel_bandwidth(x) == 2.
    actual = rbf_kernel(torch.tensor(x), torch.tensor(y), 2.).numpy()
    distances = ((x[:, None] - y[None]) ** 2).mean(-1)
    expected = np.mean(np.exp(-distances[..., None] / (2 * 2. * np.array([.25, .5, 1., 2., 4.]))), axis=-1)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    baseline = rbf_mmd(torch.tensor(x), torch.tensor(y), 2.)
    for copies in [1, 8, 32]:
        xx, yy = np.tile(x, (1, copies)), np.tile(y, (1, copies))
        assert kernel_bandwidth(xx) == 2.
        np.testing.assert_allclose(rbf_kernel(torch.tensor(xx), torch.tensor(yy), kernel_bandwidth(xx)), expected, atol=1e-14)
        torch.testing.assert_close(rbf_mmd(torch.tensor(xx), torch.tensor(yy), kernel_bandwidth(xx)), baseline)
    np.savez(tmp_path / "kernel-units.npz", actual=actual, expected=expected, bandwidth=2., mmd=float(baseline))
    for invalid in [np.ones((3, 2)), np.array([[0., 0.], [np.nan, 1.]])]:
        with pytest.raises(ValueError):
            kernel_bandwidth(invalid)


def test_known24_and_validation_permissions():
    base = fixture_data()
    data = base.training_view(known24_only=True, validation_fraction=.2)
    assert all(data.conditions.loc[key].time_point == 24 for key in data.condition_ids('T') + data.condition_ids('V'))
    for key in data.condition_ids('T'):
        train = data.sample_target(key, role='T')
        val = data.sample_target(key, role='V')
        assert len(train) == 2 and len(val) == 1
        assert not any(np.array_equal(a, b) for a in train for b in val)
    with pytest.raises(PermissionError, match='PRETRAIN24_ONLY'):
        data.sample_target(base.condition_ids(drugs=['d'])[0], role='T')
    for drug in ('c', 'e'):
        with pytest.raises(PermissionError):
            base.training_view(validation_fraction=.2).sample_target(base.condition_ids(drugs=[drug])[0], role='V')
    assert len(base.sample_target(base.condition_ids('T')[0], role='T')) == 3
