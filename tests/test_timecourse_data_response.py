"""Synthetic numerical and permission tests; no real dataset fitting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.timecourse.data import TimeData
from lung.timecourse.response import train_response, predict_response
from lung.engine.checkpoint import load_checkpoint
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


def tiny_config():
    return {"synthetic": True, "epochs": 2, "device": "cpu", "response": {
        "members": 1, "mmd_weight": 1., "conditions_per_update": 8,
        "ot_samples": 2, "cfm_samples": 2, "mmd_samples": 2,
        "microbatch": 2, "mmd_microbatch": 2, "inference_microbatch": 2,
        "train_ode_steps": 1, "inference_ode_steps": 1,
        "model": {"n_genes": 4, "d_model": 8, "n_heads": 2, "encoder_layers": 1,
            "pad_layers": 1, "query_chunk": 4, "context_dim": 2,
            "checkpoint_blocks": False, "checkpoint_attention": False}}}


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


def test_micro_response_recovery_and_prediction(tmp_path):
    torch.set_num_threads(1)
    data, config = fixture_data(), tiny_config()
    def interrupt(phase, current, total):
        if phase == "time_response_member_1" and current == 1:
            raise KeyboardInterrupt("synthetic interruption")
    with pytest.raises(KeyboardInterrupt):
        train_response(data, config, tmp_path / "resumed", progress=interrupt)
    checkpoint = tmp_path / "resumed/member_1/last.pt"
    assert load_checkpoint(checkpoint)["epoch"] == 1
    result = train_response(data, config, tmp_path / "resumed", resume=True)
    uninterrupted = train_response(data, config, tmp_path / "direct")
    resumed_weights = load_checkpoint(result["checkpoints"][0])["model"]
    direct_weights = load_checkpoint(uninterrupted["checkpoints"][0])["model"]
    for key, expected in direct_weights.items():
        actual = resumed_weights[key]
        if expected.is_sparse:
            assert torch.equal(actual.coalesce().indices(), expected.coalesce().indices()), key
            assert torch.equal(actual.coalesce().values(), expected.coalesce().values()), key
        else:
            assert torch.equal(actual, expected), key
    cid = data.condition_ids(drugs=["e"])[0]
    prediction = predict_response(result["checkpoints"], data, cid, 3, 123, "cpu")
    assert prediction.shape == (3, 4) and np.isfinite(prediction).all()
    np.save(tmp_path / "prediction.npy", prediction)
    assert len(pd.read_csv(tmp_path / "resumed/member_1/metrics.csv")) == 2
    assert list((tmp_path / "resumed/member_1/events").glob("events.out.tfevents.*"))
    files_before = {str(path): (path.stat().st_mtime_ns, path.stat().st_size)
                    for path in (tmp_path / "resumed").rglob("*") if path.is_file()}
    train_response(data, config, tmp_path / "resumed", resume=True)
    files_after = {str(path): (path.stat().st_mtime_ns, path.stat().st_size)
                   for path in (tmp_path / "resumed").rglob("*") if path.is_file()}
    assert files_after == files_before


def test_synthetic_entrypoint_has_real_permissions_and_separate_manifest(tmp_path):
    from lung.timecourse.data import synthetic_data, prepare_time_data
    config = tiny_config()
    config["timecourse"] = {"synthetic_assets": str(tmp_path / "assets"), "synthetic_development_drugs": 4}
    data = synthetic_data(config)
    assert len(data.drug_groups["D"]) == 4
    assert data.conditions.loc[data.conditions.time_point.eq(72), "culture_plate"].nunique() == 4
    assert data.conditions.reference_n.min() == 3
    path = prepare_time_data(tmp_path, config)
    assert path.is_file()
    reader = TimeData(tmp_path, config, [], [], "inference")
    assert reader.identity == data.identity
    with pytest.raises(PermissionError):
        reader.sample_target(reader.condition_ids(drugs=data.drug_groups["E"])[0])
    with pytest.raises(PermissionError):
        TimeData(tmp_path, config, ["unregistered"], ["unregistered"], "fit")


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


def test_sparse64_real_micro_training(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260926)
    rows, counts = [], []
    for time in (24, 72):
        for drug in ("S0000", "d"):
            for _ in range(128):
                rows.append(dict(catalog_number=drug, time_point=time, dose_uM=0 if drug == "S0000" else 1,
                    replicate="rep1", culture_plate=f"plate{time}", author_hash_qc=True))
                counts.append(rng.poisson(2., 64))
    counts = np.asarray(counts)
    assert np.any(counts == 0) and counts.std() > 1
    data = TimeData.from_arrays(counts, rows, [f"g{i}" for i in range(64)],
        {drug: np.zeros(2048) for drug in ("S0000", "d")}, fit_drugs=["d"],
        allowed_target_drugs=["d"], synthetic=True, drug_groups={"D": ["d"], "C": [], "E": []})
    config = tiny_config()
    config["epochs"] = 1
    config["response"].update(ot_samples=16, cfm_samples=4, mmd_samples=4)
    config["response"]["model"].update(n_genes=64, query_chunk=64)
    result = train_response(data, config, tmp_path / "sparse64")
    assert load_checkpoint(result["checkpoints"][0])["epoch"] == 1
    assert np.isfinite(pd.read_csv(tmp_path / "sparse64/member_1/metrics.csv").loss).all()
    np.savez(tmp_path / "sparse64-input.npz", counts=counts, bandwidth=data.bandwidth)
