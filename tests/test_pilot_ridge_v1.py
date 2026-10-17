"""Independent small numerical/role regressions; no real fitting is started."""

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.data import prepare_training
from lung.engine.config import config_hash, file_hash, read_yaml
from lung.engine.data import EngineData
from lung.engine.pilot_ridge import (LAMBDAS, POINT_SEMANTICS, chemical_mean_rmse, fit_ridge,
                                    raw_mean_points, run_ridge, select_ridge)
from lung.engine.pilot_ridge_data import features, mean_plan, require_ridge_data, stream_means
from lung.models import physical_projection
from test_training_data_v1 import tiny_bundle


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_data(tmp_path):
    tiny_bundle(tmp_path)
    prepare_training(tmp_path)
    with patch("lung.engine.data.ROOT", tmp_path):
        return EngineData({"data": str(tmp_path / "configs/dataset.yaml")}, roles=("T", "V"))


def test_dual_solution_matches_independent_primal_with_unpenalized_intercept():
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(7, 4))
    inputs[:, 0] = 9.
    targets = rng.normal(size=(7, 3)) + np.array([5., -8., 17.])
    regularization = .7
    model = fit_ridge(inputs, targets, regularization)
    mean = inputs.mean(0)
    mean[0] = 9.
    scale = inputs.std(0)
    scale[0] = 1.
    scaled = (inputs - mean) / scale
    design = np.column_stack([np.ones(len(inputs)), scaled])
    penalty = np.diag([0., *([regularization] * 4)])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    probe = rng.normal(size=(6, 4))
    probe[:, 0] = 9.
    expected = np.column_stack([np.ones(len(probe)), (probe - mean) / scale]) @ coefficients
    np.testing.assert_allclose(model.predict(probe), expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(model.predict(inputs).mean(0), targets.mean(0), atol=2e-12)
    assert all(value.dtype == np.float64 for value in model.arrays().values())


def test_constant_inputs_predict_target_mean_even_with_huge_penalty():
    model = fit_ridge(np.full((4, 3), 7.), np.array([[2., 8.], [4., 6.], [6., 4.], [8., 2.]]), 1e12)
    np.testing.assert_array_equal(model.predict(np.full((2, 3), 70.)), np.full((2, 2), 5.))
    np.testing.assert_array_equal(model.x_scale.numpy(), np.ones(3))


def test_scaling_and_intercept_are_t_only_when_v_changes():
    train_x = np.array([[1., 3.], [2., 5.], [3., 7.]])
    train_y = np.array([[-.2, .2], [0., 0.], [.2, -.2]])
    fixed = dict(control_mean=np.zeros((1, 2)), group_ids=["v"])
    first, _ = select_ridge(train_x, train_y, [[2., 5.]], target_mean=[[0., 0.]], **fixed)
    second, _ = select_ridge(train_x, train_y, [[100., -100.]], target_mean=[[.01, -.01]], **fixed)
    for best in (first, second):
        np.testing.assert_array_equal(best["model"].x_mean.numpy(), train_x.mean(0))
        np.testing.assert_allclose(best["model"].x_scale.numpy(), train_x.std(0), atol=1e-15)
        np.testing.assert_array_equal(best["model"].y_mean.numpy(), train_y.mean(0))


def test_v_only_selection_and_chemical_equal_weighting():
    train_x = np.array([[-1.], [1.]])
    train_y = np.array([[-.3, .3], [.3, -.3]])
    control = np.zeros((1, 2))
    # Independent 1-D closed form: beta = X'Y / (X'X + lambda).
    true_lambda = 100.
    expected_delta = np.array([[.7 * .6 / (2 + true_lambda), -.7 * .6 / (2 + true_lambda)]])
    target = control + expected_delta
    best, report = select_ridge(train_x, train_y, [[.7]], control, target, ["entity"])
    assert best["model"].regularization == true_lambda
    assert report["selection_role"] == "V" and report["selected_lambda"] == true_lambda
    assert [row["lambda"] for row in report["candidates"]] == list(LAMBDAS)
    score, _, groups = chemical_mean_rmse([[1.], [3.], [8.]], [[0.], [0.], [0.]], ["g1", "g1", "g2"])
    assert score == 5. and groups == {"g1": 2., "g2": 8.}
    with pytest.raises(ValueError, match="FIXED_LAMBDAS"):
        select_ridge(train_x, train_y, [[.7]], control, target, ["entity"], lambdas=[1.])


def test_full_unique_stream_matches_direct_means_and_never_samples(tiny_data):
    data, accesses = tiny_data, []
    original = data.reader.rows.read

    def checked_read(rows):
        assert set(rows.role) == {"fit"}
        accesses.extend(rows.cell_id.tolist())
        return original(rows)

    plan = mean_plan(data, "T")
    with patch.object(data.reader.rows, "read", side_effect=checked_read), patch.object(
            data, "sample_response", side_effect=AssertionError("No mean by resampling")):
        result = stream_means(data, plan, batch_size=7)
    assert len(accesses) == len(set(accesses)) == sum(plan["pool_counts"].values())
    for i, row in enumerate(plan["conditions"]):
        target_rows = data.reader.cell_metadata(row["condition_id"], role="fit")
        values = data.reader.normalize(original(target_rows))
        np.testing.assert_allclose(result["target_mean"][i], values.mean(0, dtype=np.float64), atol=1e-14)
    metadata = json.loads(result["metadata_json"].item())
    assert metadata["control_summary_scope"] == "full_unique_T_control_pool_mean"
    assert metadata["accumulation_dtype"] == "float64"
    assert result["pool_mean"].dtype == np.float64


def test_physical_identity_set_is_enforced_across_role_reads(tiny_data):
    plan = mean_plan(tiny_data, "T")
    metadata = tiny_data.reader.cell_metadata(plan["conditions"][0]["condition_id"], role="fit")
    row = metadata.iloc[0]
    already_read = {(str(row.matrix_path), int(row.matrix_row))}
    with pytest.raises(ValueError, match="DUPLICATE_PHYSICAL_CELL"):
        stream_means(tiny_data, plan, batch_size=7, seen_physical=already_read)


def test_ridge_reader_rejects_e_and_incomplete_assets_before_expression(tiny_data):
    with patch.object(tiny_data.reader.rows, "read", side_effect=AssertionError("No expression")):
        with pytest.raises(PermissionError, match="ONLY_T_V"):
            mean_plan(tiny_data, "E")
        with pytest.raises(PermissionError):
            tiny_data.reader.cell_metadata("anything", role="final_test")
        with patch.object(tiny_data.reader, "allowed_roles", frozenset({"fit", "selection", "final_test"})):
            with pytest.raises(PermissionError, match="T_V_ONLY"):
                require_ridge_data(tiny_data)
        with patch.dict(tiny_data.manifest, {"status": "metadata_ready"}):
            with pytest.raises(ValueError, match="PREPROCESSING_REQUIRED"):
                require_ridge_data(tiny_data)


def test_feature_axis_is_full_gene_with_frozen_drug_and_known_background(tiny_data):
    ids = tiny_data.condition_ids("T")[:2]
    controls = np.arange(2 * tiny_data.n_genes).reshape(2, -1)
    matrix, names = features(tiny_data, ids, controls)
    np.testing.assert_array_equal(matrix[:, :tiny_data.n_genes], controls)
    assert names[:tiny_data.n_genes] == [f"control:{gene}" for gene in tiny_data.reader.gene_ids]
    assert names.count("log1p_dose_um") == 1
    assert sum(name.startswith("morgan_log1p_dose:") for name in names) == 2048
    for field in ("source_id", "background_id", "protocol_id"):
        assert any(name.startswith(field + ":") for name in names)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_end_to_end_artifact_preserves_gene_axis_and_raw_means_without_pickle(tiny_data, tmp_path, device):
    output = tmp_path / "experiments" / "ridge"
    output.mkdir(parents=True)
    config = {"device": device, "stream_batch_size": 13, "lambdas": list(LAMBDAS)}
    identity = {"code_sha256": {"test": "a" * 64}}
    metadata = run_ridge(tiny_data, config, output, identity)
    assert metadata["fit_artifact_sha256"] == file_hash(output / "ridge_model.npz")
    assert metadata["fit_data_sha256"] == file_hash(output / "t_condition_means.npz")
    assert metadata["fit_code_sha256"] == config_hash(identity["code_sha256"])
    with np.load(output / "v_delta.npz", allow_pickle=False) as artifact:
        arrays = {key: artifact[key].copy() for key in artifact.files}
        for key in artifact.files:
            assert not artifact[key].dtype.hasobject
        np.testing.assert_array_equal(artifact["gene_ids"], tiny_data.reader.gene_ids)
        assert set(artifact["condition_ids"].astype(str)) == set(tiny_data.condition_ids("V"))
        assert artifact["point_valid"].all()
    with np.load(output / "v_condition_means.npz", allow_pickle=False) as population:
        np.testing.assert_array_equal(arrays["mean_prediction"], population["control_mean"] + arrays["delta"])
    provenance = json.loads(arrays["metadata_json"].item())
    assert provenance["prediction_semantics"] == POINT_SEMANTICS
    assert provenance["control_summary_scope"] == "full_unique_V_control_pool_mean"
    report = json.loads((output / "p1_full_population_metrics.json").read_text())
    assert all(row["mmd_u"] is None and row["variance_rmse"] is None for row in report["conditions"])
    with pytest.raises(FileExistsError):
        run_ridge(tiny_data, config, output, identity)


def ridge_cli():
    spec = importlib.util.spec_from_file_location("ridge_cli", ROOT / "scripts/fit_response_ridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_forbids_e_before_opening_data(tmp_path):
    path = tmp_path / "e.yaml"
    path.write_text("fit_role: T\nselection_role: E\nevaluation_role: E\n")
    with patch("lung.engine.data.EngineData", side_effect=AssertionError("No data opening")):
        with pytest.raises(PermissionError, match="ONLY_SUPPORTS_T_FIT_V_SELECTION"):
            ridge_cli().main(["--config", str(path), "--output", str(tmp_path / "unused")])


def test_cli_preserves_failed_state_and_refuses_overwrite(tmp_path):
    cli = ridge_cli()
    cli.ROOT = tmp_path
    config = {**read_yaml(ROOT / "configs/pilot_ridge.yaml"), "device": "cpu"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "experiments" / "failed"
    with patch("lung.engine.data.EngineData",
            side_effect=ValueError("PREPROCESSING_REQUIRED")) as opened:
        with pytest.raises(ValueError, match="PREPROCESSING_REQUIRED"):
            cli.main(["--config", str(path), "--output", str(output)])
        assert opened.call_args.kwargs == {"roles": ("T", "V"), "allow_test": False, "require_preprocessing": True}
        status = json.loads((output / "status.json").read_text())
        assert status["status"] == "failed" and status["partial_artifacts_preserved"]
        with pytest.raises(FileExistsError, match="OUTPUT_EXISTS"):
            cli.main(["--config", str(path), "--output", str(output)])
        assert opened.call_count == 1


def test_true_heterogeneous_group_mean_has_zero_rmse_without_cell_projection():
    # Synthetic counterexample: both cells have library mass 10000, but their
    # mean(log1p expression) does not satisfy the single-cell mass equality.
    counts = np.array([[9000., 1000.], [1000., 9000.]])
    log_expression = np.log1p(counts)
    mean, scale = np.array([2., 3.]), np.array([.7, 1.3])
    cells = (log_expression - mean) / scale
    target_mean = cells.mean(0, keepdims=True)
    assert np.allclose(np.expm1(log_expression).sum(1), 10000.)
    assert np.expm1(log_expression.mean(0)).sum() < 10000.
    # The unpenalized intercept exactly predicts this group mean for every lambda.
    best, report = select_ridge(np.ones((2, 1)), np.repeat(target_mean, 2, axis=0),
                               [[1.]], np.zeros((1, 2)), target_mean, ["heterogeneous"])
    np.testing.assert_array_equal(best["mean_prediction"], target_mean)
    assert best["score"] == 0. and all(row["score"] == 0. for row in report["candidates"])
    forced = physical_projection(torch.tensor(target_mean), torch.tensor(mean), torch.tensor(scale))
    assert forced.valid.all()
    forced_rmse = np.sqrt(np.mean((forced.x.numpy() - target_mean) ** 2))
    assert forced_rmse > .1
    # P2's single-cell projection stays applicable to actual cell predictions.
    projected_cells = physical_projection(torch.tensor(cells), torch.tensor(mean), torch.tensor(scale))
    np.testing.assert_allclose(projected_cells.x.numpy().mean(0), target_mean[0], atol=1e-12)


def test_raw_mean_checks_finiteness_without_single_cell_constraints():
    expected = np.array([[-1000., 1000.]])
    points, valid = raw_mean_points(np.zeros((1, 2)), expected)
    np.testing.assert_array_equal(points, expected)
    assert valid.tolist() == [True]
    invalid, valid = raw_mean_points(np.zeros((1, 2)), [[float("inf"), 1.]])
    assert valid.tolist() == [False] and np.isnan(invalid).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional for tiny numerical parity")
def test_cuda_keeps_float64_and_matches_cpu_small_solution():
    rng = np.random.default_rng(71)
    inputs, targets = rng.normal(size=(8, 6)), rng.normal(size=(8, 3))
    cpu, cuda = fit_ridge(inputs, targets, .01), fit_ridge(inputs, targets, .01, "cuda:0")
    assert cuda.dual.dtype == torch.float64 and cuda.dual.device.type == "cuda"
    np.testing.assert_allclose(cuda.predict(inputs), cpu.predict(inputs), rtol=1e-10, atol=1e-10)
    x = (inputs - inputs.mean(0)) / inputs.std(0)
    coefficients = np.linalg.solve(x.T @ x + .01 * np.eye(x.shape[1]), x.T @ (targets - targets.mean(0)))
    np.testing.assert_allclose(cuda.predict(inputs), x @ coefficients + targets.mean(0), rtol=1e-10, atol=1e-10)
