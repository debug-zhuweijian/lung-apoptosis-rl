"""Small synthetic regressions only; no real training or biological claims."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lung.engine.checkpoint import seed_all
from lung.engine.config import load_config
from lung.engine.data import EngineData
from lung.engine.pilot import (METRICS, baseline_metrics, make_plan, evaluate_plan, load_shifts,
                               require_v_only, summarize)
from lung.engine.response import (RESPONSE_RNG_POLICY, response_checkpoint, response_rngs,
                                  train_condition, train_response)
from lung.engine.synthetic import SyntheticData, smoke_config
from lung.models import ResponseModel, physical_projection, rbf_mmd
from test_training_data_v1 import tiny_bundle
from lung.data import prepare_training
from lung.data.preprocess import standardized


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_training():
    config = smoke_config(load_config())
    config["response"]["members"] = 1
    data = SyntheticData(config)
    from lung.models import ModelConfig
    shape = ModelConfig(**config["response"]["model"])
    seed_all(13)
    model = ResponseModel(shape, torch.as_tensor(data.neighbors))
    return config, data, model


def test_zero_weight_bypasses_second_sampling_ode_and_mmd():
    config, data, model = small_training()
    cfg = {**config["response"], "mmd_weight": 0.}
    cfg.pop("mmd_samples")
    cfg.pop("train_ode_steps")
    with patch.object(data, "sample_response", wraps=data.sample_response) as sample:
        with patch("lung.engine.response.exact_mmd_backward", side_effect=AssertionError("must bypass")):
            result = train_condition(model, data, "T-0", np.random.default_rng(13), cfg, "cpu", 1)
    assert sample.call_count == 1
    assert result["mmd_loss"] is None and result["weighted_mmd_loss"] is None
    assert result["mmd_computed"] == 0 and result["loss"] == result["cfm_loss"]


def test_shared_condition_cell_ot_tau_epsilon_streams_match_with_or_without_mmd():
    config, data, initial = small_training()
    records = []
    for weight in (0., 1.):
        model = copy.deepcopy(initial)
        seed_all(37)
        cfm_rng, extra_rng, extra_torch = response_rngs(37, "cpu")
        cells, cfm_inputs = [], []
        original_sample, original_loss = data.sample_response, model.cfm_loss

        def sample(key, n, rng, role):
            value = original_sample(key, n, rng, role)
            if rng is cfm_rng:
                cells.append((key, value["control"].copy(), value["target"].copy()))
            return value

        def loss(reference, target, condition):
            epsilon = torch.randn_like(target)
            tau = torch.rand(len(target), device=target.device)
            cfm_inputs.append(tuple(value.detach().clone() for value in (reference, target, epsilon, tau)))
            return original_loss(reference, target, condition, noise=epsilon, tau=tau)

        keys = list(data.iter_response_epoch("T", 37)) * 2
        with patch.object(data, "sample_response", side_effect=sample), patch.object(model, "cfm_loss", side_effect=loss):
            metrics = [train_condition(model, data, key, cfm_rng,
                       {**config["response"], "mmd_weight": weight}, "cpu", 1,
                       mmd_rng=extra_rng, mmd_generator=extra_torch) for key in keys]
        records.append((keys, cells, cfm_inputs, cfm_rng.bit_generator.state,
                        torch.get_rng_state().clone(), metrics, extra_torch.get_state().clone()))
    a, b = records
    assert a[0] == b[0] and a[3] == b[3]
    for left, right in zip(a[1], b[1]):
        assert left[0] == right[0]
        for x, y in zip(left[1:], right[1:]):
            np.testing.assert_array_equal(x, y)
    for left, right in zip(a[2], b[2]):
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(a[4], b[4], rtol=0, atol=0)
    assert not torch.equal(a[6], b[6])
    assert [m["cfm_loss"] for m in a[5]] == [m["cfm_loss"] for m in b[5]]


def test_extra_branch_weight_and_stream_requirements():
    config, data, model = small_training()
    shared, extra, torch_extra = response_rngs(14, "cpu")
    cfg = {**config["response"], "mmd_weight": .25}
    with pytest.raises(ValueError, match="separate"):
        train_condition(model, data, "T-0", shared, cfg, "cpu", 2)
    with patch("lung.engine.response.exact_mmd_backward", return_value=4.) as mmd:
        result = train_condition(model, data, "T-0", shared, cfg, "cpu", 2,
                                 mmd_rng=extra, mmd_generator=torch_extra)
    assert mmd.call_args.kwargs["scale"] == .125
    assert mmd.call_args.kwargs["noise"].shape == (cfg["mmd_samples"], data.n_genes)
    assert result["weighted_mmd_loss"] == 1.
    assert result["loss"] == result["cfm_loss"] + 1.


def test_checkpoint_policy_rejects_legacy_resume(tmp_path):
    path = tmp_path / "legacy.pt"
    saved = {"schema_version": 1, "metadata": {}, "identity": "new"}
    torch.save(saved, path)
    with pytest.raises(ValueError, match="RESPONSE_RNG_POLICY_CHANGED"):
        response_checkpoint(path, "new")
    torch.save({**saved, "metadata": {"rng_policy": RESPONSE_RNG_POLICY}}, path)
    assert response_checkpoint(path, "new")["identity"] == "new"
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        response_checkpoint(path, "other")


def test_cfm_only_epoch_records_missing_mmd_and_resumes(tmp_path):
    config, data, _ = small_training()
    config["response"]["mmd_weight"] = 0.
    config["progress"] = False
    monitor = Mock()
    monitor.weights = tmp_path / "weights"
    monitor.logdir = tmp_path / "logs"
    monitor.config = config
    monitor.progress.side_effect = lambda iterable=None, **kw: iterable if iterable is not None else Mock()
    paths = train_response(data, config, monitor)
    saved = torch.load(paths[0], weights_only=False)
    assert saved["metadata"]["rng_policy"] == RESPONSE_RNG_POLICY
    assert saved["metadata"]["mmd_training"] == "not_computed"
    assert monitor.record.call_args.args[2]["train/mmd_loss"] is None
    assert monitor.record.call_args.args[2]["val/mmd_loss"] is not None
    # Current epoch-complete checkpoint loads without silently changing its objective.
    assert train_response(data, config, monitor, resume=True) == paths


@pytest.fixture
def tiny_v_data(tmp_path):
    tiny_bundle(tmp_path)
    prepare_training(tmp_path)
    with patch("lung.engine.data.ROOT", tmp_path):
        return EngineData({"data": str(tmp_path / "configs/dataset.yaml")}, roles=("V",))


def test_v_sampling_has_no_replacement_reads_only_v_and_shares_observations(tiny_v_data):
    data = tiny_v_data
    before = []
    original = data.reader.rows.read
    def check_rows(frame):
        before.extend(frame.role.unique().tolist())
        assert set(frame.role).issubset({"selection"})
        return original(frame)
    with patch.object(data.reader.rows, "read", side_effect=check_rows):
        plan = make_plan(data, [1, 2])
        assert before == []  # Sampling identity is fixed before expression is read.
        for sample in plan["samples"]:
            for population in ("target", "control"):
                ids = sample[f"{population}_cell_ids"]
                assert len(ids) == len(set(ids)) == sample[f"{population}_unique_n"]
        shifts = {key: np.zeros(data.n_genes) for key in plan["conditions"]}
        rows = evaluate_plan(data, plan, shifts)
    assert set(before) == {"selection"}
    assert len(rows) == 2 * len(plan["samples"])
    for p0, p2 in zip(rows[::2], rows[1::2]):
        for metric in METRICS:
            assert p0[metric] == p2[metric]
    assert make_plan(data, [1, 2]) == plan
    with pytest.raises(PermissionError):
        data.reader.cell_metadata("anything", role="final_test")
    with pytest.raises(PermissionError):
        evaluate_plan(data, {**plan, "role": "E"})
    with patch.object(data.reader, "allowed_roles", frozenset({"fit", "selection"})):
        with pytest.raises(PermissionError, match="V_ONLY"):
            require_v_only(data)


def test_main_statistics_and_na_are_explicit():
    x = np.array([[1., 2., 3.], [2., 3., 4.]])
    mean, scale = np.full(3, 3.), np.ones(3)
    y = x + .2
    result = baseline_metrics(x, y, mean, scale, .7)
    predicted = physical_projection(torch.tensor(x), torch.tensor(mean), torch.tensor(scale)).x
    assert result["mean_rmse"] == pytest.approx(float((predicted.mean(0) - torch.tensor(y).mean(0)).square().mean().sqrt()))
    assert result["mmd_u"] == pytest.approx(float(rbf_mmd(predicted, torch.tensor(y), .7, unbiased=True)))
    singleton = baseline_metrics(x[:1], y, mean, scale, .7)
    assert singleton["mean_rmse"] is not None and singleton["mmd_u"] is None
    assert singleton["variance_rmse"] is None
    invalid = baseline_metrics(x, y, mean, scale, .7, delta=np.full(3, -1000.))
    assert invalid["mean_rmse"] is None and invalid["mmd_u"] is None
    assert invalid["output_valid_rate"] == 0. and invalid["status"] == "invalid_prediction"
    # Identical finite empirical populations can have a negative U estimate.
    same = baseline_metrics(predicted.numpy(), predicted.numpy(), mean, scale, .7)
    assert same["mmd_u"] < 0


def test_p0_float32_reprojection_cannot_create_a_delta_direction():
    from scipy import sparse
    mean, scale = np.array([4.2, 6.1, 5.7]), np.array([.7, .9, 1.2])
    x = standardized(sparse.csr_matrix([[1., 4., 7.], [2., 5., 8.], [4., 2., 3.]]), mean, scale)
    y = standardized(sparse.csr_matrix([[8., 1., 2.], [5., 7., 4.]]), mean, scale)
    projection = physical_projection(torch.tensor(x, dtype=torch.float64), torch.tensor(mean), torch.tensor(scale)).x
    assert not torch.equal(projection, torch.tensor(x, dtype=torch.float64))
    p0 = baseline_metrics(x, y, mean, scale, .7)
    zero_shift = baseline_metrics(x, y, mean, scale, .7, delta=np.zeros(3))
    for report in (p0, zero_shift):
        assert report["delta_correlation"] is None
        assert report["na_reasons"]["delta_correlation"] == "constant_delta_vector"
        assert report["direction_agreement"] == 0.
        assert report["mean_rmse"] == pytest.approx(.9609934657057823)
        assert report["mmd_u"] == pytest.approx(-.0011507527086750402)


def write_shift(path, data, **changes):
    metadata = {"schema_version": 1, "model_family": "full_gene_ridge", "fit_role": "T",
                "selection_role": "V", "evaluation_role": "V", "data_identity": data.identity,
                "prediction_inputs": ["V_controls", "condition_metadata"],
                "fit_condition_ids": data.reader.role_counts.query("role == 'fit'").condition_id.tolist(),
                "fit_code_sha256": "a" * 64, "fit_artifact_sha256": "b" * 64}
    treated = set(data.reader.condition_table.query("condition_type == 'treated'").condition_id)
    metadata["fit_condition_ids"] = sorted(set(metadata["fit_condition_ids"]) & treated)
    metadata.update(changes)
    ids = data.condition_ids("V")[:1]
    np.savez(path, metadata_json=json.dumps(metadata), condition_ids=ids,
             gene_ids=data.reader.gene_ids, delta=np.zeros((1, data.n_genes)))


def test_shift_provenance_roles_and_partial_coverage(tiny_v_data, tmp_path):
    data, path = tiny_v_data, tmp_path / "delta.npz"
    write_shift(path, data)
    shifts, provenance = load_shifts(path, data)
    assert "not_training_audit" in provenance["verification"]
    rows = evaluate_plan(data, make_plan(data, [1, 2], max_cells=1), shifts)
    assert any(row["status"] == "missing_shift" for row in rows)
    summary = summarize(rows, bootstrap_repeats=10)
    assert summary["methods"]["P0_no_change"]["mmd_u"]["estimate"] is None
    coverage = summary["methods"]["P2_mean_shift"]["mean_rmse"]
    assert coverage["computed_rows"] < coverage["total_rows"]
    assert summary["paired_comparisons"]["P2_mean_shift_minus_P0/mmd_u"]["difference"] is None
    for bad in ({"fit_role": "V"}, {"fit_role": "E"}, {"evaluation_role": "E"},
                {"fit_condition_ids": data.condition_ids("V")}, {"prediction_inputs": ["V_response"]}):
        write_shift(path, data, **bad)
        with pytest.raises(PermissionError):
            load_shifts(path, data)


def test_missing_controls_remain_in_plan_and_coverage(tiny_v_data):
    data = tiny_v_data
    missing = data.condition_ids("V")[0]
    original = data.reader.controls
    without_control = original.loc[~original.condition_id.eq(missing)].copy()
    with patch.object(data.reader, "controls", without_control):
        plan = make_plan(data, [17])
        assert missing in plan["conditions"]
        sample = next(row for row in plan["samples"] if row["condition_id"] == missing)
        assert sample["control_unique_n"] == 0
        rows = evaluate_plan(data, plan)
    failed = next(row for row in rows if row["condition_id"] == missing)
    assert failed["status"] == "missing_population" and failed["generated_n"] is None
    assert all(failed[metric] is None for metric in METRICS)
    coverage = summarize(rows)["methods"]["P0_no_change"]["mean_rmse"]
    assert coverage["computed_rows"] < coverage["total_rows"]


def test_shift_data_identity_and_gene_axis_cannot_drift(tiny_v_data, tmp_path):
    data, path = tiny_v_data, tmp_path / "delta.npz"
    write_shift(path, data, data_identity="another_split")
    with pytest.raises(ValueError, match="DATA_IDENTITY"):
        load_shifts(path, data)
    write_shift(path, data)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    np.savez(path, **{**arrays, "gene_ids": arrays["gene_ids"][::-1]})
    with pytest.raises(ValueError, match="GENE_ORDER"):
        load_shifts(path, data)


def test_summary_uses_equal_chemical_entities_and_paired_bootstrap():
    rows = []
    for group, condition, value in (("g1", "a", 1.), ("g1", "b", 3.), ("g2", "c", 8.)):
        for seed in (1, 2):
            for method, offset in (("P0_no_change", 0.), ("P2_mean_shift", -1.)):
                rows.append({"group_id": group, "condition_id": condition, "seed": seed,
                             "method": method, **{name: value + offset for name in METRICS}})
    summary = summarize(rows, bootstrap_repeats=25)
    assert summary["methods"]["P0_no_change"]["mean_rmse"]["estimate"] == 5.
    comparison = summary["paired_comparisons"]["P2_mean_shift_minus_P0/mean_rmse"]
    assert comparison["difference"] == -1. and comparison["ci95"] == [-1., -1.]
    with pytest.raises(ValueError, match="positive"):
        summarize(rows, bootstrap_repeats=0)


def test_independent_configs_keep_full_model_and_shared_sampling():
    default = load_config()
    off, on = (load_config(f"configs/pilot_cfm_{name}.yaml") for name in ("only", "mmd"))
    assert (default["stage"], default["algorithm"], default["epochs"]) == ("all", "both", 1000)
    assert off["response"]["mmd_weight"] == 0 and on["response"]["mmd_weight"] == 1
    for config in (off, on):
        assert config["epochs"] == 1000 and config["response"]["members"] == 5
        assert config["response"]["model"]["n_genes"] == 32461
    assert {k: v for k, v in off["response"].items() if k != "mmd_weight"} == {
        k: v for k, v in on["response"].items() if k != "mmd_weight"}


def test_manual_report_rejects_e_config_before_opening_data(tmp_path):
    spec = importlib.util.spec_from_file_location("pilot_cli", ROOT / "scripts/report_response_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "e.yaml"
    config.write_text("role: E\n", encoding="utf-8")
    with patch("lung.engine.data.EngineData", side_effect=AssertionError("must not read data")):
        with pytest.raises(PermissionError, match="ONLY_SUPPORTS_V"):
            module.main(["--config", str(config), "--output", "experiments/not-created"])
