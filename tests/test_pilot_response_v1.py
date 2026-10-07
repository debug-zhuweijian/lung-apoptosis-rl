"""Temporary tiny checkpoints and synthetic populations, never real evaluation."""

import json
from dataclasses import asdict
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

from lung.engine.config import config_hash, load_config
from lung.engine.pilot import make_plan, evaluate_plan, summarize, point_metrics, load_shifts
from lung.engine.pilot_response import (inspect_response_set, shared_inference_steps, mixture_metrics,
                                        kernel_sum, noise_seed, evaluate_response_sets)
from lung.engine.response import RESPONSE_RNG_POLICY, load_response
from lung.engine.synthetic import smoke_config
from lung.models import ModelConfig, build_response_model, physical_projection, rbf_mmd, rbf_kernel
from test_pilot_interfaces_v1 import tiny_v_data


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_checkpoints(data, method="P3_cfm_mmd", run_id="tiny-p3"):
    root = data.reader.project_root
    data.synthetic = True
    config = smoke_config(load_config())
    cfg = {**config["response"], "members": 5, "mmd_weight": 1. if method == "P3_cfm_mmd" else 0.,
           "model": {**config["response"]["model"], "n_genes": data.n_genes}}
    config = {**config, "response": cfg}
    shape = ModelConfig(**cfg["model"])
    experiment = root / "experiments" / run_id
    experiment.mkdir(parents=True)
    (experiment / "run_manifest.json").write_text(json.dumps({"run_id": run_id, "synthetic": True,
        "config_hash": config_hash(config), "status": "complete"}), encoding="utf-8")
    (experiment / "data_manifest.json").write_text(json.dumps({"identity": data.identity}), encoding="utf-8")
    (experiment / "active_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    identity = config_hash({"data": data.identity, "response": cfg, "task": "RESPONSE-A549-V1",
                            "rng_policy": RESPONSE_RNG_POLICY})
    paths = []
    for member in range(5):
        torch.manual_seed(31 + member)
        model = build_response_model(shape, torch.as_tensor(data.neighbors))
        path = root / "checkpoints" / run_id / "response" / f"member_{member+1}" / "selected.pt"
        path.parent.mkdir(parents=True)
        # Epoch and counters are artificial checkpoint fixtures, not training evidence.
        torch.save({"schema_version": 1, "epoch": 1, "identity": identity, "model": model.state_dict(),
            "counters": {"optimizer_step": 1},
            "metadata": {"stage": "response", "member": member, "model_config": asdict(shape),
                "response_config": cfg, "rng_policy": RESPONSE_RNG_POLICY,
                "mmd_training": "computed" if cfg["mmd_weight"] else "not_computed",
                "data_identity": data.identity, "synthetic": True, "train_role": "T", "selection_role": "V"}}, path)
        paths.append(path)
    return paths


def test_five_member_real_identity_and_common_precision(tiny_v_data):
    data = tiny_v_data
    p3 = inspect_response_set(fixture_checkpoints(data)[::-1], data, "P3_cfm_mmd", root=data.reader.project_root)
    p4 = inspect_response_set(fixture_checkpoints(data, "P4_cfm_only", "tiny-p4"), data,
                              "P4_cfm_only", root=data.reader.project_root)
    assert [member["member"] for member in p3["members"]] == list(range(5))
    assert p3["gene_axis_sha256"] == p4["gene_axis_sha256"]
    assert shared_inference_steps([p3, {**p4, "inference_steps": 4}]) == 4
    altered = {**p4, "response_config": {**p4["response_config"], "seed": 1}}
    with pytest.raises(ValueError, match="PAIRED_CFM_CONFIG"):
        shared_inference_steps([p3, altered])


@pytest.mark.parametrize("change, expected", [
    ({"rng_policy": "old_shared_rng"}, "OBJECTIVE_OR_RNG"),
    ({"mmd_training": "not_computed"}, "OBJECTIVE_OR_RNG"),
    ({"member": 4}, "MEMBER_INDEX"),
    ({"data_identity": "other"}, "TRAINED_T_V"),
    ({"train_role": "E"}, "TRAINED_T_V"),
    ({"selection_role": "E"}, "TRAINED_T_V"),
])
def test_checkpoint_content_not_filename_controls_method(tiny_v_data, change, expected):
    data = tiny_v_data
    paths = fixture_checkpoints(data)
    saved = torch.load(paths[0], weights_only=False)
    torch.save({**saved, "metadata": {**saved["metadata"], **change}}, paths[0])
    with pytest.raises(ValueError, match=expected):
        inspect_response_set(paths, data, "P3_cfm_mmd", root=data.reader.project_root)


def test_missing_mixed_duplicate_and_wrong_objective_are_rejected(tiny_v_data):
    data = tiny_v_data
    first = fixture_checkpoints(data)
    other = fixture_checkpoints(data, run_id="other")
    for paths, label in ((first[:4], "FIVE_RESPONSE"), ([first[0]]*5, "FIVE_UNIQUE"),
                         ([*first[:4], other[4]], "SHARE_RUN")):
        with pytest.raises(ValueError, match=label):
            inspect_response_set(paths, data, "P3_cfm_mmd", root=data.reader.project_root)
    with pytest.raises(ValueError, match="OBJECTIVE_OR_RNG"):
        inspect_response_set(first, data, "P4_cfm_only", root=data.reader.project_root)


def test_ode_audit_can_raise_but_never_lower_inference_steps(tiny_v_data):
    data = tiny_v_data
    paths = fixture_checkpoints(data)
    selected = inspect_response_set(paths, data, "P3_cfm_mmd", root=data.reader.project_root)
    audit = data.reader.project_root / "experiments/tiny-p3/ode_audit.json"
    record = {"identity": config_hash({"response": [item["sha256"] for item in selected["members"]],
                                      "data": data.identity}), "passed": True, "selected_steps": 4}
    audit.write_text(json.dumps(record), encoding="utf-8")
    assert inspect_response_set(paths, data, "P3_cfm_mmd", root=data.reader.project_root)["inference_steps"] == 4
    audit.write_text(json.dumps({**record, "selected_steps": 1}), encoding="utf-8")
    assert inspect_response_set(paths, data, "P3_cfm_mmd", root=data.reader.project_root)["inference_steps"] == 2
    audit.write_text(json.dumps({**record, "identity": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="ODE_AUDIT"):
        inspect_response_set(paths, data, "P3_cfm_mmd", root=data.reader.project_root)


def populations():
    generator = torch.Generator().manual_seed(84)
    raw = [torch.randn((3, 6), generator=generator, dtype=torch.float64) * .1 for _ in range(5)]
    control = np.zeros((3, 6))
    target = torch.randn((4, 6), generator=generator, dtype=torch.float64).numpy() * .1
    return raw, control, target, np.full(6, 5.), np.ones(6)


def test_block_mmd_contains_all_cross_members_and_matches_dense_formula():
    raw, control, target, mu, scale = populations()
    actual = mixture_metrics(raw, control, target, mu, scale, .7, block_size=2)
    outputs = [physical_projection(x, torch.tensor(mu), torch.tensor(scale)).x for x in raw]
    y = torch.tensor(target)
    pp = sum((rbf_kernel(a, b, .7).sum() - rbf_kernel(a, b, .7).diagonal().sum()) / 6
             for a in outputs for b in outputs) / 25
    yy = (rbf_kernel(y, y, .7).sum() - rbf_kernel(y, y, .7).diagonal().sum()) / 12
    py = sum(rbf_kernel(x, y, .7).mean() for x in outputs) / 5
    assert actual["mmd_u"] == pytest.approx(float(pp + yy - 2*py), abs=1e-12)
    expected_mean = torch.stack(outputs).mean((0, 1))
    assert actual["mean_rmse"] == pytest.approx(float((expected_mean - y.mean(0)).square().mean().sqrt()))
    dense = kernel_sum(outputs[0], outputs[1], .7, block_size=1, exclude_block_diagonal=True)
    expected = rbf_kernel(outputs[0], outputs[1], .7)
    assert float(dense) == pytest.approx(float(expected.sum() - expected.diagonal().sum()), abs=1e-12)


def test_identical_members_reduce_to_n_block_u_and_preserve_negative_values():
    raw, control, _, mu, scale = populations()
    outputs = physical_projection(raw[0], torch.tensor(mu), torch.tensor(scale)).x
    actual = mixture_metrics([raw[0]]*5, control, outputs.numpy(), mu, scale, .7, block_size=1)
    expected = rbf_mmd(outputs, outputs, .7, unbiased=True)
    pooled = rbf_mmd(outputs.repeat(5, 1), outputs, .7, unbiased=True)
    assert actual["mmd_u"] == pytest.approx(float(expected), abs=1e-12)
    assert actual["mmd_u"] < 0 and abs(actual["mmd_u"] - float(pooled)) > 1e-7


def test_block_permutation_and_member_permutation_do_not_change_statistics():
    raw, control, target, mu, scale = populations()
    original = mixture_metrics(raw, control, target, mu, scale, .7, block_size=1)
    changed = mixture_metrics([raw[index][[2, 0, 1]] for index in [4, 2, 1, 3, 0]],
                               control[[2, 0, 1]], target[::-1].copy(), mu, scale, .7, block_size=2)
    for name in ("mean_rmse", "mmd_u", "output_valid_rate"):
        assert original[name] == pytest.approx(changed[name], abs=1e-12)
    for n, m in ((1, 4), (3, 1)):
        missing = mixture_metrics([value[:n] for value in raw], control[:n], target[:m], mu, scale, .7)
        assert missing["mean_rmse"] is not None and missing["mmd_u"] is None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable for the tiny inference check"))])
def test_shared_noise_population_and_heun_steps_across_methods(tiny_v_data, device):
    data = tiny_v_data
    sets = [inspect_response_set(fixture_checkpoints(data, method, name), data, method,
                                 root=data.reader.project_root)
            for method, name in (("P3_cfm_mmd", "p3"), ("P4_cfm_only", "p4"))]
    sets[1] = {**sets[1], "inference_steps": 4}
    plan = make_plan(data, [9], max_cells=2)
    key = plan["conditions"][0]
    plan = {**plan, "conditions": [key], "samples": [row for row in plan["samples"] if row["condition_id"] == key]}
    traces = []
    displays = []
    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix_str(self, value, **kwargs):
            displays.append(value)
        def close(self):
            pass
    def traced_load(path, source, device):
        model = load_response(path, source, device)
        original = model.sample
        def sample(reference, condition, **kwargs):
            traces.append((reference.cpu().clone(), kwargs["noise"].cpu().clone(), kwargs["steps"]))
            return original(reference, condition, **kwargs)
        model.sample = sample
        return model
    with patch("lung.engine.pilot_response.load_response", side_effect=traced_load):
        mixtures, members = evaluate_response_sets(data, plan, sets, device=device, block_size=1, progress=Progress)
    assert len(mixtures) == 2 and len(members) == 10
    assert {row["member"] for row in members} == set(range(5))
    for x, noise, steps in traces:
        torch.testing.assert_close(x, traces[0][0], rtol=0, atol=0)
        torch.testing.assert_close(noise, traces[0][1], rtol=0, atol=0)
        assert steps == 4
    assert all(row["noise_seed"] == noise_seed(plan["samples"][0]) for row in members)
    assert all(row["n_blocks"] == 2 and row["correlated_generated_outputs"] == 10 for row in mixtures)
    assert any("member=0" in value and "ODE" in value for value in displays)
    assert any("mixture kernel" in value for value in displays)
    baseline = evaluate_plan(data, plan, device=device)
    summary = summarize([*baseline, *mixtures], bootstrap_repeats=10)
    assert "P3_cfm_mmd_minus_P4_cfm_only/mmd_u" in summary["paired_comparisons"]
    assert summary["methods"]["P3_cfm_mmd"]["mmd_u"]["total_rows"] == 1


def test_p1_mean_point_na_is_not_a_distribution_failure(tiny_v_data):
    data = tiny_v_data
    plan = make_plan(data, [7], max_cells=2)
    points = {key: np.ones(data.n_genes) for key in plan["conditions"]}
    rows = evaluate_plan(data, plan, mean_predictions=points)
    summary = summarize(rows, bootstrap_repeats=10)
    metric = summary["methods"]["P1_ridge_mean"]["mmd_u"]
    assert metric["status"] == "not_applicable" and metric["missing_or_failed_rows"] == 0
    assert metric["not_applicable_rows"] == len(plan["samples"])
    assert all(row["status"] == "ok" for row in rows if row["method"] == "P1_ridge_mean")
    assert summary["methods"]["P1_ridge_mean"]["output_valid_rate"]["definitions"] == [
        "finite_mean_vector_only_not_cell_physical_validity"]
    assert summary["methods"]["P0_no_change"]["output_valid_rate"]["definitions"] == ["per_cell_physical_projection"]
    bad = point_metrics(np.full(data.n_genes, np.nan), np.ones((2, data.n_genes)))
    assert bad["mean_rmse"] is None and bad["output_valid_rate"] == 0.


def test_actual_tiny_ridge_artifact_enters_p1_p2_v_report(tiny_v_data):
    from lung.engine.pilot_ridge import run_ridge, load_mean_predictions, LAMBDAS
    data = tiny_v_data
    output = data.reader.project_root / "experiments" / "tiny-ridge"
    output.mkdir(parents=True)
    with patch.object(data.reader, "allowed_roles", frozenset({"fit", "selection"})):
        run_ridge(data, {"device": "cpu", "stream_batch_size": 13, "lambdas": list(LAMBDAS)}, output,
                   {"code_sha256": {"synthetic_fixture": "a"*64}})
    shifts, provenance = load_shifts(output / "v_delta.npz", data)
    points, mean_provenance = load_mean_predictions(output / "v_delta.npz", data)
    plan = make_plan(data, [32], max_cells=2)
    rows = evaluate_plan(data, plan, shifts, mean_predictions=points,
                         shift_control_scope=provenance["metadata"]["control_summary_scope"])
    assert {row["method"] for row in rows} == {"P0_no_change", "P1_ridge_mean", "P2_mean_shift"}
    for row in rows:
        if row["method"] == "P1_ridge_mean":
            assert row["control_summary_scope"] == "full_unique_V_control_pool_mean"
            assert row["mmd_u"] is None and "mmd_u" in row["not_applicable"]
        elif row["method"] == "P2_mean_shift":
            assert row["control_summary_scope"] == "sampled_V_controls_plus_external_delta"
            assert row["delta_control_summary_scope"] == "full_unique_V_control_pool_mean"
    assert mean_provenance["control_summary_scope"] == "full_unique_V_control_pool_mean"
