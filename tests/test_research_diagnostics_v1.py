"""Small deterministic fixtures; never run real-data diagnostics or fit a model."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.research import audit
from lung.engine import diagnostics
from lung.engine.config import config_hash, file_hash, StageUnavailable
from lung.engine.modular_budget import StagePaused
from lung.rl import stream_seed
from tests.test_world_execution_v1 import world_factory
from lung.research.sciplex_diagnostics import accumulate_development, cycle_diagnostics, CYCLE, LAYOUT
from lung.research.tidyi_diagnostics import (
    LABELS, MODELS, EXPRESSION_MODELS, binary_metrics, constrained_moment, mean_decomposition, mixtures, run_tidyi,
    normalize_counts, ordered_probabilities, require_gene_order, validate_gene_axis,
)


def test_gene_identity_and_order_cannot_be_silently_changed():
    frame = pd.DataFrame({"common_gene_index": [0, 1], "ensembl_id": ["a", "b"]})
    validate_gene_axis(frame, 2)
    with pytest.raises(ValueError, match="order"):
        validate_gene_axis(frame.iloc[::-1], 2)
    with pytest.raises(ValueError, match="order"):
        require_gene_order(["b", "a"], ["a", "b"])
    with pytest.raises(ValueError):
        validate_gene_axis(frame.assign(ensembl_id=["a", "a"]), 2)


def test_csv_column_order_uses_original_last_two_classes_as_positive():
    frame = pd.DataFrame({"true_state": list(LABELS), "p_PI+APC+": [.1, .2, .7],
                          "p_PI-APC+": [.1, .7, .2], "p_PI-APC-": [.8, .1, .1]})
    np.testing.assert_allclose(ordered_probabilities(frame), [[.8, .1, .1], [.1, .7, .2], [.1, .2, .7]])
    result = binary_metrics(frame)
    assert result["positive_n"] == 2
    assert result["auroc"] == 1
    assert result["brier"] == pytest.approx(.02)
    assert result["class_recall"] == dict.fromkeys(LABELS, 1.)
    prior = frame.copy()
    for label in LABELS:
        prior["p_" + label] = 1 / 3
    assert binary_metrics(prior)["auroc"] == .5
    with pytest.raises(ValueError):
        ordered_probabilities(frame.assign(true_state="new biology"))


def test_constrained_moment_interior_boundary_and_singular():
    matrix = np.array([[.8, .1, .2], [.1, .8, .1], [.1, .1, .7]])
    expected = np.array([.2, .3, .5])
    np.testing.assert_allclose(constrained_moment(matrix, matrix @ expected), expected, atol=1e-12)
    np.testing.assert_allclose(constrained_moment(np.eye(3), [-1, .8, 1.2]), [0, .3, .7], atol=1e-12)
    result = constrained_moment(np.ones((3, 3)) / 3, [.2, .3, .5])
    assert min(result) >= 0
    assert result.sum() == pytest.approx(1)


def test_mixture_pools_disjoint_and_66_exact_proportions():
    states = list(LABELS) * 2
    frame = pd.DataFrame({"cell_id": list("abcdef"), "true_state": states})
    for i, label in enumerate(LABELS):
        frame["p_" + label] = np.tile(np.eye(3)[:, i], 2)
    metadata = pd.DataFrame({"cell_id": list("abcdef"), "Well": ["A2"] * 3 + ["A3"] * 3})
    summary, table = mixtures(frame, metadata)
    assert summary["estimation_n"] == summary["mixture_pool_n"] == 3
    assert len(table) == 66
    assert summary["moment_positive_mae"] < 1e-12
    np.testing.assert_allclose(table[["pi_negative", "pi_middle", "pi_last"]].sum(axis=1), 1)
    with pytest.raises(ValueError, match="Every class"):
        mixtures(frame, metadata.assign(Well="A2"))


def test_normalization_full_axis_and_exact_mean_identity():
    counts = np.array([[1, 3], [2, 2], [3, 1]] * 2)
    dense = normalize_counts(counts)
    np.testing.assert_allclose(dense, normalize_counts(sparse.csr_matrix(counts)).toarray())
    np.testing.assert_allclose(dense[0], np.log1p([2500, 7500]))
    expression = dense.copy()
    expression[3:] += [.1, .2]
    result, _ = mean_decomposition(expression, np.array([8] * 3 + [1] * 3), np.array(list(LABELS) * 2))
    assert result["composition_rms"] == 0
    assert result["residual_rms"] == pytest.approx(np.sqrt(.025))
    assert result["max_identity_residual"] < 1e-14
    with pytest.raises(ValueError, match="Zero"):
        normalize_counts([[0, 0]])


def test_only_d72_qc_rows_enter_computation_even_in_shared_shards(tmp_path):
    meta = pd.DataFrame({"catalog_number": ["S1007", "S1103", "S1085", "S1007", "S1007"],
                         "time_point": [72, 72, 72, 24, 72], "author_hash_qc": [True, True, True, True, False],
                         "shard_id": [0] * 5, "row_in_shard": list(range(5)),
                         "common_totals_umi": [4, -2, -2, -2, -2], "dose_uM": [.01] * 5,
                         "replicate": ["rep1"] * 5, "culture_plate": ["plate50"] * 5})
    # Invalid expressions poison excluded rows: normalization would fail if used.
    matrix = sparse.csr_matrix([[1, 3], [-1, -1], [-1, -1], [-1, -1], [-1, -1]])
    seen = []
    def loader(path):
        seen.append(path.name)
        return matrix
    shards = [{"shard_id": 0, "common": {"file": "shared.npz", "rows": 5}},
              {"shard_id": 1, "common": {"file": "CE_only_must_not_open.npz", "rows": 1}}]
    records, rows = accumulate_development(meta, shards, tmp_path, lambda p, expected=None: p, 2, loader)
    assert seen == ["shared.npz"]
    assert rows.row_in_shard.tolist() == [0]
    assert rows.catalog_number.tolist() == ["S1007"]
    np.testing.assert_allclose(next(iter(records.values()))[1], np.log1p([2500, 7500]))


def test_cycle_additive_null_and_missing_low_recovery():
    records = {}
    for dose, rep, sign in CYCLE:
        plate = LAYOUT[(rep, dose)]
        mean = np.array([dose, dose * 2]) + int(plate[-2:])
        values = np.stack([mean - .1, mean + .1])
        records[("S1007", dose, rep, plate)] = (2, values.sum(0), (values ** 2).sum(0))
    wells, cycles = cycle_diagnostics(records, ["S1007"], 2)
    assert len(wells) == 8
    assert cycles.iloc[0].cycle_rms < 1e-12
    assert cycles.iloc[0].all_wells_at_least_two
    key = next(iter(records))
    low = {**records, key: (1, np.ones(2), np.ones(2))}
    wells, cycles = cycle_diagnostics(low, ["S1007"], 2)
    assert cycles.iloc[0].complete_cycle
    assert not cycles.iloc[0].all_wells_at_least_two
    assert cycles.iloc[0].cell_independent_noise_rms is None
    missing = {k: v for k, v in records.items() if k != key}
    wells, cycles = cycle_diagnostics(missing, ["S1007"], 2)
    assert wells.hash_qc_cell_count.eq(0).sum() == 1
    assert cycles.iloc[0].cycle_rms is None


def test_output_refuses_overwrite_and_failure_never_complete(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/audit_research_progress.py").write_text("fixture", encoding="utf-8")
    output = root / "experiments/research-diagnostics/new"
    def failure(root, tracker):
        raise ValueError("fixture failed")
    monkeypatch.setattr(audit, "run_tidyi", failure)
    with pytest.raises(ValueError, match="fixture failed"):
        audit.run_audit(root, output, "tidyi")
    assert not (output / "report.json").exists()
    with pytest.raises(FileExistsError):
        audit.run_audit(root, output, "tidyi")
    with pytest.raises(ValueError, match="new run directory"):
        audit.run_audit(root, root / "08-datasets/new", "tidyi")


def test_report_complete_only_after_tables_and_input_hashes(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/audit_research_progress.py").write_text("fixture", encoding="utf-8")
    source = tmp_path / "fixture.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    def diagnostic(root, tracker):
        tracker(source)
        return {"no_model_fit": True}, {"values": pd.DataFrame({"a": [1]})}
    monkeypatch.setattr(audit, "run_tidyi", diagnostic)
    output = tmp_path / "experiments/research-diagnostics/success"
    report = json.loads(audit.run_audit(tmp_path, output, "tidyi").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["no_model_fit"] and report["formal_training_started"] is False
    assert report["inputs"][0]["sha256"] == audit.sha256(source)
    assert report["artifacts"][0]["sha256"] == audit.sha256(output / "values.csv")
    assert not (output / "report.json.tmp").exists()
    assert report["seed"] is None


def test_input_hash_mismatch_or_mutation_is_rejected(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"fixture")
    tracker = audit.InputTracker()
    with pytest.raises(ValueError, match="SHA256"):
        tracker(source, "0" * 64)
    source.write_bytes(b"changed fixture")
    with pytest.raises(RuntimeError, match="changed"):
        tracker.verify_unchanged()


def test_cli_help_exits_without_creating_output(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/audit_research_progress.py"
    result = subprocess.run([sys.executable, str(script), "--project-root", str(tmp_path), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--only {tidyi,sciplex,all}" in result.stdout
    assert not list(tmp_path.iterdir())


def test_all_five_historical_models_reported_but_only_expression_mixtures(tmp_path):
    base = tmp_path / "08-datasets/TIDYI/E-MTAB-16354"
    sources = base / "audit_sources"
    sources.mkdir(parents=True)
    genes_path = tmp_path / "08-datasets/LungCancer_A549_Integration/prepared/v1"
    genes_path.mkdir(parents=True)
    width = 32461
    genes = np.array([f"GENE{i}" for i in range(width)])
    pd.DataFrame({"common_gene_index": np.arange(width), "ensembl_id": genes,
                  "tidyi_column": np.arange(width)}).to_parquet(genes_path / "gene_map.parquet")
    ids = np.array([f"cell{i}" for i in range(12)])
    labels = np.array(list(LABELS) * 4)
    plates = np.array([1] * 6 + [8] * 6)
    pd.DataFrame({"cell_id": ids, "cell_state": labels, "Plate": plates,
                  "Well": (["A2"] * 3 + ["A3"] * 3) * 2}).to_csv(sources / "human_ss3_A549_metadata.csv", index=False)
    counts = np.ones((12, width), dtype=np.float32)
    np.savez(sources / "A549_author_collection_counts.npz", expression=counts,
             cell_ids=ids, gene_ids=genes, plates=plates, states=labels)
    rows = []
    for i in range(12):
        for model in MODELS:
            rows.append({"cell_id": ids[i], "true_state": labels[i], "model": model,
                         "test_plate": int(plates[i]), "train_plate": 9 - int(plates[i]),
                         "direction": "primary" if plates[i] == 1 else "reverse",
                         **{"p_" + label: .8 if label == labels[i] else .1 for label in LABELS}})
    pd.DataFrame(rows).to_csv(base / "A549_cross_plate_out_of_sample_predictions.csv", index=False)
    report, tables = run_tidyi(tmp_path, audit.InputTracker())
    assert len(report["binary_metrics"]) == 10
    assert set(report["binary_metric_models"]) == {"prior", "well_position_only", "qc_only",
                                                 "expression_uncalibrated", "expression_sigmoid_calibrated"}
    assert len(report["mixture_diagnostics"]) == 4
    assert {r["model"] for r in report["mixture_diagnostics"]} == set(EXPRESSION_MODELS)
    assert len(tables["tidyi_binary_metrics"]) == 10
    assert all(len(t) == 66 for k, t in tables.items() if k.startswith("tidyi_mixtures_"))


def test_explicit_population_scoring_is_bounded_and_cached_by_n_and_steps(world_factory, monkeypatch):
    owner, _, _, _ = world_factory(2)
    root = owner.roots("T")[0]
    sequence, seeds = (next(iter(owner.actions)),), (11, 7, 23, 7, 31)
    calls, original = [], owner._score_seed_batch

    def record(root, sequence, seeds, n, steps):
        calls.append((tuple(seeds), n, steps))
        return original(root, sequence, seeds, n, steps)

    monkeypatch.setattr(owner, "_score_seed_batch", record)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = torch.tensor([owner.score_one(root, sequence, seed, n=3, ode_steps=3)
                                 for seed in seeds], dtype=torch.float64)
        actual = owner.score(root, sequence, seeds, n=3, ode_steps=3)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        assert [len(row[0]) for row in calls] == [4, 1]
        assert all(row[1:] == (3, 3) for row in calls)
        owner.score(root, sequence, seeds, n=5, ode_steps=3)
        owner.score(root, sequence, seeds, n=3, ode_steps=2)
        assert {row[1:] for row in calls} == {(3, 3), (5, 3), (3, 2)}
        previous = len(calls)
        torch.testing.assert_close(owner.score(root, sequence, seeds, n=3, ode_steps=3),
                                   actual, rtol=0, atol=0)
        assert len(calls) == previous
        key = next(key for key in owner._score_cache.values if key[-3:] == (3, 3, 23))
        owner._score_cache.values.pop(key)
        torch.testing.assert_close(owner.score(root, sequence, seeds, n=3, ode_steps=3),
                                   actual, rtol=0, atol=0)
        assert calls[previous:] == [(seeds[:4], 3, 3)]
        for n, steps in ((0, 3), (True, 3), (3, 0), (3, True)):
            with pytest.raises(ValueError):
                owner.score(root, sequence, seeds, n=n, ode_steps=steps)
    finally:
        torch.set_num_threads(threads)


class NumericalMonitor:
    def __init__(self, directory, limit=None):
        self.experiment, self.limit, self.calls = directory, limit, 0

    def stop_check(self, *_):
        self.calls += 1
        return self.limit is not None and self.calls > self.limit

    def progress(self, iterable, **_):
        return iterable


@pytest.mark.parametrize("factor,passed", [(1e-6, True), (1e-4, False)])
def test_population_batches_keep_registration_seed_order_and_resume(tmp_path, monkeypatch, factor, passed):
    from lung.engine import auxiliary
    monkeypatch.setattr(auxiliary, "calibration_audit_binding", lambda *args: {"unit_test_only": True})
    calls = []

    class World:
        config, data = {"synthetic": False}, SimpleNamespace()
        biological_identity, score_batch_size = "unit-test-world", 4
        registry = SimpleNamespace(candidates=[SimpleNamespace(action_id="a")])
        protocol = SimpleNamespace(position=lambda registry, prefix: (None, None, bool(prefix)),
                                   legal_mask=lambda registry, prefix: torch.tensor([True]))

        def roots(self, role):
            assert role == "V"
            return [SimpleNamespace(root_id="V-root")]

        def score(self, root, sequence, seeds, *, n):
            assert (tmp_path / "population_preregistration.json").is_file()
            assert sequence == ("a",)
            calls.append((tuple(seeds), n))
            return torch.tensor([.4 + (seed % 7) * 1e-5 + n * factor for seed in seeds],
                                dtype=torch.float64)

    with pytest.raises(StagePaused):
        diagnostics.population_diagnostic(World(), NumericalMonitor(tmp_path, 5), {})
    cache = tmp_path / f"population-{config_hash('V-root')[:16]}.json"
    assert len(json.loads(cache.read_text())) == 5
    calls.clear()
    if passed:
        diagnostics.population_diagnostic(World(), NumericalMonitor(tmp_path), {})
    else:
        with pytest.raises(StageUnavailable, match="POPULATION_RESOLUTION_UNRESOLVED"):
            diagnostics.population_diagnostic(World(), NumericalMonitor(tmp_path), {})
    expected_seeds = [stream_seed(20260923, "population-size", "V-root", repeat)
                      for repeat in range(1024)]
    expected = [[.4 + (seed % 7) * 1e-5 + n * factor for n in (512, 1024, 2048)]
                for seed in expected_seeds]
    np.testing.assert_array_equal(json.loads(cache.read_text()), expected)
    for n in (512, 1024, 2048):
        assert [seed for seeds, size in calls if size == n for seed in seeds] == expected_seeds[4:]
    assert all(len(seeds) <= 4 for seeds, _ in calls)
    report = json.loads((tmp_path / "population_resolution.json").read_text())
    assert report["repeats"] == 1024 and report["sizes"] == [512, 1024, 2048]
    assert report["adjacent_tolerance"] == .01 and report["passed"] is passed
    assert json.loads((tmp_path / "population_batch_layout.json").read_text())["batch_size"] == 4


class PointwiseAuditModel:
    def __init__(self, mode="observed"):
        self.config, self.calls = SimpleNamespace(reference_mode=mode), []

    def encode_condition(self, **kwargs):
        return kwargs["dose"][:, None]

    def sample_noise(self, reference, generator):
        return torch.randn((len(reference), 3), generator=generator)

    def sample(self, reference, condition, *, noise, steps, checkpoint):
        self.calls.append((reference.clone(), noise.clone(), steps))
        if self.config.reference_mode == "population_mean":
            reference = reference.mean(0, keepdim=True).expand_as(reference)
        return reference + noise[:, :2].abs() * .01 + condition * .001


def audit_sample(index):
    return {"control": np.array([[.2 + index * .01, .4 + index * .01]], np.float32),
            "fingerprint": np.array([1., index % 2], np.float32), "dose": float(index),
            "duration": float(index % 3 + 1), "background_id": 0, "source_id": 0, "protocol_id": 0}


@pytest.mark.parametrize("mode,batch_lengths", [("observed", [3, 2]), ("population_mean", [1] * 5)])
def test_ode_audit_batch_keeps_each_noise_condition_and_sample_order(mode, batch_lengths):
    data = SimpleNamespace(mu=np.zeros(2), scale=np.ones(2), n_genes=2)
    samples = [(index, audit_sample(index)) for index in (6, 2, 6, 11, 13)]
    scalar, batched = PointwiseAuditModel(mode), PointwiseAuditModel(mode)
    expected = diagnostics.ode_audit_batch(scalar, data, samples, device="cpu", member=2,
                                         steps=128, batch_size=1)
    actual = diagnostics.ode_audit_batch(batched, data, samples, device="cpu", member=2,
                                       steps=128, batch_size=3)
    noises = torch.cat([torch.randn((1, 3), generator=torch.Generator().manual_seed(
        stream_seed(20260923, "ODE", 2, index))) for index, _ in samples])
    torch.testing.assert_close(torch.cat([row[1] for row in batched.calls]), noises, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(actual), torch.cat(expected), rtol=0, atol=0)
    assert [len(row[0]) for row in batched.calls] == batch_lengths
    assert all(row[2] == 128 for row in batched.calls)
    assert all(row.untyped_storage().nbytes() == row.numel() * row.element_size() for row in actual)
    with pytest.raises(ValueError):
        diagnostics.ode_audit_batch(batched, data, samples, device="cpu", member=2, steps=128, batch_size=0)


def test_ode_audit_batches_resume_existing_per_sample_cache(tmp_path, monkeypatch):
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"unit-test-only")
    data = SimpleNamespace(identity="unit-test", n_genes=2, mu=np.zeros(2), scale=np.ones(2),
        condition_ids=lambda role: ["V-0", "V-1"],
        sample_controls=lambda key, *_: audit_sample(int(key[-1])))
    model = PointwiseAuditModel()
    monkeypatch.setattr(diagnostics, "load_response", lambda *args: model)
    config = {"device": "cpu", "synthetic": False,
              "response": {"inference_ode_steps": 128, "inference_microbatch": 4}}
    pipeline = {"response": [str(weight)]}
    with pytest.raises(StagePaused):
        diagnostics.resolve_ode_steps(data, config, NumericalMonitor(tmp_path, 3), pipeline)
    assert len(list(tmp_path.glob("ode_progress-*/member-0/steps-128/sample-*.pt"))) == 3
    assert not (tmp_path / "ode_audit.json").exists()
    result = diagnostics.resolve_ode_steps(data, config, NumericalMonitor(tmp_path), pipeline)
    report = json.loads(Path(result["ode_audit"]).read_text())
    assert report["n_fixed_V_noise"] == 256 and report["tolerance"] == .001
    assert report["selected_steps"] == 128 and report["passed"]
    assert max(len(row[0]) for row in model.calls) == 4
    assert len(model.calls) < 512
    for steps in (128, 256):
        paths = list(tmp_path.glob(f"ode_progress-*/member-0/steps-{steps}/sample-*.pt"))
        assert len(paths) == 256
        for path in paths:
            record = torch.load(path, map_location="cpu", weights_only=False)
            assert record["expression"].shape == (1, 2) and record["steps"] == steps


def test_legacy_scalar_ode_cache_keeps_its_layout_and_rejects_changed_binding(tmp_path, monkeypatch):
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"unit-test-only")
    data = SimpleNamespace(identity="unit-test", n_genes=2, mu=np.zeros(2), scale=np.ones(2),
        condition_ids=lambda role: ["V-0"], sample_controls=lambda *_: audit_sample(0))
    model = PointwiseAuditModel()
    monkeypatch.setattr(diagnostics, "load_response", lambda *args: model)
    identity = config_hash({"response": [file_hash(weight)], "data": data.identity})
    folder = tmp_path / f"ode_progress-{identity[:20]}" / "member-0"
    cached = folder / "steps-128/sample-0.pt"
    cached.parent.mkdir(parents=True)
    output = diagnostics.ode_audit_batch(model, data, [(0, audit_sample(0))],
                                       device="cpu", member=0, steps=128)[0]
    torch.save({"identity": identity, "index": 0, "steps": 128, "member": 0, "expression": output}, cached)
    model.calls.clear()
    config = {"device": "cpu", "synthetic": False,
              "response": {"inference_ode_steps": 128, "inference_microbatch": 4}}
    pipeline = {"response": [str(weight)]}
    with pytest.raises(StagePaused):
        diagnostics.resolve_ode_steps(data, config, NumericalMonitor(tmp_path, 2), pipeline)
    assert [len(row[0]) for row in model.calls] == [1]
    layout = folder / "batch_layout.json"
    record = json.loads(layout.read_text())
    assert record["batch_size"] == 1
    layout.write_text(json.dumps({**record, "identity": "changed"}))
    with pytest.raises(StageUnavailable, match="BATCH_LAYOUT_IDENTITY_MISMATCH"):
        diagnostics.resolve_ode_steps(data, config, NumericalMonitor(tmp_path), pipeline)
