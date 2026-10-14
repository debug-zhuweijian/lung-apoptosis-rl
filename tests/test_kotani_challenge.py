"""Synthetic source/arithmetic fixtures only, never real Kotani model predictions."""

import json

import pandas as pd
import pytest

from scripts import build_kotani_challenge as builder
from scripts import evaluate_kotani_challenge as evaluation


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def refresh_inventory(source):
    save_json(source / "local_file_inventory.json", [
        {"file": name, "sha256": builder.sha(source / name), "bytes": (source / name).stat().st_size}
        for name in builder.SOURCE_FILES])


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "synthetic-source"
    source.mkdir()
    for name in builder.SOURCE_FILES:
        (source / name).write_text("SYNTHETIC engineering fixture, not a scientific source", encoding="utf-8")
    means, panels = [], []
    grid = [(builder.CONTROL, None), (builder.CONTROL, 24), (builder.PRETREATED, None),
            (builder.PRETREATED, 6), (builder.PRETREATED, 12), (builder.PRETREATED, 24)]
    for column, (state, hour) in enumerate(grid, 1):
        control = state == builder.CONTROL
        cid = "Kotani2024_A549_" + ("control" if control else "axitinib") + "_"
        cid += "no_ABT" if hour is None else f"ABT_{hour}h"
        common = {"condition_id": cid, "cell_line": "A549", "state": state,
            "pretreatment_compound": "none" if control else "axitinib",
            "pretreatment_dose_uM": 0 if control else 2.5, "pretreatment_duration_h": 72,
            "transition_handling": "harvest_then_reculture;washout_and_carryover_unspecified",
            "challenge_compound": "none" if hour is None else "ABT-263",
            "challenge_dose_uM": 0 if hour is None else 1, "challenge_duration_h": hour,
            "no_ABT_reference_clock": "not_stated" if hour is None else "not_applicable",
            "vehicle_identity": "not_stated", "source_column_1based": column,
            "pairing_level": "same_study_defined_population_state_not_same_cell",
            "source_image": builder.IMAGE, "source_pdf_page": 9}
        for endpoint in builder.DEFINITIONS:
            value = 80 - column if endpoint == "ND" else 5 + column
            means.append({**common, "figure": "Fig4C", "endpoint": endpoint, "value": value,
                "readout_lower": value - 1, "readout_upper": value + 1, "unit": "percent",
                "denominator": builder.DEFINITIONS[endpoint]["denominator"],
                "value_origin": "digitized_published_bar_mean",
                "summary_statistic": "mean_of_3_replicates_per_caption", "author_reported_n": 3,
                "replicate_kind": "not_specified", "author_error_bar": "SD_not_digitized",
                "interval_kind": "pixel_readout_range_not_statistical"})
        for endpoint in ("ND", "Annexin_negative_PI_positive", "Annexin_positive_PI_positive", "Annexin_positive_PI_negative"):
            panels.append({**common, "figure": "Fig4B", "endpoint": endpoint,
                "value": 70 if endpoint == "ND" else 2, "unit": "percent",
                "denominator": builder.DEFINITIONS["ND"]["denominator"] if endpoint == "ND" else "ND_gate_events",
                "value_origin": "author_printed_flow_plot_label",
                "summary_statistic": "single_displayed_flow_plot_not_Fig4C_mean",
                "source_tube_label": f"synthetic_tube_{column}", "replicate_id": None,
                "raw_events_or_counts_available": False})
    pd.DataFrame(means).to_csv(source / "Fig4C_digitized_means.csv", index=False)
    pd.DataFrame(panels).to_csv(source / "Fig4B_author_printed_values.csv", index=False)
    save_json(source / "Fig4_extraction_audit.json", {"source_image": builder.IMAGE,
        "source_image_sha256": builder.sha(source / builder.IMAGE), "B_and_C_kept_separate": True,
        "replicate_values_recovered": False, "standard_deviations_recovered": False,
        "natural_population_apoptosis_inferred": False, "no_ABT_rows_assumed_time_matched_vehicle": False})
    refresh_inventory(source)
    return source


@pytest.fixture
def package(source, tmp_path):
    target = tmp_path / "challenge-v1"
    builder.build(target, source, tmp_path)
    return target, evaluation.load_challenge(target, tmp_path)


def predictions(bundle):
    metadata = {"model_id": "synthetic_arithmetic_fixture", "model_version": "no_scientific_model",
        "model_sha256": "a" * 64, "preprocessing_sha256": "b" * 64,
        "data_exposure_statement": "All synthetic Y used for arithmetic tests; not independent evaluation",
        "unvalidated_transfer_flags": "synthetic_fixture_only", "evaluation_is_retrospective": True}
    observed = bundle["observations"].loc[lambda f: f.endpoint.eq("Annexin_given_ND")].set_index("condition_id")
    definition = bundle["contract"]["endpoint_definitions"]["Annexin_given_ND"]
    rows = [{"condition_id": row["condition_id"], "state": row["state"],
             "challenge_duration_h": row["challenge_duration_h"], "protocol_id": builder.PROTOCOL,
             **definition, "prediction_unit": "fraction", "prediction_status": "available", "missing_reason": "",
             "prediction_fraction": float(observed.loc[row["condition_id"], "value"]) + index / 100,
             "state_input_kind": "declared_group_category", "state_input_provenance": "synthetic category only",
             "same_sample_expression_endpoint": False}
            for index, row in enumerate(bundle["conditions"], 1)]
    return {"metadata": metadata, "predictions": rows}


def evaluate(tmp_path, bundle, declared):
    path = tmp_path / "synthetic_predictions.json"
    save_json(path, declared)
    return evaluation.evaluate_predictions(bundle, path)


def test_real_reader_preserves_four_active_two_unknown_and_separate_figures(package):
    _, bundle = package
    assert len(bundle["conditions"]) == 4 and len(bundle["references"]) == 2
    assert len(bundle["observations"]) == 18 and len(bundle["panels"]) == 24
    assert bundle["observations"].observation_id.is_unique
    assert len({row["condition_id"] for row in bundle["conditions"]}) == 4
    for row in bundle["references"]:
        assert row["challenge_duration_h"] is None and row["no_ABT_reference_clock"] == "not_stated"
        assert row["initial_expression_sample_id"] is None and not row["same_sample_expression_endpoint"]
    assert set(bundle["observations"].interval_kind) == {"pixel_readout_range_not_statistical"}
    report = evaluation.readiness(bundle)
    assert report["status"] == "input-ready/prediction-missing"
    assert report["primary_24h_metrics"]["mae_percent_points"] is None
    assert all(row["prediction_fraction"] is None for row in report["active_annexin_points"])
    json.dumps(report, allow_nan=False)


def test_point_errors_and_24h_state_comparison_do_not_pool_times(package, tmp_path):
    _, bundle = package
    report = evaluate(tmp_path, bundle, predictions(bundle))
    assert report["primary_24h_metrics"]["mae_percent_points"] == pytest.approx(2.5)
    assert report["primary_24h_state_comparison"]["difference_error_percent_points"] == pytest.approx(3)
    assert [row["error_percent_points"] for row in report["active_annexin_points"]] == pytest.approx([1, 2, 3, 4])
    assert not report["prediction_provenance_independently_verified"]
    assert not report["timepoints_pooled_as_replicates"] and not report["duration_selection_performed"]
    assert not report["p_values_computed"] and not report["representative_panels_pooled_with_means"]
    assert "not_SD_SE_CI" in report["primary_24h_state_comparison"]["bound_kind"]


@pytest.mark.parametrize(("field", "value", "error"), [
    ("denominator", "all_seeded_cells", "denominator"),
    ("protocol_id", "another_protocol", "protocol"),
    ("endpoint", "PI_given_ND", "endpoint"),
    ("sampling_gate", "ungated", "gate"),
    ("challenge_duration_h", None, "time"),
    ("challenge_duration_h", 6, "time"),
    ("condition_id", "unknown_id", "Unknown prediction"),
    ("condition_id", "Kotani2024_A549_control_no_ABT", "Unknown-clock"),
    ("state", builder.PRETREATED, "state"),
    ("prediction_unit", "percent", "fraction units"),
    ("prediction_fraction", 1.1, "finite fraction"),
    ("prediction_fraction", True, "finite fraction"),
    ("state_input_provenance", "", "State-input"),
    ("same_sample_expression_endpoint", True, "matched RNA"),
])
def test_bad_prediction_identity_and_semantics_rejected(package, tmp_path, field, value, error):
    _, bundle = package
    declared = predictions(bundle)
    declared["predictions"][0][field] = value
    with pytest.raises(ValueError, match=error):
        evaluate(tmp_path, bundle, declared)


def test_duplicates_and_undeclared_model_provenance_rejected(package, tmp_path):
    _, bundle = package
    declared = predictions(bundle)
    declared["predictions"].append(declared["predictions"][0].copy())
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(tmp_path, bundle, declared)
    declared = predictions(bundle)
    del declared["metadata"]["data_exposure_statement"]
    with pytest.raises(ValueError, match="provenance declarations"):
        evaluate(tmp_path, bundle, declared)


def test_missing_predictions_remain_missing_and_primary_score_requires_both(package, tmp_path):
    _, bundle = package
    declared = predictions(bundle)
    declared["predictions"][0].update(prediction_status="missing", prediction_fraction=None, missing_reason="unsupported_state")
    report = evaluate(tmp_path, bundle, declared)
    assert report["status"] == "prediction-incomplete"
    assert report["primary_24h_metrics"]["mae_percent_points"] is None
    assert report["primary_24h_state_comparison"]["predicted_difference_percent_points"] is None
    declared["predictions"][0]["prediction_fraction"] = .2
    with pytest.raises(ValueError, match="null value"):
        evaluate(tmp_path, bundle, declared)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("denominator", "all_seeded_cells", "denominators"),
    ("interval_kind", "author_SD", "raw repeats or SD"),
    ("replicate_kind", "independent_experiment", "raw repeats or SD"),
])
def test_source_semantics_rejected_even_with_refreshed_hashes(source, tmp_path, field, value, error):
    path = source / "Fig4C_digitized_means.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.endpoint.eq("Annexin_positive"), field] = value
    frame.to_csv(path, index=False)
    refresh_inventory(source)
    with pytest.raises(ValueError, match=error):
        builder.build(tmp_path / "bad", source, tmp_path)


def test_time_ids_cannot_collapse_and_unknown_reference_clock_cannot_be_filled(source, tmp_path):
    for filename in ("Fig4C_digitized_means.csv", "Fig4B_author_printed_values.csv"):
        frame = pd.read_csv(source / filename)
        frame.loc[frame.challenge_duration_h.eq(6), "condition_id"] = "Kotani2024_A549_axitinib_ABT_12h"
        frame.to_csv(source / filename, index=False)
    refresh_inventory(source)
    with pytest.raises(ValueError, match="unique means"):
        builder.build(tmp_path / "merged", source, tmp_path)


def test_unknown_reference_clock_not_assumed_24h(source, tmp_path):
    for filename in ("Fig4C_digitized_means.csv", "Fig4B_author_printed_values.csv"):
        frame = pd.read_csv(source / filename)
        frame.loc[frame.challenge_compound.eq("none"), "challenge_duration_h"] = 24
        frame.to_csv(source / filename, index=False)
    refresh_inventory(source)
    with pytest.raises(ValueError, match="Unknown reference clock"):
        builder.build(tmp_path / "filled", source, tmp_path)


def test_hashes_and_source_alignment_are_both_checked(package, tmp_path):
    target, _ = package
    path = target / "observations.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "value"] = .123
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluation.load_challenge(target, tmp_path)
    manifest = builder.read_json(target / "manifest.json")
    manifest["files_sha256"][path.name] = builder.sha(path)
    save_json(target / "manifest.json", manifest)
    with pytest.raises(ValueError, match="reviewed source"):
        evaluation.load_challenge(target, tmp_path)


def test_no_predictions_entry_writes_readiness_and_refuses_overwrite(package, tmp_path):
    target, _ = package
    output = tmp_path / "input-readiness"
    report = evaluation.run(target, output, root=tmp_path)
    assert report["status"] == "input-ready/prediction-missing"
    manifest = builder.read_json(output / "manifest.json")
    assert not manifest["predictions_supplied"] and not manifest["model_executed"] and not manifest["training_started"]
    with pytest.raises(FileExistsError):
        evaluation.run(target, output, root=tmp_path)


def test_source_change_during_preparation_cannot_receive_ready_manifest(source, tmp_path, monkeypatch):
    original = builder.prepare_tables

    def changed_source(path):
        tables = original(path)
        (path / "protocol_evidence.json").write_text("changed after reading", encoding="utf-8")
        return tables

    monkeypatch.setattr(builder, "prepare_tables", changed_source)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="changed during"):
        builder.build(output, source, tmp_path)
    assert not (output / "manifest.json").exists()
