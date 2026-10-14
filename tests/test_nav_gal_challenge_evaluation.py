"""Synthetic arithmetic fixtures only; these are not Nav-Gal model predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_nav_gal_challenge as evaluation


def write_manifest(directory, root, inputs, outputs, **extra):
    value = {"inputs_sha256": {str(path.relative_to(root)): evaluation.sha(path) for path in inputs},
             "files_sha256": {name: evaluation.sha(directory / name) for name in outputs}, **extra}
    (directory / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def package(tmp_path):
    root = tmp_path
    challenge, digitized, chemistry = [root / name for name in ["challenge-v1", "digitized-v1", "chemical_review"]]
    for directory in [challenge, digitized, chemistry]:
        directory.mkdir()
    source = root / "synthetic_primary_source.txt"
    source.write_text("SYNTHETIC fixture: no real publication values or chemical structures", encoding="utf-8")
    observations, conditions = [], []
    for state, density in [("nonsenescent", 6000), ("cisplatin_induced_senescent", 4000)]:
        for drug in ["vehicle", "navitoclax", "nav_gal"]:
            curve = f"synthetic:{state}:{drug}"
            mean = {("nonsenescent", "vehicle"): 2, ("cisplatin_induced_senescent", "vehicle"): 3,
                    ("nonsenescent", "navitoclax"): 60, ("nonsenescent", "nav_gal"): 20,
                    ("cisplatin_induced_senescent", "navitoclax"): 90, ("cisplatin_induced_senescent", "nav_gal"): 91}[state, drug]
            for hour in range(0, 37, 4):
                observations.append({"observation_id": f"{curve}:h{hour}", "curve_id": curve, "state_id": state,
                                     "challenge_treatment": drug, "hours_after_challenge": hour,
                                     "digitized_mean_percent": mean, "reading_lower_percent": max(0, mean - 3),
                                     "reading_upper_percent": min(100, mean + 3)})
            conditions.append({"condition_group_id": curve, "state_id": state, "challenge_treatment": drug,
                               "is_vehicle": drug == "vehicle", "baseline_observation_id": f"{curve}:h0",
                               "reference_expression_sample_ids": [], "same_sample_expression_endpoint": False,
                               "nominal_seeded_cells_per_well": density,
                               "state_evidence": "synthetic_group_category_not_RNA"})
    observed = pd.DataFrame(observations)
    observed.to_csv(digitized / "published_means.csv", index=False)
    (digitized / "calibration.json").write_text('{"kind":"synthetic_interval_not_CI"}')
    write_manifest(digitized, root, [source], ["published_means.csv", "calibration.json"])
    observed.to_csv(challenge / "observations.csv", index=False)
    observed.loc[observed.hours_after_challenge.eq(24)].to_csv(challenge / "primary_24h.csv", index=False)
    (challenge / "conditions.json").write_text(json.dumps(conditions))
    contract = {"primary_readout_hours": 24, "dose_scenarios": {"figure_caption": 10, "methods": 11},
                "endpoint_definition": "synthetic_image_Annexin_fraction", "uncertainty_policy": "reading_bounds_not_CI",
                "state_claim_boundary": "group_context_not_causal_effect", "duration_boundary": "continuous_readouts_not_withdrawal",
                "gene_expression_pairing": "none_verified"}
    (challenge / "challenge_contract.json").write_text(json.dumps(contract))
    write_manifest(challenge, root, [digitized / "manifest.json"], ["observations.csv", "primary_24h.csv", "conditions.json", "challenge_contract.json"])
    # The synthetic ethanol/ammonia labels below exercise joining, not real chemical science.
    chemicals = pd.DataFrame([{"source_id": f"synthetic:{drug}", "challenge_treatment": drug, "identity_id": f"synthetic_structure_{index}",
                               "fingerprint_row": index, "eligible_for_structural_features": True,
                               "canonical_full_smiles": ["CCO", "N"][index]} for index, drug in enumerate(["navitoclax", "nav_gal"])])
    chemicals.to_csv(chemistry / "chemical_identity.csv", index=False)
    fp = np.zeros((2, 2048), dtype=np.uint8)
    fp[0, 1], fp[1, 2] = 1, 1
    np.save(chemistry / "morgan2048_chiral.npy", fp)
    (chemistry / "fingerprint_metadata.json").write_text(json.dumps({"radius": 2, "bits": 2048, "includeChirality": True,
                                                                  "row_order": chemicals.source_id.tolist()}))
    write_manifest(chemistry, root, [source], ["chemical_identity.csv", "morgan2048_chiral.npy", "fingerprint_metadata.json"])
    return root, challenge, chemistry


def loaded(package):
    root, challenge, chemistry = package
    return evaluation.load_challenge(challenge, chemistry, root)


def prediction_frame(bundle):
    active = bundle["primary"].loc[bundle["primary"].challenge_treatment.ne("vehicle")]
    rows = []
    for dose, factor in [("10", 1), ("11", 2)]:
        for index, row in enumerate(active.to_dict("records"), 1):
            rows.append({"observation_id": row["observation_id"], "dose_scenario_uM": dose,
                         "prediction_percent": row["digitized_mean_percent"] + index * factor,
                         "prediction_status": "available", "missing_reason": "", "prediction_unit": "percent",
                         "model_id": "synthetic_arithmetic_fixture", "model_version": "no_scientific_model_v1",
                         "model_sha256": "a" * 64, "preprocessing_sha256": "b" * 64,
                         "state_input_provenance": "synthetic category only", "unvalidated_transfer_flags": "not_applicable_synthetic",
                         "data_exposure_statement": "All synthetic Y values used to construct arithmetic checks; no science result",
                         "evaluation_is_retrospective": "true", "endpoint_definition": bundle["contract"]["endpoint_definition"]})
    return pd.DataFrame(rows).astype(str)


def save_predictions(root, frame):
    path = root / "synthetic_predictions.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_preserves_full_conditions_exact_chemistry_and_vehicle(package):
    bundle = loaded(package)
    assert (len(bundle["observations"]), len(bundle["primary"])) == (60, 6)
    for condition in bundle["conditions"]:
        assert condition["nominal_seeded_cells_per_well"] in [4000, 6000]
        assert condition["reference_expression_sample_ids"] == []
        if condition["is_vehicle"]:
            assert condition["chemical"] is None
        else:
            chemical = condition["chemical"]
            assert chemical["challenge_treatment"] == condition["challenge_treatment"]
            assert bundle["fingerprints"][chemical["fingerprint_row"]].sum() == 1
    assert evaluation.readiness(bundle)["status"] == "input-ready/prediction-missing"


def test_upstream_source_hash_change_is_rejected(package):
    (package[0] / "synthetic_primary_source.txt").write_text("modified source")
    with pytest.raises(ValueError, match="hash mismatch"):
        loaded(package)


def test_self_consistent_hashes_cannot_hide_fingerprint_row_alias(package):
    root, _, chemistry = package
    frame = pd.read_csv(chemistry / "chemical_identity.csv")
    frame.loc[1, "fingerprint_row"] = 0
    frame.to_csv(chemistry / "chemical_identity.csv", index=False)
    write_manifest(chemistry, root, [root / "synthetic_primary_source.txt"], ["chemical_identity.csv", "morgan2048_chiral.npy", "fingerprint_metadata.json"])
    with pytest.raises(ValueError, match="fingerprint row"):
        loaded(package)


def test_arithmetic_uses_four_unique_targets_not_eight_and_keeps_overlap_undetermined(package):
    bundle = loaded(package)
    report = evaluation.evaluate_predictions(bundle, save_predictions(package[0], prediction_frame(bundle)))
    assert report["n_unique_active_observations"] == 4
    assert report["scenarios"]["10"]["metrics"]["mae_percent_points"] == 2.5
    assert report["scenarios"]["10"]["metrics"]["rmse_percent_points"] == pytest.approx(np.sqrt(7.5))
    assert report["across_reported_dose_scenarios"]["aggregate_metric_ranges"]["mae_percent_points"] == [2.5, 5.0]
    senescent = next(row for row in report["within_state_order_challenge"] if row["state_id"] == "cisplatin_induced_senescent")
    assert senescent["published_mean_order"] == "undetermined"
    assert senescent["scenarios"]["10"]["agrees_with_resolved_published_order"] is None
    assert report["vehicle"]["metrics"]["available_predictions"] == 0
    assert not report["p_values_computed"] and not report["timepoints_pooled_as_replicates"]
    json.dumps(report, allow_nan=False)


def test_missing_scenario_is_explicit_and_does_not_produce_best_scenario_score(package):
    bundle = loaded(package)
    frame = prediction_frame(bundle)
    frame.loc[4, ["prediction_status", "prediction_percent", "missing_reason"]] = ["missing", "", "unsupported_synthetic_input"]
    report = evaluation.evaluate_predictions(bundle, save_predictions(package[0], frame))
    assert report["status"] == "prediction-incomplete"
    assert report["scenarios"]["10"]["metrics"]["mae_percent_points"] == 2.5
    assert report["scenarios"]["11"]["metrics"]["mae_percent_points"] is None
    assert report["across_reported_dose_scenarios"]["aggregate_metric_ranges"]["mae_percent_points"] is None


@pytest.mark.parametrize(("column", "value", "all_rows"), [
    ("prediction_unit", "fraction", True), ("prediction_percent", "nan", False),
    ("prediction_percent", "101", False), ("prediction_percent", "-1", False),
    ("evaluation_is_retrospective", "false", True), ("data_exposure_statement", "", True),
    ("model_sha256", "wrong", True), ("model_version", "different", False),
    ("endpoint_definition", "SYTOX", True), ("dose_scenario_uM", "12", False),
    ("observation_id", "unknown", False), ("prediction_status", "missing", False),
])
def test_invalid_units_scope_identity_and_exposure_are_rejected(package, column, value, all_rows):
    bundle = loaded(package)
    frame = prediction_frame(bundle)
    if all_rows:
        frame[column] = value
    else:
        frame.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluation.evaluate_predictions(bundle, save_predictions(package[0], frame))


def test_duplicate_and_omitted_scenario_rows_rejected(package):
    bundle = loaded(package)
    frame = prediction_frame(bundle)
    for altered in [pd.concat([frame, frame.iloc[[0]]]), frame.iloc[1:]]:
        with pytest.raises(ValueError):
            evaluation.evaluate_predictions(bundle, save_predictions(package[0], altered))


def test_vehicle_is_separate_and_nonprimary_timepoints_cannot_expand_sample_size(package):
    bundle = loaded(package)
    frame = prediction_frame(bundle)
    vehicle = []
    for row in bundle["primary"].loc[bundle["primary"].challenge_treatment.eq("vehicle")].to_dict("records"):
        vehicle.append({**frame.iloc[0].to_dict(), "observation_id": row["observation_id"],
                        "dose_scenario_uM": "vehicle", "prediction_percent": str(row["digitized_mean_percent"] + 1)})
    extended = pd.concat([frame, pd.DataFrame(vehicle)], ignore_index=True)
    report = evaluation.evaluate_predictions(bundle, save_predictions(package[0], extended))
    assert report["vehicle"]["metrics"]["mae_percent_points"] == 1
    assert report["scenarios"]["10"]["metrics"]["mae_percent_points"] == 2.5
    extended.loc[len(extended)] = {**frame.iloc[0].to_dict(), "observation_id": frame.iloc[0].observation_id.replace(":h24", ":h4")}
    with pytest.raises(ValueError, match="24 h"):
        evaluation.evaluate_predictions(bundle, save_predictions(package[0], extended))
