"""Read the frozen Nav-Gal package and evaluate declared external 24 h predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round4/nav_gal"
STATES = {"nonsenescent", "cisplatin_induced_senescent"}
DRUGS = {"navitoclax", "nav_gal"}
METADATA = ("model_id", "model_version", "model_sha256", "preprocessing_sha256",
            "state_input_provenance", "unvalidated_transfer_flags", "data_exposure_statement",
            "evaluation_is_retrospective", "endpoint_definition", "prediction_unit")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(directory, root, required_outputs):
    """Check declared data dependencies, ignoring historical Python provenance."""
    path = directory / "manifest.json"
    manifest = read_json(path)
    outputs = manifest.get("files_sha256", manifest.get("outputs_sha256", {}))
    if not set(required_outputs).issubset(outputs):
        raise ValueError(f"Required output hashes missing: {path}")
    inputs_base = directory / manifest["input_base"] if "input_base" in manifest else root
    checked = {str(path.resolve()): sha(path)}
    for base, entries in [(inputs_base, manifest.get("inputs_sha256", {})),
                          (root, manifest.get("sources_sha256", {})), (directory, outputs)]:
        for name, expected in entries.items():
            if Path(name).suffix.lower() == ".py":
                continue
            source = (base / name.replace("\\", "/")).resolve()
            if not re.fullmatch(r"[a-f0-9]{64}", expected) or sha(source) != expected:
                raise ValueError(f"Frozen dependency hash mismatch: {source}")
            checked[str(source)] = expected
    return manifest, checked


def load_challenge(challenge=BASE / "challenge-v1", chemistry=BASE / "chemical_review", root=ROOT):
    challenge, chemistry, root = Path(challenge), Path(chemistry), Path(root)
    _, binding = verify_manifest(challenge, root, ["conditions.json", "observations.csv", "primary_24h.csv", "challenge_contract.json"])
    digitized = challenge.parent / "digitized-v1"
    if str((digitized / "manifest.json").resolve()) not in binding:
        raise ValueError("Digitization manifest is not bound by the challenge")
    _, digitized_binding = verify_manifest(digitized, root, ["published_means.csv", "calibration.json"])
    _, chemistry_binding = verify_manifest(chemistry, root, ["chemical_identity.csv", "morgan2048_chiral.npy", "fingerprint_metadata.json"])
    binding = {**binding, **digitized_binding, **chemistry_binding}
    contract, conditions = read_json(challenge / "challenge_contract.json"), read_json(challenge / "conditions.json")
    observed, primary = (pd.read_csv(challenge / name) for name in ["observations.csv", "primary_24h.csv"])
    if contract["primary_readout_hours"] != 24 or contract["dose_scenarios"] != {"figure_caption": 10, "methods": 11}:
        raise ValueError("This evaluator supports the frozen 24 h, 10/11 uM challenge")
    if len(conditions) != 6 or len(observed) != 60 or len(primary) != 6 or observed.observation_id.duplicated().any():
        raise ValueError("Expected six conditions, sixty unique observations and six primary rows")
    pd.testing.assert_frame_equal(primary.reset_index(drop=True), observed.loc[observed.hours_after_challenge.eq(24)].reset_index(drop=True))
    pd.testing.assert_frame_equal(observed, pd.read_csv(digitized / "published_means.csv"))
    bounds = observed[["reading_lower_percent", "digitized_mean_percent", "reading_upper_percent"]].to_numpy(float)
    if not np.isfinite(bounds).all() or (bounds < 0).any() or (bounds > 100).any() or (np.diff(bounds, axis=1) < 0).any():
        raise ValueError("Invalid percentage or digitization bounds")
    condition_ids = [row["condition_group_id"] for row in conditions]
    expected_groups = {(state, drug) for state in STATES for drug in DRUGS | {"vehicle"}}
    if len(set(condition_ids)) != 6 or set(zip(observed.state_id, observed.challenge_treatment)) != expected_groups:
        raise ValueError("Condition state/treatment identities are missing or duplicated")
    chemicals = pd.read_csv(chemistry / "chemical_identity.csv")
    fingerprints = np.load(chemistry / "morgan2048_chiral.npy", allow_pickle=False)
    metadata = read_json(chemistry / "fingerprint_metadata.json")
    if len(chemicals) != 2 or set(chemicals.challenge_treatment) != DRUGS or not chemicals.identity_id.is_unique or not chemicals.source_id.is_unique:
        raise ValueError("Two distinct exact treatment structures are required")
    if fingerprints.shape != (2, 2048) or fingerprints.dtype != np.uint8 or not np.isin(fingerprints, [0, 1]).all():
        raise ValueError("Expected two binary 2048-bit fingerprints")
    if metadata["radius"] != 2 or metadata["bits"] != 2048 or metadata["includeChirality"] is not True or len(set(metadata["row_order"])) != 2:
        raise ValueError("Fingerprint definition changed")
    chemical_by_treatment = {}
    for chemical in json.loads(chemicals.to_json(orient="records")):
        index = chemical["fingerprint_row"]
        if not isinstance(index, int) or index not in (0, 1) or metadata["row_order"][index] != chemical["source_id"]:
            raise ValueError("Chemical fingerprint row identity mismatch")
        if chemical["eligible_for_structural_features"] is not True or not chemical["canonical_full_smiles"]:
            raise ValueError("Chemical structure is not available for input")
        chemical_by_treatment[chemical["challenge_treatment"]] = chemical
    if len({row["fingerprint_row"] for row in chemical_by_treatment.values()}) != 2 or np.array_equal(*fingerprints):
        raise ValueError("The two active structures must not share a fingerprint row")
    joined = []
    for condition in conditions:
        rows = observed.loc[observed.curve_id.eq(condition["condition_group_id"])]
        drug, state = condition["challenge_treatment"], condition["state_id"]
        if sorted(rows.hours_after_challenge.tolist()) != list(range(0, 37, 4)) or not rows.challenge_treatment.eq(drug).all() or not rows.state_id.eq(state).all():
            raise ValueError("Observation-to-condition identity or time mismatch")
        if condition["baseline_observation_id"] != rows.loc[rows.hours_after_challenge.eq(0), "observation_id"].item():
            raise ValueError("Baseline must be the same curve's plotted zero-hour value")
        if condition["is_vehicle"] != (drug == "vehicle") or condition["reference_expression_sample_ids"] or condition["same_sample_expression_endpoint"]:
            raise ValueError("Vehicle or RNA pairing boundary changed")
        joined.append({**condition, "chemical": None if drug == "vehicle" else chemical_by_treatment[drug]})
    return {"conditions": joined, "observations": observed, "primary": primary, "contract": contract,
            "fingerprints": fingerprints, "fingerprint_metadata": metadata, "source_hashes": binding}


def readiness(bundle):
    return {"status": "input-ready/prediction-missing", "condition_count": len(bundle["conditions"]),
            "published_observation_positions": len(bundle["observations"]), "primary_24h_rows": len(bundle["primary"]),
            "primary_active_unique_observations": 4, "primary_vehicle_unique_observations": 2,
            "conditions": bundle["conditions"], "source_hashes": bundle["source_hashes"],
            "gene_expression_pairing": "none_verified", "scientific_model_predictions_evaluated": False,
            "formal_training_ready": False}


def read_predictions(path, bundle):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = set(METADATA) | {"observation_id", "dose_scenario_uM", "prediction_percent", "prediction_status", "missing_reason"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Prediction CSV is empty or missing required columns")
    declared = {}
    for key in METADATA:
        if frame[key].nunique() != 1 or not frame[key].iloc[0].strip():
            raise ValueError(f"One nonempty declaration per prediction file required: {key}")
        declared[key] = frame[key].iloc[0]
    for key in ["model_sha256", "preprocessing_sha256"]:
        if not re.fullmatch(r"[a-f0-9]{64}", declared[key]):
            raise ValueError(f"Invalid SHA256 declaration: {key}")
    if declared["prediction_unit"] != "percent" or declared["evaluation_is_retrospective"].lower() != "true":
        raise ValueError("Predictions must explicitly use percent and acknowledge retrospective evaluation")
    if declared["endpoint_definition"] != bundle["contract"]["endpoint_definition"]:
        raise ValueError("Prediction endpoint definition mismatch")
    primary = bundle["primary"].set_index("observation_id")
    active = set(primary.index[primary.challenge_treatment.ne("vehicle")])
    required_keys = {(obs, dose) for obs in active for dose in ["10", "11"]}
    if frame.duplicated(["observation_id", "dose_scenario_uM"]).any():
        raise ValueError("Duplicate observation/scenario prediction")
    if not required_keys.issubset(set(zip(frame.observation_id, frame.dose_scenario_uM))):
        raise ValueError("Both dose scenarios are required; use explicit missing prediction rows")
    for row in frame.to_dict("records"):
        obs, scenario = row["observation_id"], row["dose_scenario_uM"]
        if obs not in primary.index:
            raise ValueError("Only known 24 h primary observation IDs are accepted")
        vehicle = primary.loc[obs, "challenge_treatment"] == "vehicle"
        if scenario not in ({"vehicle"} if vehicle else {"10", "11"}):
            raise ValueError("Invalid dose scenario; vehicle has one separate row")
        if row["prediction_status"] == "missing":
            if row["prediction_percent"] or not row["missing_reason"].strip():
                raise ValueError("Missing predictions need a blank value and an explicit reason")
        elif row["prediction_status"] == "available":
            value = float(row["prediction_percent"])
            if not np.isfinite(value) or not 0 <= value <= 100 or row["missing_reason"]:
                raise ValueError("Available predictions must be finite percentages with no missing reason")
        else:
            raise ValueError("Prediction status must be available or missing")
    return frame, declared


def metrics(rows, expected_count):
    available = [row for row in rows if row["prediction_status"] == "available"]
    complete = len(available) == expected_count
    errors = np.array([row["error_percent_points"] for row in available])
    return {"expected_unique_observations": expected_count, "available_predictions": len(available), "complete": complete,
            "mae_percent_points": float(np.abs(errors).mean()) if complete else None,
            "rmse_percent_points": float(np.sqrt(np.mean(errors ** 2))) if complete else None}


def published_order(left, right, left_name, right_name):
    if left.reading_lower_percent > right.reading_upper_percent:
        return left_name
    if right.reading_lower_percent > left.reading_upper_percent:
        return right_name
    return "undetermined"


def evaluate_predictions(bundle, path):
    frame, declarations = read_predictions(path, bundle)
    primary = bundle["primary"].set_index("observation_id")
    by_scenario = {"10": [], "11": [], "vehicle": []}
    for row in frame.to_dict("records"):
        observed = primary.loc[row["observation_id"]]
        available = row["prediction_status"] == "available"
        prediction = float(row["prediction_percent"]) if available else None
        by_scenario[row["dose_scenario_uM"]].append({"observation_id": row["observation_id"],
            "prediction_status": row["prediction_status"], "missing_reason": row["missing_reason"],
            "prediction_percent": prediction,
            "error_percent_points": prediction - observed.digitized_mean_percent if available else None,
            "absolute_error_reading_envelope": [max(observed.reading_lower_percent - prediction, 0., prediction - observed.reading_upper_percent),
                                                max(abs(prediction - observed.reading_lower_percent), abs(prediction - observed.reading_upper_percent))] if available else None})
    scenarios = {dose: {"metrics": metrics(by_scenario[dose], 4), "predictions": by_scenario[dose]} for dose in ["10", "11"]}
    lookup = {dose: {row["observation_id"]: row for row in rows} for dose, rows in by_scenario.items()}
    across = []
    active = primary.loc[primary.challenge_treatment.ne("vehicle")]
    for obs in active.index:
        rows = [lookup[dose][obs] for dose in ["10", "11"]]
        complete = all(row["prediction_status"] == "available" for row in rows)
        across.append({"observation_id": obs, "both_scenarios_available": complete,
                       "prediction_percent_range": [min(row["prediction_percent"] for row in rows), max(row["prediction_percent"] for row in rows)] if complete else None,
                       "absolute_error_percent_points_range": [min(abs(row["error_percent_points"]) for row in rows), max(abs(row["error_percent_points"]) for row in rows)] if complete else None})
    aggregate_range = {}
    for key in ["mae_percent_points", "rmse_percent_points"]:
        values = [scenarios[dose]["metrics"][key] for dose in ["10", "11"]]
        aggregate_range[key] = [min(values), max(values)] if all(value is not None for value in values) else None
    order_checks = []
    for state in sorted(STATES):
        rows = active.loc[active.state_id.eq(state)].set_index("challenge_treatment")
        nav, gal = rows.loc["navitoclax"], rows.loc["nav_gal"]
        observed_order = published_order(nav, gal, "navitoclax", "nav_gal")
        state_ids = {drug: active.index[(active.state_id.eq(state) & active.challenge_treatment.eq(drug))].item() for drug in DRUGS}
        orders = {}
        for dose in ["10", "11"]:
            predictions = {drug: lookup[dose][state_ids[drug]]["prediction_percent"] for drug in DRUGS}
            if any(value is None for value in predictions.values()):
                winner = "missing"
            elif predictions["navitoclax"] == predictions["nav_gal"]:
                winner = "tie"
            else:
                winner = max(predictions, key=predictions.get)
            orders[dose] = {"predicted_larger_Annexin": winner,
                            "agrees_with_resolved_published_order": winner == observed_order if observed_order != "undetermined" and winner != "missing" else None}
        order_checks.append({"state_id": state, "published_mean_order": observed_order, "scenarios": orders})
    return {"status": "retrospective-prediction-evaluated" if all(scenarios[d]["metrics"]["complete"] for d in scenarios) else "prediction-incomplete",
            "primary_readout_hours": 24, "n_unique_active_observations": 4, "declarations": declarations,
            "declaration_verification": "recorded_not_independent_verification_of_model_training_or_data_exposure",
            "prediction_file_sha256": sha(path), "source_hashes": bundle["source_hashes"],
            "observations": json.loads(bundle["primary"].to_json(orient="records")), "scenarios": scenarios,
            "across_reported_dose_scenarios": {"per_observation": across, "aggregate_metric_ranges": aggregate_range,
                                              "meaning": "two_reported_input_scenarios_not_CI_or_selected_best_dose"},
            "within_state_order_challenge": order_checks,
            "vehicle": {"metrics": metrics(by_scenario["vehicle"], 2), "predictions": by_scenario["vehicle"]},
            "conditions": bundle["conditions"],
            "boundaries": {key: bundle["contract"][key] for key in ["uncertainty_policy", "state_claim_boundary", "duration_boundary", "gene_expression_pairing"]},
            "reading_error_envelope_is_biological_CI": False, "timepoints_pooled_as_replicates": False,
            "p_values_computed": False, "formal_training_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, default=BASE / "challenge-v1")
    parser.add_argument("--chemistry", type=Path, default=BASE / "chemical_review")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_challenge(args.challenge, args.chemistry)
    report = evaluate_predictions(bundle, args.predictions) if args.predictions else readiness(bundle)
    result = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
