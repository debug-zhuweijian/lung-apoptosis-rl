"""Read Kotani's frozen ND-gated challenge; evaluate supplied predictions only."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_kotani_challenge as build_module
from scripts.build_kotani_challenge import (CONTROL, PRETREATED, OUTPUT_FILES, SOURCE,
                                            SOURCE_FILES, prepare_tables, read_json, sha)

METADATA = ("model_id", "model_version", "model_sha256", "preprocessing_sha256",
            "data_exposure_statement", "unvalidated_transfer_flags")
STATE_INPUT_KINDS = {"state_blind", "declared_group_category", "external_reference_expression", "model_predicted_state"}


def inside(base, name):
    base = Path(base).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Dependency path escapes its declared root")
    return path


def load_challenge(challenge=SOURCE / "challenge-v1", root=ROOT):
    """Check actual bytes, reconstruct source alignment, and expose no model claim."""
    challenge, root = Path(challenge).resolve(), Path(root).resolve()
    manifest = read_json(challenge / "manifest.json")
    expected = {"schema_version": 1, "status": "input-ready/prediction-missing",
                "active_conditions": 4, "unknown_clock_references": 2,
                "published_mean_observations": 18, "representative_panel_values": 24,
                "formal_training_ready": False, "core_G_compatible": False}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Kotani challenge manifest scope changed")
    source = inside(root, manifest["source_directory"])
    expected_inputs = {(source / n).relative_to(root).as_posix() for n in (*SOURCE_FILES, "local_file_inventory.json")}
    if set(manifest["inputs_sha256"]) != expected_inputs or set(manifest["files_sha256"]) != set(OUTPUT_FILES):
        raise ValueError("Required Kotani source/output binding is incomplete")
    builder = Path(build_module.__file__)
    if manifest.get("implementation_sha256") != {builder.name: sha(builder)}:
        raise ValueError("Challenge builder hash mismatch")
    binding = {str(challenge / "manifest.json"): sha(challenge / "manifest.json"), str(builder): sha(builder)}
    for base, entries in ((root, manifest["inputs_sha256"]), (challenge, manifest["files_sha256"])):
        for name, digest in entries.items():
            path = inside(base, name)
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) or sha(path) != digest:
                raise ValueError("Frozen Kotani dependency hash mismatch: " + str(path))
            binding[str(path)] = digest
    source_tables = prepare_tables(source)
    loaded = {"conditions": read_json(challenge / "conditions.json"),
              "references": read_json(challenge / "reference_conditions.json"),
              "contract": read_json(challenge / "challenge_contract.json"),
              "observations": pd.read_csv(challenge / "observations.csv"),
              "panels": pd.read_csv(challenge / "representative_panels.csv")}
    for key in ("conditions", "references", "contract"):
        if loaded[key] != source_tables[key]:
            raise ValueError("Challenge differs from reviewed source: " + key)
    for key in ("observations", "panels"):
        try:
            pd.testing.assert_frame_equal(loaded[key], source_tables[key], check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValueError("Challenge differs from reviewed source: " + key) from error
    if any(sha(path) != digest for path, digest in binding.items()):
        raise ValueError("Frozen Kotani dependency changed during reading")
    return {**loaded, "manifest": manifest, "binding": binding}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_predictions(bundle, path):
    declared = read_json(path)
    metadata, rows = declared.get("metadata"), declared.get("predictions")
    if (not isinstance(metadata, dict) or any(not _text(metadata.get(k)) for k in METADATA)
            or metadata.get("evaluation_is_retrospective") is not True
            or any(not re.fullmatch(r"[a-f0-9]{64}", metadata[k]) for k in ("model_sha256", "preprocessing_sha256"))):
        raise ValueError("Model/preprocessing identity and retrospective provenance declarations are required")
    if not isinstance(rows, list):
        raise ValueError("Predictions must be a list")
    conditions = {row["condition_id"]: row for row in bundle["conditions"]}
    references = {row["condition_id"] for row in bundle["references"]}
    contract = bundle["contract"]
    definition = contract["endpoint_definitions"][contract["primary_endpoint"]]
    checked = {}
    for row in rows:
        if not isinstance(row, dict) or not _text(row.get("condition_id")):
            raise ValueError("Prediction condition ID is required")
        condition = row["condition_id"]
        if condition in references:
            raise ValueError("Unknown-clock reference is not an evaluable active condition")
        if condition not in conditions:
            raise ValueError("Unknown prediction condition ID")
        if condition in checked:
            raise ValueError("Duplicate prediction condition ID")
        actual = conditions[condition]
        if (row.get("protocol_id") != contract["protocol_id"]
                or any(row.get(k) != definition[k] for k in ("endpoint", "denominator", "sampling_gate"))):
            raise ValueError("Prediction endpoint/protocol/denominator/gate mismatch")
        time = row.get("challenge_duration_h")
        if not _number(time) or time != actual["challenge_duration_h"]:
            raise ValueError("Prediction time is unknown or differs from the active condition")
        if row.get("state") != actual["state"]:
            raise ValueError("Prediction state differs from the source condition")
        if (row.get("state_input_kind") not in STATE_INPUT_KINDS or not _text(row.get("state_input_provenance"))
                or row.get("same_sample_expression_endpoint") is not False):
            raise ValueError("State-input provenance must not claim unverified matched RNA")
        if row.get("prediction_unit") != "fraction":
            raise ValueError("Predictions must use fraction units")
        value, status, reason = row.get("prediction_fraction"), row.get("prediction_status"), row.get("missing_reason")
        if status == "available":
            if not _number(value) or not 0 <= value <= 1 or reason != "":
                raise ValueError("Available prediction must be a finite fraction with no missing reason")
        elif status == "missing":
            if value is not None or not _text(reason):
                raise ValueError("Missing prediction requires a null value and explicit reason")
        else:
            raise ValueError("Prediction status must be available or missing")
        checked[condition] = row
    return checked, metadata


def _report(bundle, predictions, metadata=None):
    contract = bundle["contract"]
    active_ids = {row["condition_id"] for row in bundle["conditions"]}
    observed = bundle["observations"].loc[lambda f: f.condition_id.isin(active_ids) & f.endpoint.eq(contract["primary_endpoint"])]
    points = []
    for row in observed.to_dict("records"):
        prediction = predictions.get(row["condition_id"], {})
        available = prediction.get("prediction_status") == "available"
        value = prediction.get("prediction_fraction") if available else None
        points.append({"condition_id": row["condition_id"], "observation_id": row["observation_id"],
            "state": row["state"], "challenge_duration_h": row["challenge_duration_h"],
            "observed_mean_fraction": row["value"], "readout_lower_fraction": row["readout_lower"],
            "readout_upper_fraction": row["readout_upper"], "prediction_fraction": value,
            "prediction_status": "available" if available else "missing",
            "missing_reason": "" if available else prediction.get("missing_reason", "prediction_not_supplied"),
            "error_percent_points": 100 * (value - row["value"]) if available else None,
            "state_input_kind": prediction.get("state_input_kind"),
            "state_input_provenance": prediction.get("state_input_provenance")})
    indexed = {row["condition_id"]: row for row in points}
    control = indexed[contract["primary_condition_ids"][CONTROL]]
    pretreated = indexed[contract["primary_condition_ids"][PRETREATED]]
    complete = all(row["prediction_status"] == "available" for row in (control, pretreated))
    observed_delta = 100 * (pretreated["observed_mean_fraction"] - control["observed_mean_fraction"])
    predicted_delta = 100 * (pretreated["prediction_fraction"] - control["prediction_fraction"]) if complete else None
    available_count = sum(row["prediction_status"] == "available" for row in points)
    status = "input-ready/prediction-missing" if not available_count else (
        "evaluated-declared-predictions" if available_count == 4 else "prediction-incomplete")
    return {"status": status, "input_ready": True, "protocol_id": contract["protocol_id"],
        "endpoint_definition": contract["endpoint_definitions"][contract["primary_endpoint"]],
        "active_conditions": 4, "unknown_clock_reference_conditions": 2,
        "published_mean_observations": 18, "representative_panel_values": 24,
        "available_active_annexin_predictions": available_count, "active_annexin_points": points,
        "primary_24h_metrics": {"expected_predictions": 2, "complete": complete,
            "mae_percent_points": sum(abs(r["error_percent_points"]) for r in (control, pretreated)) / 2 if complete else None},
        "primary_24h_state_comparison": {"direction": "axitinib_pretreated_minus_no_axitinib",
            "control_condition_id": control["condition_id"], "pretreated_condition_id": pretreated["condition_id"],
            "observed_difference_percent_points": observed_delta,
            "reading_difference_bounds_percent_points": [
                100 * (pretreated["readout_lower_fraction"] - control["readout_upper_fraction"]),
                100 * (pretreated["readout_upper_fraction"] - control["readout_lower_fraction"])],
            "bound_kind": "pixel_readout_range_not_SD_SE_CI_or_statistical_significance",
            "predicted_difference_percent_points": predicted_delta,
            "difference_error_percent_points": predicted_delta - observed_delta if complete else None},
        "prediction_metadata_declarations": metadata, "prediction_provenance_independently_verified": False,
        "same_sample_RNA_verified": False, "unknown_clock_references_scored": False,
        "representative_panels_pooled_with_means": False, "timepoints_pooled_as_replicates": False,
        "p_values_computed": False, "duration_selection_performed": False,
        "formal_training_ready": False, "core_G_compatible": False,
        "boundaries": {key: contract[key] for key in ("uncertainty_policy", "replicate_policy",
            "gene_expression_pairing", "state_claim_boundary", "sampling_boundary", "secondary_scope", "reference_scope")}}


def readiness(bundle):
    return _report(bundle, {})


def evaluate_predictions(bundle, prediction_json):
    predictions, metadata = validate_predictions(bundle, prediction_json)
    return _report(bundle, predictions, metadata)


def run(challenge, output, predictions=None, root=ROOT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Evaluation output exists; choose a new version")
    bundle = load_challenge(challenge, root)
    prediction_hash = None if predictions is None else sha(predictions)
    report = readiness(bundle) if predictions is None else evaluate_predictions(bundle, predictions)
    inputs = {**bundle["binding"], str(Path(__file__).resolve()): sha(__file__)}
    if predictions is not None:
        if sha(predictions) != prediction_hash:
            raise ValueError("Predictions changed during evaluation")
        inputs[str(Path(predictions).resolve())] = prediction_hash
    output.mkdir(parents=True)
    target = output / "evaluation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "status": report["status"], "inputs_sha256": inputs,
                "files_sha256": {target.name: sha(target)}, "predictions_supplied": predictions is not None,
                "model_executed": False, "training_started": False}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, default=SOURCE / "challenge-v1")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.challenge, args.output, args.predictions)
    print(json.dumps({k: result[k] for k in ("status", "active_conditions", "unknown_clock_reference_conditions",
        "published_mean_observations", "representative_panel_values", "available_active_annexin_predictions")}, ensure_ascii=False))
