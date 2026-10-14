"""Source-bound full-P summaries -> fit-only G/baseline -> joint-choice view.

This is an independent retrospective experiment. The private terminal ridge Q
is a diagnostic learned approximation to frozen G feedback, not DA-AC, a
qualified environment reward, or evidence that reinforcement learning helps.
Preparation never parses Y, loads PT, fits, or invokes a response model.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from lung.data.joint_selection_evaluation import (
    DEFAULT_VIEW, JointSelectionEvaluation, OBJECTIVE, SCORE_SCHEMA)
from lung.models.full_protocol import DOSE_UNITS, EVENT_TYPES, SCALAR_FIELDS
from . import full_protocol_response as full_p
from .endpoint_protocol_learning import FixedRidge, LAMBDA, ridge_parameters
from .endpoint_protocol_model import CATEGORIES, NUMERIC_COVARIATES

ROOT = full_p.ROOT
INDEX_SCHEMA = "joint-selection-full-P-index-v1"
STATE_INDEX_SCHEMA = "joint-selection-raw-state-sidecar-index-v1"
STATE_CLASSES = ("PI-APC-", "PI-APC+", "PI+APC+")
SCHEMA = "joint-selection-source-bound-learning-v1"
FREEZE_SCHEMA = "joint-selection-learning-score-execution-v1"
DEFAULT_INDEX = ROOT / "experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json"
COMMON_KEYS = ("bindings", "model_metadata", "production_registration", "reference", "seed", "steps",
    "outlet", "summary", "training_protocol_coverage", "candidate_protocol_OOD",
    "candidate_protocol_scientifically_validated", "reference_target_cell_pairing",
    "source_state_numerically_encoded", "cross_source_transfer_validated")
FILES = ("information.json", "response_features.npz", "G.json", "simple.json", "Q.json", "terminal_decisions.json")


def require(ok, reason):
    if not ok:
        raise ValueError("JOINT_LEARNING_" + reason)


def canonical(value):
    return full_p.canonical(value)


def digest(value):
    return full_p.digest(value)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")


def project_path(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to((ROOT / "experiments").resolve()), "PROJECT_EXPERIMENT_PATH_REQUIRED")
    return path


def binding(path):
    path = project_path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def check_binding(item):
    require(type(item) is dict and set(item) == {"path", "sha256"}, "ARTIFACT_BINDING_REQUIRED")
    path = project_path(item["path"])
    require(sha(path) == item["sha256"], "ARTIFACT_BYTES_CHANGED")
    return path


def dependencies(include_state=False):
    files = {"learning": Path(__file__), "fixed_ridge": Path(__file__).with_name("endpoint_protocol_learning.py"),
        "condition_metadata": Path(__file__).with_name("endpoint_protocol_model.py"),
        "full_P_reader": Path(full_p.__file__),
        "joint_view": Path(__file__).parents[1] / "data/joint_selection_evaluation.py"}
    if include_state:
        files["raw_state_reader"] = Path(__file__).with_name("raw_state_scoring.py")
    return {name: sha(path) for name, path in files.items()}


def metadata_context(view_directory=DEFAULT_VIEW):
    require(Path(view_directory).resolve().is_relative_to((full_p.inputs.ROOT / "experiments").resolve()),
            "REGISTERED_VIEW_PROJECT_BOUNDARY_REQUIRED")
    view = JointSelectionEvaluation.load(view_directory)
    reader, requests, context, full_bindings = full_p.bound_inputs()
    units = sorted(view.metadata_units(), key=lambda r: r["candidate_id"])
    ids = [r["candidate_id"] for r in units]
    require(len(ids) == len(set(ids)) == 20 and len(context["gene_ids"]) == 32461
        and len(set(context["gene_ids"])) == 32461, "EXACT_TWENTY_CANDIDATES_AND_GENE_AXIS_REQUIRED")
    by_id = {r["candidate_id"]: r for r in requests}
    require(len(by_id) == len(requests) and set(ids).issubset(by_id), "SOURCE_REQUEST_IDENTITIES_REQUIRED")
    for unit in units:
        row, candidate = by_id[unit["candidate_id"]], unit["complete_candidate"]
        require(unit["weight"] == 1 and row["base_kwargs"] is not None and row["physical"] is not None,
                "COMPLETE_LEGAL_SOURCE_REQUEST_REQUIRED")
        require(row["source_id"] == unit["source_id"] == candidate["source_id"]
            and row["source_state_id"] == candidate["state_id"]
            and row["observation_id"] == unit["observation_id"]
            and row["actual_protocol"] == candidate["protocol"]
            and row["base_kwargs"]["dose"] == candidate["dose_uM"], "SOURCE_CANDIDATE_OR_UNKNOWN_IDENTITY_CHANGED")
    roles = {role: sorted(r["observation_id"] for r in units if r["role"] == role)
             for role in ("fit", "selection", "final_test")}
    require([len(roles[r]) for r in roles] == [8, 4, 8] and len(set(sum(roles.values(), []))) == 20,
            "FROZEN_FUNCTIONAL_ROLE_IDENTITIES_REQUIRED")
    return view, reader, units, by_id, context, full_bindings, roles


def build_prediction_index(prediction_directory, output, *, view_directory=DEFAULT_VIEW):
    _, _, units, _, _, _, _ = metadata_context(view_directory)
    records = []
    for unit in units:
        path = project_path(Path(prediction_directory) / unit["candidate_id"] / "prediction.json")
        require(path.is_file(), "REAL_FULL_P_PREDICTIONS_PENDING")
        before = sha(path)
        body = full_p.read_prediction(path)
        require(body["candidate_id"] == unit["candidate_id"] and sha(path) == before,
                "PREDICTION_CHANGED_DURING_INDEX_READ")
        records.append({"candidate_id": unit["candidate_id"], **binding(path),
                        "prediction_identity": body["prediction_identity"]})
    value = {"schema": INDEX_SCHEMA, "predictions": records}
    write(project_path(output), value)
    return value


def read_state_sidecar(path):
    from .raw_state_scoring import read_state_sidecar as reader
    return reader(path)


def build_state_sidecar_index(directory, output, *, view_directory=DEFAULT_VIEW):
    view, _, units, _, _, _, _ = metadata_context(view_directory)
    records = []
    for unit in units:
        path = project_path(Path(directory) / unit["candidate_id"] / "state_sidecar.json")
        require(path.is_file(), "REAL_RAW_STATE_SIDECARS_PENDING")
        before = sha(path)
        value = read_state_sidecar(path)
        require(value["candidate_id"] == unit["candidate_id"] and sha(path) == before,
                "STATE_SIDECAR_CHANGED_DURING_INDEX_READ")
        records.append({"candidate_id": unit["candidate_id"], **binding(path)})
    value = {"schema": STATE_INDEX_SCHEMA, "view_identity": view.identity(), "sidecars": records}
    write(project_path(output), value)
    return value


def state_features(index_path, ids, predictions, view):
    path = project_path(index_path)
    before, index = sha(path), read(path)
    require(set(index) == {"schema", "view_identity", "sidecars"}
        and index["schema"] == STATE_INDEX_SCHEMA and index["view_identity"] == view.identity(),
        "RAW_STATE_INDEX_SCHEMA_REQUIRED")
    rows = index["sidecars"]
    require(type(rows) is list and len(rows) == 20 and all(type(r) is dict
        and set(r) == {"candidate_id", "path", "sha256"} for r in rows), "EXACT_STATE_INDEX_REQUIRED")
    lookup = {r["candidate_id"]: r for r in rows}
    require(len(lookup) == 20 and set(lookup) == set(ids), "COMPLETE_UNIQUE_STATE_DOMAIN_REQUIRED")
    features, common = [], None
    for cid in ids:
        record, prediction = lookup[cid], predictions[cid]
        source = check_binding({k: record[k] for k in ("path", "sha256")})
        value = read_state_sidecar(source)
        p = value["prediction"]
        require(value["schema"] == "raw-sorting-state-sidecar-v1" and value["candidate_id"] == cid
            and project_path(p["path"]) == project_path(prediction["path"])
            and p["sha256"] == prediction["sha256"] and p["prediction_identity"] == prediction["prediction_identity"]
            and sha(source) == record["sha256"], "STATE_TO_ACTUAL_PREDICTION_IDENTITY_REQUIRED")
        require(value["class_order"] == list(STATE_CLASSES) and type(value["sample_count"]) is int
            and value["sample_count"] == 2431 and value["predict_domain"] == full_p.OUTLET
            and all(value[k] is False for k in ("source_C_transfer_validated",
                "cross_source_C_calibration_verified", "Annexin_interpretation", "scientific_qualified")),
            "ORIGINAL_CLASSES_NOT_ANNEXIN_OR_TRANSFER_QUALIFICATION")
        moments = np.asarray(value["probability_mean"])
        require(moments.shape == (3,) and moments.dtype.kind == "f" and np.isfinite(moments).all()
            and ((moments >= 0) & (moments <= 1)).all(), "STATE_PROBABILITY_FEATURES_REQUIRED")
        current = {k: value[k] for k in ("state_asset", "class_order", "sample_count", "predict_domain", "sampling")}
        require(common is None or canonical(current) == canonical(common), "SAME_FROZEN_C_AND_SAMPLING_REQUIRED")
        common = current
        features.append(moments.astype(np.float64))
    require(sha(path) == before, "STATE_INDEX_CHANGED_DURING_READ")
    return np.stack(features), {"index": binding(path), "sidecars": rows, "common_C_identity": common,
        "raw_classes_only_not_measured_Annexin": True, "source_C_transfer_validated": False}


def condition_vector(unit, kwargs, physical, slots):
    """Consume original units plus explicit known bits and ordered event slots."""
    names, values = [], []

    def scalar(name, value, known=True):
        require(type(known) is bool and (not known or (type(value) in (int, float)
                and np.isfinite(value))), "FINITE_KNOWN_CONDITION_REQUIRED")
        names.extend((name, name + ".known"))
        values.extend((float(value) if known else 0., float(known)))

    row, protocol = unit["complete_candidate"], unit["complete_candidate"]["protocol"]
    scalar("candidate.nominal_dose[uM]", row["dose_uM"])
    for field in ("exposure_duration_h", "washout_at_h", "drug_free_recovery_h", "terminal_time_h"):
        scalar("candidate." + field + "[h]", protocol[field], protocol[field] is not None)
    for field, unit_name in NUMERIC_COVARIATES:
        value = row["state_covariates"].get(field)
        scalar("source." + field + "[" + unit_name + "]", value, value is not None)
    for key in ("washout_status", "prior_drug_retention", "floating_at_washout_recovered", "state_kind", "vessel"):
        value = protocol.get(key, row.get(key, row["state_covariates"].get(key)))
        for item in (*CATEGORIES[key], "unknown"):
            names.append("source_category." + key + "=" + item)
            values.append(float(value == item if item != "unknown" else value not in CATEGORIES[key]))
    # Source state ID is identity, never a fabricated numerical matched RNA state.
    for name, value in (("source_state_RNA.known", False), ("proxy_transfer_validated", False),
            ("candidate_protocol_validated", False), ("candidate_protocol_OOD", True),
            ("nominal_dose_verified", row["actual_dose_verified"]),
            ("historical_chemical_batch_verified", False), ("source_stock_form.known", False)):
        names.append(name)
        values.append(float(value))
    fingerprint = kwargs["fingerprint"][0].cpu().numpy().astype(float)
    require(fingerprint.shape == (2048,) and np.isin(fingerprint, [0, 1]).all(), "EXACT_SOURCE_FINGERPRINT_REQUIRED")
    names.extend("chemical.Morgan2048." + str(i) for i in range(2048))
    values.extend(fingerprint.tolist())
    length = int(physical.lengths[0])
    require(0 < length <= slots, "COMPLETE_EVENT_SLOTS_REQUIRED")
    for slot in range(slots):
        active = slot < length
        prefix = "physical.event" + str(slot) + "."
        names.extend((prefix + "present", prefix + "fingerprint_known", prefix + "vehicle"))
        values.extend((float(active), float(bool(physical.fingerprint_known[0, slot])) if active else 0.,
                       float(bool(physical.vehicle[0, slot])) if active else 0.))
        for field, vocabulary in (("event_type", EVENT_TYPES), ("dose_unit", DOSE_UNITS)):
            selected = int(getattr(physical, field)[0, slot]) if active else -1
            for index, value in enumerate(vocabulary):
                names.append(prefix + field + "=" + value)
                values.append(float(selected == index))
        for index, field in enumerate(SCALAR_FIELDS):
            known = active and bool(physical.known[0, slot, index])
            value = float(physical.values[0, slot, index]) if known else None
            scalar(prefix + field + "[declared_unit]", value, known)
    scalar("physical.endpoint_hours[h]", float(physical.endpoint_hours[0]), bool(physical.endpoint_known[0]))
    return tuple(names), np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class PreparedJointInputs:
    view: object
    units: tuple
    full: np.ndarray
    condition: np.ndarray
    full_names: tuple
    condition_names: tuple
    identity: dict


def prepare_inputs(prediction_index=DEFAULT_INDEX, *, view_directory=DEFAULT_VIEW, state_sidecar_index=None):
    view, reader, units, requests, context, full_bindings, roles = metadata_context(view_directory)
    path = project_path(prediction_index)
    index_sha, index = sha(path), read(path)
    require(set(index) == {"schema", "predictions"} and index["schema"] == INDEX_SCHEMA,
            "FULL_P_INDEX_SCHEMA_REQUIRED")
    records = index["predictions"]
    ids = [u["candidate_id"] for u in units]
    require(type(records) is list and len(records) == 20 and all(type(r) is dict and
        set(r) == {"candidate_id", "path", "sha256", "prediction_identity"} for r in records), "EXACT_SOURCE_INDEX_REQUIRED")
    lookup = {r["candidate_id"]: r for r in records}
    require(len(lookup) == 20 and set(lookup) == set(ids), "COMPLETE_UNIQUE_PREDICTION_DOMAIN_REQUIRED")
    summaries, common = [], None
    for cid in ids:
        record = lookup[cid]
        prediction_path = check_binding({k: record[k] for k in ("path", "sha256")})
        body = full_p.read_prediction(prediction_path)
        require(sha(prediction_path) == record["sha256"] and body["prediction_identity"] == record["prediction_identity"]
            and body["candidate_id"] == cid and body["request"] == requests[cid]
            and body["bindings"] == full_bindings, "PREDICTION_CONTENT_OR_CURRENT_REQUEST_CHANGED")
        current = {k: body[k] for k in COMMON_KEYS}
        require(common is None or canonical(current) == canonical(common), "SAME_P_MODEL_REFERENCE_SAMPLING_REQUIRED")
        common = current
        summaries.append(np.concatenate((np.asarray(body["expression_mean"], dtype=float),
                                         np.asarray(body["expression_variance"], dtype=float))))
    require(sha(path) == index_sha, "PREDICTION_INDEX_CHANGED_DURING_READ")
    slots = max(len(requests[cid]["physical"]["events"]) for cid in ids)
    condition, condition_names = [], None
    for unit in units:
        numeric, physical = full_p.numeric_batch(reader, unit["candidate_id"], 1)
        names, vector = condition_vector(unit, numeric, physical, slots)
        require(condition_names is None or names == condition_names, "FEATURE_SCHEMA_CHANGED")
        condition_names = names
        condition.append(vector)
    mean_names = tuple("fullP.expression_" + field + "." + gene for field in ("mean", "variance") for gene in context["gene_ids"])
    condition = np.stack(condition)
    state_identity, state_names = None, ()
    vectors = [np.stack(summaries), condition]
    if state_sidecar_index is not None:
        state, state_identity = state_features(state_sidecar_index, ids, lookup, view)
        state_names = tuple("raw_sorting_C.probability_mean." + label for label in STATE_CLASSES)
        vectors.append(state)
    full_names = mean_names + condition_names + state_names
    full = np.concatenate(vectors, axis=1)
    require(np.isfinite(full).all(), "NONFINITE_PREDICTION_FEATURES")
    full.setflags(write=False)
    condition.setflags(write=False)
    identity = {"view": view.identity(), "view_directory": str(Path(view_directory).resolve()),
        "prediction_index": binding(path), "predictions": records, "common_P_identity": common,
        "ordered_candidate_ids": ids, "gene_ids_sha256": digest(context["gene_ids"]),
        "role_observation_ids": roles, "dependencies": dependencies(state_sidecar_index is not None), "lambda": LAMBDA,
        "full_feature_names_sha256": digest(full_names), "raw_state_C": state_identity,
        "condition_feature_names_sha256": digest(condition_names), "event_slots": slots,
        "source_state_numerically_encoded": False, "cross_source_transfer_validated": False,
        "P_unseen_drug_or_protocol_inferred_from_functional_roles": False}
    return PreparedJointInputs(view, tuple(units), full, condition, full_names, condition_names, identity)


def preflight(prediction_index=DEFAULT_INDEX, *, view_directory=DEFAULT_VIEW, state_sidecar_index=None):
    view, _, units, _, _, bindings, _ = metadata_context(view_directory)
    exists = project_path(prediction_index).is_file()
    state_ready = state_sidecar_index is None or project_path(state_sidecar_index).is_file()
    prepared = prepare_inputs(prediction_index, view_directory=view_directory,
        state_sidecar_index=state_sidecar_index) if exists and state_ready else None
    status = "source_predictions_ready_model_fit_pending" if prepared else (
        "raw_state_sidecars_pending" if exists else "real_full_P_predictions_pending")
    return {"schema": SCHEMA, "status": status,
        "execution_dependencies": dependencies(state_sidecar_index is not None),
        "raw_state_C_requested": state_sidecar_index is not None,
        "raw_state_C_index_ready": state_ready if state_sidecar_index is not None else None,
        "view": view.preflight(), "full_P_bindings": bindings, "prediction_index_exists": exists,
        "real_predictions_consumed": 20 if prepared else 0, "missing_prediction_candidate_ids": [] if exists else [u["candidate_id"] for u in units],
        "input_identity": prepared.identity if prepared else None,
        "PT_loads": 0, "response_model_forwards": 0, "ridge_solves": 0,
        "fit_Y_deserialized": False, "selection_Y_deserialized": False, "final_Y_deserialized": False,
        "G_outputs": 0, "Q_outputs": 0, "G_reward_qualified": False, "whole_project_qualified": False}


def information_record(prepared):
    return {"schema": SCHEMA, "input_identity": prepared.identity, "objective": deepcopy(OBJECTIVE),
        "full_feature_names": list(prepared.full_names), "condition_feature_names": list(prepared.condition_names),
        "final_Y_used_for_fitting_or_selection": False, "fit_source_means": 8,
        "selection_source_means": 4, "reserved_final_source_means": 8,
        "condition_only_is_same_available_information_feature_subset": True,
        "raw_sorting_C_probabilities_included": prepared.identity["raw_state_C"] is not None,
        "C_is_measured_Annexin": False, "C_source_transfer_validated": False,
        "source_state_RNA_available": False, "predicted_common_gene_outlet_not_source_RNA": True,
        "design": "retrospective_public_Y_previously_reviewed_not_blind", "G_reward_qualified": False}


def ridge_scores(parameters, vectors):
    """Reject arithmetic overflow before clipping can turn it into a score."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = FixedRidge(parameters)(vectors)
    except FloatingPointError as error:
        raise ValueError("JOINT_LEARNING_NONFINITE_RIDGE_ARITHMETIC") from error
    require(np.isfinite(result).all(), "NONFINITE_RIDGE_ARITHMETIC")
    return result


def fit_models(prediction_index, output, *, view_directory=DEFAULT_VIEW, state_sidecar_index=None):
    """Future explicit fit: eight source means, no selection/final Y access."""
    prepared = prepare_inputs(prediction_index, view_directory=view_directory, state_sidecar_index=state_sidecar_index)
    output = project_path(output)
    require(not output.exists(), "REFUSING_TRAINING_DIRECTORY_OVERWRITE")
    fit = [i for i, u in enumerate(prepared.units) if u["role"] == "fit"]
    labels = prepared.view.measured_units("fit")
    require(set(labels) == {prepared.units[i]["candidate_id"] for i in fit}, "EXACT_FIT_Y_IDENTITIES_REQUIRED")
    y = np.asarray([labels[prepared.units[i]["candidate_id"]]["label_fraction"] for i in fit], dtype=float)
    require(y.shape == (8,) and np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all(), "FINITE_FIT_SOURCE_FRACTIONS_REQUIRED")
    models = {}
    for name, vectors, names in (("G", prepared.full, prepared.full_names),
                                  ("simple", prepared.condition, prepared.condition_names)):
        models[name] = {"schema": SCHEMA, "model_kind": "endpoint_G" if name == "G" else "condition_only_endpoint_baseline",
            "algorithm": "fixed_lambda_ridge", "lambda": LAMBDA, "feature_names": list(names),
            "parameters": ridge_parameters(vectors[fit], y), "input_identity": prepared.identity,
            "fit_observation_ids": prepared.identity["role_observation_ids"]["fit"],
            "weight_per_source_mean": 1, "normalization_role": "fit", "actual_fit_completed": True,
            "fit_operation": "closed_form_ridge_solve", "parameter_selection": "fixed_lambda_no_outcome_selection",
            "G_reward_qualified": False, "scientific_effect_verified": False}
    models["G"]["content_identity"] = digest(models["G"])
    models["simple"]["content_identity"] = digest(models["simple"])
    feedback = ridge_scores(models["G"]["parameters"], prepared.full[fit])
    decisions = {"schema": SCHEMA, "unit_kind": "model_simulated_single_terminal_candidate",
        "feedback_G_content_identity": models["G"]["content_identity"], "objective": deepcopy(OBJECTIVE),
        "units": [{"candidate_id": prepared.units[i]["candidate_id"], "source_role": "fit", "terminal": True,
                   "reward": float(value), "reward_source": "frozen_G_prediction", "bootstrap": False, "weight": 1}
                  for i, value in zip(fit, feedback)], "measured_reward": False, "G_reward_qualified": False}
    models["Q"] = {**{k: v for k, v in models["G"].items() if k not in {"content_identity", "parameters"}},
        "model_kind": "private_single_terminal_ridge_Q_diagnostic", "parameters": ridge_parameters(prepared.full[fit], feedback),
        "feedback_G_content_identity": models["G"]["content_identity"], "simulated_decisions_identity": digest(decisions),
        "actual_Q_fit_completed": True, "reward_source": "model_simulated_frozen_G",
        "formal_World_or_Actor_qualified": False, "RL_benefit_verified": False}
    models["Q"]["content_identity"] = digest(models["Q"])
    output.mkdir(parents=True)
    write(output / "information.json", information_record(prepared))
    with (output / "response_features.npz").open("xb") as stream:
        np.savez(stream, full=prepared.full, condition=prepared.condition)
    write(output / "terminal_decisions.json", decisions)
    for name, model in models.items():
        write(output / (name + ".json"), model)
    manifest = {"schema": SCHEMA, "input_identity": prepared.identity,
        "files_sha256": {name: sha(output / name) for name in FILES}, "actual_G_fit_completed": True,
        "actual_Q_fit_completed": True, "fit_Y_used": True, "selection_Y_used": False, "final_Y_used": False,
        "G_reward_qualified": False, "scientific_effect_verified": False, "formal_World_or_Actor_qualified": False}
    write(output / "manifest.json", manifest)
    return manifest


def load_models(directory, prepared):
    directory = project_path(directory)
    manifest = read(directory / "manifest.json")
    require(manifest["schema"] == SCHEMA and manifest["input_identity"] == prepared.identity
        and set(manifest["files_sha256"]) == set(FILES), "CURRENT_TRAINING_INPUT_IDENTITY_REQUIRED")
    require(manifest["actual_G_fit_completed"] is True and manifest["actual_Q_fit_completed"] is True
        and manifest["fit_Y_used"] is True and manifest["selection_Y_used"] is False and manifest["final_Y_used"] is False
        and manifest["G_reward_qualified"] is False and manifest["scientific_effect_verified"] is False
        and manifest["formal_World_or_Actor_qualified"] is False, "NO_FINAL_Y_OR_FORMAL_QUALIFICATION_REQUIRED")
    for name, expected in manifest["files_sha256"].items():
        require(sha(directory / name) == expected, "TRAINED_ARTIFACT_BYTES_CHANGED")
    require(canonical(read(directory / "information.json")) == canonical(information_record(prepared)),
            "INFORMATION_SEMANTICS_OR_FINAL_Y_CLAIM_CHANGED")
    with np.load(directory / "response_features.npz", allow_pickle=False) as data:
        require(set(data.files) == {"full", "condition"} and np.array_equal(data["full"], prepared.full)
            and np.array_equal(data["condition"], prepared.condition), "RECOMPUTED_FEATURES_CHANGED")
    models = {}
    fit = [i for i, u in enumerate(prepared.units) if u["role"] == "fit"]
    expected_kinds = {"G": "endpoint_G", "simple": "condition_only_endpoint_baseline", "Q": "private_single_terminal_ridge_Q_diagnostic"}
    for name, kind in expected_kinds.items():
        value = read(directory / (name + ".json"))
        require(value["content_identity"] == digest({k: v for k, v in value.items() if k != "content_identity"})
            and value["schema"] == SCHEMA and value["input_identity"] == prepared.identity
            and value["model_kind"] == kind and value["algorithm"] == "fixed_lambda_ridge" and value["lambda"] == LAMBDA
            and type(value["lambda"]) in (int, float)
            and value["fit_observation_ids"] == prepared.identity["role_observation_ids"]["fit"]
            and value["normalization_role"] == "fit" and type(value["weight_per_source_mean"]) is int
            and value["weight_per_source_mean"] == 1
            and value["actual_fit_completed"] is True and value["G_reward_qualified"] is False
            and value["scientific_effect_verified"] is False and value["fit_operation"] == "closed_form_ridge_solve"
            and value["parameter_selection"] == "fixed_lambda_no_outcome_selection", "STRICT_SCORER_SCHEMA_AND_ROLES_REQUIRED")
        names = prepared.condition_names if name == "simple" else prepared.full_names
        require(value["feature_names"] == list(names), "EXACT_SCORER_FEATURE_NAMES_REQUIRED")
        params = value["parameters"]
        require(type(params) is dict and set(params) == {"mean", "scale", "beta", "intercept"}
            and type(params["intercept"]) in (int, float) and np.isfinite(params["intercept"])
            and all(type(params[k]) is list and len(params[k]) == len(names)
                and all(type(v) in (int, float) and np.isfinite(v) for v in params[k])
                for k in ("mean", "scale", "beta")), "STRICT_NUMERIC_RIDGE_PARAMETERS_REQUIRED")
        vectors = prepared.condition if name == "simple" else prepared.full
        mean, scale = vectors[fit].mean(0), vectors[fit].std(0)
        scale = np.where(scale == 0, 1., scale)
        require(np.array_equal(np.asarray(params["mean"], dtype=float), mean)
            and np.array_equal(np.asarray(params["scale"], dtype=float), scale),
            "EXACT_FIT_ONLY_NORMALIZATION_REQUIRED")
        models[name] = value
    decisions = read(directory / "terminal_decisions.json")
    feedback = ridge_scores(models["G"]["parameters"], prepared.full[fit])
    expected_units = [{"candidate_id": prepared.units[i]["candidate_id"], "source_role": "fit", "terminal": True,
        "reward": float(value), "reward_source": "frozen_G_prediction", "bootstrap": False, "weight": 1}
        for i, value in zip(fit, feedback)]
    require(models["Q"]["feedback_G_content_identity"] == models["G"]["content_identity"]
        and decisions["feedback_G_content_identity"] == models["G"]["content_identity"]
        and models["Q"]["simulated_decisions_identity"] == digest(decisions)
        and models["Q"]["actual_Q_fit_completed"] is True and models["Q"]["reward_source"] == "model_simulated_frozen_G"
        and models["Q"]["formal_World_or_Actor_qualified"] is False and models["Q"]["RL_benefit_verified"] is False
        and canonical(decisions["units"]) == canonical(expected_units)
        and decisions["objective"] == OBJECTIVE and decisions["measured_reward"] is False
        and decisions["G_reward_qualified"] is False,
        "INDEPENDENT_TERMINAL_Q_FEEDBACK_REQUIRED")
    return models


def score_packet(directory, prepared, role):
    require(role in {"selection", "final_test"}, "EXPLICIT_SCORING_ROLE_REQUIRED")
    directory = project_path(directory)
    models = load_models(directory, prepared)
    rows = [i for i, unit in enumerate(prepared.units) if unit["role"] == role]
    info, features, G = [binding(directory / name) for name in ("information.json", "response_features.npz", "G.json")]
    information_id = digest(read(directory / "information.json"))
    methods = {}
    for method, name in (("direct_G", "G"), ("simple_rank", "simple"), ("learned_Q", "Q")):
        vectors = prepared.condition if name == "simple" else prepared.full
        scores = ridge_scores(models[name]["parameters"], vectors[rows])
        methods[method] = {"scores": {prepared.units[i]["candidate_id"]: float(value) for i, value in zip(rows, scores)},
            "model_artifact": binding(directory / (name + ".json")), "information_set_id": information_id,
            "feedback_G_artifact": G, "feature_use": "condition_only_subset_of_same_available_inputs" if name == "simple"
                else ("full_P_mean_variance_condition_unknown_indicators_plus_original_C_class_probabilities"
                    if prepared.identity["raw_state_C"] is not None
                    else "full_P_mean_variance_complete_nominal_condition_and_unknown_indicators"),
            "score_definition": "candidate_value_same_terminal_Annexin_reward" if name == "Q" else "estimated_terminal_Annexin_fraction"}
    return {"schema": SCORE_SCHEMA, "view_identity": prepared.view.identity(), "role": role, "objective": deepcopy(OBJECTIVE),
        "information_set_id": information_id, "information_artifact": info, "response_feature_artifact": features,
        "feedback_G_artifact": G, "fit_observation_ids": prepared.identity["role_observation_ids"]["fit"],
        "selection_observation_ids": prepared.identity["role_observation_ids"]["selection"],
        "final_Y_used_for_fitting_or_selection": False, "synthetic_predictions": False,
        "model_updates_during_scoring": 0, "methods": methods}


def freeze_scores(directory, output, *, role, prediction_index=DEFAULT_INDEX, view_directory=DEFAULT_VIEW, state_sidecar_index=None):
    prepared = prepare_inputs(prediction_index, view_directory=view_directory, state_sidecar_index=state_sidecar_index)
    packet = score_packet(directory, prepared, role)
    value = {"schema": FREEZE_SCHEMA, "training_manifest": binding(project_path(directory) / "manifest.json"),
        "dependencies": dependencies(state_sidecar_index is not None), "prediction_index": binding(prediction_index),
        "state_sidecar_index": binding(state_sidecar_index) if state_sidecar_index is not None else None,
        "view_directory": str(Path(view_directory).resolve()),
        "score_receipt": prepared.view.freeze_scores(packet, root=ROOT),
        "model_updates_during_scoring": 0, "scoring_performed": True,
        "execution_independently_certified": False, "G_reward_qualified": False,
        "formal_World_or_Actor_qualified": False, "whole_project_qualified": False}
    write(project_path(output), value)
    return value


def evaluate_frozen(receipt_path, output):
    value = read(project_path(receipt_path))
    state_index = check_binding(value["state_sidecar_index"]) if value["state_sidecar_index"] is not None else None
    require(value["schema"] == FREEZE_SCHEMA and value["dependencies"] == dependencies(state_index is not None)
        and type(value["model_updates_during_scoring"]) is int and value["model_updates_during_scoring"] == 0
        and value["scoring_performed"] is True and value["execution_independently_certified"] is False
        and value["G_reward_qualified"] is False and value["formal_World_or_Actor_qualified"] is False,
        "FROZEN_LEARNING_EXECUTION_CONTRACT_REQUIRED")
    directory = check_binding(value["training_manifest"]).parent
    index = check_binding(value["prediction_index"])
    prepared = prepare_inputs(index, view_directory=value["view_directory"], state_sidecar_index=state_index)
    receipt = value["score_receipt"]
    packet = score_packet(directory, prepared, receipt["packet"]["role"])
    require(canonical(receipt["packet"]) == canonical(packet), "SUPPLIED_SCORES_DIFFER_FROM_ACTUAL_FROZEN_SCORERS")
    result = prepared.view.evaluate_frozen(receipt, root=ROOT)
    result = {**result, "learning_source_receipt": binding(receipt_path),
        "Q_algorithm": "independent_single_terminal_ridge_diagnostic", "Q_feedback": "model_simulated_frozen_G",
        "raw_sorting_C_probability_features_used": prepared.identity["raw_state_C"] is not None,
        "C_source_transfer_validated": False, "C_is_measured_Annexin": False,
        "P_heldout_drug_or_protocol_verified": False, "formal_World_or_Actor_qualified": False}
    write(project_path(output), result)
    return result
