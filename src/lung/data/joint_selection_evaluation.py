"""Independent retrospective joint-choice view; no fitting or model inference.

Metadata construction never deserializes Y. Explicit frozen-result evaluation
uses published condition means, not independent biological samples.
"""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .endpoint_protocol_evaluation import check_supervision_bytes, DEFAULT_PLAN
from .endpoint_protocol_inputs import BASE
from .science_evaluation_comparison import canonical, read, require, sha
from ..engine.endpoint_protocol_features_v2 import EndpointProtocolFeatureInputsV2

ROOT = BASE.parents[1]
INPUT = BASE / "endpoint-protocol-inputs-v1"
SUPERVISION = BASE / "endpoint-protocol-supervision-v1"
SOURCE = BASE / "endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration"
DEFAULT_VIEW = ROOT / "experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/view-v1"
SCHEMA = "retrospective-joint-selection-v1"
SCORE_SCHEMA = "joint-selection-frozen-scores-v1"
OLD_FILES = {
    "plan.json": "c5621b3d4efbaaf46fdfba2a2d8fd23edd6527fdf21e4bab74f1089f4133a6d8",
    "manifest.json": "87bf7f09cbd913d69faabc386ecbf50fd757a9ce4bac9dbe0913b0fae0c599a4",
}
OBJECTIVE = {"endpoint": "early_plus_late_Annexin_positive",
             "unit": "fraction", "direction": "maximize", "terminal_time_h": 24,
             "denominator": "post_protocol_harvested_parent_population",
             "cost_or_shaping_terms": []}


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), "SOURCE_OUTSIDE_PROJECT")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def make_view():
    """Fix roles from sorted dose metadata: extremes fit, middle selection.

    Every drug+dose pair retains both protocols in one role. Public numerical
    outcomes were already inspected historically; this is not a blinded plan.
    """
    bundle = EndpointProtocolFeatureInputsV2.load(INPUT)
    manifest = read(INPUT / "manifest.json")
    mapping_path = INPUT / "observation_mapping.json"
    require(sha(mapping_path) == manifest["files_sha256"][mapping_path.name], "MAPPING_CHANGED")
    mapping_rows = read(mapping_path)
    require(all(set(r) == {"candidate_id", "observation_id"} for r in mapping_rows), "MAPPING_NOT_ID_ONLY")
    mapping = {r["candidate_id"]: r["observation_id"] for r in mapping_rows}
    require(len(mapping) == len(mapping_rows) == 28 and set(mapping) == set(bundle.candidate_ids), "MAPPING_IDENTITIES")
    condition_path = SOURCE / "bocnho_conditions.csv"
    require(sha(condition_path) == manifest["inputs_sha256"][str(condition_path.resolve())], "SOURCE_METADATA_CHANGED")
    with condition_path.open(encoding="utf-8-sig", newline="") as stream:
        conditions = list(csv.DictReader(stream))
    by_condition = {r["condition_id"]: r for r in conditions}
    require(len(conditions) == len(by_condition) == 22, "CONDITION_METADATA_COVERAGE")
    candidates = [bundle.model_input(cid)["candidate_record"] for cid in bundle.candidate_ids
                  if cid.startswith("vielhauer2013_")]
    compounds = sorted({r["chemical"]["compound_id"] for r in candidates})
    doses = sorted({r["dose_uM"] for r in candidates})
    arms = {"continuous_24h", "pulse_6h_washout_recovery18h"}
    require(len(candidates) == 20 and len(compounds) == 2 and len(doses) == 5, "JOINT_GRID_SIZE")
    require({(r["chemical"]["compound_id"], r["dose_uM"], r["protocol_arm"]) for r in candidates}
            == {(c, d, a) for c in compounds for d in doses for a in arms}, "COMPLETE_SOURCE_GRID_REQUIRED")
    endpoint = candidates[0]["endpoint_definition"]
    units = []
    for row in sorted(candidates, key=lambda r: r["candidate_id"]):
        source = by_condition[row["condition_id"]]
        require(row["source_id"] == "Vielhauer2013" and row["cell_line"] == "A549"
                and row["state_id"] == "a549_parental_standard_culture"
                and row["endpoint_definition"] == endpoint and row["protocol"]["terminal_time_h"] == 24,
                "INCOMPATIBLE_SOURCE_STATE_ENDPOINT_OR_CLOCK")
        rank = doses.index(row["dose_uM"])
        role = "fit" if rank in {0, 4} else "selection" if rank == 2 else "final_test"
        source_fields = {k: source[k] for k in ("source_figure", "published_n", "n_status",
            "denominator", "sampling_gate", "preincubation_after_seeding_h",
            "cross_figure_replicate_pairing", "washout_discarded_floating_cells_retained",
            "initial_RNA_reference")}
        require(source_fields["denominator"] == endpoint["denominator"]
                and source_fields["cross_figure_replicate_pairing"] == "not_established", "SOURCE_COMPARABILITY_CHANGED")
        expected_figure = "Fig3C" if row["protocol_arm"] == "continuous_24h" else "Fig5C_left"
        require(source_fields["source_figure"] == expected_figure
                and source["cell_line"] == row["cell_line"] and source["state_id"] == row["state_id"]
                and float(source["dose_nM"]) / 1000 == row["dose_uM"], "SOURCE_CONDITION_MISMATCH")
        units.append({"candidate_id": row["candidate_id"], "observation_id": mapping[row["candidate_id"]],
            "role": role, "dose_order_index": rank, "source_id": row["source_id"],
            "source_condition_metadata": source_fields, "complete_candidate": row, "weight": 1})
    require(len({r["observation_id"] for r in units}) == 20, "ONE_SOURCE_Y_PER_UNIT_REQUIRED")
    blocks = []
    for role in ("selection", "final_test"):
        rows = [r for r in units if r["role"] == role]
        block_doses = sorted({r["complete_candidate"]["dose_uM"] for r in rows})
        blocks.append({"block_id": "joint_" + role, "role": role,
            "candidate_ids": [r["candidate_id"] for r in rows],
            "fixed_policy_constraints": {"fixed_drug": {"compound_id": compounds[0]},
                "fixed_dose": {"dose_uM": block_doses[0]},
                "fixed_exposure": {"exposure_duration_h": 24},
                "fixed_drug_dose_exposure": {"compound_id": compounds[0],
                    "dose_uM": block_doses[0], "exposure_duration_h": 24}}})
    supervision = check_supervision_bytes(SUPERVISION)
    source_paths = [INPUT / n for n in ("manifest.json", "candidate_records.json", "observation_mapping.json")]
    source_paths += [SUPERVISION / "manifest.json", SUPERVISION / "vielhauer_source_nominal.json"]
    for name in ("Vielhauer2013_PMC_fulltext.md", "bocnho_conditions.csv", "bocnho_total_annexin.csv",
                 "cbt-14-527-g3.jpg", "figure5_C_left_highres_context.png", "pixel_landmarks.json",
                 "figure5_highres_assembly.json"):
        path = SOURCE / name
        require(sha(path) == manifest["inputs_sha256"][str(path.resolve())], "REVIEWED_SOURCE_CHANGED")
        source_paths.append(path)
    for name, expected in OLD_FILES.items():
        require(sha(DEFAULT_PLAN / name) == expected, "FROZEN_V2_CHANGED")
        source_paths.append(DEFAULT_PLAN / name)
    return {"schema": SCHEMA, "experiment_id": "vielhauer_retrospective_joint_choice_01",
        "assignment_rule": "sorted_dose_extremes_fit_middle_selection_remaining_final_metadata_only",
        "units": units, "blocks": blocks, "objective": deepcopy(OBJECTIVE),
        "source_bindings": [binding(p) for p in source_paths],
        "supervision_manifest_sha256": sha(SUPERVISION / "manifest.json"),
        "supervision_view_sha256": supervision["files_sha256"]["vielhauer_source_nominal.json"],
        "old_v2_files_sha256": deepcopy(OLD_FILES),
        "design": "retrospective_public_outcomes_previously_inspected_not_prospective_blind",
        "view_roles_are_new_experiment_only": True, "roles_frozen": True,
        "cross_fold_scores_allowed": False, "source_control_ids_shared_across_roles": True,
        "source_figure_ids": ["Fig3C", "Fig5C_left"],
        "cross_figure_biological_pairing_verified": False,
        "independent_biological_sample_count": None, "uncertainty": "pixel_reading_bounds_not_SEM_or_CI",
        "comparability_scope": "published_means_fraction_of_protocol_specific_terminal_harvested_parent",
        "initial_cohort_death_fraction_identifiable": False,
        "unseen_drug_or_state_validation": False, "model_execution_performed": False,
        "formal_training_performed": False, "formal_evaluation_performed": False,
        "whole_project_qualified": False}


def validate_view(view):
    require(canonical(view) == canonical(make_view()), "JOINT_VIEW_SOURCE_METADATA_OR_ROLE_CHANGED")
    return view


def build_view(directory=DEFAULT_VIEW):
    directory = Path(directory)
    require(not directory.exists(), "REFUSING_NEW_VIEW_OVERWRITE")
    view = make_view()
    directory.mkdir(parents=True)
    (directory / "view.json").write_text(canonical(view) + "\n", encoding="utf-8")
    manifest = {"schema": SCHEMA, "view_sha256": sha(directory / "view.json"),
                "view_contract_sha256": digest(view), "model_execution_performed": False}
    (directory / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return JointSelectionEvaluation.load(directory)


def finite(value, *, fraction=False):
    require(type(value) in (int, float), "REAL_SCORE_REQUIRED")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    require(valid and (not fraction or 0 <= value <= 1), "FINITE_SCORE_OR_FRACTION_REQUIRED")
    return value


def checked_artifact(item, root):
    require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "ARTIFACT_BINDING_REQUIRED")
    require(all(isinstance(item[k], str) and item[k] for k in item), "ARTIFACT_STRING_REQUIRED")
    path = (Path(root) / item["path"]).resolve()
    require(path.is_relative_to(Path(root).resolve()), "FROZEN_ARTIFACT_OUTSIDE_PROJECT")
    require(sha(path) == item["sha256"], "FROZEN_ARTIFACT_CHANGED")


def metric_result(scores, labels, probabilities):
    """Exact finite expectations; intervals are digitization sensitivity only."""
    require(set(scores) == set(labels) == set(probabilities) and len(scores) >= 2, "COMPLETE_METRIC_DOMAIN_REQUIRED")
    for cid, label in labels.items():
        require(set(label) == {"label_fraction", "reading_lower_fraction", "reading_upper_fraction"}, "LABEL_COORDINATE_REQUIRED")
        low, value, high = [finite(label[k], fraction=True) for k in
                            ("reading_lower_fraction", "label_fraction", "reading_upper_fraction")]
        require(low <= value <= high, "LABEL_READING_BOUNDS")
        finite(scores[cid])
        finite(probabilities[cid], fraction=True)
    require(abs(sum(probabilities.values()) - 1) <= 1e-12, "SELECTION_PROBABILITY_SUM")
    errors = [abs(scores[cid] - labels[cid]["label_fraction"]) for cid in scores]
    mae = finite(sum(errors) / len(scores))
    mse = finite(sum(error * error for error in errors) / len(scores))
    expected = lambda key: sum(probabilities[cid] * labels[cid][key] for cid in scores)
    measured = expected("label_fraction")
    best = max(r["label_fraction"] for r in labels.values())
    lo, hi = expected("reading_lower_fraction"), expected("reading_upper_fraction")
    upper_regret = max((1 - probabilities[cid]) * labels[cid]["reading_upper_fraction"]
        - sum(probabilities[j] * labels[j]["reading_lower_fraction"] for j in scores if j != cid) for cid in scores)
    ids = sorted(scores)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    sign = lambda value: (value > 0) - (value < 0)
    matches = [sign(scores[a] - scores[b]) == sign(labels[a]["label_fraction"] - labels[b]["label_fraction"]) for a, b in pairs]
    robust_pairs = [(a, b) for a, b in pairs if
        labels[a]["reading_lower_fraction"] > labels[b]["reading_upper_fraction"] or
        labels[b]["reading_lower_fraction"] > labels[a]["reading_upper_fraction"]]
    possible_best = [cid for cid in ids if labels[cid]["reading_upper_fraction"] >=
                     max(r["reading_lower_fraction"] for r in labels.values())]
    return {"selected_candidate_probabilities": {cid: p for cid, p in probabilities.items() if p > 0},
        "expected_measured_selected_fraction": measured, "best_in_same_measured_domain_fraction": best,
        "measured_candidate_regret": max(0.0, best - measured),
        "selected_fraction_reading_interval": [lo, hi],
        "regret_reading_interval": [max(0.0, max(r["reading_lower_fraction"] for r in labels.values()) - hi), max(0.0, upper_regret)],
        "mean_absolute_error": mae, "mean_squared_error": mse,
        "pairwise_order_matches": sum(matches), "pairwise_order_comparisons": len(matches),
        "pairwise_order_match_fraction": sum(matches) / len(matches),
        "pixel_robust_order_comparisons": len(robust_pairs),
        "pixel_robust_order_matches": sum(sign(scores[a] - scores[b]) ==
            sign(labels[a]["label_fraction"] - labels[b]["label_fraction"]) for a, b in robust_pairs),
        "possibly_best_under_pixel_intervals": possible_best,
        "reading_bounds_are_biological_CI": False}


def maximize(scores, allowed):
    require(bool(allowed) and set(allowed).issubset(scores), "LEGAL_POLICY_SUBSET_REQUIRED")
    best = max(scores[cid] for cid in allowed)
    selected = [cid for cid in allowed if scores[cid] == best]
    return {cid: 1 / len(selected) if cid in selected else 0.0 for cid in scores}


def source_role_rows(path, wanted, expected):
    """Scan the frozen pretty-JSON records; decode only requested role records.

    Unselected numeric label literals are scanned as characters, never parsed.
    Fixed source SHA and complete top-level IDs bind this one source format.
    """
    require(bool(wanted) and set(wanted).issubset(expected), "EXPLICIT_SOURCE_LABEL_ROLE_REQUIRED")
    rows, seen, record = {}, set(), []
    depth, quoted, escaped = 0, False, False
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            for char in line:
                if depth == 0:
                    if char != "{":
                        continue
                    depth, record = 1, [char]
                    continue
                record.append(char)
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        raw = "".join(record)
                        ids = re.findall(r'(?m)^    "candidate_id"[ \t]*:[ \t]*"([^"\\]+)"[ \t]*,?[ \t\r]*$', raw)
                        require(len(ids) == 1 and ids[0] in expected and ids[0] not in seen, "SOURCE_RECORD_IDENTITY")
                        seen.add(ids[0])
                        if ids[0] in wanted:
                            rows[ids[0]] = json.loads(raw)
                        record = []
    require(depth == 0 and seen == set(expected) and set(rows) == set(wanted), "SOURCE_ROLE_RECORD_COVERAGE")
    return rows


class JointSelectionEvaluation:
    def __init__(self, view):
        self._view = deepcopy(validate_view(view))

    @classmethod
    def load(cls, directory=DEFAULT_VIEW):
        directory = Path(directory)
        manifest, view = read(directory / "manifest.json"), read(directory / "view.json")
        require(manifest == {"schema": SCHEMA, "view_sha256": sha(directory / "view.json"),
                "view_contract_sha256": digest(view), "model_execution_performed": False}, "JOINT_MANIFEST_CHANGED")
        return cls(view)

    def identity(self):
        return {"experiment_id": self._view["experiment_id"], "view_contract_sha256": digest(self._view),
                "supervision_view_sha256": self._view["supervision_view_sha256"]}

    def metadata_units(self, role=None):
        require(role in {None, "fit", "selection", "final_test"}, "UNKNOWN_JOINT_ROLE")
        return deepcopy([r for r in self._view["units"] if role is None or r["role"] == role])

    def preflight(self):
        units = self.metadata_units()
        return {"identity": self.identity(), "role_counts": dict(Counter(r["role"] for r in units)),
            "unique_condition_means": 20, "final_candidate_count": 8,
            "final_drug_dose_protocol_grid": [2, 2, 2], "final_drug_dose_groups": 4,
            "shared_controls": sorted({r["complete_candidate"]["shared_control_id"] for r in units}),
            "old_v2_files_sha256": deepcopy(OLD_FILES), "label_values_deserialized": False,
            "independent_biological_sample_count": None, "model_execution_performed": False,
            "formal_training_performed": False, "formal_evaluation_performed": False,
            "whole_project_qualified": False, "status": "metadata_ready_frozen_model_outputs_pending"}

    def freeze_scores(self, packet, *, root=ROOT):
        fields = {"schema", "view_identity", "role", "objective", "information_set_id",
            "information_artifact", "response_feature_artifact", "feedback_G_artifact",
            "fit_observation_ids", "selection_observation_ids", "final_Y_used_for_fitting_or_selection",
            "synthetic_predictions", "model_updates_during_scoring", "methods"}
        require(isinstance(packet, dict) and set(packet) == fields and packet["schema"] == SCORE_SCHEMA, "JOINT_SCORE_SCHEMA")
        for item in self._view["source_bindings"]:
            checked_artifact(item, ROOT)
        require(packet["view_identity"] == self.identity() and packet["objective"] == self._view["objective"], "JOINT_IDENTITY_OR_OBJECTIVE_CHANGED")
        require(packet["role"] in {"selection", "final_test"}, "EXPLICIT_EVALUATION_ROLE_REQUIRED")
        require(isinstance(packet["information_set_id"], str) and packet["information_set_id"].strip(), "SAME_INFORMATION_ID_REQUIRED")
        require(packet["final_Y_used_for_fitting_or_selection"] is False and packet["synthetic_predictions"] is False
                and type(packet["model_updates_during_scoring"]) is int and packet["model_updates_during_scoring"] == 0, "FROZEN_UNSEEN_Y_EXECUTION_CLAIMS_REQUIRED")
        for role in ("fit", "selection"):
            require(packet[role + "_observation_ids"] == sorted(r["observation_id"] for r in self.metadata_units(role)), "JOINT_FIT_SELECTION_ROLE_MISMATCH")
        for name in ("information_artifact", "response_feature_artifact", "feedback_G_artifact"):
            checked_artifact(packet[name], root)
        require(isinstance(packet["methods"], dict) and "direct_G" in packet["methods"]
                and set(packet["methods"]).issubset({"direct_G", "simple_rank"}),
                "DIRECT_G_WITH_OPTIONAL_SIMPLE_RANK_REQUIRED")
        ids = {r["candidate_id"] for r in self.metadata_units(packet["role"])}
        for name, method in packet["methods"].items():
            require(set(method) == {"scores", "model_artifact", "information_set_id", "feedback_G_artifact", "feature_use", "score_definition"}, "ONE_FROZEN_MODEL_PER_METHOD_REQUIRED")
            require(method["information_set_id"] == packet["information_set_id"]
                    and method["feedback_G_artifact"] == packet["feedback_G_artifact"], "SAME_INFORMATION_AND_FEEDBACK_REQUIRED")
            require(isinstance(method["feature_use"], str) and method["feature_use"].strip(), "FEATURE_USE_REQUIRED")
            require(method["score_definition"] == "estimated_terminal_Annexin_fraction", "COMMON_TERMINAL_SCORE_COORDINATE_REQUIRED")
            checked_artifact(method["model_artifact"], root)
            require(isinstance(method["scores"], dict) and set(method["scores"]) == ids, "COMPLETE_JOINT_CANDIDATE_DOMAIN_REQUIRED")
            for value in method["scores"].values():
                finite(value, fraction=name == "direct_G")
        require(packet["methods"]["direct_G"]["model_artifact"] == packet["feedback_G_artifact"], "DIRECT_G_ARTIFACT_MISMATCH")
        return {"schema": "joint-selection-score-receipt-v1", "packet": deepcopy(packet),
                "packet_sha256": digest(packet), "execution_certified": False,
                "producer_and_no_leakage_claims_independently_verified": False}

    def _measured_units(self, role):
        """Explicit role-gated Y API; never called by construction or preflight."""
        require(role in {"fit", "selection", "final_test"}, "EXPLICIT_MEASURED_ROLE_REQUIRED")
        check_supervision_bytes(SUPERVISION)
        units = self.metadata_units(role)
        by_id = source_role_rows(SUPERVISION / "vielhauer_source_nominal.json",
            {r["candidate_id"] for r in units}, {r["candidate_id"] for r in self.metadata_units()})
        result = {}
        for unit in units:
            row = by_id[unit["candidate_id"]]
            require(row["observation_id"] == unit["observation_id"]
                    and row["endpoint_definition"] == unit["complete_candidate"]["endpoint_definition"]
                    and row["shared_control_id"] == unit["complete_candidate"]["shared_control_id"], "SOURCE_Y_IDENTITY_OR_DEFINITION_CHANGED")
            result[unit["candidate_id"]] = {k: row[k] for k in
                ("label_fraction", "reading_lower_fraction", "reading_upper_fraction")}
        return result

    def measured_units(self, role):
        """Deliberate label access: fit/selection never decode final records."""
        require(role in {"fit", "selection"}, "EXPLICIT_MEASURED_ROLE_FIT_SELECTION_ONLY_FINAL_REQUIRES_FROZEN_EVALUATION")
        return self._measured_units(role)

    def evaluate_frozen(self, receipt, *, root=ROOT):
        require(receipt == self.freeze_scores(receipt["packet"], root=root), "FROZEN_JOINT_RECEIPT_CHANGED")
        packet = receipt["packet"]
        labels = self._measured_units(packet["role"])
        block = next(b for b in self._view["blocks"] if b["role"] == packet["role"])
        G = packet["methods"]["direct_G"]["scores"]
        results = {}
        for name, method in packet["methods"].items():
            results[name] = {"policy_family": name, "candidate_domain": block["candidate_ids"],
                "allowed_policy_candidate_ids": block["candidate_ids"],
                **metric_result(method["scores"], labels, maximize(method["scores"], block["candidate_ids"]))}
        metadata = {r["candidate_id"]: r["complete_candidate"] for r in self.metadata_units(packet["role"])}
        for name, constraints in block["fixed_policy_constraints"].items():
            allowed = [cid for cid, row in metadata.items() if all(
                (row["chemical"]["compound_id"] if key == "compound_id" else row["dose_uM"]
                 if key == "dose_uM" else row["protocol"]["exposure_duration_h"]) == value
                for key, value in constraints.items())]
            results[name] = {"policy_family": "fixed_condition", "candidate_domain": block["candidate_ids"],
                "allowed_policy_candidate_ids": sorted(allowed), "constraints": constraints,
                **metric_result(G, labels, maximize(G, allowed))}
        results["uniform_legal_candidates"] = {"policy_family": "data_independent_diagnostic",
            "candidate_domain": block["candidate_ids"], "allowed_policy_candidate_ids": block["candidate_ids"],
            **metric_result(G, labels, dict.fromkeys(G, 1 / len(G)))}
        return {"identity": self.identity(), "role": packet["role"], "block_id": block["block_id"],
            "information_set_id": packet["information_set_id"], "objective": deepcopy(OBJECTIVE), "results": results,
            "source_condition_means": len(labels), "independent_biological_sample_count": None,
            "source_controls_shared_across_roles": True, "cross_figure_biological_pairing_verified": False,
            "retrospective": True, "prospective_blind": False, "execution_certified": False,
            "measured_metric_computation_performed": True, "model_execution_performed": False,
            "global_optimum_claim": False, "whole_project_qualified": False}
