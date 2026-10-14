"""Retrospective, observation-level role plan and explicit measured evaluation.

This module executes no models. A prediction receipt binds supplied artifacts;
it does not certify that a P/G/Q procedure actually ran.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from .endpoint_protocol_inputs import BASE
from .endpoint_protocol_supervision import EndpointProtocolSupervision, load_views, file_sha
from ..engine.endpoint_protocol_features_v2 import EndpointProtocolFeatureInputsV2
from ..engine.endpoint_protocol_features import REVIEWED_BUNDLE_SHA256

BRANCHES = {"main": "nav_figure_caption_10uM",
            "nav_methods_sensitivity": "nav_methods_11uM"}
DEFAULT_PLAN = BASE.parents[1] / "experiments/20261011-data-alignment-round15/evaluation-plan-v2"
REVIEWED_SUPERVISION_SHA256 = "91a34a7f0f7c5cb82975241586f8fa979ba6499f15a08aec295a63d7fd3f095c"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def check_supervision_bytes(directory):
    """Verify frozen label bytes without deserializing any measured labels."""
    directory = Path(directory)
    require(file_sha(directory / "manifest.json") == REVIEWED_SUPERVISION_SHA256,
            "SUPERVISION_REVIEWED_MANIFEST_CHANGED")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["input_manifest_sha256"] == REVIEWED_BUNDLE_SHA256, "SUPERVISION_INPUT_CHANGED")
    for name, expected in manifest["files_sha256"].items():
        require(file_sha(directory / name) == expected, "SUPERVISION_LABEL_BYTES_CHANGED")
    return manifest


def make_plan(input_directory=BASE / "endpoint-protocol-inputs-v1", *, schema_version=2):
    """Assign groups exclusively from complete candidate metadata, never Y."""
    require(schema_version in {1, 2}, "PLAN_SCHEMA")
    input_directory = Path(input_directory)
    bundle = EndpointProtocolFeatureInputsV2.load(input_directory)
    # The reviewed bridge anchors manifest/candidates; mapping contains IDs only.
    manifest = json.loads((input_directory / "manifest.json").read_text(encoding="utf-8"))
    mapping_path = input_directory / "observation_mapping.json"
    require(file_sha(mapping_path) == manifest["files_sha256"][mapping_path.name], "OBSERVATION_MAPPING_CHANGED")
    mapping_rows = json.loads(mapping_path.read_text(encoding="utf-8"))
    require(all(set(r) == {"candidate_id", "observation_id"} for r in mapping_rows), "MAPPING_ID_ONLY_SCHEMA")
    mapping = {r["candidate_id"]: r["observation_id"] for r in mapping_rows}
    require(len(mapping) == len(mapping_rows) == len(bundle.candidate_ids)
            and set(mapping) == set(bundle.candidate_ids), "MAPPING_CANDIDATE_IDENTITY")
    candidates = [bundle.model_input(cid)["candidate_record"] for cid in bundle.candidate_ids]
    groups = {}
    for row in candidates:
        if row["source_id"] == "Vielhauer2013":
            key = (row["chemical"]["compound_id"], row["dose_uM"])
            groups.setdefault(key, []).append(row["candidate_id"])
    require(len(groups) == 10 and all(len(v) == 2 for v in groups.values()), "PAIRED_GROUP_COVERAGE")
    folds = []
    ordered_groups = sorted(groups.items())
    for i, (key, heldout) in enumerate(ordered_groups, 1):
        selection_key, selection_ids = ordered_groups[i % len(ordered_groups)]
        branches = {}
        for branch, view in BRANCHES.items():
            scenario = "figure_caption" if branch == "main" else "methods"
            units = []
            for row in candidates:
                if row["source_id"] != "Vielhauer2013" and row["dose_scenario"] != scenario:
                    continue
                role = ("external_test" if row["source_id"] != "Vielhauer2013" else
                        "final_test" if row["candidate_id"] in heldout else
                        "selection" if schema_version == 2 and row["candidate_id"] in selection_ids else "fit")
                units.append({"observation_id": mapping[row["candidate_id"]],
                    "candidate_id": row["candidate_id"], "role": role,
                    "view_id": "vielhauer_source_nominal" if row["source_id"] == "Vielhauer2013" else view,
                    "source_id": row["source_id"], "state_id": row["state_id"],
                    "dose_scenario": row["dose_scenario"], "condition_id": row["condition_id"],
                    "p_request_id": row["p_request_id"], "shared_control_id": row["shared_control_id"],
                    "endpoint_definition": row["endpoint_definition"],
                    "terminal_time_h": row["protocol"]["terminal_time_h"],
                    "complete_candidate": row})
            blocks = [{"block_id": "paired_protocol_holdout", "role": "final_test",
                "candidate_ids": sorted(heldout), "scope_status": "prospective_execution_pending_retrospective_plan"}]
            if schema_version == 2:
                blocks.append({"block_id": "paired_protocol_reward_validation", "role": "selection",
                    "candidate_ids": sorted(selection_ids),
                    "scope_status": "limited_two_conditions_reward_validation_pending_not_final_test"})
            for state in sorted({r["state_id"] for r in units if r["role"] == "external_test"}):
                blocks.append({"block_id": "nav_state:" + state, "role": "external_test",
                    "candidate_ids": sorted(r["candidate_id"] for r in units
                        if r["role"] == "external_test" and r["state_id"] == state),
                    "scope_status": "pending_scope_specific_G_and_independent_calibration"})
            branches[branch] = {"units": units, "blocks": blocks}
        fold = {"fold_id": f"viel_leave_group_{i:02d}",
            "heldout_metadata_group": {"compound_id": key[0], "dose_uM": key[1]}, "branches": branches}
        if schema_version == 2:
            fold["selection_metadata_group"] = {"compound_id": selection_key[0], "dose_uM": selection_key[1]}
        folds.append(fold)
    plan = {"schema_version": schema_version, "folds": folds,
        "role_plan_frozen": True, "model_execution_performed": False,
        "formal_training_performed": False, "formal_evaluation_performed": False,
        "design": "retrospective_public_outcomes_previously_inspected",
        "selection_rule": "sorted_complete_compound_dose_metadata_pair_leave_one_group_out",
        "unique_source_observations": 24, "biological_replicate_ids_recovered": False,
        "folds_are_biological_replicates": False, "cross_fold_global_ranking_allowed": False,
        "alternative_branches_independent_Y": False,
        "correlation": "shared_figure_and_controls_retained_no_independence_or_CI_claim",
        "state_functional_increment": "pending_matched_high_dimensional_initial_state_and_independent_functional_validation",
        "state_duration_coverage": "Viel_one_state_two_protocols_Nav_two_confounded_states_one_duration_not_factorial"}
    if schema_version == 2:
        plan.update({"selection_rule": "final_sorted_group_selection_next_cyclic_group_fit_remaining_metadata_only",
            "reward_qualification_role": "selection",
            "final_or_external_Y_may_qualify_Q_training": False,
            "selection_validation_scope": "two_published_conditions_shared_source_not_independent_biological_validation",
            "G_reward_qualified": False, "Q_training_performed": False})
    return plan


def validate_plan(plan):
    require(plan["schema_version"] in {1, 2}, "PLAN_SCHEMA")
    require(plan["role_plan_frozen"] is True and not plan["model_execution_performed"]
            and not plan["formal_evaluation_performed"] and not plan["formal_training_performed"], "PLAN_EXECUTION_CLAIM")
    for fold in plan["folds"]:
        roles_by_branch = []
        for branch in BRANCHES:
            data = fold["branches"][branch]
            units = data["units"]
            require(len(units) == 24 and len({r["observation_id"] for r in units}) == 24,
                    "SOURCE_OBSERVATION_DUPLICATE_OR_MISSING")
            roles = {"fit", "final_test", "external_test"} | ({"selection"} if plan["schema_version"] == 2 else set())
            require(all(r["role"] in roles for r in units), "EXPLICIT_ROLE_REQUIRED")
            expected_counts = {"fit": 16, "selection": 2, "final_test": 2, "external_test": 4} if plan["schema_version"] == 2 else {"fit": 18, "final_test": 2, "external_test": 4}
            require({role: sum(r["role"] == role for r in units) for role in roles} == expected_counts, "ROLE_COUNTS")
            roles_by_branch.append({r["observation_id"]: r["role"] for r in units})
            role_groups = {}
            for row in units:
                c = row["complete_candidate"]
                require(row["candidate_id"] == c["candidate_id"] and row["endpoint_definition"] == c["endpoint_definition"], "UNIT_SCOPE_CHANGED")
                if row["source_id"] == "Vielhauer2013":
                    key = (c["chemical"]["compound_id"], c["dose_uM"])
                    role_groups.setdefault(key, set()).add(row["role"])
            require(all(len(v) == 1 for v in role_groups.values()), "COMPLETE_CONDITION_GROUP_ROLE_LEAKAGE")
            for block in data["blocks"]:
                rows = [r for r in units if r["candidate_id"] in block["candidate_ids"]]
                require(len(rows) == len(block["candidate_ids"]) == 2 and all(r["role"] == block["role"] for r in rows), "BLOCK_COVERAGE_OR_ROLE")
                require(len({digest([r["endpoint_definition"], r["state_id"], r["dose_scenario"], r["terminal_time_h"]]) for r in rows}) == 1, "ENDPOINT_STATE_SCENARIO_SCOPE")
        require(roles_by_branch[0] == roles_by_branch[1], "ALTERNATIVE_VIEW_CROSS_ROLE_LEAKAGE")


def build_plan(output=DEFAULT_PLAN, input_directory=BASE / "endpoint-protocol-inputs-v1",
               supervision_directory=BASE / "endpoint-protocol-supervision-v1", *, schema_version=2):
    output, input_directory, supervision_directory = map(Path, (output, input_directory, supervision_directory))
    require(not output.exists(), "REFUSING_FROZEN_PLAN_OVERWRITE")
    # Construction performs explicit full source/Y audit; serving/load does not.
    load_views(supervision_directory)
    plan = make_plan(input_directory, schema_version=schema_version)
    validate_plan(plan)
    manifest = {"schema_version": schema_version, "input_directory": str(input_directory.resolve()),
        "supervision_directory": str(supervision_directory.resolve()),
        "input_manifest_sha256": file_sha(input_directory / "manifest.json"),
        "supervision_manifest_sha256": file_sha(supervision_directory / "manifest.json"),
        "role_contract_sha256": digest(plan), "role_plan_frozen": True,
        "model_execution_performed": False, "formal_evaluation_performed": False,
        "formal_training_performed": False}
    output.mkdir(parents=True)
    for name, value in (("plan.json", plan), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return EndpointProtocolEvaluationPlan.load(output)


class EndpointProtocolEvaluationPlan:
    def __init__(self, plan, manifest):
        self._plan, self.manifest = copy.deepcopy(plan), copy.deepcopy(manifest)

    @classmethod
    def load(cls, directory=DEFAULT_PLAN):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        require(manifest["schema_version"] in {1, 2} and plan["schema_version"] == manifest["schema_version"], "PLAN_SCHEMA")
        require(file_sha(Path(manifest["input_directory"]) / "manifest.json") == manifest["input_manifest_sha256"]
            and file_sha(Path(manifest["supervision_directory"]) / "manifest.json") == manifest["supervision_manifest_sha256"], "PLAN_SOURCE_CHANGED")
        check_supervision_bytes(manifest["supervision_directory"])
        validate_plan(plan)
        require(plan == make_plan(manifest["input_directory"], schema_version=manifest["schema_version"]) and digest(plan) == manifest["role_contract_sha256"], "PLAN_METADATA_OR_HASH_CHANGED")
        for field in ("role_plan_frozen", "model_execution_performed", "formal_evaluation_performed", "formal_training_performed"):
            require(manifest[field] == plan[field], "MANIFEST_EXECUTION_CLAIM")
        return cls(plan, manifest)

    @property
    def fold_ids(self):
        return tuple(f["fold_id"] for f in self._plan["folds"])

    def _branch(self, fold_id, branch):
        require(branch in BRANCHES, "UNKNOWN_BRANCH")
        matches = [f for f in self._plan["folds"] if f["fold_id"] == fold_id]
        require(len(matches) == 1, "UNKNOWN_FOLD")
        return copy.deepcopy(matches[0]["branches"][branch])

    def identity(self, fold_id, branch="main"):
        self._branch(fold_id, branch)
        return {k: self.manifest[k] for k in ("role_contract_sha256", "input_manifest_sha256", "supervision_manifest_sha256")} | {"fold_id": fold_id, "branch": branch}

    def metadata_units(self, fold_id, branch="main", role=None):
        rows = self._branch(fold_id, branch)["units"]
        return [r for r in rows if role is None or r["role"] == role]

    def blocks(self, fold_id, branch="main"):
        return self._branch(fold_id, branch)["blocks"]

    def _label_units(self, fold_id, branch, role):
        # Explicit label API: legacy source audit reads Y, but only selected role
        # units leave this method. Never invoke it during metadata preflight.
        source = EndpointProtocolSupervision.load(self.manifest["input_directory"])
        result = []
        for unit in self.metadata_units(fold_id, branch, role):
            matches = [r for r in source.view(unit["view_id"]) if r["candidate_id"] == unit["candidate_id"]]
            require(len(matches) == 1 and matches[0]["observation_id"] == unit["observation_id"], "LABEL_IDENTITY_MISMATCH")
            result.append({**matches[0], "role": role, "evaluation_plan_identity": self.identity(fold_id, branch)})
        return result

    def fit_units(self, fold_id, branch="main"):
        return self._label_units(fold_id, branch, "fit")

    def evaluation_units(self, fold_id, branch="main", role="final_test"):
        require(role in {"selection", "final_test", "external_test"}, "EVALUATION_ROLE_REQUIRED")
        return self._label_units(fold_id, branch, role)

    def freeze_predictions(self, fold_id, block_id, predictions, *, branch="main", artifacts):
        """Bind complete supplied scores and artifact hashes without reading Y."""
        blocks = [b for b in self.blocks(fold_id, branch) if b["block_id"] == block_id]
        require(len(blocks) == 1, "UNKNOWN_BLOCK")
        block = blocks[0]
        require(set(predictions) == set(block["candidate_ids"]), "INCOMPLETE_OR_EXTRA_LEGAL_CANDIDATE_COVERAGE")
        require(all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in predictions.values()), "FRACTION_PREDICTION_REQUIRED")
        require(artifacts and all(file_sha(path) == expected for path, expected in artifacts.items()), "ARTIFACT_BINDING_MISMATCH")
        return {"identity": self.identity(fold_id, branch), "block_id": block_id,
            "predictions": copy.deepcopy(predictions), "predictions_sha256": digest(predictions),
            "artifacts": copy.deepcopy(artifacts), "execution_certified": False,
            "scope_status": block["scope_status"]}

    def evaluate_frozen(self, receipt, *, endpoint_definition):
        """Explicit measured-label API; metrics only on one complete legal block."""
        identity = receipt["identity"]
        fold, branch = identity["fold_id"], identity["branch"]
        require(identity == self.identity(fold, branch), "PREDICTION_ROLE_CONTRACT_MISMATCH")
        require(receipt["predictions_sha256"] == digest(receipt["predictions"]), "PREDICTION_CHANGED_AFTER_FREEZE")
        checked = self.freeze_predictions(fold, receipt["block_id"], receipt["predictions"], branch=branch, artifacts=receipt["artifacts"])
        require(receipt == checked, "PREDICTION_RECEIPT_CHANGED")
        block = next(b for b in self.blocks(fold, branch) if b["block_id"] == receipt["block_id"])
        metadata = [r for r in self.metadata_units(fold, branch) if r["candidate_id"] in block["candidate_ids"]]
        require(all(r["endpoint_definition"] == endpoint_definition for r in metadata), "ENDPOINT_DEFINITION_MISMATCH")
        require(block["role"] != "external_test", "EXTERNAL_SCOPE_SPECIFIC_G_CALIBRATION_PENDING")
        labels = {r["candidate_id"]: r for r in self.evaluation_units(fold, branch, block["role"]) if r["candidate_id"] in block["candidate_ids"]}
        predictions = receipt["predictions"]
        best_pred = max(predictions.values())
        selected = sorted(cid for cid, value in predictions.items() if value == best_pred)
        best_y = max(r["label_fraction"] for r in labels.values())
        a, b = block["candidate_ids"]
        sign = lambda x: (x > 0) - (x < 0)
        return {"identity": identity, "block_id": block["block_id"], "role": block["role"],
            "reward_qualification_role_eligible": block["role"] == "selection",
            "G_reward_qualified": False,
            "absolute_error": {cid: abs(predictions[cid] - r["label_fraction"]) for cid, r in labels.items()},
            "mean_absolute_error": sum(abs(predictions[cid] - r["label_fraction"]) for cid, r in labels.items()) / len(labels),
            "pairwise_order_matches": sign(predictions[a] - predictions[b]) == sign(labels[a]["label_fraction"] - labels[b]["label_fraction"]),
            "selected_candidate_ids": selected,
            "gap_to_best_measured_candidate": {cid: best_y - labels[cid]["label_fraction"] for cid in selected},
            "reading_bound_sensitivity": {cid: [predictions[cid] - r["reading_upper_fraction"], predictions[cid] - r["reading_lower_fraction"]] for cid, r in labels.items()},
            "reading_bounds_are_SD_or_CI": False, "biological_replicate_count": None,
            "evaluation_unit": "source_published_group_means_complete_paired_protocol_block",
            "global_optimum_claim": False, "execution_certified": False,
            "external_validation": False, "retrospective": True}
