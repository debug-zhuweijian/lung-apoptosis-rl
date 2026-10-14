"""Metadata audit and paired frozen-result comparison; never execute models.

Caller metrics are not producer certification. No biological independence,
significance or complete A/B/C/D validation is inferred from result rows.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "science-frozen-comparison-v1"
METRICS = {
    "A": {"population_mmd_v": "minimize", "mean_mse": "minimize"},
    "B": {"population_mmd_v": "minimize", "mean_mse": "minimize"},
    "C": {"mean_absolute_error": "minimize", "relative_caspase_MAE": "minimize",
          "annexin_fold_MAE": "minimize"},
    "D": {"measured_candidate_regret": "minimize"},
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("NONFINITE_JSON_CONSTANT")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique, parse_constant=constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def artifact(item, root):
    require(set(item) == {"path", "sha256"}, "EXPLICIT_ARTIFACT_BINDING_REQUIRED")
    path = (Path(root) / item["path"]).resolve()
    require(path.is_relative_to(Path(root).resolve()), "ARTIFACT_OUTSIDE_PROJECT")
    require(sha(path) == item["sha256"], "FROZEN_ARTIFACT_CHANGED")
    return read(path)


def number(value):
    require(type(value) in (int, float), "JSON_REAL_METRIC_REQUIRED")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, "FINITE_METRIC_REQUIRED")
    require(value >= 0, "NONNEGATIVE_ERROR_MMD_V_OR_REGRET_REQUIRED")
    return value


def compare_frozen_metrics(packet, root):
    """Compare already computed metrics on identical declared source units.

    The metadata artifact must be curated from a role-gated evaluation source.
    The loader checks its source bytes, unit identities and report consistency;
    it cannot certify the caller's model execution or unseen information use.
    """
    require(packet.get("schema") == SCHEMA, "COMPARISON_SCHEMA")
    hypothesis, metric = packet["hypothesis"], packet["metric"]
    require(metric in METRICS.get(hypothesis, {}), "METRIC_NOT_FOR_THIS_HYPOTHESIS")
    metadata = artifact(packet["metadata_artifact"], root)
    require(bool(metadata["source_bindings"]) and len({b["path"] for b in metadata["source_bindings"]})
            == len(metadata["source_bindings"]), "UNIQUE_REAL_SOURCE_BINDINGS_REQUIRED")
    for binding in metadata["source_bindings"]:
        path = (Path(root) / binding["path"]).resolve()
        require(path.is_relative_to(Path(root).resolve()), "SOURCE_OUTSIDE_PROJECT")
        require(sha(path) == binding["sha256"], "SOURCE_CHANGED")
    scope = metadata["scope"]
    fields = ("source_id", "fold_id", "role", "block_id", "measurement_domain_id",
              "endpoint_definition", "target_coordinate", "information_set_id")
    require(all(scope.get(field) is not None for field in fields), "COMPLETE_SCOPE_REQUIRED")
    require(all(type(scope[field]) is str and scope[field].strip() for field in fields
                if field != "endpoint_definition") and type(scope["endpoint_definition"]) is dict
                and bool(scope["endpoint_definition"]), "EXPLICIT_NONEMPTY_SCOPE_IDENTITIES_REQUIRED")
    require(scope["role"] in {"final_test", "external_test", "E"}, "FINAL_EVALUATION_ROLE_REQUIRED")
    require(metadata.get("evidence") in {"measured_expression", "measured_functional"},
            "REAL_MEASUREMENT_METADATA_REQUIRED")
    require((hypothesis in {"A", "B"}) == (metadata["evidence"] == "measured_expression"),
            "EXPRESSION_FUNCTIONAL_TASK_MISMATCH")
    if hypothesis == "C":
        if metric == "mean_absolute_error":
            compatible = (scope["measurement_domain_id"] == "Annexin_fraction"
                          and scope["target_coordinate"] == "fraction")
        elif metric == "relative_caspase_MAE":
            from .martin_state_function import DOMAIN_IDS, SCALE
            compatible = (scope["measurement_domain_id"] in DOMAIN_IDS and scope["target_coordinate"] == SCALE
                and scope["endpoint_definition"].get("signal") == "distinct_green_caspase_3_7_spots/confluence"
                and scope["endpoint_definition"].get("summary") == "source_reported_72h_AUC")
        else:
            compatible = (scope["measurement_domain_id"] == "Sarin2017_Annexin_fold"
                and scope["target_coordinate"] == "fold_change_relative_to_respective_untreated_control"
                and scope["endpoint_definition"].get("signal") == "FITC_Annexin_V"
                and scope["endpoint_definition"].get("normalization") == "respective_untreated_control")
        require(compatible, "ENDPOINT_METRIC_SCALE_MISMATCH")
    units = metadata["units"]
    require(bool(units) and len({u["unit_id"] for u in units}) == len(units), "UNIQUE_SOURCE_UNITS_REQUIRED")
    require(all(type(u["unit_id"]) is str and u["unit_id"].strip()
                and all(type(identity) is str and identity.strip()
                        for identity in u["observation_ids"] + u["candidate_ids"]) for u in units),
            "EXPLICIT_NONEMPTY_UNIT_IDENTITIES_REQUIRED")
    observations = [oid for u in units for oid in u["observation_ids"]]
    require(all(type(u["observation_ids"]) is list and type(u["candidate_ids"]) is list
                and u["observation_ids"] and u["candidate_ids"]
                and type(u.get("weight")) is int and u["weight"] == 1
                and len(u["candidate_ids"]) == len(set(u["candidate_ids"])) for u in units),
            "SOURCE_UNIT_WEIGHT_AND_CANDIDATES_REQUIRED")
    require(len(observations) == len(set(observations)), "REPEATED_Y_OR_TARGET_UNIT")
    if hypothesis == "D":
        require(all(len(u["candidate_ids"]) >= 2 for u in units), "SELECTION_REQUIRES_COMPARABLE_CANDIDATES")
    method_ids = [row["method_id"] for row in packet["methods"]]
    require(len(method_ids) >= 2 and len(set(method_ids)) == len(method_ids), "UNIQUE_BASELINE_AND_METHOD_REQUIRED")
    baseline = packet["baseline_method_id"]
    require(baseline in method_ids, "BASELINE_MISSING")
    values = {}
    for method in packet["methods"]:
        report = artifact(method["evaluation_artifact"], root)
        require(canonical(report["scope"]) == canonical(scope), "INFORMATION_FOLD_ROLE_ENDPOINT_MISMATCH")
        require(report.get("synthetic") is False and report.get("used_for_selection") is False
                and type(report.get("model_updates")) is int and report["model_updates"] == 0,
                "FROZEN_FINAL_RESULTS_REQUIRED")
        require(report["metric"] == metric and report["hypothesis"] == hypothesis, "REPORT_TASK_METRIC_MISMATCH")
        require(canonical(report["units"]) == canonical(units), "CANDIDATE_OR_OBSERVATION_COVERAGE_MISMATCH")
        require(method.get("feature_use") in {"full_state", "same_reference_mean", "drug_only", "state_ID_only",
                "direct_score", "candidate_enumeration", "learned_Q", "fixed_condition", "no_change", "ridge_shift"},
                "DECLARED_FEATURE_OR_DECISION_RULE_REQUIRED")
        rows = report["metric_rows"]
        require(len(rows) == len(units) and {r["unit_id"] for r in rows} == {u["unit_id"] for u in units}
                and len({r["unit_id"] for r in rows}) == len(rows), "EXACT_METRIC_UNIT_COVERAGE_REQUIRED")
        values[method["method_id"]] = {row["unit_id"]: number(row["value"]) for row in rows}
    comparisons = []
    for method in method_ids:
        if method == baseline:
            continue
        paired = [{"unit_id": u["unit_id"], "method_value": values[method][u["unit_id"]],
                   "baseline_value": values[baseline][u["unit_id"]],
                   "improvement": values[baseline][u["unit_id"]] - values[method][u["unit_id"]]}
                  for u in units]
        require(all(math.isfinite(r["improvement"]) for r in paired), "NONFINITE_COMPUTED_COMPARISON")
        comparisons.append({"method_id": method, "baseline_method_id": baseline,
            "paired_units": paired, "mean_improvement": math.fsum(r["improvement"] / len(paired) for r in paired)})
    return {"schema": SCHEMA, "hypothesis": hypothesis, "metric": metric,
        "direction": METRICS[hypothesis][metric], "scope": deepcopy(scope),
        "source_unit_count": len(units), "unique_observation_count": len(observations),
        "comparisons": comparisons, "execution_certified": False,
        "caller_metrics_only": True, "biological_replicate_count": None,
        "significance_claim": False, "whole_project_qualified": False,
        "cross_fold_global_ranking_allowed": False}


def metadata_preflight(root):
    """Read existing validated metadata interfaces, never values or model weights."""
    from .endpoint_protocol_evaluation import EndpointProtocolEvaluationPlan, DEFAULT_PLAN
    from .martin_state_function import MartinStateFunctionInputs
    from .native_response_units import NativeResponseUnits
    from ..engine.native_response_producer import protocol_status
    root = Path(root).resolve()
    native_path = root / "experiments/20261011-data-alignment-round17/native-response-units-v1"
    native = NativeResponseUnits(root, native_path)
    requests = read(native_path / "requests.json")
    plan = EndpointProtocolEvaluationPlan.load(DEFAULT_PLAN)
    martin_path = root / "08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/bridge_v2"
    martin = MartinStateFunctionInputs(martin_path, project_root=root)
    blocks = [{"fold_id": fold, "branch": branch, **block}
              for fold in plan.fold_ids for branch in ("main", "nav_methods_sensitivity")
              for block in plan.blocks(fold, branch)]
    role_counts = Counter(r["role"] for r in requests)
    return {"schema": "science-metadata-preflight-v1",
        "source_bindings": [{"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha(path)}
            for path in (native_path / "manifest.json", DEFAULT_PLAN / "manifest.json", martin_path / "manifest.json")],
        "native_request_count": len(requests), "native_role_counts": dict(role_counts),
        "native_protocol_counts": dict(Counter(protocol_status(r) for r in requests)),
        "endpoint_fold_count": len(plan.fold_ids), "endpoint_blocks": blocks,
        "endpoint_unique_observation_count": 24, "endpoint_biological_replicates_recovered": False,
        "martin_counts": deepcopy(martin.manifest["counts"]),
        "hypotheses": {
            "A": "expression_state_ablation_comparable_inputs_available_execution_pending;_A549_Annexin_state_increment_unresolved",
            "B": "source_specific_held_conditions_registered_prediction_and_transfer_pending",
            "C": "limited_endpoint_blocks_evaluable_after_frozen_predictions;_expression_Annexin_bridge_unresolved",
            "D": "two_protocol_candidates_per_final_endpoint_block;_full_state_drug_dose_duration_validation_unresolved"},
        "layers": {"single_selection": "limited_measured_blocks_pending_models",
                   "multistep_rollout": "model_error_and_independent_rollout_validation_pending",
                   "true_sequence": "matched_A549_sequence_expression_Annexin_evidence_unresolved"},
        "labels_parsed": False, "expression_values_read": False, "model_weights_read": False,
        "model_forward_called": False, "formal_training_performed": False,
        "whole_project_qualified": False,
        "next_required_assets": ["matching_P_descriptors", "scope_bound_G_checkpoint_and_selection_validation",
                                 "frozen_baseline_and_Q_results", "matched_A549_state_Annexin_candidate_matrix"]}
