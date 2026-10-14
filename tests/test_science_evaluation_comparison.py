"""Synthetic frozen-result fixtures exercise scientific comparison boundaries."""
import json

import pytest

from lung.data.science_evaluation_comparison import SCHEMA, compare_frozen_metrics, read, sha


def write(path, data):
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    return {"path": path.name, "sha256": sha(path)}


@pytest.fixture
def packet(tmp_path):
    source = tmp_path / "source.json"
    write(source, {"fixture": "not a biological experiment"})
    scope = {"source_id": "source", "fold_id": "fold1", "role": "final_test", "block_id": "block1",
        "measurement_domain_id": "Annexin_fraction", "endpoint_definition": {"assay": "Annexin"},
        "target_coordinate": "fraction", "information_set_id": "same frozen inputs"}
    units = [{"unit_id": "block1", "observation_ids": ["Y1", "Y2"],
              "candidate_ids": ["candidate1", "candidate2"], "weight": 1}]
    metadata = {"scope": scope, "units": units, "evidence": "measured_functional",
                "source_bindings": [{"path": source.name, "sha256": sha(source)}]}
    methods = []
    for name, value, rule in (("enumeration", .2, "candidate_enumeration"), ("Q", .1, "learned_Q")):
        report = {"scope": scope, "units": units, "hypothesis": "D", "metric": "measured_candidate_regret",
                  "synthetic": False, "used_for_selection": False, "model_updates": 0,
                  "metric_rows": [{"unit_id": "block1", "value": value}]}
        methods.append({"method_id": name, "feature_use": rule,
                        "evaluation_artifact": write(tmp_path / (name + ".json"), report)})
    return {"schema": SCHEMA, "hypothesis": "D", "metric": "measured_candidate_regret",
            "baseline_method_id": "enumeration", "metadata_artifact": write(tmp_path / "metadata.json", metadata),
            "methods": methods}


def replace_artifact(tmp_path, binding, mutation):
    value = read(tmp_path / binding["path"])
    mutation(value)
    return write(tmp_path / binding["path"], value)


def test_paired_lower_regret_is_positive_but_never_execution_or_science_certificate(packet, tmp_path):
    result = compare_frozen_metrics(packet, tmp_path)
    assert result["comparisons"][0]["mean_improvement"] == pytest.approx(.1)
    assert result["unique_observation_count"] == 2
    assert result["biological_replicate_count"] is None
    assert not result["execution_certified"] and not result["whole_project_qualified"]


@pytest.mark.parametrize("field", ["fold_id", "role", "block_id", "measurement_domain_id",
                                   "endpoint_definition", "target_coordinate", "information_set_id"])
def test_same_information_role_fold_and_target_are_required(packet, tmp_path, field):
    binding = packet["methods"][1]["evaluation_artifact"]
    packet["methods"][1]["evaluation_artifact"] = replace_artifact(
        tmp_path, binding, lambda value: value["scope"].update({field: "other"}))
    with pytest.raises(ValueError, match="INFORMATION_FOLD_ROLE_ENDPOINT"):
        compare_frozen_metrics(packet, tmp_path)


def test_one_Y_cannot_be_expanded_to_RNA_replicates(packet, tmp_path):
    def duplicate(value):
        value["units"].append({**value["units"][0], "unit_id": "RNA_rep2"})
    packet["metadata_artifact"] = replace_artifact(tmp_path, packet["metadata_artifact"], duplicate)
    with pytest.raises(ValueError, match="REPEATED_Y"):
        compare_frozen_metrics(packet, tmp_path)


def test_same_complete_candidates_required(packet, tmp_path):
    def missing(value):
        value["units"][0]["candidate_ids"].pop()
    binding = packet["methods"][1]["evaluation_artifact"]
    packet["methods"][1]["evaluation_artifact"] = replace_artifact(tmp_path, binding, missing)
    with pytest.raises(ValueError, match="CANDIDATE_OR_OBSERVATION"):
        compare_frozen_metrics(packet, tmp_path)


@pytest.mark.parametrize("value", [True, "0.1", None, 10**1000])
def test_invalid_numeric_metric_rejected(packet, tmp_path, value):
    binding = packet["methods"][1]["evaluation_artifact"]
    packet["methods"][1]["evaluation_artifact"] = replace_artifact(
        tmp_path, binding, lambda report: report["metric_rows"][0].update(value=value))
    with pytest.raises(ValueError, match="METRIC_REQUIRED"):
        compare_frozen_metrics(packet, tmp_path)


@pytest.mark.parametrize("flag,value", [("synthetic", True), ("used_for_selection", True), ("model_updates", 1), ("model_updates", False)])
def test_final_label_tuning_or_model_updates_are_rejected(packet, tmp_path, flag, value):
    binding = packet["methods"][1]["evaluation_artifact"]
    packet["methods"][1]["evaluation_artifact"] = replace_artifact(
        tmp_path, binding, lambda report: report.update({flag: value}))
    with pytest.raises(ValueError, match="FROZEN_FINAL_RESULTS"):
        compare_frozen_metrics(packet, tmp_path)


def test_expression_metric_not_replaced_by_Annexin_MAE(packet, tmp_path):
    packet["hypothesis"] = "A"
    packet["metric"] = "mean_absolute_error"
    with pytest.raises(ValueError, match="METRIC_NOT_FOR_THIS_HYPOTHESIS"):
        compare_frozen_metrics(packet, tmp_path)


def test_changed_source_or_result_bytes_rejected(packet, tmp_path):
    (tmp_path / "source.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        compare_frozen_metrics(packet, tmp_path)


def test_duplicate_json_key_never_silently_accepted(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"same":1,"same":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        read(path)


@pytest.mark.parametrize("metric", ["mean_mse", "population_mmd_v", "mean_absolute_error", "measured_candidate_regret"])
def test_negative_error_metric_is_never_a_gain(packet, tmp_path, metric):
    if metric in {"mean_mse", "population_mmd_v"}:
        hypothesis, evidence = "A", "measured_expression"
    else:
        hypothesis, evidence = ("C" if metric == "mean_absolute_error" else "D"), "measured_functional"
    packet.update(hypothesis=hypothesis, metric=metric)
    packet["metadata_artifact"] = replace_artifact(tmp_path, packet["metadata_artifact"],
        lambda value: value.update(evidence=evidence))
    for method in packet["methods"]:
        method["evaluation_artifact"] = replace_artifact(tmp_path, method["evaluation_artifact"],
            lambda value: value.update(hypothesis=hypothesis, metric=metric))
    binding = packet["methods"][1]["evaluation_artifact"]
    packet["methods"][1]["evaluation_artifact"] = replace_artifact(tmp_path, binding,
        lambda value: value["metric_rows"][0].update(value=-.1))
    with pytest.raises(ValueError, match="NONNEGATIVE"):
        compare_frozen_metrics(packet, tmp_path)


@pytest.mark.parametrize("domain", ["SYTOX_lethal_fraction", "growth", "Annexin_fraction", "invented_caspase"])
def test_caspase_metric_rejects_other_measurement_domains(packet, tmp_path, domain):
    packet.update(hypothesis="C", metric="relative_caspase_MAE")
    packet["metadata_artifact"] = replace_artifact(tmp_path, packet["metadata_artifact"],
        lambda value: value["scope"].update(measurement_domain_id=domain))
    with pytest.raises(ValueError, match="ENDPOINT_METRIC_SCALE"):
        compare_frozen_metrics(packet, tmp_path)


def test_supported_Annexin_fold_stays_on_its_original_scale(packet, tmp_path):
    packet.update(hypothesis="C", metric="annexin_fold_MAE")
    def scope(value):
        value["scope"].update(measurement_domain_id="Sarin2017_Annexin_fold",
            target_coordinate="fold_change_relative_to_respective_untreated_control",
            endpoint_definition={"signal": "FITC_Annexin_V", "normalization": "respective_untreated_control"})
    packet["metadata_artifact"] = replace_artifact(tmp_path, packet["metadata_artifact"], scope)
    for method in packet["methods"]:
        def update(value):
            scope(value)
            value.update(hypothesis="C", metric="annexin_fold_MAE")
        method["evaluation_artifact"] = replace_artifact(tmp_path, method["evaluation_artifact"], update)
    assert compare_frozen_metrics(packet, tmp_path)["metric"] == "annexin_fold_MAE"


@pytest.mark.parametrize("valid", [True, False])
def test_caspase_needs_source_domain_coordinate_and_readout(packet, tmp_path, valid):
    from lung.data.martin_state_function import SCALE
    packet.update(hypothesis="C", metric="relative_caspase_MAE")
    def scope(value):
        value["scope"].update(measurement_domain_id="Martin2022_S2F", target_coordinate=SCALE,
            endpoint_definition={"signal": "distinct_green_caspase_3_7_spots/confluence",
                                 "summary": "source_reported_72h_AUC" if valid else "growth"})
    packet["metadata_artifact"] = replace_artifact(tmp_path, packet["metadata_artifact"], scope)
    for method in packet["methods"]:
        def update(value):
            scope(value)
            value.update(hypothesis="C", metric="relative_caspase_MAE")
        method["evaluation_artifact"] = replace_artifact(tmp_path, method["evaluation_artifact"], update)
    if valid:
        assert compare_frozen_metrics(packet, tmp_path)["metric"] == "relative_caspase_MAE"
    else:
        with pytest.raises(ValueError, match="ENDPOINT_METRIC_SCALE"):
            compare_frozen_metrics(packet, tmp_path)


def test_state_ID_baseline_can_isolate_high_dimensional_increment(packet, tmp_path):
    packet.update(hypothesis="A", metric="mean_mse")
    packet["methods"][0]["feature_use"] = "state_ID_only"
    packet["methods"][1]["feature_use"] = "full_state"
    packet["metadata_artifact"] = replace_artifact(tmp_path, packet["metadata_artifact"],
        lambda value: value.update(evidence="measured_expression"))
    for method in packet["methods"]:
        method["evaluation_artifact"] = replace_artifact(tmp_path, method["evaluation_artifact"],
            lambda value: value.update(hypothesis="A", metric="mean_mse"))
    result = compare_frozen_metrics(packet, tmp_path)
    assert result["comparisons"][0]["mean_improvement"] == pytest.approx(.1)
    assert result["whole_project_qualified"] is False
