"""Metadata-only real readback and synthetic fixtures; no real final Y/model."""
from copy import deepcopy
from pathlib import Path

import pytest

from lung.data import endpoint_protocol_inputs, endpoint_protocol_supervision
from lung.data.joint_selection_evaluation import (
    JointSelectionEvaluation, OBJECTIVE, SCORE_SCHEMA, build_view, make_view,
    metric_result, sha, source_role_rows, validate_view)


@pytest.fixture(scope="module")
def view():
    return JointSelectionEvaluation(make_view())


def packet(view, tmp_path, role="final_test"):
    files = {}
    for name in ("information", "response", "G", "simple"):
        path = tmp_path / (name + ".fixture")
        path.write_text("synthetic fixture only, no scientific execution: " + name, encoding="utf-8")
        files[name] = {"path": path.name, "sha256": sha(path)}
    ids = sorted(r["candidate_id"] for r in view.metadata_units(role))
    methods = {}
    for name, model in (("direct_G", "G"), ("simple_rank", "simple")):
        methods[name] = {"scores": {cid: (i + 1) / 20 for i, cid in enumerate(ids)},
            "model_artifact": files[model], "information_set_id": "shared-fixture-information",
            "feedback_G_artifact": files["G"], "feature_use": "declared-fixture-feature-use",
            "score_definition": "estimated_terminal_Annexin_fraction"}
    return {"schema": SCORE_SCHEMA, "view_identity": view.identity(), "role": role,
        "objective": deepcopy(OBJECTIVE), "information_set_id": "shared-fixture-information",
        "information_artifact": files["information"], "response_feature_artifact": files["response"],
        "feedback_G_artifact": files["G"],
        "fit_observation_ids": sorted(r["observation_id"] for r in view.metadata_units("fit")),
        "selection_observation_ids": sorted(r["observation_id"] for r in view.metadata_units("selection")),
        "final_Y_used_for_fitting_or_selection": False, "synthetic_predictions": False,
        "model_updates_during_scoring": 0, "methods": methods}


def fixture_labels(view, role):
    ids = sorted(r["candidate_id"] for r in view.metadata_units(role))
    return {cid: {"label_fraction": (i + 1) / 20,
                  "reading_lower_fraction": i / 20, "reading_upper_fraction": (i + 2) / 20}
            for i, cid in enumerate(ids)}


def forbid_Y(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata path reached labels or source reconstruction")
    monkeypatch.setattr(endpoint_protocol_inputs, "prepare", forbidden)
    monkeypatch.setattr(endpoint_protocol_inputs.EndpointProtocolInputs, "load", forbidden)
    monkeypatch.setattr(endpoint_protocol_supervision.EndpointProtocolSupervision, "load", forbidden)
    monkeypatch.setattr(JointSelectionEvaluation, "_measured_units", forbidden)
    original = Path.read_text
    names = {"observations.json", "reference_observations.json", "vielhauer_source_nominal.json",
             "nav_figure_caption_10uM.json", "nav_methods_11uM.json", "pixel_landmarks.json"}
    def guarded(path, *args, **kwargs):
        assert path.name not in names, "measured Y JSON deserialized during metadata/freeze"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)


def test_real_build_load_preflight_freeze_never_deserialize_Y(monkeypatch, tmp_path):
    forbid_Y(monkeypatch)
    directory = tmp_path / "new_view"
    view = build_view(directory)
    result = JointSelectionEvaluation.load(directory).preflight()
    assert result["role_counts"] == {"fit": 8, "selection": 4, "final_test": 8}
    assert result["final_drug_dose_protocol_grid"] == [2, 2, 2]
    assert result["label_values_deserialized"] is False
    assert not result["whole_project_qualified"]
    assert len(result["shared_controls"]) == 2
    assert not view.freeze_scores(packet(view, tmp_path), root=tmp_path)["execution_certified"]


def test_each_complete_drug_dose_group_stays_one_new_role(view):
    groups = {}
    for unit in view.metadata_units():
        candidate = unit["complete_candidate"]
        key = (candidate["chemical"]["compound_id"], candidate["dose_uM"])
        groups.setdefault(key, []).append(unit)
    assert len(groups) == 10
    assert all(len(units) == 2 and len({r["role"] for r in units}) == 1 for units in groups.values())
    final = view.metadata_units("final_test")
    assert {r["complete_candidate"]["dose_uM"] for r in final} == {0.0005, 0.04}
    assert all(r["source_condition_metadata"]["cross_figure_replicate_pairing"] == "not_established" for r in final)


def test_constructor_cannot_bypass_source_bound_roles():
    changed = make_view()
    changed["units"][0]["role"] = "final_test"
    with pytest.raises(ValueError, match="METADATA_OR_ROLE"):
        JointSelectionEvaluation(changed)


@pytest.mark.parametrize("mutation", ["role", "duplicate_Y", "endpoint", "control", "clock", "cross_fold", "pairing", "cost"])
def test_view_rejects_semantic_tampering_even_without_hash_guard(mutation):
    value = make_view()
    unit = value["units"][0]
    if mutation == "role":
        unit["role"] = "final_test"
    elif mutation == "duplicate_Y":
        value["units"][1]["observation_id"] = unit["observation_id"]
    elif mutation in {"endpoint", "control", "clock"}:
        key = {"endpoint": "endpoint_definition", "control": "shared_control_id", "clock": "protocol"}[mutation]
        unit["complete_candidate"][key] = {}
    elif mutation == "cross_fold":
        value["cross_fold_scores_allowed"] = True
    elif mutation == "pairing":
        value["cross_figure_biological_pairing_verified"] = True
    else:
        value["objective"]["cost_or_shaping_terms"] = ["dose_penalty"]
    with pytest.raises(ValueError, match="METADATA_OR_ROLE"):
        validate_view(value)


@pytest.mark.parametrize("mutation", ["missing", "extra", "fold", "information", "feedback", "role", "finalY", "model_per_candidate", "missing_method", "coordinate", "model_update_bool"])
def test_complete_same_information_same_feedback_single_model_boundaries(view, tmp_path, mutation):
    value = packet(view, tmp_path)
    G = value["methods"]["direct_G"]
    if mutation == "missing":
        G["scores"].pop(next(iter(G["scores"])))
    elif mutation == "extra":
        G["scores"]["unmeasured_candidate"] = .5
    elif mutation == "fold":
        value["view_identity"]["experiment_id"] = "viel_leave_group_01"
    elif mutation == "information":
        value["methods"]["simple_rank"]["information_set_id"] = "extra-test-information"
    elif mutation == "feedback":
        value["methods"]["simple_rank"]["feedback_G_artifact"] = G["model_artifact"] | {"sha256": "wrong"}
    elif mutation == "role":
        value["fit_observation_ids"].append(view.metadata_units("final_test")[0]["observation_id"])
    elif mutation == "finalY":
        value["final_Y_used_for_fitting_or_selection"] = True
    elif mutation == "model_per_candidate":
        G["candidate_model_artifacts"] = dict.fromkeys(G["scores"], G["model_artifact"])
    elif mutation == "missing_method":
        value["methods"].pop("direct_G")
    elif mutation == "coordinate":
        value["methods"]["simple_rank"]["score_definition"] = "arbitrary_rank_energy"
    else:
        value["model_updates_during_scoring"] = False
    with pytest.raises(ValueError):
        view.freeze_scores(value, root=tmp_path)


@pytest.mark.parametrize("invalid", [True, False, "0.2", float("nan"), float("inf"), 10 ** 400, -0.01, 1.01])
def test_G_fraction_score_boundaries(view, tmp_path, invalid):
    value = packet(view, tmp_path)
    G = value["methods"]["direct_G"]["scores"]
    G[next(iter(G))] = invalid
    with pytest.raises(ValueError):
        view.freeze_scores(value, root=tmp_path)


@pytest.mark.parametrize("include_simple_rank", [True, False])
def test_explicit_evaluation_uses_mock_Y_full_domain_and_prefrozen_fixed_subsets(view, tmp_path, monkeypatch, include_simple_rank):
    monkeypatch.setattr(JointSelectionEvaluation, "_measured_units", lambda self, role: fixture_labels(self, role))
    value = packet(view, tmp_path)
    if not include_simple_rank:
        value["methods"].pop("simple_rank")
    frozen = view.freeze_scores(value, root=tmp_path)
    result = view.evaluate_frozen(frozen, root=tmp_path)
    methods = result["results"]
    assert ("simple_rank" in methods) is include_simple_rank
    full = {r["candidate_id"] for r in view.metadata_units("final_test")}
    assert all(set(r["candidate_domain"]) == full for r in methods.values())
    assert methods["direct_G"]["measured_candidate_regret"] == 0
    assert methods["direct_G"]["mean_squared_error"] == 0
    assert methods["direct_G"]["pairwise_order_comparisons"] == 28
    assert [len(methods[n]["allowed_policy_candidate_ids"]) for n in
            ("fixed_drug", "fixed_dose", "fixed_exposure", "fixed_drug_dose_exposure")] == [4, 4, 4, 1]
    assert methods["uniform_legal_candidates"]["expected_measured_selected_fraction"] == pytest.approx(.225)
    assert not result["execution_certified"] and not result["whole_project_qualified"]
    assert result["independent_biological_sample_count"] is None


def test_ties_exact_uniform_expectation_and_pixel_sensitivity():
    labels = {"a": {"label_fraction": .4, "reading_lower_fraction": .3, "reading_upper_fraction": .5},
              "b": {"label_fraction": .6, "reading_lower_fraction": .5, "reading_upper_fraction": .7}}
    result = metric_result({"a": .5, "b": .5}, labels, {"a": .5, "b": .5})
    assert result["expected_measured_selected_fraction"] == .5
    assert result["measured_candidate_regret"] == pytest.approx(.1)
    assert result["mean_absolute_error"] == pytest.approx(.1)
    assert result["mean_squared_error"] == pytest.approx(.01)
    assert result["selected_fraction_reading_interval"] == [.4, .6]
    assert result["regret_reading_interval"] == pytest.approx([0, .2])
    assert result["pixel_robust_order_comparisons"] == 0
    assert result["possibly_best_under_pixel_intervals"] == ["a", "b"]
    assert not result["reading_bounds_are_biological_CI"]


def test_changed_artifact_or_receipt_rejected_before_Y(view, tmp_path, monkeypatch):
    forbid_Y(monkeypatch)
    frozen = view.freeze_scores(packet(view, tmp_path), root=tmp_path)
    changed = deepcopy(frozen)
    changed_scores = changed["packet"]["methods"]["simple_rank"]["scores"]
    changed_scores[next(iter(changed_scores))] = -.2
    with pytest.raises(ValueError, match="RECEIPT_CHANGED"):
        view.evaluate_frozen(changed, root=tmp_path)
    (tmp_path / "simple.fixture").write_text("changed fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="ARTIFACT_CHANGED"):
        view.evaluate_frozen(frozen, root=tmp_path)


def test_interval_box_upper_regret_is_valid_for_all_corner_readings():
    from itertools import product
    labels = {"a": {"label_fraction": .3, "reading_lower_fraction": .2, "reading_upper_fraction": .4},
              "b": {"label_fraction": .6, "reading_lower_fraction": .5, "reading_upper_fraction": .7},
              "c": {"label_fraction": .5, "reading_lower_fraction": .4, "reading_upper_fraction": .6}}
    probabilities = {"a": .5, "b": .5, "c": 0.0}
    result = metric_result({"a": .5, "b": .5, "c": .3}, labels, probabilities)
    regrets = []
    for values in product(*[(r["reading_lower_fraction"], r["reading_upper_fraction"]) for r in labels.values()]):
        regrets.append(max(values) - sum(probabilities[cid] * value for cid, value in zip(labels, values)))
    lower, upper = result["regret_reading_interval"]
    assert lower <= min(regrets) and upper == pytest.approx(max(regrets))


@pytest.mark.parametrize("mutation", ["bounds_reversed", "negative_Y", "bool_probability", "zero_sum"])
def test_invalid_label_coordinate_or_probability_rejected(mutation):
    labels = {"a": {"label_fraction": .4, "reading_lower_fraction": .3, "reading_upper_fraction": .5},
              "b": {"label_fraction": .6, "reading_lower_fraction": .5, "reading_upper_fraction": .7}}
    probabilities = {"a": .5, "b": .5}
    if mutation == "bounds_reversed":
        labels["a"]["reading_lower_fraction"] = .6
    elif mutation == "negative_Y":
        labels["a"]["label_fraction"] = -.4
    elif mutation == "bool_probability":
        probabilities["a"] = True
    else:
        probabilities = {"a": 0.0, "b": 0.0}
    with pytest.raises(ValueError):
        metric_result({"a": .5, "b": .5}, labels, probabilities)


def test_role_reader_does_not_decode_final_records_during_fit_or_selection(tmp_path, monkeypatch):
    import json
    from lung.data import joint_selection_evaluation as module
    path = tmp_path / "source.fixture.json"
    rows = [{"candidate_id": cid, "label_fraction": .5,
             "source_record": {"escaped_context": 'a quote" and a brace }', "nested": [{"another": "{"}]}}
            for cid in ("fit_candidate", "final_candidate", "selection_candidate")]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    original = json.loads
    decoded = []
    def guarded(raw, *args, **kwargs):
        assert '"candidate_id": "final_candidate"' not in raw, "final Y decoded before frozen final evaluation"
        result = original(raw, *args, **kwargs)
        decoded.append(result["candidate_id"])
        return result
    monkeypatch.setattr(module.json, "loads", guarded)
    for cid in ("fit_candidate", "selection_candidate"):
        assert set(source_role_rows(path, {cid}, {r["candidate_id"] for r in rows})) == {cid}
    assert decoded == ["fit_candidate", "selection_candidate"]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown", "truncated"])
def test_source_role_reader_rejects_wrong_complete_source_ID_coverage(tmp_path, mutation):
    import json
    rows = [{"candidate_id": cid, "label_fraction": .5} for cid in ("fit", "selection", "final")]
    if mutation == "duplicate":
        rows[1]["candidate_id"] = "fit"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unknown":
        rows[1]["candidate_id"] = "invented"
    raw = json.dumps(rows, indent=2)
    if mutation == "truncated":
        raw = raw[:-15]
    path = tmp_path / "broken.fixture.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        source_role_rows(path, {"fit"}, {"fit", "selection", "final"})


def test_public_measured_API_demands_role_before_label_io(view, monkeypatch):
    import lung.data.joint_selection_evaluation as module
    monkeypatch.setattr(module, "check_supervision_bytes", lambda *args: pytest.fail("invalid role reached source labels"))
    for role in (None, "E", "unassigned", "all", "final_test"):
        with pytest.raises(ValueError, match="EXPLICIT_MEASURED_ROLE"):
            view.measured_units(role)


def test_finite_but_numerically_overflowing_squared_error_rejected():
    labels = {cid: {"label_fraction": .5, "reading_lower_fraction": .4, "reading_upper_fraction": .6}
              for cid in ("a", "b")}
    with pytest.raises(ValueError, match="FINITE_SCORE"):
        metric_result({"a": 1e308, "b": .5}, labels, {"a": 1.0, "b": 0.0})
