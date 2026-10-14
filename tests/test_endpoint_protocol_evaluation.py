"""Metadata and synthetic scores only; no model training or inference."""
import copy

import pytest

from lung.data.endpoint_protocol_evaluation import (
    EndpointProtocolEvaluationPlan, build_plan, make_plan, validate_plan,
)
from lung.data.endpoint_protocol_supervision import file_sha


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return build_plan(tmp_path_factory.mktemp("evaluation") / "v1")


def test_two_explicit_roles_for_viel_and_complete_protocol_pairs(plan):
    assert len(plan.fold_ids) == 10
    heldout = []
    for fold in plan.fold_ids:
        rows = plan.metadata_units(fold)
        assert len(rows) == 24 and len(plan.fit_units(fold)) == 16
        selection = plan.evaluation_units(fold, role="selection")
        assert len(selection) == 2
        role_ids = [{r["observation_id"] for r in plan.metadata_units(fold, role=role)}
                    for role in ("fit", "selection", "final_test", "external_test")]
        assert sum(len(s) for s in role_ids) == len(set.union(*role_ids)) == 24
        test = plan.evaluation_units(fold)
        assert len(test) == 2 and {r["role"] for r in test} == {"final_test"}
        assert len({r["p_request_id"] for r in test}) == 1
        heldout.extend(r["observation_id"] for r in test)
        assert "label_fraction" not in rows[0] and "source_record" not in rows[0]
        assert not {r["observation_id"] for r in test} & {r["observation_id"] for r in plan.fit_units(fold)}
    assert len(set(heldout)) == 20


def test_nav_alternatives_share_Y_and_roles_without_extra_observations(plan):
    main = plan.metadata_units(plan.fold_ids[0])
    alt = plan.metadata_units(plan.fold_ids[0], "nav_methods_sensitivity")
    assert {r["observation_id"]: r["role"] for r in main} == {r["observation_id"]: r["role"] for r in alt}
    assert len(plan.evaluation_units(plan.fold_ids[0], role="external_test")) == 4


@pytest.mark.parametrize("mutation", ["duplicate", "pair_leak", "alternative_leak", "endpoint_mix"])
def test_role_and_scope_rejections(mutation):
    data = make_plan()
    fold = data["folds"][0]
    rows = fold["branches"]["main"]["units"]
    if mutation == "duplicate":
        rows[1]["observation_id"] = rows[0]["observation_id"]
    elif mutation == "pair_leak":
        next(r for r in rows if r["role"] == "final_test")["role"] = "fit"
    elif mutation == "alternative_leak":
        next(r for r in fold["branches"]["nav_methods_sensitivity"]["units"] if r["role"] == "external_test")["role"] = "fit"
    else:
        rows[0]["endpoint_definition"] = {}
    with pytest.raises(ValueError):
        validate_plan(data)


def receipt(plan, tmp_path, block_id="paired_protocol_holdout", scores=None):
    fold = plan.fold_ids[0]
    block = next(b for b in plan.blocks(fold) if b["block_id"] == block_id)
    artifact = tmp_path / "mock-frozen-artifact.txt"
    artifact.write_text("synthetic fixture, no model", encoding="utf-8")
    predictions = {cid: 0.5 for cid in block["candidate_ids"]} if scores is None else scores
    return plan.freeze_predictions(fold, block_id, predictions,
        artifacts={str(artifact): file_sha(artifact)})


def test_ties_retained_and_mock_measured_metrics_do_not_certify_models(plan, tmp_path):
    frozen = receipt(plan, tmp_path)
    rows = plan.metadata_units(plan.fold_ids[0], role="final_test")
    result = plan.evaluate_frozen(frozen, endpoint_definition=rows[0]["endpoint_definition"])
    assert len(result["selected_candidate_ids"]) == 2
    assert result["biological_replicate_count"] is None
    assert not result["execution_certified"] and result["retrospective"]
    assert not result["reading_bounds_are_SD_or_CI"]


def test_incomplete_and_extra_candidate_scores_rejected(plan, tmp_path):
    ids = plan.blocks(plan.fold_ids[0])[0]["candidate_ids"]
    for scores in ({ids[0]: 0.2}, {**dict.fromkeys(ids, 0.2), "invented": .4}):
        with pytest.raises(ValueError, match="COVERAGE"):
            receipt(plan, tmp_path, scores=scores)


@pytest.mark.parametrize("boolean", [False, True])
def test_boolean_is_not_a_numeric_fraction_prediction(plan, tmp_path, boolean):
    ids = plan.blocks(plan.fold_ids[0])[0]["candidate_ids"]
    with pytest.raises(ValueError, match="FRACTION_PREDICTION_REQUIRED"):
        receipt(plan, tmp_path, scores={ids[0]: boolean, ids[1]: .5})


def test_prediction_and_endpoint_and_fold_binding(plan, tmp_path):
    frozen = receipt(plan, tmp_path)
    definition = plan.metadata_units(plan.fold_ids[0], role="final_test")[0]["endpoint_definition"]
    with pytest.raises(ValueError, match="ENDPOINT"):
        plan.evaluate_frozen(frozen, endpoint_definition={"assay": "other"})
    mutated = copy.deepcopy(frozen)
    mutated["predictions"][next(iter(mutated["predictions"]))] = .9
    with pytest.raises(ValueError, match="PREDICTION_CHANGED"):
        plan.evaluate_frozen(mutated, endpoint_definition=definition)
    mutated = copy.deepcopy(frozen)
    mutated["identity"]["fold_id"] = plan.fold_ids[1]
    with pytest.raises(ValueError):
        plan.evaluate_frozen(mutated, endpoint_definition=definition)


def test_nav_functional_challenge_requires_scope_specific_G(plan, tmp_path):
    block = next(b for b in plan.blocks(plan.fold_ids[0]) if b["role"] == "external_test")
    frozen = receipt(plan, tmp_path, block_id=block["block_id"])
    definition = plan.metadata_units(plan.fold_ids[0], role="external_test")[0]["endpoint_definition"]
    with pytest.raises(ValueError, match="CALIBRATION_PENDING"):
        plan.evaluate_frozen(frozen, endpoint_definition=definition)


def test_tampered_plan_rejected_even_if_caller_rehashes(plan, tmp_path):
    import json
    from lung.data.endpoint_protocol_evaluation import digest
    output = tmp_path / "tamper"
    build_plan(output)
    data = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    data["folds"].reverse()
    (output / "plan.json").write_text(json.dumps(data), encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["role_contract_sha256"] = digest(data)
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="METADATA"):
        EndpointProtocolEvaluationPlan.load(output)


def test_selection_block_metrics_are_limited_and_do_not_grant_qualification(plan, tmp_path):
    block_id = "paired_protocol_reward_validation"
    frozen = receipt(plan, tmp_path, block_id=block_id)
    rows = plan.metadata_units(plan.fold_ids[0], role="selection")
    result = plan.evaluate_frozen(frozen, endpoint_definition=rows[0]["endpoint_definition"])
    assert result["role"] == "selection"
    assert result["reward_qualification_role_eligible"]
    assert result["G_reward_qualified"] is False
    final_receipt = receipt(plan, tmp_path)
    final = plan.evaluate_frozen(final_receipt, endpoint_definition=rows[0]["endpoint_definition"])
    assert final["reward_qualification_role_eligible"] is False


def test_selection_is_cyclic_next_metadata_group_and_same_alternative_roles():
    data = make_plan()
    folds = data["folds"]
    for i, fold in enumerate(folds):
        assert fold["selection_metadata_group"] == folds[(i + 1) % len(folds)]["heldout_metadata_group"]
        main = fold["branches"]["main"]["units"]
        alt = fold["branches"]["nav_methods_sensitivity"]["units"]
        assert {r["observation_id"]: r["role"] for r in main} == {r["observation_id"]: r["role"] for r in alt}


def test_historical_v1_reconstructs_original_hash_without_modifying_files():
    from lung.data.endpoint_protocol_evaluation import DEFAULT_PLAN
    historical = EndpointProtocolEvaluationPlan.load(DEFAULT_PLAN.with_name("evaluation-plan-v1"))
    assert historical.manifest["role_contract_sha256"] == "426171fe139101f0805d7733fffe4bb01462ab70c136f2d11df4d915aad600e3"
    assert len(historical.fit_units(historical.fold_ids[0])) == 18


@pytest.mark.parametrize("version", ["evaluation-plan-v1", "evaluation-plan-v2"])
def test_real_metadata_loader_never_deserializes_Y_or_calls_legacy_prepare(monkeypatch, version):
    from pathlib import Path
    from lung.data import endpoint_protocol_inputs, endpoint_protocol_supervision
    from lung.data.endpoint_protocol_evaluation import DEFAULT_PLAN
    import lung.data.endpoint_protocol_evaluation as evaluation

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata loader reached measured labels/source reconstruction")

    monkeypatch.setattr(endpoint_protocol_inputs, "prepare", forbidden)
    monkeypatch.setattr(endpoint_protocol_inputs.EndpointProtocolInputs, "load", forbidden)
    monkeypatch.setattr(endpoint_protocol_supervision.EndpointProtocolSupervision, "load", forbidden)
    monkeypatch.setattr(evaluation, "load_views", forbidden)
    original = Path.read_text
    forbidden_names = {"observations.json", "reference_observations.json",
        "vielhauer_source_nominal.json", "nav_figure_caption_10uM.json", "nav_methods_11uM.json"}

    def guarded_read(path, *args, **kwargs):
        assert path.name not in forbidden_names, "measured label JSON deserialized"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    plan = EndpointProtocolEvaluationPlan.load(DEFAULT_PLAN.with_name(version))
    assert len(plan.metadata_units(plan.fold_ids[0])) == 24
    assert plan.identity(plan.fold_ids[0])["role_contract_sha256"] == plan.manifest["role_contract_sha256"]
