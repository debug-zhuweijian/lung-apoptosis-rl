"""Synthetic response moments/labels only; real source metadata is read-only."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from lung.data.joint_selection_evaluation import JointSelectionEvaluation
from lung.engine import full_protocol_response as producer
from lung.engine import joint_selection_learning as learning
from lung.models.full_protocol import DOSE_UNITS, EVENT_TYPES, FullPhysicalProtocolBatch, SCALAR_FIELDS


@pytest.fixture
def synthetic_state_index(synthetic_env, monkeypatch):
    """Original-class probability fixture; no C weights or real RNA inference."""
    env = synthetic_env
    predictions = learning.read(env["index"])["predictions"]
    source = env["root"] / "experiments/synthetic_state_sidecars"
    for i, record in enumerate(predictions):
        value = {"schema": "raw-sorting-state-sidecar-v1", "candidate_id": record["candidate_id"],
            "prediction": {k: record[k] for k in ("path", "sha256", "prediction_identity")},
            "state_asset": {"synthetic_fixture": True, "state_identity": "fixture-original-three-classes"},
            "class_order": list(learning.STATE_CLASSES), "sample_count": 2431, "predict_domain": producer.OUTLET,
            "sampling": {"seed": 77, "steps": 32, "microbatch": 128, "device": "cpu",
                "policy": "exact_float64_P_moments_replay"},
            "probability_mean": [.2 + i * .005, .4 - i * .005, .4],
            "probability_variance": [.01, .01, .01], "source_C_transfer_validated": False,
            "cross_source_C_calibration_verified": False, "Annexin_interpretation": False, "scientific_qualified": False}
        learning.write(source / record["candidate_id"] / "state_sidecar.json", value)
    monkeypatch.setattr(learning, "read_state_sidecar", learning.read)
    index = env["root"] / "experiments/state_index.json"
    learning.build_state_sidecar_index(source, index)
    return env, index


def rewrite_state_fixture(index, env, mutate):
    body = learning.read(index)
    record = body["sidecars"][0]
    path = env["root"] / record["path"]
    value = learning.read(path)
    mutate(value)
    path.write_text(learning.canonical(value), encoding="utf-8")
    record["sha256"] = learning.sha(path)
    index.write_text(learning.canonical(body), encoding="utf-8")


def test_raw_C_features_preserve_original_classes_and_functional_roles(synthetic_state_index):
    env, index = synthetic_state_index
    base = learning.prepare_inputs(env["index"])
    prepared = learning.prepare_inputs(env["index"], state_sidecar_index=index)
    assert prepared.full.shape == (20, base.full.shape[1] + 3)
    assert prepared.full_names[-3:] == tuple("raw_sorting_C.probability_mean." + c for c in learning.STATE_CLASSES)
    assert np.array_equal(prepared.full[:, :-3], base.full)
    assert prepared.identity["role_observation_ids"] == base.identity["role_observation_ids"]
    assert learning.information_record(prepared)["C_is_measured_Annexin"] is False
    assert env["calls"] == []


@pytest.mark.parametrize("change,reason", [
    (lambda v: v.update(Annexin_interpretation=True), "NOT_ANNEXIN"),
    (lambda v: v["prediction"].update(prediction_identity="wrong-P"), "ACTUAL_PREDICTION"),
    (lambda v: v["state_asset"].update(state_identity="different-C"), "SAME_FROZEN_C"),
    (lambda v: v.update(probability_mean=[True, False, False]), "PROBABILITY_FEATURES")])
def test_optional_C_rejects_semantic_or_binding_drift(synthetic_state_index, change, reason):
    env, index = synthetic_state_index
    rewrite_state_fixture(index, env, change)
    with pytest.raises(ValueError, match=reason):
        learning.prepare_inputs(env["index"], state_sidecar_index=index)
    assert env["calls"] == []


def test_optional_C_frozen_evaluation_reuses_same_bound_features(synthetic_state_index):
    env, index = synthetic_state_index
    output = env["root"] / "experiments/models_with_C"
    learning.fit_models(env["index"], output, state_sidecar_index=index)
    assert env["calls"] == ["fit"]
    receipt = env["root"] / "experiments/with_C_frozen.json"
    frozen = learning.freeze_scores(output, receipt, role="final_test", prediction_index=env["index"], state_sidecar_index=index)
    assert frozen["state_sidecar_index"]["sha256"] == learning.sha(index)
    assert env["calls"] == ["fit"]
    metrics = learning.evaluate_frozen(receipt, env["root"] / "experiments/with_C_metrics.json")
    assert env["calls"] == ["fit", "final_test"]
    assert metrics["formal_World_or_Actor_qualified"] is False


@pytest.fixture(scope="module")
def actual_metadata():
    view = JointSelectionEvaluation.load()
    reader, requests, context, bindings = producer.bound_inputs()
    return view, reader, requests, context, bindings


def synthetic_numeric(request):
    events = request["physical"]["events"]
    fp = np.zeros((1, 4, 2048), dtype=np.float32)
    known_fp, vehicle = np.zeros((1, 4), dtype=bool), np.zeros((1, 4), dtype=bool)
    event_type, dose_unit = np.zeros((1, 4), dtype=np.int64), np.zeros((1, 4), dtype=np.int64)
    values, known = np.full((1, 4, 8), np.nan, dtype=np.float32), np.zeros((1, 4, 8), dtype=bool)
    for slot, event in enumerate(events):
        if event["fingerprint"]:
            fp[0, slot, event["fingerprint"]["row"]] = 1
            known_fp[0, slot] = True
        vehicle[0, slot] = event["vehicle"]
        event_type[0, slot] = EVENT_TYPES.index(event["event_type"])
        dose_unit[0, slot] = DOSE_UNITS.index(event["dose_unit"])
        for index, name in enumerate(SCALAR_FIELDS):
            value = event["values"][name]
            if value is not None:
                values[0, slot, index], known[0, slot, index] = value, True
    physical = FullPhysicalProtocolBatch(*(torch.from_numpy(array) for array in (
        fp, known_fp, vehicle, event_type, dose_unit, values, known,
        np.array([len(events)], dtype=np.int64), np.array([24.], dtype=np.float32), np.array([True]))))
    kwargs = {"fingerprint": physical.fingerprint[:, 0].clone()}
    for key in ("dose", "duration", "elapsed", "horizon", "vehicle", "background", "source", "protocol"):
        kwargs[key] = torch.tensor([request["base_kwargs"][key]], dtype=torch.bool if key == "vehicle"
            else torch.long if key in {"source", "background", "protocol"} else torch.float32)
    producer.validate_batch(kwargs, physical)
    return kwargs, physical


@pytest.fixture
def synthetic_env(monkeypatch, tmp_path, actual_metadata):
    view, reader, requests, context, bindings = actual_metadata
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    monkeypatch.setattr(producer, "bound_inputs", lambda: (reader, deepcopy(requests), context, deepcopy(bindings)))
    registration = {"registry_path": "fixture_registry_only", "registry_sha256": "0" * 64,
                    "audit_id": "synthetic_authority_mock_not_production"}
    monkeypatch.setattr(producer, "registered_audit", lambda metadata, bindings: deepcopy(registration))
    by_id = {r["candidate_id"]: r for r in requests}
    monkeypatch.setattr(producer, "numeric_batch", lambda reader, cid, n: synthetic_numeric(by_id[cid]))
    assets = tmp_path / "experiments/fixture_assets"
    assets.mkdir(parents=True)
    models = {}
    for name, stage in (("representation", "latent_ae"), ("response", producer.STAGE)):
        path = assets / (name + ".fixture")
        path.write_bytes(b"synthetic asset bytes only; not a PT or model\n")
        models[name] = {"stage": stage, "synthetic": False, "data_identity": bindings["data_identity"],
            "optimizer_step": 1, "checkpoint": path.relative_to(tmp_path).as_posix(), "checkpoint_sha256": learning.sha(path)}
    models["response"].update(representation_hash=models["representation"]["checkpoint_sha256"],
                             training_protocol_coverage=deepcopy(producer.PARTIAL_PROTOCOL_COVERAGE),
                             reference_contract=producer.reference_contract("observed"))
    metadata = {**models, "bindings": deepcopy(bindings), "registered_audit_id": "fixture_only"}
    reference = {"reference_id": bindings["reference_id"], "reference_rows_sha256": producer.digest(context["reference_rows"]),
        "reference_cell_ids": [r["cell_id"] for r in context["reference_rows"]], "shape": [2431, 32461],
        "reference_role": "fit", "target_expression_read": False}
    source = tmp_path / "experiments/synthetic_predictions"
    units = sorted(view.metadata_units(), key=lambda r: r["candidate_id"])
    for index, unit in enumerate(units):
        cid = unit["candidate_id"]
        body = {"schema": producer.VERSION, "candidate_id": cid, "request": by_id[cid], "bindings": deepcopy(bindings),
            "model_metadata": deepcopy(metadata), "production_registration": deepcopy(registration), "reference": reference,
            "reference_contract": producer.reference_contract("observed"),
            "seed": 77, "steps": 32, "outlet": producer.OUTLET, "execution_kind": "matching_full_protocol_weights",
            "summary": "equal_cell_weighted_full_fit_proxy_common_gene_log1p_mean_variance",
            "training_protocol_coverage": deepcopy(producer.PARTIAL_PROTOCOL_COVERAGE),
            "candidate_protocol_OOD": True, "candidate_protocol_scientifically_validated": False,
            "expression_mean": [0.1 + index / 100] * 32461, "expression_variance": [0.01 + index / 1000] * 32461,
            "reference_target_cell_pairing": False, "source_state_numerically_encoded": False,
            "cross_source_transfer_validated": False, "source_target_RNA_read": False, "functional_Y_read": False,
            "G_outputs": 0, "Q_outputs": 0, "new_roles_assigned": False, "scientific_effect_verified": False}
        body["prediction_identity"] = producer.digest(body)
        learning.write(source / cid / "prediction.json", body)
    index = tmp_path / "experiments/index.json"
    learning.build_prediction_index(source, index)
    calls = []

    def labels(self, role):
        calls.append(role)
        ids = sorted(r["candidate_id"] for r in self.metadata_units(role))
        return {cid: {"label_fraction": .12 + i / 20, "reading_lower_fraction": .1 + i / 20,
                      "reading_upper_fraction": .14 + i / 20} for i, cid in enumerate(ids)}

    monkeypatch.setattr(JointSelectionEvaluation, "_measured_units", labels)
    return {"view": view, "index": index, "source": source, "root": tmp_path, "calls": calls, "by_id": by_id}


def rewrite_prediction(env, mutate, cid=None, *, update_index=True, update_digest=True):
    index = learning.read(env["index"])
    record = next(r for r in index["predictions"] if cid is None or r["candidate_id"] == cid)
    path = env["root"] / record["path"]
    body = learning.read(path)
    mutate(body)
    if update_digest:
        body["prediction_identity"] = producer.digest({k: v for k, v in body.items() if k != "prediction_identity"})
    path.write_text(learning.canonical(body), encoding="utf8")
    if update_index:
        record.update(sha256=learning.sha(path), prediction_identity=body["prediction_identity"])
        env["index"].write_text(learning.canonical(index), encoding="utf8")


def test_pending_preflight_never_reads_Y_or_models(monkeypatch, tmp_path, actual_metadata):
    view, reader, requests, context, bindings = actual_metadata
    monkeypatch.setattr(producer, "bound_inputs", lambda: (reader, requests, context, bindings))
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    monkeypatch.setattr(JointSelectionEvaluation, "_measured_units", lambda *a: pytest.fail("Y read during preflight"))
    monkeypatch.setattr(producer, "read_prediction", lambda *a: pytest.fail("missing prediction read"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("PT read"))
    monkeypatch.setattr(learning, "ridge_parameters", lambda *a: pytest.fail("fit in preflight"))
    result = learning.preflight(tmp_path / "experiments/missing.json")
    assert result["status"] == "real_full_P_predictions_pending" and result["real_predictions_consumed"] == 0
    assert len(result["missing_prediction_candidate_ids"]) == 20 and result["final_Y_deserialized"] is False


def test_full_predictions_prepare_retains_axis_units_history_and_unknowns(synthetic_env, actual_metadata):
    prepared = learning.prepare_inputs(synthetic_env["index"])
    assert prepared.full.shape == (20, 64922 + prepared.condition.shape[1])
    assert prepared.full_names[:2] == tuple("fullP.expression_mean." + g for g in actual_metadata[3]["gene_ids"][:2])
    ids = [u["candidate_id"] for u in prepared.units]
    pulse = next(i for i, cid in enumerate(ids) if "pulse6" in cid)
    continuous = next(i for i, cid in enumerate(ids) if "cont24" in cid)
    names = prepared.condition_names
    assert prepared.condition[pulse, names.index("candidate.exposure_duration_h[h]")] == 6
    assert prepared.condition[continuous, names.index("candidate.exposure_duration_h[h]")] == 24
    assert prepared.condition[pulse, names.index("physical.event3.recovery_hours[declared_unit]")] == 18
    assert prepared.condition[pulse, names.index("physical.event1.wash_count[declared_unit].known")] == 0
    assert prepared.condition[pulse, names.index("source_state_RNA.known")] == 0
    assert prepared.identity["source_state_numerically_encoded"] is False
    assert synthetic_env["calls"] == [] and not prepared.full.flags.writeable


@pytest.mark.parametrize("mutation,reason", [("bytes", "ARTIFACT_BYTES"), ("digest", "CONTENT_CHANGED"),
    ("seed", "SAME_P_MODEL_REFERENCE_SAMPLING"), ("unknown", "REQUEST_OR_BINDINGS"),
    ("negative_variance", "MOMENTS_INVALID"), ("gene_shape", "MOMENTS_INVALID"),
    ("source_state", "OUTLET_OR_SCIENCE")])
def test_prediction_bytes_identity_domain_and_same_sampling_reject(synthetic_env, mutation, reason):
    mutations = {"bytes": lambda b: b.update(steps=64), "digest": lambda b: b.update(steps=64),
        "seed": lambda b: b.update(seed=78),
        "unknown": lambda b: b["request"]["physical"]["events"][0]["values"].update(wash_count=0),
        "negative_variance": lambda b: b["expression_variance"].__setitem__(0, -1),
        "gene_shape": lambda b: b["expression_mean"].pop(), "source_state": lambda b: b.update(source_state_numerically_encoded=True)}
    rewrite_prediction(synthetic_env, mutations[mutation], update_index=mutation != "bytes", update_digest=mutation != "digest")
    with pytest.raises(ValueError, match=reason):
        learning.prepare_inputs(synthetic_env["index"])
    assert synthetic_env["calls"] == []


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_incomplete_source_prediction_domain_rejected_before_Y(synthetic_env, kind):
    value = learning.read(synthetic_env["index"])
    if kind == "missing":
        value["predictions"].pop()
    else:
        value["predictions"][-1] = value["predictions"][0]
    synthetic_env["index"].write_text(learning.canonical(value), encoding="utf8")
    with pytest.raises(ValueError, match="SOURCE_INDEX|UNIQUE_PREDICTION"):
        learning.fit_models(synthetic_env["index"], synthetic_env["root"] / "experiments/no_models")
    assert synthetic_env["calls"] == []


def test_fit_only_normalization_independent_Q_and_full_domains_freeze_evaluate(synthetic_env):
    env = synthetic_env
    directory = env["root"] / "experiments/models"
    manifest = learning.fit_models(env["index"], directory)
    assert env["calls"] == ["fit"]
    prepared = learning.prepare_inputs(env["index"])
    fit = [i for i, u in enumerate(prepared.units) if u["role"] == "fit"]
    G, simple, Q = [learning.read(directory / (name + ".json")) for name in ("G", "simple", "Q")]
    assert np.array_equal(np.asarray(G["parameters"]["mean"]), prepared.full[fit].mean(0))
    assert np.array_equal(np.asarray(simple["parameters"]["mean"]), prepared.condition[fit].mean(0))
    assert G["content_identity"] != Q["content_identity"] and G["parameters"] != Q["parameters"]
    assert Q["feedback_G_content_identity"] == G["content_identity"] and Q["formal_World_or_Actor_qualified"] is False
    assert manifest["selection_Y_used"] is False and manifest["final_Y_used"] is False
    for role, count in (("selection", 4), ("final_test", 8)):
        receipt_path = env["root"] / ("experiments/" + role + "_scores.json")
        receipt = learning.freeze_scores(directory, receipt_path, role=role, prediction_index=env["index"])
        assert env["calls"] == (["fit"] if role == "selection" else ["fit", "selection"])
        methods = receipt["score_receipt"]["packet"]["methods"]
        assert all(len(m["scores"]) == count for m in methods.values())
        assert methods["direct_G"]["model_artifact"] == methods["learned_Q"]["feedback_G_artifact"]
        result = learning.evaluate_frozen(receipt_path, env["root"] / ("experiments/" + role + "_metrics.json"))
        assert len(result["results"]) == 8 and result["source_condition_means"] == count
        assert result["G_reward_qualified"] is False and result["P_heldout_drug_or_protocol_verified"] is False
    assert env["calls"] == ["fit", "selection", "final_test"]


def test_supplied_scores_cannot_bypass_actual_frozen_scorer(synthetic_env):
    env = synthetic_env
    models, receipt = env["root"] / "experiments/models", env["root"] / "experiments/scores.json"
    learning.fit_models(env["index"], models)
    value = learning.freeze_scores(models, receipt, role="final_test", prediction_index=env["index"])
    packet = value["score_receipt"]["packet"]
    cid = next(iter(packet["methods"]["direct_G"]["scores"]))
    packet["methods"]["direct_G"]["scores"][cid] = .99
    value["score_receipt"] = env["view"].freeze_scores(packet, root=env["root"])
    receipt.write_text(learning.canonical(value), encoding="utf8")
    with pytest.raises(ValueError, match="SUPPLIED_SCORES"):
        learning.evaluate_frozen(receipt, env["root"] / "experiments/no_metrics.json")
    assert env["calls"] == ["fit"]


def test_model_byte_drift_refuses_before_final_Y(synthetic_env):
    env = synthetic_env
    models = env["root"] / "experiments/models"
    learning.fit_models(env["index"], models)
    with (models / "Q.json").open("a", encoding="utf8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="ARTIFACT_BYTES"):
        learning.freeze_scores(models, env["root"] / "experiments/no_scores.json", role="final_test", prediction_index=env["index"])
    assert env["calls"] == ["fit"]


def test_dependency_drift_rejects_before_final_Y(synthetic_env, monkeypatch):
    env = synthetic_env
    models = env["root"] / "experiments/models"
    learning.fit_models(env["index"], models)
    original_dependencies = learning.dependencies
    monkeypatch.setattr(learning, "dependencies", lambda include_state=False:
        {**original_dependencies(include_state), "fixed_ridge": "0" * 64})
    with pytest.raises(ValueError, match="TRAINING_INPUT_IDENTITY"):
        learning.freeze_scores(models, env["root"] / "experiments/no_scores.json", role="final_test", prediction_index=env["index"])
    assert env["calls"] == ["fit"]


@pytest.mark.parametrize("field", ["lambda", "weight_per_source_mean", "intercept", "beta", "parameter_keys"])
def test_self_rehashed_ridge_json_rejects_bool_and_wrong_schema(synthetic_env, field):
    env = synthetic_env
    directory = env["root"] / "experiments/models"
    learning.fit_models(env["index"], directory)
    value = learning.read(directory / "simple.json")
    if field in {"lambda", "weight_per_source_mean"}:
        value[field] = True
    elif field == "intercept":
        value["parameters"][field] = True
    elif field == "beta":
        value["parameters"][field][0] = True
    else:
        value["parameters"]["extra_parameter"] = 1
    value["content_identity"] = learning.digest({k: v for k, v in value.items() if k != "content_identity"})
    path = directory / "simple.json"
    path.write_text(learning.canonical(value), encoding="utf8")
    manifest = learning.read(directory / "manifest.json")
    manifest["files_sha256"][path.name] = learning.sha(path)
    (directory / "manifest.json").write_text(learning.canonical(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="SCORER_SCHEMA|NUMERIC_RIDGE"):
        learning.score_packet(directory, learning.prepare_inputs(env["index"]), "final_test")
    assert env["calls"] == ["fit"]


def test_finite_parameters_cannot_clip_overflow_to_finite_score():
    params = {"mean": [0., 0.], "scale": [1., 1.], "beta": [1e308, 1e308], "intercept": 0.}
    with pytest.raises(ValueError, match="NONFINITE_RIDGE_ARITHMETIC"):
        learning.ridge_scores(params, np.array([[2., 2.]]))


def test_finite_predictions_keep_existing_fraction_clip():
    params = {"mean": [0.], "scale": [1.], "beta": [2.], "intercept": .5}
    assert learning.ridge_scores(params, np.array([[-1.], [0.], [1.]])).tolist() == [0., .5, 1.]
