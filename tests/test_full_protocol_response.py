"""Real metadata/input checks and separately marked synthetic E/P/D checks."""
from dataclasses import replace
from collections import OrderedDict
import json

import numpy as np
import pytest
import torch

from lung.engine import full_protocol_response as module
from lung.models import ModelConfig
from lung.models.full_protocol import (FullPhysicalProtocolBatch, FullProtocolConditioner,
                                      FullProtocolLatentAdapter)
from lung.models.latent_staged import SharedAutoencoder, LatentMember


@pytest.fixture(scope="module")
def actual():
    return module.bound_inputs()


def physical(batch=2, pulse=True):
    values = torch.zeros(batch, 4, 8)
    known = torch.zeros_like(values, dtype=torch.bool)
    values[:, 0, :4] = torch.tensor([1000., 0., 6. if pulse else 24., 6. if pulse else 24.])
    known[:, 0, :4] = True
    for i in (1, 2, 3):
        values[:, i, 1:3] = torch.tensor([6., 24. if i == 3 else 6.])
        known[:, i, 1:3] = True
    values[:, 2, 6] = 1
    known[:, 2, 6] = True
    values[:, 3, 7] = 18
    known[:, 3, 7] = True
    fp = torch.zeros(batch, 4, 2048)
    fp[:, 0] = 1
    fk = torch.zeros(batch, 4, dtype=torch.bool)
    fk[:, 0] = True
    return FullPhysicalProtocolBatch(fp, fk, torch.zeros(batch, 4, dtype=torch.bool),
        torch.tensor([[0, 2, 4, 3]]).repeat(batch, 1), torch.tensor([[2, 0, 0, 0]]).repeat(batch, 1),
        values, known, torch.full((batch,), 4 if pulse else 1), torch.full((batch,), 24.),
        torch.ones(batch, dtype=torch.bool))


def numeric(batch=2, pulse=True):
    return {"fingerprint": torch.ones(batch, 2048), "dose": torch.ones(batch),
        "duration": torch.full((batch,), 6. if pulse else 24.), "elapsed": torch.zeros(batch),
        "horizon": torch.full((batch,), 24.), "vehicle": torch.zeros(batch, dtype=torch.bool),
        **{k: torch.zeros(batch, dtype=torch.long) for k in ("source", "background", "protocol")}}


def adapter(reference_variant="observed"):
    torch.manual_seed(55)
    config = ModelConfig(n_genes=4, d_model=8, n_heads=2, context_dim=2, latent_dim=3,
        n_sources=2, n_protocols=2, pad_layers=1, backbone="compact_full_gene", checkpoint_blocks=False)
    shared = SharedAutoencoder(config, torch.arange(4)[:, None])
    member = LatentMember(config, shared)
    full = FullProtocolLatentAdapter(member, FullProtocolConditioner(config))
    return module.FullProtocolResponseAdapter(shared, full, np.ones(4), np.ones(4), reference_variant=reference_variant)


def test_preflight_preserves_all_candidates_and_pending_sources_without_model(actual):
    report = module.preflight()
    assert report["candidate_count"] == 32 and report["numeric_ready"] == 20
    assert {r["candidate_id"] for r in report["candidates"]} == {r["candidate_id"] for r in actual[1]}
    assert sum(r["request_status"] == "pending_physical_input" for r in report["candidates"]) == 8
    assert sum(r["request_status"] == "pending_bulk_reference_context_decoder" for r in report["candidates"]) == 4
    assert report["PT_loads"] == report["model_forwards"] == report["P_outputs"] == 0
    assert not any(r["production_qualified"] or r["source_state_numerically_encoded"] for r in report["candidates"])


def test_real_pulse_numeric_physical_input_retains_masks_and_explicit_proxy(actual):
    reader, rows, _, _ = actual
    request = next(r for r in rows if r["base_kwargs"] is not None and r["base_kwargs"]["duration"] == 6)
    kw, p = module.numeric_batch(reader, request["candidate_id"], 2)
    assert kw["fingerprint"].shape == (2, 2048) and kw["source"].tolist() == [0, 0]
    assert p.lengths.tolist() == [4, 4] and kw["horizon"].tolist() == [24, 24]
    assert not p.known[:, :, 5].any() and not p.known[:, 1, 4].any()
    assert request["source_id"] == "Vielhauer2013" and request["proxy_context"]["source_id"] == "Tahoe-100M"
    assert request["source_state_id"] == "a549_parental_standard_culture"


def test_zero_h_is_actual_no_RNA_state_input_and_never_calls_encoder(monkeypatch):
    model = adapter("zero_h").eval()
    monkeypatch.setattr(model.shared, "encode_h", lambda *a: pytest.fail("zero_h cannot consume E(RNA)"))
    noise = torch.ones(2, 3)
    a = model(torch.ones(2, 4), physical(), numeric(), noise=noise, steps=1).y
    b = model(torch.arange(8).reshape(2, 4).float(), physical(), numeric(), noise=noise, steps=1).y
    assert torch.equal(a, b)
    h = module.prepare_reference_h(model.shared, np.ones((3, 4), np.float32), "zero_h", microbatch=1)
    assert h.shape == (3, 8) and (h == 0).all()


def test_mean_h_is_entire_frozen_latent_population_not_expression_mean_or_chunk_mean():
    from types import SimpleNamespace
    class NonlinearE:
        config = SimpleNamespace(n_genes=2, d_model=2)
        def encode_h(self, x):
            return x[:, None], x.square()
    values = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    expected = torch.from_numpy(values).square().mean(0).expand(3, 2)
    for microbatch in (1, 2, 3):
        h = module.prepare_reference_h(NonlinearE(), values, "mean_h", microbatch=microbatch)
        assert torch.equal(h, expected)
    assert not torch.equal(expected[0], torch.from_numpy(values).mean(0).square())


@pytest.mark.parametrize("variant", module.REFERENCE_VARIANTS)
def test_prepared_global_h_adapter_and_metadata_keep_exact_variant(variant):
    model = adapter(variant).eval()
    x = torch.arange(8).reshape(2, 4).float()
    h = module.prepare_reference_h(model.shared, x, variant, microbatch=2)
    noise = torch.ones(2, 3)
    direct = model(x, physical(), numeric(), noise=noise, steps=1).y
    prepared = model.predict_h(h, physical(), numeric(), noise=noise, steps=1).y
    assert torch.equal(direct, prepared)
    assert module.declared_reference_variant({"reference_contract": model.reference_contract}) == variant


def test_missing_or_rewritten_reference_metadata_is_rejected_before_weights(actual, monkeypatch):
    meta = declaration(actual[3])
    del meta["response"]["reference_contract"]
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No PT for missing variant"))
    with pytest.raises(ValueError, match="REFERENCE_VARIANT_INVALID"):
        module.validate_metadata(meta, actual[3])
    contract = module.reference_contract("zero_h")
    contract["RNA_state_used_by_P"] = True
    meta["response"]["reference_contract"] = contract
    with pytest.raises(ValueError, match="REFERENCE_CONTRACT_MISMATCH"):
        module.validate_metadata(meta, actual[3])
    contract = module.reference_contract("mean_h")
    contract["population_definition_by_stage"]["CFM_fit"] = "all_cells_in_infinite_population"
    meta["response"]["reference_contract"] = contract
    with pytest.raises(ValueError, match="REFERENCE_CONTRACT_MISMATCH"):
        module.validate_metadata(meta, actual[3])


def test_pending_nav_and_bothe_rejected_before_weights(actual, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No checkpoint read allowed for pending source")
    monkeypatch.setattr(torch, "load", forbidden)
    for status in ("pending_physical_input", "pending_bulk_reference_context_decoder"):
        request = next(r for r in actual[1] if r["request_status"] == status)
        with pytest.raises(ValueError, match="SOURCE_PHYSICAL_REFERENCE_OR_OUTLET_PENDING"):
            module.produce(request["candidate_id"], None, tmp_path / status, seed=1)
    assert not list(tmp_path.iterdir())


def test_viel_missing_matching_assets_rejected_before_weights_and_reference(actual, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing metadata cannot read weights or real reference")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(module, "read_proxy_reference", forbidden)
    cid = next(r["candidate_id"] for r in actual[1] if r["base_kwargs"] is not None)
    with pytest.raises(ValueError, match="MATCHING_MODEL_METADATA_PENDING"):
        module.produce(cid, None, tmp_path / "missing", seed=1)
    assert not list(tmp_path.iterdir())


def test_old_response_stage_and_static_receipt_cannot_qualify(actual):
    bindings = actual[3]
    metadata = {"bindings": bindings, "qualification": "static_inspection_only"}
    with pytest.raises(ValueError, match="MATCHING_STAGE_DATA_PENDING"):
        module.validate_metadata(metadata, bindings)
    metadata["qualification"] = "safe_tensor_execution_audited"
    metadata["representation"] = {"stage": "latent_ae", "synthetic": False,
        "data_identity": bindings["data_identity"], "optimizer_step": 1,
        "checkpoint": "dummy.pt", "checkpoint_sha256": "a" * 64}
    metadata["response"] = {**metadata["representation"], "stage": "latent_cfm"}
    with pytest.raises(ValueError, match="MATCHING_STAGE_DATA_PENDING"):
        module.validate_metadata(metadata, bindings)


def declaration(bindings):
    ae = {"stage": "latent_ae", "synthetic": False, "data_identity": bindings["data_identity"],
          "optimizer_step": 1, "checkpoint": "dummy-ae.pt", "checkpoint_sha256": "a" * 64}
    return {"bindings": bindings, "qualification": "safe_tensor_execution_audited",
        "representation": ae, "response": {**ae, "stage": module.STAGE,
        "checkpoint": "dummy-response.pt", "checkpoint_sha256": "b" * 64,
        "representation_hash": ae["checkpoint_sha256"],
        "training_protocol_coverage": module.PARTIAL_PROTOCOL_COVERAGE,
        "reference_contract": module.reference_contract("observed")}}


def test_parameter_namespace_does_not_grant_washout_or_recovery_supervision(actual):
    meta = declaration(actual[3])
    meta["response"]["training_protocol_coverage"] = {**module.PARTIAL_PROTOCOL_COVERAGE,
        "washout_RNA_supervision": True}
    with pytest.raises(ValueError, match="PROTOCOL_SUPERVISION_COVERAGE_REQUIRED"):
        module.validate_model_declaration(meta, actual[3])


def test_prediction_consumer_rejects_tamper_before_registry_or_PT(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "validate_metadata", lambda *a: pytest.fail("Reject content before registration"))
    folder = tmp_path / "experiments"
    folder.mkdir()
    path = folder / "prediction.json"
    path.write_text(json.dumps({"schema": module.VERSION, "prediction_identity": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="PREDICTION_CONTENT_CHANGED"):
        module.read_prediction(path)


def test_prediction_consumer_rejects_request_semantic_rewrite_without_model(tmp_path, monkeypatch, actual):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "bound_inputs", lambda: actual)
    monkeypatch.setattr(module, "validate_metadata", lambda *a: pytest.fail("Reject request before registration"))
    folder = tmp_path / "experiments"
    folder.mkdir()
    request = next(r for r in actual[1] if r["base_kwargs"] is not None)
    changed = {**request, "source_state_id": "invented_initial_state"}
    body = {"schema": module.VERSION, "candidate_id": request["candidate_id"],
        "bindings": actual[3], "request": changed}
    body["prediction_identity"] = module.digest(body)
    path = folder / "prediction.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="PREDICTION_REQUEST_OR_BINDINGS_CHANGED"):
        module.read_prediction(path)


def test_prediction_moment_reader_without_real_PT_qualification(tmp_path, monkeypatch, actual):
    # Pure synthetic container check: mocked registration grants no real asset.
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "bound_inputs", lambda: actual)
    registration = {"synthetic_mock_only": True}
    monkeypatch.setattr(module, "validate_metadata", lambda *a: registration)
    monkeypatch.setattr(module.torch, "load", lambda *a, **k: pytest.fail("Reader must not deserialize PT"))
    folder = tmp_path / "experiments"
    folder.mkdir()
    for name in ("AE", "P"):
        (folder / name).write_bytes(b"synthetic mocked asset")
    request = next(r for r in actual[1] if r["base_kwargs"] is not None)
    reference = {"reference_id": actual[3]["reference_id"], "reference_rows_sha256": module.digest(actual[2]["reference_rows"]),
        "reference_cell_ids": [r["cell_id"] for r in actual[2]["reference_rows"]],
        "shape": [2431, len(actual[2]["gene_ids"])], "reference_role": "fit", "target_expression_read": False}
    body = {"schema": module.VERSION, "candidate_id": request["candidate_id"], "request": request,
        "bindings": actual[3], "reference": reference, "model_metadata": {k: {"checkpoint": str(folder / n),
            "checkpoint_sha256": module.inputs.sha(folder / n)} for k, n in (("representation", "AE"), ("response", "P"))},
        "production_registration": registration, "seed": 4, "steps": 2, "outlet": module.OUTLET,
        "execution_kind": "matching_full_protocol_weights", "training_protocol_coverage": module.PARTIAL_PROTOCOL_COVERAGE,
        "candidate_protocol_OOD": True, "candidate_protocol_scientifically_validated": False,
        "cross_source_transfer_validated": False, "functional_Y_read": False, "reference_target_cell_pairing": False,
        "source_state_numerically_encoded": False, "source_target_RNA_read": False, "new_roles_assigned": False,
        "G_outputs": 0, "Q_outputs": 0, "scientific_effect_verified": False,
        "summary": "equal_cell_weighted_full_fit_proxy_common_gene_log1p_mean_variance",
        "expression_mean": [.1] * len(actual[2]["gene_ids"]), "expression_variance": [.05] * len(actual[2]["gene_ids"])}
    path = folder / "prediction.json"
    body["reference_contract"] = module.reference_contract("observed")
    body["model_metadata"]["response"]["reference_contract"] = body["reference_contract"]
    def store():
        body.pop("prediction_identity", None)
        body["prediction_identity"] = module.digest(body)
        path.write_text(json.dumps(body), encoding="utf-8")
    store()
    assert module.read_prediction(path)["candidate_protocol_OOD"] is True
    body["expression_variance"][0] = -1
    store()
    with pytest.raises(ValueError, match="PREDICTION_MOMENTS_INVALID"):
        module.read_prediction(path)
    body["expression_mean"] = [True] * len(actual[2]["gene_ids"])
    body["expression_variance"] = [False] * len(actual[2]["gene_ids"])
    store()
    with pytest.raises(ValueError, match="PREDICTION_MOMENTS_INVALID"):
        module.read_prediction(path)


def test_caller_audited_text_and_caller_receipt_path_do_not_register_assets(actual, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REGISTRY", tmp_path / "fixed_missing_ledger.json")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Unregistered audit must fail before PT"))
    metadata = {**declaration(actual[3]), "registered_audit_id": "self_approved",
                "audit_receipt_path": str(tmp_path / "self.json"), "audit_passed": True}
    module.validate_model_declaration(metadata, actual[3])  # necessary static check only
    with pytest.raises(ValueError, match="REGISTERED_EXECUTION_AUDIT_PENDING"):
        module.qualified_model(metadata, actual[3], actual[2])


def test_boolean_actual_optimizer_counter_rejected_without_PT_or_model(actual, monkeypatch):
    metadata = declaration(actual[3])
    monkeypatch.setattr(module.inputs, "sha", lambda path: "a" * 64 if "dummy-ae" in str(path) else "b" * 64)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {
        "metadata": {"stage": "latent_ae", "data_identity": actual[3]["data_identity"], "synthetic": False,
                     "full_response_bindings": actual[3]}, "counters": {"optimizer_step": True}})
    with pytest.raises(ValueError, match="ACTUAL_TRAINED_COUNTER_MISMATCH"):
        module._load_matching_model(metadata, actual[3], actual[2])


def test_boolean_context_metadata_cannot_equal_integer_bindings(actual):
    metadata = {"bindings": {**actual[3], "model_context": {"source_id": False, "background_id": 0, "protocol_id": 1}}}
    with pytest.raises(ValueError, match="MODEL_BINDINGS_MISMATCH"):
        module.validate_metadata(metadata, actual[3])


@pytest.mark.parametrize("change", ["fingerprint", "vehicle", "dose", "duration", "horizon", "elapsed"])
def test_base_and_full_protocol_cannot_disagree(change):
    kw = numeric()
    kw[change] = (~kw[change]) if change == "vehicle" else kw[change] + 1
    with pytest.raises(ValueError, match="BASE_PHYSICAL"):
        module.validate_batch(kw, physical())


def test_synthetic_full_epd_consumes_reference_ordered_history_and_decodes_once():
    model = adapter()
    calls = []
    velocity, decode = model.full.member.velocity, model.full.member.decode
    def capture(state, tau, ref, condition):
        calls.append((ref.clone(), condition.clone(), tau.clone()))
        return velocity(state, tau, ref, condition)
    decoded = []
    def capture_decode(latent):
        decoded.append(latent.shape)
        return decode(latent)
    model.full.member.velocity = capture
    model.full.member.decode = capture_decode
    x = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    y = model(x, physical(), numeric(), noise=torch.ones(2, 3), steps=2)
    expected = model.shared.encode_h(x)[1]
    assert all(torch.equal(c[0], expected) for c in calls)
    assert all((c[2] >= 0).all() and (c[2] <= 1).all() for c in calls)
    assert decoded == [torch.Size([2, 3])] and y.x.shape == y.y.shape == (2, 4)
    assert torch.allclose(y.y.expm1().sum(1), torch.full((2,), 10000.), atol=.01)
    assert not torch.equal(calls[0][0][0], calls[0][0][1])
    previous = calls[0][1]
    calls.clear()
    model(x, physical(pulse=False), numeric(pulse=False), noise=torch.ones(2, 3), steps=2)
    assert not torch.equal(previous, calls[0][1])


def test_synthetic_unknown_placeholder_changes_no_values_and_does_not_become_known_zero():
    model = adapter()
    p, kw = physical(), numeric()
    values = p.values.clone()
    values[:, 1, 4] = float("nan")
    unknown = replace(p, values=values)
    known = p.known.clone()
    known[:, 1, 4] = True
    measured_zero = replace(p, known=known)
    x = torch.ones(2, 4)
    noise = torch.ones(2, 3)
    baseline = model(x, p, kw, noise=noise, steps=2)
    same = model(x, unknown, kw, noise=noise, steps=2)
    different = model(x, measured_zero, kw, noise=noise, steps=2)
    assert torch.equal(baseline.y, same.y)
    assert not torch.equal(same.y, different.y)
    assert not unknown.known[:, 1, 4].any()


def test_single_cell_outlet_refuses_martin_and_bothe_domain():
    model = adapter()
    for domain in ("log2_TPM_plus_0.01", "bulk_raw_counts", "source_estimated_counts"):
        with pytest.raises(ValueError, match="SOURCE_DOMAIN_NOT_SUPPORTED"):
            module.FullProtocolResponseAdapter(model.shared, model.full, np.ones(4), np.ones(4), input_domain=domain)


def test_reference_missing_or_nonfinite_cannot_be_silently_filled():
    model = adapter()
    for x in (torch.ones(2, 3), torch.full((2, 4), float("nan"))):
        with pytest.raises(ValueError, match="REFERENCE_DOMAIN_SHAPE_INVALID"):
            model(x, physical(), numeric(), noise=torch.ones(2, 3), steps=2)


def test_prepare_refuses_existing_output_before_source_read(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "bound_inputs", lambda: pytest.fail("Existing output must fail before source read"))
    with pytest.raises(ValueError, match="REFUSING_EXISTING_OUTPUT"):
        module.prepare(tmp_path)


def test_sparse_graph_checkpoint_state_finiteness_and_exact_identity_without_loading():
    graph = torch.sparse_coo_tensor(torch.tensor([[0, 1], [0, 1]]), torch.tensor([1., 1.]), (2, 2), check_invariants=True)
    module.finite_weights({"cell_encoder.graph": graph, "weight": torch.ones(1)})
    module.finite_weights(OrderedDict([("cell_encoder.graph", graph), ("weight", torch.ones(1))]))
    assert module.equal_weight(graph, graph.clone())
    assert not module.equal_weight(graph, graph * 2)
    assert not module.equal_weight(graph, graph.to_dense())
    with pytest.raises(ValueError, match="FINITE_WEIGHTS_REQUIRED"):
        module.finite_weights({"cell_encoder.graph": graph * float("nan")})


@pytest.mark.parametrize("state", [{}, {"weight": None}])
def test_empty_or_non_tensor_weights_cannot_qualify(state):
    with pytest.raises(ValueError, match="TENSOR"):
        module.finite_weights(state)
