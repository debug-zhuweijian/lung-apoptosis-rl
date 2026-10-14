"""Pure function mocks only: no world network or production checkpoint inference."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lung.engine import endpoint_protocol_producer as p
from lung.engine.endpoint_protocol_features_v2 import EndpointProtocolFeatureInputsV2
from lung.engine.config import StageUnavailable


@pytest.fixture
def bridge():
    return EndpointProtocolFeatureInputsV2.load()


def test_real_label_free_dedup_and_actual_clock_preserved(bridge):
    requests = p.standard_requests(bridge)
    assert len(requests) == 14
    values = [bridge.model_input(cid) for cid in bridge.candidate_ids]
    assert {v["candidate_record"]["protocol"]["exposure_duration_h"] for v in values} == {6, 24}
    assert all(v["request"]["reference_terminal_time_h"] == 24 for v in requests.values())


@pytest.mark.parametrize("field", ["request_id", "fingerprint", "reference", "schema", "elapsed", "nan", "coverage", "flatten_history"])
def test_fixed_row_contract_rejects_changes(bridge, field):
    requests = p.standard_requests(bridge)
    rows = [{"request_id": rid, **{k: item["request"][k] for k in
        ("compound_id", "dose_uM", "reference_protocol_arm", "reference_terminal_time_h")},
        "population": [0, 0], "bio": [0]*27, "context": [0]*3, "history": [[0]*6],
        "elapsed": 24, "horizon": 24, "descriptor_definition": p.DESCRIPTOR_DEFINITION,
        "reference_identity": "ref", "feature_identity": "identity"} for rid, item in requests.items()]
    p.validate_rows(rows, requests, {"d_model": 2, "context_dim": 1, "history_input_dim": 6}, "ref", "identity")
    if field == "request_id": rows[0]["request_id"] = "changed"
    elif field == "fingerprint": rows[0]["compound_id"] = "changed"
    elif field == "reference": rows[0]["reference_identity"] = "changed"
    elif field == "schema": rows[0]["Annexin"] = 0.5
    elif field == "elapsed": rows[0]["elapsed"] = 6
    elif field == "nan": rows[0]["population"][0] = float("nan")
    elif field == "flatten_history": rows[0]["history"] = rows[0]["history"][0]
    else: rows.pop()
    with pytest.raises(StageUnavailable):
        p.validate_rows(rows, requests, {"d_model": 2, "context_dim": 1, "history_input_dim": 6}, "ref", "identity")


@pytest.mark.parametrize("bad", ["epoch", "optimizer_step", "synthetic", "stage", "data_identity", "missing"])
def test_actual_checkpoint_qualification(monkeypatch, tmp_path, bad):
    saved = {"epoch": 1, "model": {"w": np.array([1])}, "counters": {"optimizer_step": 1},
             "metadata": {"stage": "pool", "data_identity": "real", "synthetic": False}}
    if bad == "epoch": saved["epoch"] = 0
    elif bad == "optimizer_step": saved["counters"]["optimizer_step"] = 0
    elif bad == "synthetic": saved["metadata"][bad] = True
    elif bad == "stage": saved["metadata"][bad] = "state"
    elif bad == "data_identity": saved["metadata"][bad] = "other"
    pipeline = {}
    for stage in ("response", "pool", "state", "representation"):
        checkpoint = tmp_path / (stage + ".pt")
        checkpoint.write_bytes(b"mock bytes; never torch loaded")
        pipeline[stage] = str(checkpoint)
    def checkpoint_reader(path):
        value = copy.deepcopy(saved)
        if bad != "stage":
            value["metadata"]["stage"] = "latent_ae" if Path(path).stem == "representation" else Path(path).stem
        return value
    monkeypatch.setattr(p, "load_checkpoint", checkpoint_reader)
    if bad == "missing": del pipeline["response"]
    with pytest.raises((StageUnavailable, ValueError)):
        p.validate_weights(pipeline, "real")


def test_actions_are_request_keyed_standard24_only(monkeypatch, bridge):
    requests = p.standard_requests(bridge)
    import pandas as pd
    ref = {"condition_id": "fit", "reference_plate": "plate1", "reference_id": "ref",
           "source_id": "Tahoe", "background_id": "A549", "protocol_id": "continuous"}
    data = SimpleNamespace(synthetic=False, condition_ids=lambda role: ["fit"],
        action_features=lambda cid: {"source_id": 0, "background_id": 0, "protocol_id": 0},
        reader=SimpleNamespace(conditions=pd.DataFrame([{**ref, "plate": "plate1"}])))
    actions, provenance = p.make_actions(requests, data, ref)
    assert len(actions) == 14 and set(actions) == set(requests)
    assert all(a["duration"] == 24 and a["dose"] == requests[rid]["request"]["dose_uM"]
               for rid, a in actions.items())
    assert all(v["background_transfer_validated"] is False for v in provenance.values())


def test_no_label_json_is_deserialized(monkeypatch):
    from lung.engine import endpoint_protocol_features as inputs
    real = inputs._read
    def guarded(path):
        assert Path(path).name not in {"observations.json", "reference_observations.json", "observation_mapping.json"}
        return real(path)
    monkeypatch.setattr(inputs, "_read", guarded)
    assert len(p.standard_requests(EndpointProtocolFeatureInputsV2.load())) == 14


def test_current_empty_registration_never_builds_world(monkeypatch, bridge):
    root = Path(__file__).resolve().parents[1]
    pre = root / "experiments/20261010-endpoint-feature-preflight-joint"
    def forbidden(*a, **k): raise AssertionError("World must not be instantiated")
    monkeypatch.setattr(p, "ExternalConditionWorld", forbidden)
    report = p.preflight_standard_requests(bridge,
        reference_path=pre / "final-v3/reference_inputs/reference.json",
        mapping_path=root / "experiments/20261011-data-alignment-round14/producer_context_map.json",
        data_config_path=root / "configs/data_joint_sciplex_v1.json", pipeline={},
        config={"synthetic": False, "history_support": "single_exposure_only"})
    assert report["status"] == "pending" and report["model_execution_performed"] is False


def test_mock_generation_and_cache_tamper(monkeypatch, bridge, tmp_path):
    """Exercise producer/loader I/O, with qualification/data/world replaced by pure mocks."""
    import pandas as pd
    from lung.engine import data as data_module
    root = Path(__file__).resolve().parents[1]
    pre = root / "experiments/20261010-endpoint-feature-preflight-joint"
    reference_path = pre / "final-v3/reference_inputs/reference.json"
    ref = p.read(reference_path)
    draw_ids = pd.read_parquet(ref["cell_index"]["path"]).cell_id.iloc[:2].tolist()
    assets = root / "08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1"
    fake_data = SimpleNamespace(synthetic=False, identity=ref["data_identity"], assets_dir=assets,
        condition_ids=lambda _: [ref["condition_id"]],
        action_features=lambda _: {"source_id": 0, "background_id": 0, "protocol_id": 0},
        reader=SimpleNamespace(conditions=pd.DataFrame([{**ref, "plate": "plate1"}])))
    monkeypatch.setattr(data_module, "EngineData", lambda *a, **k: fake_data)
    monkeypatch.setattr(p, "preflight_standard_requests", lambda *a, **k: {
        "missing_stages": [], "data_identity": ref["data_identity"], "reference_identity": ref["reference_id"], "checkpoints": {}})
    monkeypatch.setattr(p, "validate_weights", lambda *a, **k: {})
    mock_bytes = tmp_path / "mock-only.bin"
    mock_bytes.write_bytes(b"not a checkpoint; qualification function is mocked")
    pipeline = {k: str(mock_bytes) for k in ("response", "pool", "state", "representation")}
    called = []
    def world_factory(data, config, pipeline, actions):
        assert len(actions) == 14 and all(a["duration"] == 24 for a in actions.values())
        called.append(actions)
        from lung.models import ModelConfig
        import torch
        shape = ModelConfig(n_genes=2, d_model=2, n_heads=1, context_dim=1)
        root_context = SimpleNamespace(payload=SimpleNamespace(identity="mock_root", initial_seed=123, population_size=2))
        return SimpleNamespace(actions=actions, shape=shape, n_pop=2,
            root=lambda cid, role: root_context,
            _initial=lambda root, n: (torch.tensor([[0., 1.], [1., 0.]]), {}),
            _initial_cache=(None, None, {"control_cell_ids": draw_ids}))
    monkeypatch.setattr(p, "ExternalConditionWorld", world_factory)
    def rows(world, initial, provenance, **kwargs):
        assert initial.payload.identity == "mock_root"
        return [{"experimental_condition_id": rid, "population": [1., 2.], "bio": [0.]*27,
                 "context": [0.]*3, "history": [[0.]*6], "elapsed": 24., "horizon": 24.} for rid in world.actions]
    monkeypatch.setattr(p, "generate_feature_rows", rows)
    monkeypatch.setattr(p, "load_checkpoint", lambda _: {"metadata": {"model_config": {"n_genes": 2, "d_model": 2, "n_heads": 1, "context_dim": 1}}})
    output = tmp_path / "cache"
    p.generate_standard_cache(bridge, reference_path=reference_path,
        mapping_path=root / "experiments/20261011-data-alignment-round14/producer_context_map.json",
        data_config_path=root / "configs/data_joint_sciplex_v1.json", pipeline=pipeline,
        config={"synthetic": False, "seed": 1, "history_support": "single_exposure_only"}, output=output)
    loaded = p.load_standard_cache(output / "manifest.json", bridge)
    assert len(loaded) == 14 and len(called) == 1
    table = p.read(output / "features.json")
    table[0]["population"][0] += 1
    p.atomic_json(output / "features.json", table)
    with pytest.raises(StageUnavailable, match="SOURCE_MISSING_OR_CHANGED"):
        p.load_standard_cache(output / "manifest.json", bridge)
    m = p.read(output / "manifest.json")
    m["features"] = p.record(output / "features.json")
    p.atomic_json(output / "manifest.json", m)
    with pytest.raises(StageUnavailable, match="TABLE_CONTENT_IDENTITY_MISMATCH"):
        p.load_standard_cache(output / "manifest.json", bridge)
    m["descriptor_definition"] = {"version": "arbitrary_annexin"}
    p.atomic_json(output / "manifest.json", m)
    with pytest.raises(StageUnavailable, match="PRODUCER_UNVERIFIED"):
        p.load_standard_cache(output / "manifest.json", bridge)

