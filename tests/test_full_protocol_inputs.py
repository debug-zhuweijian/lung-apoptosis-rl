"""Actual source mapping probes; CPU tensor validation, no member forward."""
import pytest
import torch

from src.lung.data import full_protocol_inputs as module


@pytest.fixture(scope="module")
def inputs():
    return module.FullProtocolInputs()


def test_real_source_coverage(inputs):
    assert len(inputs.candidate_ids) == 24
    batch = inputs.tensor_batch(inputs.candidate_ids)
    batch.validate()
    assert batch.fingerprint.device.type == "cpu"
    assert batch.lengths.tolist().count(1) == 10
    assert batch.lengths.tolist().count(4) == 10
    assert batch.lengths.tolist().count(5) == 4
    assert batch.fingerprint_known.sum() == 20


def test_viel_original_units_timeline_and_unknowns(inputs):
    cid = next(x for x in inputs.candidate_ids if "bocnho_50pm" in x and "pulse" in x)
    b = inputs.tensor_batch([cid])
    assert b.dose_unit[0, 0] == module.DOSE_UNITS.index("nM")
    assert b.values[0, 0, 0].item() == pytest.approx(.05)
    assert b.values[0, :, 1:3].tolist() == [[0, 6], [6, 6], [6, 6], [6, 24]]
    assert not b.known[..., 4].any()  # wash count absent, not zero/one
    assert not b.known[..., 5].any()  # biological retention unknown
    assert b.known[0, 0, 1] and b.values[0, 0, 1] == 0  # known zero origin
    assert b.known[0, 2, 6] and b.values[0, 2, 6] == 1


def test_bothe_order_and_unknown_drug_identity(inputs):
    cid = "Bothe2021:E-MTAB-9923:ethanol+washout+dexamethasone"
    b = inputs.tensor_batch([cid])
    assert b.event_type.tolist() == [[1, 2, 4, 3, 0]]
    assert b.values[0, :, 1:3].tolist() == [[0, 4], [4, 4], [4, 4], [4, 28], [28, 32]]
    assert not b.fingerprint_known.any()
    assert b.known[0, 1, 4] and b.values[0, 1, 4] == 2
    assert b.endpoint_hours.tolist() == [32]
    assert len(inputs.candidate(cid)["source_samples"]) == 3


@pytest.mark.parametrize("ids", [[], ["continuous24"], torch.zeros(2048), [True]])
def test_caller_vectors_and_nominal_protocol_rejected(inputs, ids):
    with pytest.raises(ValueError):
        inputs.tensor_batch(ids)


def test_candidate_returns_independent_metadata(inputs):
    cid = inputs.candidate_ids[0]
    row = inputs.candidate(cid)
    row["events"][0]["values"]["dose"] = 123
    assert inputs.candidate(cid)["events"][0]["values"]["dose"] != 123


def test_pinned_source_mismatch_fails_before_input_decode(monkeypatch):
    reads = []
    monkeypatch.setattr(module, "sha", lambda path: "0" * 64)
    monkeypatch.setattr(module, "read", lambda path: reads.append(path))
    with pytest.raises(ValueError, match="Source hash mismatch"):
        module.FullProtocolInputs()
    assert reads == []


def test_no_label_json_decoder_or_model_path(inputs):
    forbidden = {"value", "observations", "observation_mapping", "labels_seen_in_design", "Y"}
    for cid in inputs.candidate_ids:
        assert not forbidden.intersection(inputs.candidate(cid))
        assert inputs.candidate(cid)["condition_kwargs_status"].startswith("pending")


def test_metadata_decoder_selects_only_input_json(monkeypatch):
    paths = []
    original = module.read
    def tracked(path):
        paths.append(module.Path(path).name)
        return original(path)
    monkeypatch.setattr(module, "read", tracked)
    module.FullProtocolInputs()
    assert paths == ["manifest.json", "candidate_records.json"]


def test_frozen_mapped_package_load(inputs):
    loaded = module.FullProtocolInputs.load()
    assert loaded.candidate_ids == inputs.candidate_ids
    assert loaded.source_bindings == inputs.source_bindings


def test_new_asset_overwrite_refused(tmp_path):
    with pytest.raises(ValueError, match="refusing overwrite"):
        module.build(tmp_path)


@pytest.mark.parametrize("surface", ["dose", "role", "event_type", "bindings"])
def test_mutable_cache_tampering_rejected_before_tensor(surface):
    reader = module.FullProtocolInputs()
    cid = reader.candidate_ids[0]
    if surface == "bindings":
        reader.source_bindings.clear()
    elif surface == "role":
        reader._records[0]["role"] = "caller_promoted_fit"
    elif surface == "event_type":
        reader._records[0]["events"][0]["event_type"] = "vehicle_interval"
    else:
        reader._records[0]["events"][0]["values"]["dose"] = 123
    with pytest.raises(ValueError, match="Cached .* changed"):
        reader.tensor_batch([cid])
    with pytest.raises(ValueError, match="Cached .* changed"):
        reader.candidate(cid)
