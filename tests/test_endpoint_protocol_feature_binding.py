"""Production bridge regressions; fabricated receipts stay in pytest tmp dirs."""

import copy
import json
from pathlib import Path

import pytest

from src.lung.data.endpoint_protocol_inputs import BASE, request_for, sha
from src.lung.engine.config import StageUnavailable
from src.lung.engine.endpoint_protocol_features import EndpointProtocolFeatureInputs, INPUT_FILES


@pytest.fixture
def bridge():
    return EndpointProtocolFeatureInputs.load()


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def envelope(bridge):
    row = bridge.model_input(bridge.candidate_ids[0])["candidate_record"]
    return {"candidate_record": row, "request": request_for(row),
            "input_files_sha256": {k: bridge._manifest["files_sha256"][k] for k in INPUT_FILES}}


def test_real_bundle_is_label_free_and_pending(monkeypatch):
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path.name not in {"observations.json", "reference_observations.json", "observation_mapping.json"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    import src.lung.data.endpoint_protocol_inputs as legacy
    monkeypatch.setattr(legacy, "prepare", lambda **kw: pytest.fail("legacy prepare reads Y"))
    bridge = EndpointProtocolFeatureInputs.load()
    inputs = [bridge.model_input(cid) for cid in bridge.candidate_ids]
    assert len(inputs) == 28
    assert len({r["candidate_record"]["p_request_id"] for r in inputs}) == 14
    assert bridge._manifest["unique_active_observations"] == 24
    assert all(r["p_descriptor"] is None and r["chemical_fingerprint"].shape == (2048,) for r in inputs)
    with pytest.raises(StageUnavailable, match="P_CACHE_REQUIRED"):
        bridge.require_response_features(bridge.candidate_ids[0])


def test_old_p2_annexin_descriptor_is_rejected(bridge, tmp_path):
    # Old shape's arbitrary label-like vector is not read as production features.
    payload = [{**request_for(bridge._rows[0]), "feature_names": ["observed_Annexin_fraction"],
                "values": [0.5], "model_sha256": "a" * 64}]
    path = dump(tmp_path / "legacy.json", payload)
    with pytest.raises(StageUnavailable, match="LEGACY_DESCRIPTOR"):
        bridge.model_input(bridge.candidate_ids[0], cache_manifest=path)


def test_missing_cache_is_rejected(bridge, tmp_path):
    with pytest.raises(StageUnavailable, match="ARTIFACT_MISSING"):
        bridge.model_input(bridge.candidate_ids[0], cache_manifest=tmp_path / "missing.json")


@pytest.mark.parametrize("field", ["request_id", "dose_uM", "reference_protocol_arm"])
def test_cross_request_and_clock_receipt_rejected(bridge, tmp_path, field):
    receipt = envelope(bridge)
    receipt["request"][field] = "different"
    with pytest.raises(StageUnavailable, match="REQUEST_OR_PROTOCOL_MISMATCH"):
        bridge.model_input(bridge.candidate_ids[0], cache_manifest=dump(tmp_path / "receipt.json", receipt))


def test_actual_pulse_protocol_mismatch_rejected(bridge, tmp_path):
    receipt = envelope(bridge)
    receipt["candidate_record"]["protocol"]["terminal_time_h"] = 6
    with pytest.raises(StageUnavailable, match="REQUEST_OR_PROTOCOL_MISMATCH"):
        bridge.model_input(bridge.candidate_ids[0], cache_manifest=dump(tmp_path / "receipt.json", receipt))


def test_hash_shaped_strings_do_not_replace_real_artifacts(bridge, tmp_path):
    receipt = envelope(bridge)
    receipt["artifacts"] = {key: {"path": "missing.bin", "sha256": "a" * 64} for key in (
        "producer_source", "checkpoint", "gene_axis", "preprocessing", "reference",
        "descriptor_definition", "feature_table")}
    with pytest.raises(StageUnavailable, match="ARTIFACT_MISSING"):
        bridge.model_input(bridge.candidate_ids[0], cache_manifest=dump(tmp_path / "receipt.json", receipt))


def test_self_consistent_fabricated_artifacts_still_unsupported(bridge, tmp_path):
    receipt = envelope(bridge)
    artifact = tmp_path / "synthetic_not_measured.bin"
    artifact.write_bytes(b"pytest synthetic artifact, never trained or measured")
    receipt["artifacts"] = {key: {"path": artifact.name, "sha256": sha(artifact)} for key in (
        "producer_source", "checkpoint", "gene_axis", "preprocessing", "reference",
        "descriptor_definition", "feature_table")}
    with pytest.raises(StageUnavailable, match="UNSUPPORTED_PRODUCER"):
        bridge.model_input(bridge.candidate_ids[0], cache_manifest=dump(tmp_path / "receipt.json", receipt))


def test_changed_candidate_file_rejected_before_parsing(bridge, tmp_path):
    directory = BASE / "endpoint-protocol-inputs-v1"
    (tmp_path / "manifest.json").write_bytes((directory / "manifest.json").read_bytes())
    rows = copy.deepcopy(bridge._rows)
    rows[0]["protocol"]["terminal_time_h"] = 6
    dump(tmp_path / "candidate_records.json", rows)
    with pytest.raises(StageUnavailable, match="ARTIFACT_MISSING_OR_CHANGED"):
        EndpointProtocolFeatureInputs.load(tmp_path)
