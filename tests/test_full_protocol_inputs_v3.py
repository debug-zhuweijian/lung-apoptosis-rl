"""Actual nominal graph enrichment probes; no ML or checkpoint paths."""
import json

import numpy as np
import pytest

from src.lung.data import full_protocol_inputs_v3 as module


@pytest.fixture(scope="module")
def inputs():
    return module.FullProtocolInputsV3()


def test_actual_source_tensor_and_old_v2_unchanged(inputs):
    batch = inputs.tensor_batch(inputs.candidate_ids)
    assert len(inputs.candidate_ids) == 24 and batch.lengths.sum() == 70
    assert batch.fingerprint_known.sum() == 24
    assert batch.known.sum((0, 1)).tolist() == [24, 70, 70, 28, 4, 0, 14, 14]
    assert batch.fingerprint.device.type == "cpu"
    batch.validate()
    old = module.base.FullProtocolInputs.load()
    assert old.tensor_batch(old.candidate_ids).fingerprint_known.sum() == 20
    assert module.base.sha(module.base.__file__) == module.BASE_CODE_HASH


def test_four_actual_dex_events_match_source_array(inputs):
    fp = np.load(module.ENRICHMENT / "morgan2048_chiral.npy", allow_pickle=False)[0]
    count = 0
    for cid in inputs.candidate_ids:
        row = inputs.candidate(cid)
        if row["source_id"] != "Bothe2021:E-MTAB-9923":
            continue
        batch = inputs.tensor_batch([cid])
        first, _, second = row["source_condition_id"].split("+")
        for index, compound in ((0, first), (4, second)):
            if compound == "dexamethasone":
                np.testing.assert_array_equal(batch.fingerprint[0, index].numpy(), fp)
                assert batch.fingerprint_known[0, index]
                count += 1
            else:
                assert not batch.fingerprint_known[0, index] and not batch.known[0, index, 0]
        assert row["stock_form"] is None
        assert "fingerprint" not in row["condition_kwargs_status"]
        assert row["condition_kwargs_status"].startswith("pending")
        assert row["source_vehicle_dose"]["unit"] == "percent_unspecified_basis"
    assert count == 4


@pytest.mark.parametrize("surface", ["dose", "role", "action", "fingerprint", "bindings"])
def test_cached_mutation_is_rejected(surface):
    reader = module.FullProtocolInputsV3()
    cid = next(x for x in reader.candidate_ids if "dexamethasone+washout+dexamethasone" in x)
    row = next(r for r in reader._records if r["candidate_id"] == cid)
    if surface == "bindings":
        reader.source_bindings.clear()
    elif surface == "role":
        row["role"] = "caller_fit"
    elif surface == "action":
        row["events"][0]["event_type"] = "vehicle_interval"
    elif surface == "fingerprint":
        row["events"][0]["fingerprint"]["row"] = 1
    else:
        row["events"][0]["values"]["dose"] = 123
    for access in (lambda: reader.candidate_ids, lambda: reader.candidate(cid), lambda: reader.tensor_batch([cid])):
        with pytest.raises(ValueError, match="Cached v3 .* changed"):
            access()


def test_enrichment_hash_fails_before_any_input_decoder(monkeypatch):
    reads = []
    original_sha = module.base.sha
    monkeypatch.setattr(module.base, "sha", lambda path: "0" * 64 if module.Path(path) == module.ENRICHMENT / "manifest.json" else original_sha(path))
    monkeypatch.setattr(module.base, "read", lambda path: reads.append(path))
    with pytest.raises(ValueError, match="Source hash mismatch"):
        module.FullProtocolInputsV3()
    assert reads == []


def test_changed_records_with_updated_asset_hash_rejected(tmp_path):
    reader = module.FullProtocolInputsV3.load()
    records = json.loads((module.OUTPUT / "records.json").read_text(encoding="utf-8"))
    records[0]["events"][0]["values"]["dose"] = 123
    altered = tmp_path / "records.json"
    altered.write_text(json.dumps(records), encoding="utf-8")
    manifest = module.base.read(module.OUTPUT / "manifest.json")
    manifest["files_sha256"]["records.json"] = module.base.sha(altered)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from actual sources"):
        module.FullProtocolInputsV3.load(tmp_path)
    assert len(reader.candidate_ids) == 24


def test_no_label_json_or_checkpoint_decoder(monkeypatch):
    names = []
    original = module.base.read
    def read(path):
        names.append(module.Path(path).name)
        return original(path)
    monkeypatch.setattr(module.base, "read", read)
    reader = module.FullProtocolInputsV3()
    reader.tensor_batch([reader.candidate_ids[0]])
    assert not {"observations.json", "active_conditions.json", "reference_conditions.json"}.intersection(names)


def test_candidate_copy_and_overwrite_refusal(inputs, tmp_path):
    cid = inputs.candidate_ids[0]
    row = inputs.candidate(cid)
    row["role"] = "different"
    assert inputs.candidate(cid)["role"] != "different"
    with pytest.raises(ValueError, match="refusing overwrite"):
        module.build(tmp_path)
