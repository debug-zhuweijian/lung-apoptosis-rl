"""Batch tests use synthetic outlets, never registered real models or labels."""
import json

import pytest

from lung.engine import full_protocol_batch as batch


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    ids = [f"candidate-{i}" for i in range(20)]
    identity = {"schema": "fixture-batch", "candidate_ids": ids, "producer_bindings": {"fixture": True}}
    monkeypatch.setattr(batch.producer, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "context", lambda _: (ids, identity))
    monkeypatch.setattr(batch.producer, "validate_metadata", lambda *_: None)
    calls = []
    def produce(cid, meta, output, **kwargs):
        calls.append(cid)
        output.mkdir()
        value = {"candidate_id": cid, "model_metadata": meta, "bindings": identity["producer_bindings"],
            "seed": kwargs["seed"], "steps": kwargs["steps"], "prediction_identity": cid}
        (output / "prediction.json").write_text(json.dumps(value), encoding="utf-8")
        return value
    monkeypatch.setattr(batch.producer, "produce", produce)
    monkeypatch.setattr(batch.producer, "read_prediction", lambda p: json.loads(p.read_text(encoding="utf-8")))
    return tmp_path / "experiments" / "batch", calls


def execute(output, **kwargs):
    return batch.produce_all({"fixture": True}, output, seed=23, steps=2, microbatch=1,
                            device="cpu", **kwargs)


def test_new_batch_produces_all_candidates_once(fixture):
    output, calls = fixture
    first = execute(output)
    assert first["new_P_outputs_in_invocation"] == len(calls) == 20
    with pytest.raises(ValueError, match="NEW_BATCH_OUTPUT_REQUIRED"):
        execute(output)
    assert len(calls) == 20


def test_preflight_never_produces(fixture, monkeypatch):
    _, calls = fixture
    monkeypatch.setattr(batch.producer, "produce", lambda *_a, **_k: pytest.fail("real production"))
    value = batch.preflight()
    assert value["model_forwards"] == value["PT_loads"] == value["P_outputs"] == len(calls) == 0
