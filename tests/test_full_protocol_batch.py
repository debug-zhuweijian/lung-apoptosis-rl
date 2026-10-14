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


def test_completed_resume_does_not_forward(fixture):
    output, calls = fixture
    first = execute(output)
    assert first["new_P_outputs_in_invocation"] == len(calls) == 20
    second = execute(output, resume=True)
    assert second["new_P_outputs_in_invocation"] == 0 and len(calls) == 20
    assert second["reused_P_outputs_in_invocation"] == 20


def test_resume_plan_drift_rejected_before_production(fixture):
    output, calls = fixture
    execute(output)
    plan_path = output / "batch_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["steps"] = 99
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="PRE_PT_RESUME_IDENTITY"):
        execute(output, resume=True)
    assert len(calls) == 20


def test_incomplete_candidate_preserved(fixture):
    output, calls = fixture
    execute(output)
    path = output / "candidate-3" / "prediction.json"
    path.unlink()  # Only the current test's own fixture; no project data.
    with pytest.raises(ValueError, match="INCOMPLETE_CANDIDATE_PRESERVED"):
        execute(output, resume=True)
    assert path.parent.exists() and len(calls) == 20
    assert "INCOMPLETE_CANDIDATE_PRESERVED" in (output / "failures.jsonl").read_text()


def test_preflight_never_produces(fixture, monkeypatch):
    _, calls = fixture
    monkeypatch.setattr(batch.producer, "produce", lambda *_a, **_k: pytest.fail("real production"))
    value = batch.preflight()
    assert value["model_forwards"] == value["PT_loads"] == value["P_outputs"] == len(calls) == 0
