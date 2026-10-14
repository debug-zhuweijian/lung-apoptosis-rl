"""Caller values are mock predictions; no fit, scorer, or model inference."""
import json
from pathlib import Path

import pytest

from lung.data.martin_state_function import MartinStateFunctionInputs, SCALE, sha
from lung.data.state_function_evaluation import SCHEMA, number, preflight, freeze, evaluate, block_metrics

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/bridge_v2"


@pytest.fixture
def source():
    return MartinStateFunctionInputs(BRIDGE, project_root=ROOT)


def packet(source, model):
    rows = []
    for fid, fold in source.folds.items():
        for cid in fold["final_test_candidate_ids"]:
            c = source.candidates[cid]
            rows.append({**{k: c[k] for k in ("candidate_id", "decision_block_id", "measurement_domain_id",
                                            "state_reference_id", "cell_background")},
                         "fold_id": fid, "prediction": 2.0})
    return {"schema": SCHEMA, "value_scale": SCALE, "rules_tuned_with_final_Y": False,
            "source_manifest_sha256": sha(BRIDGE / "manifest.json"),
            "model_artifact_sha256": sha(model), "predictions": rows}


def files(tmp_path, source):
    model = tmp_path / "explicit_mock_artifact.bin"
    model.write_bytes(b"mock artifact; never trained")
    predictions = tmp_path / "caller.json"
    predictions.write_text(json.dumps(packet(source, model)), encoding="utf-8")
    return predictions, model


@pytest.mark.parametrize("value", [True, False, "1", None, float("nan"), float("inf"), -1, 1j])
def test_bad_numbers(value):
    with pytest.raises(ValueError):
        number(value)


def test_no_fraction_clipping():
    assert number(12.7) == 12.7


def test_metadata_and_freeze_never_parse_Y(source, tmp_path, monkeypatch):
    predictions, model = files(tmp_path, source)
    original = Path.read_text
    def guarded(path, *a, **kw):
        if path.name in ("observations.json", "published_means.csv"):
            raise AssertionError("Y parsed")
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "read_text", guarded)
    monkeypatch.setattr(MartinStateFunctionInputs, "observation_units", lambda *a: pytest.fail("Y API"))
    assert preflight(source)["labels_parsed"] is False
    result = freeze(source, predictions, model, tmp_path / "frozen")
    assert result["execution_certified"] is False


@pytest.mark.parametrize("change", ["subset", "duplicate", "domain", "scale", "tuning", "model"])
def test_invalid_identity(source, tmp_path, change):
    predictions, model = files(tmp_path, source)
    p = json.loads(predictions.read_text())
    if change == "subset": p["predictions"].pop()
    elif change == "duplicate": p["predictions"].append(p["predictions"][0])
    elif change == "domain": p["predictions"][0]["measurement_domain_id"] = "other"
    elif change == "scale": p["value_scale"] = "Annexin_fraction"
    elif change == "tuning": p["rules_tuned_with_final_Y"] = True
    else: p["model_artifact_sha256"] = "bad"
    predictions.write_text(json.dumps(p))
    with pytest.raises(ValueError): freeze(source, predictions, model, tmp_path / "frozen")


@pytest.mark.parametrize("asset", ["model", "caller", "snapshot", "receipt"])
def test_tamper_before_final_labels(source, tmp_path, monkeypatch, asset):
    predictions, model = files(tmp_path, source)
    directory = tmp_path / "frozen"
    freeze(source, predictions, model, directory)
    paths = {"model": model, "caller": predictions, "snapshot": directory / "predictions.json",
             "receipt": directory / "receipt.json"}
    with paths[asset].open("ab") as stream: stream.write(b" ")
    monkeypatch.setattr(MartinStateFunctionInputs, "observation_units", lambda *a: pytest.fail("early Y"))
    with pytest.raises(ValueError): evaluate(directory)


def test_real_Y_explicit_readback_with_mock_predictions(source, tmp_path):
    predictions, model = files(tmp_path, source)
    directory = tmp_path / "frozen"
    freeze(source, predictions, model, directory)
    with pytest.raises(FileExistsError): freeze(source, predictions, model, directory)
    result = evaluate(directory)
    assert len(result["blocks"]) == 8
    assert sum(b["observation_count"] for b in result["blocks"]) == 20
    assert result["execution_certified"] is result["reward_qualified"] is result["Q_qualified"] is False
    for b in result["blocks"]:
        y = source.observation_units(b["fold_id"], "final_test", b["measurement_domain_id"])
        assert b["MAE"] == pytest.approx(sum(abs(2-r["value"]) for r in y)/len(y))


def test_pixel_overlap_and_tie_regret():
    predictions = [{"candidate_id": "a", "prediction": 8}, {"candidate_id": "b", "prediction": 8}]
    labels = [{"candidate_id": "a", "value": 10, "reading_lower": 8, "reading_upper": 12,
               "weight": 1, "value_scale": SCALE},
              {"candidate_id": "b", "value": 11, "reading_lower": 9, "reading_upper": 13,
               "weight": 1, "value_scale": SCALE}]
    result = block_metrics(predictions, labels)
    assert result["mean_candidate_regret"] == .5
    assert result["rank_pairs"][0]["status"] == "uncertain"
    assert result["resolved_rank_pair_concordance"] is None


def test_duplicate_json_value_cannot_hide_bool(source, tmp_path):
    predictions, model = files(tmp_path, source)
    text = predictions.read_text().replace('"prediction": 2.0', '"prediction": true, "prediction": 2.0', 1)
    predictions.write_text(text)
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        freeze(source, predictions, model, tmp_path / "frozen")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_are_rejected(source, tmp_path, constant):
    predictions, model = files(tmp_path, source)
    predictions.write_text(predictions.read_text().replace('"prediction": 2.0', '"prediction": ' + constant, 1))
    with pytest.raises(ValueError, match="NONFINITE_JSON_CONSTANT"):
        freeze(source, predictions, model, tmp_path / "frozen")


def test_large_finite_predictions_have_finite_mean_metrics():
    import math
    predictions = [{"candidate_id": str(i), "prediction": 1e308} for i in range(3)]
    labels = [{"candidate_id": str(i), "value": 2., "reading_lower": 1., "reading_upper": 3.,
               "weight": 1, "value_scale": SCALE} for i in range(3)]
    result = block_metrics(predictions, labels)
    assert math.isfinite(result["MAE"]) and result["MAE"] == pytest.approx(1e308)
    assert result["mean_candidate_regret"] == 0
