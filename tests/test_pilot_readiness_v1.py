"""Pilot prerequisite and checkpoint provenance regressions, with tiny fixtures."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lung.engine.auxiliary import pipeline_identity
from lung.engine.config import StageUnavailable, file_hash
from lung.engine.evaluate import state_evaluation_config
from lung.engine.preflight import stage_readiness
from lung.engine.train import verify_completed_auxiliary


def fixture_data(tmp_path, complete=False):
    if complete:
        for name in ("normalization.npz", "neighbors.npz", "bandwidths.npz"):
            (tmp_path / name).touch()
    return SimpleNamespace(assets_dir=tmp_path, identity="fixture",
                           manifest={"status": "complete" if complete else "metadata_ready"},
                           condition_ids=lambda role: [role])


def test_metadata_ready_is_not_training_ready(tmp_path):
    config = {"stage": "response", "task": "RESPONSE-A549-V1", "auto_prepare": False}
    result = stage_readiness(config, fixture_data(tmp_path), {})
    assert result["ready_for_selected_stage"] is False
    assert result["can_start_with_automatic_preparation"] is False
    assert set(result["missing_preprocessing_assets"]) == {
        "normalization.npz", "neighbors.npz", "bandwidths.npz"}
    assert result["expression_rows_read"] == 0
    assert not list(tmp_path.iterdir())


def test_auto_preparation_is_pending_not_completed(tmp_path):
    config = {"stage": "response", "task": "RESPONSE-A549-V1", "auto_prepare": True}
    result = stage_readiness(config, fixture_data(tmp_path), {})
    assert not result["ready_for_selected_stage"]
    assert result["can_start_with_automatic_preparation"]


def test_complete_manifest_does_not_hide_missing_assets(tmp_path):
    data = fixture_data(tmp_path, complete=True)
    (tmp_path / "neighbors.npz").unlink()
    result = stage_readiness({"stage": "response", "task": "RESPONSE-A549-V1"}, data, {})
    assert not result["ready_for_selected_stage"]
    assert result["missing_preprocessing_assets"] == ["neighbors.npz"]


def test_state_requires_frozen_response_and_usable_source_split(tmp_path):
    config = {"stage": "state", "task": "RESPONSE-A549-V1", "state_fold": "8_to_1"}
    data = fixture_data(tmp_path, complete=True)
    (tmp_path / "tidyi_split_report.json").write_text(json.dumps({
        "8_to_1": {"status": "STATE_SPLIT_INSUFFICIENT", "missing_classes": {"selection": [2]}},
        "1_to_8": {"status": "ready", "missing_classes": {}}}), encoding="utf-8")
    result = stage_readiness(config, data, {})
    assert "STATE_SPLIT_INSUFFICIENT" in result["blockers"]
    assert any("TRAINED_RESPONSE_REQUIRED" in item for item in result["blockers"])
    ready = stage_readiness({**config, "state_fold": "1_to_8"}, data,
                            {"response": ["frozen.pt"], "_data_identity": "fixture"})
    assert ready["ready_for_selected_stage"]
    wrong = stage_readiness({**config, "state_fold": "1_to_8"}, data,
                            {"response": ["frozen.pt"], "_data_identity": "changed"})
    assert "PIPELINE_DATA_IDENTITY_MISMATCH" in wrong["blockers"]


def test_readiness_reports_task_and_real_endpoint_guards():
    base = {"stage": "all", "synthetic": True}
    wrong_task = stage_readiness({**base, "task": "RESPONSE-A549-V1"}, None, {})
    assert any("TASK_REQUIRED" in item for item in wrong_task["blockers"])
    missing = stage_readiness({**base, "task": "ANNEXIN-A549-V1"}, None, {})
    assert "ENDPOINT_UNAVAILABLE" in missing["blockers"]


def test_state_evaluation_uses_saved_fold_without_changing_current_config():
    config = {"state_fold": "1_to_8", "device": "cpu"}
    resolved = state_evaluation_config(config, {"state_fold": "8_to_1"})
    assert resolved["state_fold"] == "8_to_1"
    assert config["state_fold"] == "1_to_8"
    with pytest.raises(StageUnavailable, match="STATE_CHECKPOINT_FOLD_UNRECORDED"):
        state_evaluation_config(config, {})


@pytest.mark.parametrize("change", [{"state_loss": "ce"},
                                   {"state_loss": "ce_aggregate", "state_aggregate_lambda": 1.0},
                                   {"state_fold": "8_to_1"}])
def test_completed_state_cannot_skip_different_pilot(tmp_path, change):
    response = tmp_path / "response.pt"
    response.write_bytes(b"frozen response fixture")
    pipeline = {"response": [str(response)]}
    data = SimpleNamespace(identity="fixture")
    config = {"seed": 7, "state_fold": "1_to_8", "state_loss": "weighted_ce",
              "state_aggregate_lambda": 0.0, "response": {"inference_ode_steps": 4}}
    identity = pipeline_identity("state", data, config, pipeline)
    path = tmp_path / "state.pt"
    torch.save({"schema_version": 1, "identity": identity,
                "metadata": {"dependencies": {"pipeline": {"response": [file_hash(response)]},
                                               "files": []}}}, path)
    verify_completed_auxiliary("state", [path], data, config, pipeline)
    with pytest.raises(ValueError, match="CHECKPOINT_IDENTITY_MISMATCH"):
        verify_completed_auxiliary("state", [path], data, {**config, **change}, pipeline)
