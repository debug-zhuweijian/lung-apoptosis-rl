"""Root-entry routing and inherited-input recovery, without real training."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lung.engine import suite_entry
from lung.engine.config import file_hash
from lung.engine.response import condition_kwargs
from lung.engine.train import parser, read_pipeline


def test_suite_identity_binds_comparison_settings_and_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(suite_entry, "ROOT", tmp_path)
    monkeypatch.setattr(suite_entry, "project_path", lambda value: tmp_path / value)
    files = {
        "configs/dataset.yaml": "split_manifest: split.json\n",
        "split.json": "{}",
        "configs/pilot_ridge.yaml": "protocol: protocol.md\n",
        "configs/pilot_v.yaml": "bootstrap_seed: 7\n",
        "protocol.md": "Frozen protocol A",
    }
    for name in ("timecourse.yaml", "time_auxiliary.yaml", "time_decision.yaml"):
        files["configs/" + name] = "{}"
    for name in ("extraction_status.json", "a549_cells.parquet", "common_genes.parquet"):
        files["08-datasets/sciPlex3/GSM4150378/extracted/v1/" + name] = "source fixture"
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    config = {"data": "configs/dataset.yaml"}
    first = suite_entry.suite_data_identity(config)
    (tmp_path / "configs/pilot_v.yaml").write_text("bootstrap_seed: 8\n", encoding="utf-8")
    second = suite_entry.suite_data_identity(config)
    assert first != second
    (tmp_path / "protocol.md").write_text("Frozen protocol B", encoding="utf-8")
    third = suite_entry.suite_data_identity(config)
    assert second != third
    assert str((tmp_path / "protocol.md").resolve()) in third["files"]


def test_default_root_routes_to_registered_suite(monkeypatch):
    from lung.engine import research_suite
    config = {"workflow": "research_all", "synthetic": True, "seed": 7,
              "device": "cpu", "stage": "all", "task": "PROXY-A549-V1", "epochs": 1,
              "algorithm": "both"}
    observed = {}
    monkeypatch.setattr(suite_entry, "load_config", lambda path: dict(config))
    monkeypatch.setattr(suite_entry, "validate_config", lambda cfg: None)
    monkeypatch.setattr(suite_entry, "verify_interpreter", lambda: None)
    def run(cfg, arguments):
        observed.update(cfg)
        return 17
    monkeypatch.setattr(research_suite, "run_suite", run)
    assert suite_entry.dispatch_suite(parser().parse_args([])) == 17
    assert observed["suite_data_identity"]["scope"] == "synthetic_engineering_only"
    assert observed["stage"] == "all"


@pytest.mark.parametrize("option,value", [("--task", "ANNEXIN-A549-V1"), ("--algorithm", "daac")])
def test_suite_rejects_silent_task_or_algorithm_substitution(option, value, monkeypatch):
    config = {"workflow": "research_all", "synthetic": True, "seed": 7,
              "device": "cpu", "stage": "all", "task": "PROXY-A549-V1", "epochs": 1,
              "algorithm": "both"}
    monkeypatch.setattr(suite_entry, "load_config", lambda path: dict(config))
    with pytest.raises(ValueError, match="research_all requires"):
        suite_entry.dispatch_suite(parser().parse_args([option, value]))


@pytest.mark.parametrize("args", [["--stage", "response"], ["--prepare-only"]])
def test_explicit_single_stage_keeps_legacy_entry(args, monkeypatch):
    def forbidden(_):
        raise AssertionError("suite configuration must not be loaded")
    monkeypatch.setattr(suite_entry, "load_config", forbidden)
    assert suite_entry.dispatch_suite(parser().parse_args(args)) is None


def test_legacy_child_resume_does_not_become_new_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(suite_entry, "ROOT", tmp_path)
    child = tmp_path / "experiments/child"
    child.mkdir(parents=True)
    (child / "active_config.yaml").write_text("stage: state\n", encoding="utf-8")
    assert suite_entry.dispatch_suite(parser().parse_args(["--resume", "child"])) is None


def test_first_publish_interruption_keeps_frozen_parent(tmp_path, monkeypatch):
    from lung.engine import train
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "implementation_identity", lambda: "code")
    weight = tmp_path / "response.pt"
    weight.write_bytes(b"frozen dependency fixture")
    source = tmp_path / "parent.json"
    source.write_text(json.dumps({"schema_version": 1, "implementation_identity": "code",
                                 "data_identity": "data", "artifacts": {"response": [str(weight)]},
                                 "artifact_hashes": {str(weight): file_hash(weight)}}), encoding="utf-8")
    actual = read_pipeline({"pipeline": str(source)}, "unpublished-state-child")
    assert actual == {"response": [str(weight)], "_data_identity": "data"}
    weight.write_bytes(b"changed dependency")
    with pytest.raises(ValueError, match="FROZEN_PIPELINE_ARTIFACT_CHANGED"):
        read_pipeline({"pipeline": str(source)}, "unpublished-state-child")


def test_condition_clock_uses_real_endpoint_and_explicit_world_override():
    batch = {"fingerprint": [0.] * 2048, "dose": 1., "duration": 72.,
             "background_id": 0, "source_id": 1, "protocol_id": 0}
    actual = condition_kwargs(batch, 2, "cpu")
    torch.testing.assert_close(actual["horizon"], torch.full((2,), 72.))
    torch.testing.assert_close(actual["duration"], torch.full((2,), 72.))
    override = condition_kwargs(batch, 2, "cpu", horizon=96.)
    torch.testing.assert_close(override["horizon"], torch.full((2,), 96.))
    old = condition_kwargs({**batch, "duration": 24.}, 1, "cpu")
    assert old["horizon"].item() == 24.
