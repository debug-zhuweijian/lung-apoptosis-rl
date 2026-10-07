"""Task-adapter contracts and actual numerical synthetic Ridge fitting."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine.config import load_config, ROOT
from lung.engine.synthetic import smoke_config
from lung.engine.suite_jobs import execute_job
from lung.engine.suite_training import time_config, child, merge_primary


@pytest.fixture
def context(tmp_path):
    return {"root": ROOT, "suite_dir": tmp_path, "job_dir": tmp_path,
            "run_id": "unit-child", "resume": False, "synthetic": True,
            "config": smoke_config(load_config()), "dependencies": {}, "task_states": {}}


def test_synthetic_ridge_really_fits_and_exports_finite_predictions(context):
    result = execute_job("ridge", context)
    assert result["synthetic"] is True and result["proxy"] is True
    with np.load(result["shifts"], allow_pickle=False) as saved:
        assert saved["delta"].shape == (2, 8)
        assert np.isfinite(saved["delta"]).all()
        assert np.any(saved["delta"] != 0)
    report = json.loads(Path(result["report"]).read_text())
    assert report["fit_role"] == "T" and report["selection_role"] == "V"
    assert all(Path(p).is_file() for p in result["artifacts"])


def test_time_config_unwraps_defaults_and_smoke_is_small(context):
    config = time_config(context)
    assert "time_auxiliary" not in config["time_auxiliary"]
    assert config["time_auxiliary"]["outer_folds"] == 2
    assert config["time_auxiliary"]["inner_folds"] == 2
    assert config["timecourse"]["epochs"] == 1
    assert config["timecourse"]["response"]["members"] == 2
    assert config["time_decision"]["policy_seeds"] == [0]
    assert config["time_decision"]["daac"]["episodes_per_epoch"] == 6


def test_summary_preserves_failed_and_blocked_statuses(context):
    context["task_states"] = {"ridge": {"status": "failed", "reason": "numeric failure"},
                              "response_compare": {"status": "blocked", "reason": "ridge"},
                              "summary": {"status": "running"}}
    result = execute_job("summary", context)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["tasks"]["response_compare"]["status"] == "blocked"
    assert report["tasks"]["summary"]["status"] == "complete"
    markdown = (Path(context["job_dir"]) / "summary.md").read_text(encoding="utf-8")
    assert len(markdown.splitlines()) > 10
    assert markdown.startswith("# 一键研究套件汇总\n\n")
    assert "\\n" not in markdown


def test_child_dispatch_has_single_workflow_and_exact_objective(context, monkeypatch):
    from lung.engine import train
    captured = []
    def invoke(arguments):
        cfg = yaml.safe_load(Path(arguments[1]).read_text(encoding="utf-8"))
        captured.append(cfg)
        return 1
    monkeypatch.setattr(train, "main", invoke)
    with pytest.raises(RuntimeError, match="failed with exit 1"):
        child(context, "response", overrides={"task": "RESPONSE-A549-V1", "response": {"mmd_weight": 0.0}})
    assert captured[0]["workflow"] == "single"
    assert captured[0]["task"] == "RESPONSE-A549-V1"
    assert captured[0]["response"]["mmd_weight"] == 0.0
    assert captured[0]["response"]["members"] == 2


def test_primary_merge_refuses_mismatched_data(context):
    folder = Path(context["job_dir"])
    p, c = folder / "P.json", folder / "C.json"
    p.write_text(json.dumps({"data_identity": "one"}))
    c.write_text(json.dumps({"data_identity": "two"}))
    context["dependencies"] = {"response_cfm_mmd": {"pipeline": str(p)},
                              "state_weighted_ce": {"pipeline": str(c)}}
    with pytest.raises(ValueError, match="DATA_IDENTITY"):
        merge_primary(context)


def test_suite_dashboard_is_single_call_after_dry_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lung.engine import research_suite
    from lung.monitoring import dashboard
    calls = []
    monkeypatch.setattr(research_suite, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "launch_dashboard", lambda *args, **kwargs:
                        (calls.append((args, kwargs)) or (None, {"url": "http://127.0.0.1:6006"})))
    def executor(name, ctx):
        target = ctx["job_dir"] / "result.txt"
        target.write_text(name)
        return {"artifacts": [str(target)]}
    monkeypatch.setattr(research_suite, "execute_job", executor)
    config = {"dashboard": True, "open_browser": False}
    assert research_suite.run_suite(config, SimpleNamespace(dry_run=True)) == 0
    assert not calls
    assert research_suite.run_suite(config, SimpleNamespace(dry_run=False, resume=None)) == 0
    assert len(calls) == 1 and calls[0][0][0] == tmp_path / "logs"


def test_job_progress_replaces_phase_and_closes_on_failure(monkeypatch):
    from lung.engine.suite_jobs import job_progress
    import tqdm.auto
    bars = []
    class Bar:
        def __init__(self, total, **kwargs):
            self.total, self.n, self.closed = total, 0, False
            bars.append(self)
        def update(self, amount):
            self.n += amount
        def close(self):
            self.closed = True
    monkeypatch.setattr(tqdm.auto, "tqdm", Bar)
    with pytest.raises(RuntimeError, match="interrupted"):
        with job_progress({"progress": True}) as progress:
            progress("prepare", 1, 3)
            progress("prepare", 2, 3)
            progress("train", 1, 5)
            raise RuntimeError("interrupted")
    assert [bar.n for bar in bars] == [2, 1]
    assert all(bar.closed for bar in bars)
    with job_progress({"progress": False}) as progress:
        progress("quiet", 1, 2)
    assert len(bars) == 2


@pytest.mark.parametrize("name,module,entry", [
    ("prepare", "lung.engine.data", "open_data"),
    ("time_prepare", "lung.timecourse.data", "prepare_time_data"),
    ("time_nested", "lung.timecourse.nested", "run_nested"),
    ("time_decision", "lung.timecourse.decisions", "run_comparison"),
])
def test_preparation_and_time_jobs_forward_progress(context, monkeypatch, name, module, entry):
    from importlib import import_module
    from types import SimpleNamespace
    context["config"]["progress"] = False
    context["dependencies"] = {"time_nested": {}, "state_weighted_ce": {
        "pipeline_artifacts": {}, "original_config": {}}}
    calls = []
    artifact = Path(context["job_dir"]) / "fixture.json"
    artifact.write_text("{}")
    def operation(*args, **kwargs):
        callback = kwargs["progress"]
        assert callable(callback)
        callback("fixture", 1, 2)
        calls.append(True)
        if name == "prepare":
            return SimpleNamespace(identity="synthetic", manifest={}, synthetic=True)
        if name == "time_prepare":
            return artifact
        return {"artifacts": [str(artifact)]}
    monkeypatch.setattr(import_module(module), entry, operation)
    result = execute_job(name, context)
    assert calls == [True] and result["synthetic"] is True
