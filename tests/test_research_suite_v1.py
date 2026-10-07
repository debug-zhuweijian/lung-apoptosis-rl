"""Orchestration failure scenarios, not substitutes for scientific integration."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from filelock import FileLock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import research_suite as suite
from lung.engine.suite_plan import task_graph


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    calls = []
    fail = set()

    def execute(name, context):
        calls.append((name, context))
        if name in fail:
            raise RuntimeError("injected failure")
        artifact = context["job_dir"] / "result.json"
        artifact.write_text(json.dumps({"name": name, "run": context["run_id"]}), encoding="utf-8")
        return {"artifacts": [str(artifact)], "value": name}

    monkeypatch.setattr(suite, "execute_job", execute)
    return tmp_path, calls, fail


def arguments(**kwargs):
    return SimpleNamespace(dry_run=False, resume=None, **kwargs)


def latest(root):
    base = root / "experiments/research-suites"
    run = json.loads((base / "latest.json").read_text())["suite_id"]
    return base / run, json.loads((base / run / "manifest.json").read_text(encoding="utf-8"))


def test_dry_run_is_complete_graph_without_writes(rig, capsys):
    root, calls, _ = rig
    assert suite.run_suite({}, SimpleNamespace(dry_run=True)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert set(plan["tasks"]) == set(task_graph({}))
    assert all(t["status"] == "planned" for t in plan["tasks"].values())
    assert not calls and not list(root.iterdir())


def test_failure_blocks_dependents_but_time_and_summary_continue(rig):
    root, calls, failures = rig
    failures.add("response_cfm_mmd")
    assert suite.run_suite({}, arguments()) == 1
    directory, manifest = latest(root)
    assert manifest["tasks"]["state_ce"]["status"] == "blocked"
    assert manifest["tasks"]["time_nested"]["status"] == "complete"
    assert manifest["tasks"]["summary"]["status"] == "complete"
    assert (directory / "status-summary.json").is_file()
    failed_context = next(c for name, c in calls if name == "response_cfm_mmd")
    calls.clear()
    failures.clear()
    assert suite.run_suite({}, arguments()) == 0
    resumed = next(c for name, c in calls if name == "response_cfm_mmd")
    assert resumed["resume"] and resumed["run_id"] == failed_context["run_id"]
    assert "prepare" not in [name for name, _ in calls]
    assert latest(root)[0] == directory


def test_nonresumable_retry_uses_new_attempt(rig):
    root, calls, failures = rig
    failures.add("ridge")
    assert suite.run_suite({}, arguments()) == 1
    first = next(c for n, c in calls if n == "ridge")
    failures.clear()
    calls.clear()
    assert suite.run_suite({}, arguments()) == 0
    second = next(c for n, c in calls if n == "ridge")
    assert not second["resume"] and second["job_dir"] != first["job_dir"]
    assert first["job_dir"].exists()


def test_artifact_corruption_invalidates_parent_and_descendants(rig):
    root, calls, _ = rig
    assert suite.run_suite({}, arguments()) == 0
    directory, manifest = latest(root)
    Path(manifest["tasks"]["response_cfm_mmd"]["result"]["artifacts"][0]).write_text("corrupt")
    calls.clear()
    assert suite.run_suite({}, SimpleNamespace(dry_run=False, resume=directory.name)) == 0
    rerun = [name for name, _ in calls]
    assert "response_cfm_mmd" in rerun and "state_ce" in rerun
    assert "response_cfm_only" not in rerun


@pytest.mark.parametrize("change", ["data", "source", "config"])
def test_identity_change_rejects_explicit_resume_and_auto_creates_new(rig, change):
    root, _, failures = rig
    data = root / "data.txt"
    data.write_text("original")
    source = root / "train.py"
    source.write_text("# initial")
    config: dict = {"suite_data_identity": {"files": {str(data): "declared"}}}
    failures.add("ridge")
    assert suite.run_suite(config, arguments()) == 1
    directory, _ = latest(root)
    if change == "data":
        data.write_text("changed")
    elif change == "source":
        source.write_text("# changed")
    else:
        config["epochs"] = 2
    with pytest.raises(ValueError, match="identity changed"):
        suite.run_suite(config, SimpleNamespace(dry_run=False, resume=directory.name))
    assert suite.run_suite(config, arguments()) == 1
    assert latest(root)[0] != directory


def test_path_traversal_and_double_instance_rejected(rig):
    root, calls, failures = rig
    with pytest.raises(ValueError, match="plain identifier"):
        suite.run_suite({}, SimpleNamespace(dry_run=False, resume="../external"))
    failures.add("ridge")
    assert suite.run_suite({}, arguments()) == 1
    directory, _ = latest(root)
    calls.clear()
    with FileLock(str(directory / ".suite.lock"), timeout=0):
        assert suite.run_suite({}, arguments()) == 2
    assert not calls


def test_missing_artifact_is_failure_and_summary_fallback_exists(rig, monkeypatch):
    root, _, _ = rig
    monkeypatch.setattr(suite, "execute_job", lambda n, c: {"artifacts": [str(c["job_dir"] / "absent")]})
    assert suite.run_suite({}, arguments()) == 1
    directory, manifest = latest(root)
    assert manifest["tasks"]["prepare"]["status"] == "failed"
    assert manifest["tasks"]["summary"]["status"] == "failed"
    assert json.loads((directory / "status-summary.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_ui_changes_can_resume_complete_suite_without_jobs(rig):
    root, calls, _ = rig
    assert suite.run_suite({"progress": True}, arguments()) == 0
    directory, _ = latest(root)
    calls.clear()
    assert suite.run_suite({"progress": False}, SimpleNamespace(dry_run=False, resume=directory.name)) == 0
    assert not calls


def test_keyboard_interrupt_restores_same_neural_attempt(rig, monkeypatch):
    root, calls, _ = rig
    normal = suite.execute_job
    first = []

    def interrupt(name, context):
        if name == "response_cfm_mmd":
            first.append(context)
            raise KeyboardInterrupt
        return normal(name, context)

    monkeypatch.setattr(suite, "execute_job", interrupt)
    assert suite.run_suite({}, arguments()) == 130
    directory, manifest = latest(root)
    assert manifest["status"] == "interrupted"
    monkeypatch.setattr(suite, "execute_job", normal)
    assert suite.run_suite({}, arguments()) == 0
    resumed = next(c for n, c in calls if n == "response_cfm_mmd")
    assert resumed["resume"] and resumed["job_dir"] == first[0]["job_dir"]
    assert latest(root)[0] == directory


def test_os_lock_excludes_a_second_process(rig):
    root, _, failures = rig
    failures.add("ridge")
    assert suite.run_suite({}, arguments()) == 1
    directory, _ = latest(root)
    code = (
        "import sys; from filelock import FileLock, Timeout\n"
        "try:\n"
        " with FileLock(sys.argv[1], timeout=0): sys.exit(9)\n"
        "except Timeout: print('LOCK_EXCLUDED'); sys.exit(0)\n"
    )
    with FileLock(str(directory / ".suite.lock"), timeout=0):
        child = subprocess.run([sys.executable, "-c", code, str(directory / ".suite.lock")],
                               capture_output=True, text=True, timeout=20)
    assert child.returncode == 0 and "LOCK_EXCLUDED" in child.stdout
    # A surviving lock file must not prevent a subsequent process from running.
    released = subprocess.run([sys.executable, "-c", code, str(directory / ".suite.lock")],
                              capture_output=True, text=True, timeout=20)
    assert released.returncode == 9
