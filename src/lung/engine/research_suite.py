"""Recoverable, failure-aware orchestration of the complete registered study."""

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock, Timeout

from .config import ROOT
from .suite_plan import RESUMABLE, task_graph
from .suite_state import (artifact_hashes, atomic_json, digest, identity, read_json,
                          suite_path, valid_completed)


def execute_job(name, context):
    from .suite_jobs import execute_job as implementation
    return implementation(name, context)


def _select(base, current, requested, auto_resume=True):
    if requested:
        directory = suite_path(base, requested)
        manifest = read_json(directory / "manifest.json")
        if manifest.get("identity") != current:
            raise ValueError("Suite identity changed (config/source/data); start a new suite")
        return directory
    # Examine all unfinished matching suites, not only the most recent run.
    candidates = sorted(base.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True) if auto_resume else []
    for path in candidates:
        directory = suite_path(base, path.parent.name)
        try:
            manifest = read_json(directory / "manifest.json")
        except (OSError, ValueError):
            continue
        if manifest.get("identity") == current and manifest.get("status") != "complete":
            return directory
    run_id = datetime.now(timezone.utc).strftime("suite-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    return suite_path(base, run_id)


def _run(directory, config, graph, current, root):
    path = directory / "manifest.json"
    manifest = read_json(path) if path.exists() else {
        "schema_version": 1, "suite_id": directory.name, "identity": current,
        "config": config, "graph": graph,
        "primary_response": "cfm_mmd",
        "primary_state": config.get("suite", {}).get("primary_state", "weighted_ce"),
        "tasks": {name: {"status": "planned", "attempts": []} for name in graph},
    }
    if manifest.get("identity") != current or manifest.get("graph") != graph:
        raise ValueError("Suite manifest no longer matches current identity/graph")
    manifest["status"] = "running"
    atomic_json(path, manifest)
    tasks = manifest["tasks"]
    for index, (name, parents) in enumerate(graph.items(), 1):
        task = tasks[name]
        parent_identity = {p: {"status": tasks[p]["status"],
                               "hashes": tasks[p].get("artifact_hashes"),
                               "binding": tasks[p].get("binding")} for p in parents}
        binding = digest({"identity": current, "parents": parent_identity, "task": name})
        if valid_completed(task, binding):
            print(f"[{index}/{len(graph)}] {name}：已完成，产物校验通过")
            continue
        failures = [p for p in parents if tasks[p]["status"] != "complete"]
        if failures and name != "summary":
            task.update(status="blocked", reason="依赖未完成: " + ", ".join(failures))
            atomic_json(path, manifest)
            print(f"[{index}/{len(graph)}] {name}：依赖阻断")
            continue
        previous = task.get("status")
        resume = (name in RESUMABLE and previous in ("failed", "running", "interrupted")
                  and task.get("binding") == binding and bool(task.get("attempts")))
        attempts = task["attempts"]
        if resume:
            attempt = attempts[-1]
        else:
            number = len(attempts) + 1
            attempt = {"number": number, "run_id": f"{directory.name}-{name}-{number}"}
            attempts.append(attempt)
        job_dir = (directory / "jobs" / name / f"attempt-{attempt['number']:03d}").resolve()
        if not job_dir.is_relative_to(directory.resolve()):
            raise ValueError("Task directory escaped suite")
        job_dir.mkdir(parents=True, exist_ok=True)
        task.update(status="running", binding=binding, job_dir=str(job_dir),
                    run_id=attempt["run_id"], reason=None)
        atomic_json(path, manifest)
        print(f"[{index}/{len(graph)}] {name}：{'恢复' if resume else '开始'}")
        context = {"root": root, "suite_dir": directory, "job_dir": job_dir,
                   "run_id": attempt["run_id"], "config": copy.deepcopy(config),
                   "dependencies": {p: copy.deepcopy(tasks[p].get("result", {}))
                                    for p in parents if tasks[p]["status"] == "complete"},
                   "resume": resume, "synthetic": bool(config.get("synthetic", False)),
                   "task_states": copy.deepcopy(tasks)}
        try:
            result = execute_job(name, context)
            hashes = artifact_hashes(result)
            # Prove JSON publication before declaring success.
            json.dumps(result)
            task.update(status="complete", result=result, artifact_hashes=hashes)
            attempt["status"] = "complete"
        except (KeyboardInterrupt, SystemExit):
            task.update(status="interrupted", reason="进程被中断")
            attempt["status"] = "interrupted"
            manifest["status"] = "interrupted"
            atomic_json(path, manifest)
            return 130
        except Exception as error:
            task.update(status="failed", reason=f"{type(error).__name__}: {error}")
            attempt.update(status="failed", reason=task["reason"])
            print(f"{name}：失败：{task['reason']}")
        atomic_json(path, manifest)
    complete = all(task["status"] == "complete" for task in tasks.values())
    manifest["status"] = "complete" if complete else "failed"
    atomic_json(path, manifest)
    # This independent status artifact exists even when the summary executor fails.
    atomic_json(directory / "status-summary.json", {
        "suite_id": directory.name, "status": manifest["status"],
        "tasks": {name: {"status": task["status"], "reason": task.get("reason")}
                  for name, task in tasks.items()}})
    print(f"套件 {directory.name}：{manifest['status']}；清单：{path}")
    return 0 if complete else 1


def run_suite(config, arguments):
    root = Path(ROOT).resolve()
    config = copy.deepcopy(config)
    graph = task_graph(config)
    if getattr(arguments, "dry_run", False):
        print(json.dumps({"workflow": "research_all", "status": "planned",
                          "static_preparation": ["Tahoe 训练资产检查/准备", "独立 sci-Plex 时间资产准备"],
                          "tasks": {name: {"status": "planned", "dependencies": parents}
                                    for name, parents in graph.items()}}, ensure_ascii=False, indent=2))
        return 0
    current = identity(config, root)
    base = root / "experiments" / "research-suites"
    if not base.resolve().is_relative_to(root):
        raise ValueError("research-suites escaped project root")
    base.mkdir(parents=True, exist_ok=True)
    # Serialize selection/publication so simultaneous default invocations select one suite.
    try:
        with FileLock(str(base / ".selection.lock"), timeout=0):
            directory = _select(base, current, getattr(arguments, "resume", None),
                                config.get("suite", {}).get("auto_resume", True))
            directory.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(directory / ".suite.lock"), timeout=0)
            lock.acquire()
            try:
                if not (directory / "manifest.json").exists():
                    atomic_json(directory / "manifest.json", {
                        "schema_version": 1, "suite_id": directory.name, "identity": current,
                        "config": config, "graph": graph, "status": "planned",
                        "primary_response": "cfm_mmd",
                        "primary_state": config.get("suite", {}).get("primary_state", "weighted_ce"),
                        "tasks": {name: {"status": "planned", "attempts": []} for name in graph}})
                atomic_json(base / "latest.json", {"suite_id": directory.name})
            except BaseException:
                lock.release()
                raise
    except Timeout:
        print("套件已有进程执行，未启动第二个实例。")
        return 2
    dashboard_process = None
    try:
        if config.get("dashboard", False):
            from lung.monitoring.dashboard import launch_dashboard
            log_root = root / "logs"
            log_root.mkdir(parents=True, exist_ok=True)
            dashboard_process, dashboard = launch_dashboard(
                log_root, directory / "dashboard.json", open_browser=config.get("open_browser", True))
            print(f"套件 {directory.name} TensorBoard：{dashboard.get('url', dashboard.get('reason'))}")
        return _run(directory, config, graph, current, root)
    finally:
        if config.get("synthetic", False) and dashboard_process is not None and dashboard_process.poll() is None:
            dashboard_process.terminate()
            dashboard_process.wait(timeout=10)
        lock.release()
