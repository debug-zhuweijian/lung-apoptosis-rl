"""Independent response research entry (implementation task W1).

Modes:
- ``route_compare``: sequential single-member arms (full / cfm_only /
  cfm_then_full by default; condition_only, population_mean and
  one_step_then_full optional) followed by one frozen-weight V comparison.
- ``v_compare``: re-evaluate registered frozen weights on V only.
- ``latent_ae`` / ``latent_cache`` / ``latent_cfm`` / ``latent_members``:
  staged latent pipeline (task W3/W6), dispatched to ``latent_pipeline``.

Only T/fit and V/selection are opened; E/final_test, the registered 19-task
suite and downstream RL are never started here. No V winner is promoted: the
report records arms, update counts, stop reasons and wall time; promotion and
final tests stay separate research decisions, as in the suite contract.
"""

import argparse
import json
import math
import time
from pathlib import Path

from lung.monitoring import RunMonitor

from .checkpoint import atomic_json, load_checkpoint, seed_all
from .config import ROOT, StageUnavailable, merge, project_path, read_yaml, validate_config, verify_interpreter
from .numerics import PrecisionPolicy
from .timing import PhaseTimer
from .training_control import checkpoint_completion

MODES = ("route_compare", "v_compare", "latent_ae", "latent_cache", "latent_cfm", "latent_members")
ROUTE_ARMS = ("full", "cfm_only", "cfm_then_full", "one_step_then_full", "condition_only", "population_mean")
DEFAULT_ARMS = ("full", "cfm_only", "cfm_then_full")
REPORT_PROTOCOL = "response-research-v1"


def parser():
    command = argparse.ArgumentParser(description="响应研究独立入口：路线比较 / 潜空间阶段 / V 比较")
    command.add_argument("--config", default="configs/response_research.yaml")
    command.add_argument("--mode", choices=MODES)
    command.add_argument("--resume", metavar="RUN_ID")
    command.add_argument("--run-id")
    command.add_argument("--epochs", type=int)
    command.add_argument("--device")
    command.add_argument("--budget-seconds", type=float, help="Optional wall-clock development budget")
    command.add_argument("--smoke", action="store_true")
    command.add_argument("--no-dashboard", action="store_true")
    command.add_argument("--no-browser", action="store_true")
    command.add_argument("--no-progress", action="store_true")
    return command


def load_research_config(path):
    config = read_yaml(project_path(path))
    if config.get("workflow") != "response_research":
        raise ValueError("response research entry requires workflow: response_research")
    if config.get("stage") != "response":
        raise ValueError("response research runs the response stage only")
    if "suite" in config:
        raise ValueError("response research is decoupled from the registered suite; remove the suite section")
    config["response"] = merge(read_yaml(ROOT / "configs/response.yaml"), config.get("response", {}))
    research = config.setdefault("research", {})
    if not isinstance(research, dict):
        raise ValueError("research must be a mapping")
    mode = research.get("mode", "route_compare")
    if mode not in MODES:
        raise ValueError(f"RESEARCH_MODE_UNKNOWN: {mode}")
    arms = research.get("arms", list(DEFAULT_ARMS))
    if not isinstance(arms, list) or not arms or any(arm not in ROUTE_ARMS for arm in arms):
        raise ValueError(f"research.arms must be a nonempty subset of {ROUTE_ARMS}")
    if len(set(arms)) != len(arms):
        raise ValueError("DUPLICATE_RESEARCH_ARM")
    budget = research.get("budget_seconds")
    if budget is not None and (not isinstance(budget, (int, float)) or isinstance(budget, bool)
                               or not math.isfinite(budget) or budget <= 0):
        raise ValueError("research.budget_seconds must be a positive finite number or null")
    validate_config(config)
    return config


def resolve(arguments):
    if arguments.resume:
        if Path(arguments.resume).name != arguments.resume:
            raise ValueError("--resume expects a run_id, not a filesystem path")
        active = ROOT / "experiments" / arguments.resume / "active_config.yaml"
        if not active.exists():
            raise StageUnavailable(f"RESEARCH_RESUME_MISSING: {active}")
        config = read_yaml(active)
        if config.get("workflow") != "response_research":
            raise ValueError("Cannot resume a non-research run from this entry")
    else:
        config = load_research_config(arguments.config)
    if arguments.resume and arguments.smoke and not config.get("synthetic", False):
        raise ValueError("Cannot change a real research run to synthetic during resume")
    if arguments.smoke and not arguments.resume:
        from .synthetic import smoke_config
        config = smoke_config(config)
    if arguments.mode is not None:
        if arguments.resume and arguments.mode != config["research"].get("mode"):
            raise ValueError("Cannot change the research mode when resuming; create a new run")
        config["research"]["mode"] = arguments.mode
    if arguments.budget_seconds is not None:
        if arguments.budget_seconds <= 0 or not math.isfinite(arguments.budget_seconds):
            raise ValueError("--budget-seconds must be positive and finite")
        config["research"]["budget_seconds"] = arguments.budget_seconds
    for key in ("epochs", "device"):
        if getattr(arguments, key) is not None:
            if arguments.resume and getattr(arguments, key) != config.get(key):
                raise ValueError(f"Cannot change {key} when resuming; create a new run")
            config[key] = getattr(arguments, key)
    for argument, key in (("no_dashboard", "dashboard"), ("no_browser", "open_browser"),
                          ("no_progress", "progress")):
        if getattr(arguments, argument):
            config[key] = False
    validate_config(config)
    return config


def _update_state(path, updater):
    """One JSON state file for budget and arm progress; updates never drop keys."""
    path = Path(path)
    state = {"schema_version": 1, "budget_total_seconds": None,
             "budget_consumed_seconds": 0., "branches": {}}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("RESEARCH_STATE_VERSION")
    state = updater(state)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(path, state)
    return state


class ResearchBudget:
    """Optional wall-clock development budget; exhaustion is resumable, not convergence."""

    def __init__(self, total_seconds, state_path=None):
        self.total = None if total_seconds is None else float(total_seconds)
        self.consumed = 0.
        self.state_path = Path(state_path) if state_path is not None else None
        self.segment_started = time.monotonic()
        self.exhausted = False
        if self.state_path is not None and self.state_path.exists():
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.consumed = float(saved.get("budget_consumed_seconds", 0.))
            # An explicitly passed total is the NEW overall cap (CLI semantics,
            # rework R4.4): consumed time is preserved, the cap is replaced and
            # persisted; only an omitted total falls back to the saved one.
            if self.total is None:
                self.total = (None if saved.get("budget_total_seconds") is None
                              else float(saved["budget_total_seconds"]))
            elif saved.get("budget_total_seconds") != self.total:
                self.pause()
        if self.total is not None and self.remaining() <= 0:
            self.exhausted = True

    def remaining(self):
        if self.total is None:
            return math.inf
        return self.total - (self.consumed + (time.monotonic() - self.segment_started))

    def check(self, step, epoch):
        """Stop-check callback wired by the caller into the training update loop."""
        if not self.exhausted and self.remaining() <= 0:
            self.exhausted = True
        return self.exhausted

    def pause(self):
        """Close the current timed segment and persist the consumed total."""
        now = time.monotonic()
        self.consumed += now - self.segment_started
        self.segment_started = now
        if self.state_path is not None:
            _update_state(self.state_path, lambda state: {**state,
                "budget_total_seconds": self.total,
                "budget_consumed_seconds": self.consumed})

    def summary(self):
        return {"total_seconds": self.total,
                "consumed_seconds": self.consumed + (time.monotonic() - self.segment_started),
                "exhausted": self.exhausted}


class BranchMonitor:
    """Per-arm weights/logdir view; metrics merge into the parent run monitor."""

    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.weights = Path(parent.weights) / "branches" / name
        self.logdir = Path(parent.logdir) / "branches" / name
        self.weights.mkdir(parents=True, exist_ok=True)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.config = parent.config

    def progress(self, iterable=None, **kwargs):
        return self.parent.progress(iterable, **kwargs)

    def record(self, stage, step, metrics, axis="epoch"):
        return self.parent.record(f"arm/{self.name}/{stage}", step, metrics, axis=axis)

    def say(self, message):
        return self.parent.say(f"[{self.name}] {message}")

    def close(self):
        return None


def arm_overrides(arm, config):
    """One arm's response overrides; cfm-only semantics follow suite_jobs."""
    if arm == "full":
        return {"task": "RESPONSE-A549-V1", "response": {
            "mmd_weight": 1.0, "coupling": "entropic_ot_prior",
            "training_schedule": {"mode": "full"}, "model": {"reference_mode": "observed"}}}
    if arm == "cfm_only":
        return {"task": "RESPONSE-A549-V1", "response": {
            "mmd_weight": 0.0, "coupling": "entropic_ot_prior",
            "training_schedule": {"mode": "full"}, "model": {"reference_mode": "observed"}}}
    if arm in {"cfm_then_full", "one_step_then_full"}:
        from .optimization import candidate_overrides
        return candidate_overrides(arm, config)
    if arm in {"condition_only", "population_mean"}:
        return {"task": "RESPONSE-A549-V1", "response": {
            "mmd_weight": 1.0, "coupling": "entropic_ot_prior",
            "training_schedule": {"mode": "full"},
            "model": {"reference_mode": arm}}}
    raise ValueError(f"UNKNOWN_RESEARCH_ARM: {arm}")


def arm_config(arm, config):
    merged = merge(config, arm_overrides(arm, config))
    merged["workflow"] = "response_research_arm"
    merged["response"]["members"] = 1
    merged["response"]["member_workers"] = 1
    return merged


def _arm_weights(weights_root, arm):
    folder = Path(weights_root) / "branches" / arm / "response" / "member_1"
    if (folder / "best.pt").exists():
        return folder / "best.pt", "best"
    if (folder / "last.pt").exists():
        return folder / "last.pt", "last_no_full_selection"
    return None, "missing"


def _load_for_comparison(path, data, device):
    """Load response or staged-latent weights behind one comparison interface.

    Staged-latent checkpoints record stage latent_cfm/latent_members and hold
    {shared, member} state; the adapter exposes the evaluator's model contract
    (rework R5.2). Identity checks stay inside the loaders.
    """
    from .response import load_response
    saved = load_checkpoint(path)
    stage = saved.get("metadata", {}).get("stage")
    if stage == "response":
        return load_response(path, data, device)
    if stage in ("latent_cfm", "latent_members"):
        from .latent_pipeline import load_latent_adapter
        return load_latent_adapter(path, data, device)
    raise ValueError(f"V_COMPARE_UNSUPPORTED_STAGE: {stage} ({path})")


def _arm_completion(weights_path, target_epochs=None):
    """Canonical completion semantics; budget exhaustion is resumable, not complete."""
    try:
        saved = load_checkpoint(weights_path)
    except (OSError, ValueError):
        return {"status": "checkpoint_unreadable"}
    control = saved.get("extra", {}).get("training_control") or {}
    target = control.get("target_epochs", target_epochs)
    budget_flag = saved.get("extra", {}).get("budget_exhausted", False)
    try:
        completion = checkpoint_completion(saved, target)
    except (ValueError, TypeError):
        completion = {"status": "incomplete",
                      "reason": "budget_exhausted" if budget_flag else control.get("reason", "interrupted")}
    return {**completion, "budget_exhausted": bool(budget_flag),
            "actual_epochs": saved.get("epoch"),
            "actual_updates": saved.get("counters", {}).get("optimizer_step")}


def run_route_compare(config, monitor, data, budget):
    """Equal-share per-arm wall-clock screening (rework C1).

    The entry-level budget is split evenly across the arms; each arm gets its
    own resumable ResearchBudget, so one exhausted arm never starves the rest.
    This is equal-share screening, not a strict equal-wall-clock claim; the
    comparison report records the protocol explicitly.
    """
    from .response import train_response
    state_path = Path(monitor.experiment) / "research_state.json"
    branches = {}
    if state_path.exists():
        branches = json.loads(state_path.read_text(encoding="utf-8")).get("branches", {})
    arms = config["research"]["arms"]
    per_arm_total = budget.total / len(arms) if budget.total is not None else None
    results = {}
    for arm in arms:
        prior = branches.get(arm)
        if prior and prior.get("status") == "complete":
            monitor.say(f"臂 {arm} 已完成，跳过。")
            results[arm] = prior
            continue
        arm_budget = ResearchBudget(per_arm_total,
                                    Path(monitor.experiment) / f"budget_{arm}.json")
        if arm_budget.exhausted:
            monitor.say(f"臂 {arm} 额度已耗尽：未启动（可用 --budget-seconds 提高总额后恢复）。")
            weights, selection = _arm_weights(monitor.weights, arm)
            entry = {**prior, "status": "budget_exhausted"} if prior else {
                "status": "budget_exhausted", "weights": str(weights) if weights else None,
                "selection": selection, "wall_seconds": 0.,
                "completion": _arm_completion(weights, config["epochs"]) if weights else None,
                "response_config": arm_config(arm, config)["response"]}
            _update_state(state_path, lambda state: {**state,
                "branches": {**state.get("branches", {}), arm: entry}})
            results[arm] = entry
            continue
        branch_monitor = BranchMonitor(arm, monitor)
        arm_cfg = arm_config(arm, config)
        timer = PhaseTimer(config["device"])
        # An interrupted or budget-exhausted arm resumes from its member last.pt;
        # identity checks inside the response trainer reject any mismatch.
        resume_arm = _arm_weights(monitor.weights, arm)[0] is not None
        monitor.say(f"开始臂 {arm}：schedule={arm_cfg['response'].get('training_schedule', {}).get('mode', 'full')}"
                    f"，mmd_weight={arm_cfg['response'].get('mmd_weight', 1.)}，resume={resume_arm}。")
        try:
            with timer.phase(f"arm/{arm}/train"):
                train_response(data, arm_cfg, branch_monitor, resume=resume_arm,
                               stop_check=arm_budget.check)
        finally:
            # Settle this arm's budget on every handleable exit — including
            # Ctrl-C and exceptions — so a resume cannot re-obtain spent time
            # (rework S07). Hard process kills are outside any in-process
            # guarantee and are documented as such.
            arm_budget.pause()
            budget.pause()
        wall = timer.summary()["host_seconds"].get(f"arm/{arm}/train", 0.)
        weights, selection = _arm_weights(monitor.weights, arm)
        entry = {"status": "complete" if not arm_budget.exhausted else "budget_exhausted",
                 "weights": str(weights) if weights else None,
                 "selection": selection, "wall_seconds": wall,
                 "arm_budget": arm_budget.summary(),
                 "completion": _arm_completion(weights, config["epochs"]) if weights else None,
                 "timing": timer.summary(),
                 "response_config": arm_cfg["response"]}
        _update_state(state_path, lambda state: {**state,
            "branches": {**state.get("branches", {}), arm: entry}})
        results[arm] = entry
        if arm_budget.exhausted:
            monitor.say(f"臂 {arm} 额度耗尽；其余臂继续（各自的独立额度）。")
    comparison = compare_arms(config, monitor, data, results, budget)
    statuses = [entry.get("status") for entry in results.values()]
    run_status = ("complete" if statuses and all(status == "complete" for status in statuses)
                  else "budget_exhausted" if any(status == "budget_exhausted" for status in statuses)
                  else "complete")
    return {"branches": results, "comparison": comparison, "run_status": run_status,
            "budget_protocol": "equal_share_per_arm_wall_clock" if per_arm_total else "unbounded"}


def compare_arms(config, monitor, data, results, budget):
    """Frozen-weight V comparison; weights are hashed before and after."""
    from .config import file_hash
    from .evaluate import evaluate_expression
    from .response import load_response
    entries = {arm: entry for arm, entry in results.items() if entry.get("weights")}
    if not entries:
        monitor.say("没有可用权重的臂，跳过 V 比较。")
        return None
    frozen = {arm: entry["weights"] for arm, entry in entries.items()}
    before = {arm: file_hash(path) for arm, path in frozen.items()}
    folder = Path(monitor.experiment) / "v_comparison"
    folder.mkdir(parents=True, exist_ok=True)
    reports = {}
    for arm, entry in entries.items():
        arm_cfg = arm_config(arm, config)
        (folder / arm).mkdir(parents=True, exist_ok=True)  # evaluator writes directly (R5.1)
        model = load_response(entry["weights"], data, config["device"])
        reports[arm] = {"metrics": evaluate_expression([model], data, arm_cfg, folder / arm, role="V"),
                        "selection": entry["selection"],
                        "training_completion": entry.get("completion"),
                        "precision": PrecisionPolicy.from_config(config).identity(),
                        "members": 1}
        del model
    after = {arm: file_hash(path) for arm, path in frozen.items()}
    if before != after:
        raise RuntimeError("FROZEN_COMPARISON_CHANGED_WEIGHTS")
    report = folder / "research-comparison.json"
    atomic_json(report, {"protocol": REPORT_PROTOCOL, "role": "V", "reports": reports,
                         "frozen_hashes": before, "budget": budget.summary(),
                         "budget_protocol": "equal_share_per_arm_wall_clock",
                         "automatic_promotion": False,
                         "scope": "development_expression_comparison_not_apoptosis_efficacy",
                         "synthetic": bool(config.get("synthetic", False))})
    return {"report": str(report), "arms": sorted(reports)}


def run_v_compare(config, monitor, data):
    from .config import file_hash
    from .evaluate import evaluate_expression
    from .response import load_response
    declared = config["research"].get("v_compare", {}).get("weights", [])
    if not declared:
        raise ValueError("v_compare requires research.v_compare.weights: [{name, path}]")
    frozen = {}
    for item in declared:
        if not isinstance(item, dict) or not item.get("name") or not item.get("path"):
            raise ValueError("v_compare weight entries need name and path")
        path = project_path(item["path"])
        if not path.is_file():
            raise StageUnavailable(f"V_COMPARE_WEIGHT_MISSING: {path}")
        frozen[item["name"]] = str(path)
    before = {name: file_hash(path) for name, path in frozen.items()}
    folder = Path(monitor.experiment) / "v_comparison"
    folder.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, path in frozen.items():
        saved = load_checkpoint(path)
        if saved["metadata"].get("data_identity") != data.identity:
            raise ValueError(f"V_COMPARE_DATA_IDENTITY_MISMATCH: {name}")
        branch_cfg = {**config, "response": {**config["response"],
                     **saved["metadata"].get("response_config", {})}}
        (folder / name).mkdir(parents=True, exist_ok=True)  # evaluator writes directly (R5.1)
        model = _load_for_comparison(path, data, config["device"])
        reports[name] = {"metrics": evaluate_expression([model], data, branch_cfg, folder / name, role="V"),
                         "training_completion": _arm_completion(path),
                         "members": 1, "weights": path}
        del model
    if before != {name: file_hash(path) for name, path in frozen.items()}:
        raise RuntimeError("FROZEN_COMPARISON_CHANGED_WEIGHTS")
    report = folder / "research-comparison.json"
    atomic_json(report, {"protocol": REPORT_PROTOCOL, "role": "V", "reports": reports,
                         "frozen_hashes": before, "automatic_promotion": False,
                         "scope": "development_expression_comparison_not_apoptosis_efficacy",
                         "synthetic": bool(config.get("synthetic", False))})
    return {"report": str(report), "arms": sorted(reports)}


def dispatch(config, monitor, data, budget):
    mode = config["research"]["mode"]
    if mode == "route_compare":
        return {"mode": mode, **run_route_compare(config, monitor, data, budget)}
    if mode == "v_compare":
        return {"mode": mode, **run_v_compare(config, monitor, data)}
    from .latent_pipeline import run_latent_stage
    return {"mode": mode, **run_latent_stage(mode, config, monitor, data, budget)}


def main(argv=None):
    arguments = parser().parse_args(argv)
    monitor = None
    try:
        config = resolve(arguments)
        verify_interpreter()
        import torch
        if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
            raise StageUnavailable("CUDA_UNAVAILABLE: no silent CPU fallback")
        policy = PrecisionPolicy.from_config(config)
        seed_all(config["seed"])
        policy.apply()
        monitor = RunMonitor(config, run_id=arguments.resume or arguments.run_id,
                             resume=bool(arguments.resume))
        if policy.mode == "bf16" and not policy.bf16_active(config["device"]):
            monitor.say("BF16 已登记但当前设备不支持 autocast；按 FP32 执行并如实记录（precision_fallback）。")
        from .data import open_data
        data = open_data(config, "response", prepare=False)
        atomic_json(Path(monitor.experiment) / "data_manifest.json",
                    {"identity": data.identity, "manifest": data.manifest})
        budget = ResearchBudget(config["research"].get("budget_seconds"),
                                Path(monitor.experiment) / "research_state.json")
        try:
            result = dispatch(config, monitor, data, budget)
        finally:
            budget.pause()  # settle on every handleable exit path (S07)
        result.update({"protocol": REPORT_PROTOCOL,
                       "precision": policy.identity(),
                       "precision_fallback": policy.mode == "bf16" and not policy.bf16_active(config["device"]),
                       "budget": budget.summary(),
                       "implementation_note": "static delivery; entry not executed in the writing session"})
        atomic_json(Path(monitor.experiment) / "research_report.json", result)
        monitor.say(f"研究入口完成。报告：{monitor.experiment / 'research_report.json'}")
        monitor.close(result.get("run_status", "complete" if not budget.exhausted
                                 else "budget_exhausted"))
        return 0
    except KeyboardInterrupt:
        if monitor is not None:
            monitor.say("已中断。各臂从最近已保存的更新边界恢复；预算已按各臂最后结算点"
                        "持久化（进程被强杀的窗口不在进程内保证范围）。")
            monitor.close("interrupted")
        return 130
    except Exception as error:
        if monitor is not None:
            monitor.say(f"停止：{type(error).__name__}: {error}")
            monitor.close("failed", error_type=type(error).__name__, reason=str(error))
        else:
            print(f"{type(error).__name__}: {error}")
        return 1
