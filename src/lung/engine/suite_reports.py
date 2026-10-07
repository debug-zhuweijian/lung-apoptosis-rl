"""Read-only frozen comparisons and explicitly synthetic numerical smoke reports."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .checkpoint import load_checkpoint
from .config import file_hash, read_yaml
from .suite_state import atomic_json
from .inference_noise import response_noise


def synthetic_ridge(context):
    from .synthetic import SyntheticData
    from .pilot_ridge import fit_ridge, LAMBDAS, save_npz
    data = SyntheticData(context["config"])
    rng = np.random.default_rng(context["config"]["seed"])
    batches = {role: [data.sample_response(cid, 6, rng, role) for cid in data.condition_ids(role)]
               for role in ("T", "V")}
    def features(rows):
        return np.stack([np.concatenate((b["fingerprint"], [b["dose"], b["duration"]])) for b in rows])
    train, valid = batches["T"], batches["V"]
    targets = np.stack([b["target"].mean(0) - b["control"].mean(0) for b in train])
    truth = np.stack([b["target"].mean(0) - b["control"].mean(0) for b in valid])
    candidates = []
    for value in LAMBDAS:
        model = fit_ridge(features(train), targets, value, device=context["config"]["device"])
        prediction = model.predict(features(valid))
        candidates.append((float(np.mean((prediction - truth)**2)), value, model, prediction))
    loss, value, model, prediction = min(candidates, key=lambda x: x[0])
    folder = Path(context["job_dir"])
    weights, shifts, report = folder / "ridge.npz", folder / "v_delta.npz", folder / "ridge-report.json"
    save_npz(weights, **model.arrays(), regularization=np.asarray(value))
    save_npz(shifts, condition_ids=np.asarray(data.condition_ids("V")), delta=prediction)
    atomic_json(report, {"synthetic": True, "proxy": True, "fit_role": "T", "selection_role": "V",
                         "selected_lambda": value, "validation_mse": loss, "not_research_evidence": True})
    return {"artifacts": [str(p) for p in (weights, shifts, report)], "shifts": str(shifts),
            "weights": str(weights), "report": str(report)}


def response_compare(context):
    folder, root = Path(context["job_dir"]), Path(context["root"])
    config, deps = context["config"], context["dependencies"]
    if not context["synthetic"]:
        from scripts.report_response_pilot import main
        selected = read_yaml(root / "configs/pilot_v.yaml")
        selected.update({key: config[key] for key in ("device", "data")})
        cfg = folder / "report-config.yaml"
        cfg.write_text(yaml.safe_dump(selected, allow_unicode=True), encoding="utf-8")
        output = folder / "analysis"
        args = ["--config", str(cfg), "--output", str(output), "--shifts", deps["ridge"]["shifts"]]
        for label, name in (("--p3-checkpoints", "response_cfm_mmd"), ("--p4-checkpoints", "response_cfm_only")):
            args.extend([label, *deps[name]["pipeline_artifacts"]["response"]])
        if not config.get("progress", True):
            args.append("--no-progress")
        if main(args) != 0:
            raise RuntimeError("RESPONSE_COMPARISON_FAILED")
        return {"artifacts": [str(cfg), *[str(p) for p in output.rglob("*") if p.is_file()]], "report": str(output)}
    from .synthetic import SyntheticData
    from .pilot import distribution_metrics
    from .response import load_response, tensor, condition_kwargs
    data = SyntheticData(config)
    with np.load(deps["ridge"]["shifts"], allow_pickle=False) as saved:
        shifts = dict(zip(saved["condition_ids"].tolist(), saved["delta"]))
    rows = []
    with torch.no_grad():
        for cid in data.condition_ids("V"):
            batch = data.sample_response(cid, 4, np.random.default_rng(config["seed"] + 17), "V")
            control, target = batch["control"], batch["target"]
            for method, prediction in (("P0", None), ("P2", control + shifts[cid])):
                metrics = distribution_metrics(prediction, control, target, data.mu, data.scale, data.bandwidth)
                rows.append({"condition": cid, "method": method, **metrics})
            mean_prediction = control.mean(0) + shifts[cid]
            rows.append({"condition": cid, "method": "P1", "mean_rmse": float(np.mean((mean_prediction-target.mean(0))**2)**.5)})
            for method, name in (("P3", "response_cfm_mmd"), ("P4", "response_cfm_only")):
                for member, path in enumerate(deps[name]["pipeline_artifacts"]["response"]):
                    model = load_response(path, data, config["device"])
                    condition = model.encode_condition(**condition_kwargs(batch, len(control), config["device"]))
                    generator = torch.Generator(device=config["device"]).manual_seed(config["seed"] + member)
                    noise = response_noise(model, tensor(control, config["device"]), generator)
                    prediction = model.sample(tensor(control, config["device"]), condition, noise=noise,
                                              steps=config["response"]["inference_ode_steps"], checkpoint=False)
                    metrics = distribution_metrics(prediction, control, target, data.mu, data.scale, data.bandwidth,
                                                   device=config["device"])
                    rows.append({"condition": cid, "method": method, "member": member, **metrics})
    report = folder / "response-comparison.json"
    atomic_json(report, {"synthetic": True, "proxy": True, "role": "V", "rows": rows,
                         "scope": "member_metrics_engineering_smoke_not_five_member_research_protocol",
                         "model_updates": 0})
    return {"artifacts": [str(report)], "report": str(report)}


def state_compare(context):
    from .suite_evaluation import state_final_test
    rows, files = [], []
    for name, result in context["dependencies"].items():
        checkpoint = result["pipeline_artifacts"]["state"]
        saved = load_checkpoint(checkpoint)
        metrics = []
        if result.get("metrics"):
            with Path(result["metrics"]).open(encoding="utf-8-sig", newline="") as stream:
                metrics = list(csv.DictReader(stream))
        rows.append({"name": name, "checkpoint": checkpoint, "sha256": file_hash(checkpoint),
                     "metadata": saved["metadata"], "selected_epoch": saved["epoch"],
                      "best_selection": saved.get("extra", {}).get("best"), "metrics": metrics,
                      "independent_plate_test": state_final_test(result)})
    report = Path(context["job_dir"]) / "state-comparison.json"
    atomic_json(report, {"synthetic": bool(context["synthetic"]), "proxy": True,
                         "primary_state": context["config"].get("suite", {}).get("primary_state", "weighted_ce"),
                         "selection": "predeclared_not_chosen_from_this_comparison", "E_used_for_selection": False,
                         "label_semantics": "TIDYI_sorting_labels_not_apoptosis_outcomes", "rows": rows})
    return {"artifacts": [str(report)], "report": str(report)}


def decision_compare(context, result):
    from .data import open_data
    from .evaluate import evaluate_decisions
    config = result["original_config"]
    data = open_data(config, "daac", prepare=False, test=True)
    output = Path(context["job_dir"]) / "comparison"
    output.mkdir(exist_ok=True)
    before = {p: file_hash(p) for p in result["artifacts"]}
    report = evaluate_decisions(data, config, result["pipeline_artifacts"], output)
    rows = json.loads((output / "decisions.json").read_text(encoding="utf-8"))
    if not rows or any("enumeration" not in row["proposed_by"] for row in rows):
        raise ValueError("COMMON_ENUMERATION_COMPARISON_MISSING")
    if before != {p: file_hash(p) for p in before}:
        raise ValueError("FROZEN_EVALUATION_CHANGED_INPUT_ARTIFACTS")
    target = output / "report.json"
    atomic_json(target, {**report, "synthetic": bool(context["synthetic"]), "proxy": True,
                         "frozen_hashes": before, "test_used_for_selection": False})
    return {"artifacts": [str(target), str(output / "decisions.json")], "report": str(target)}


def summary(context):
    states = context["task_states"]
    folder = Path(context["job_dir"])
    json_path, markdown = folder / "summary.json", folder / "summary.md"
    failures = [name for name, item in states.items() if name != "summary" and item["status"] != "complete"]
    records = {name: {"status": item["status"], "reason": item.get("reason"),
                      "artifacts": item.get("result", {}).get("artifacts", [])} for name, item in states.items()}
    records["summary"] = {"status": "complete", "reason": None, "artifacts": [str(json_path), str(markdown)]}
    atomic_json(json_path, {"synthetic": bool(context["synthetic"]), "proxy": True,
                            "status": "failed" if failures else "complete", "tasks": records,
                            "primary_state": context["config"].get("suite", {}).get("primary_state", "weighted_ce"),
                            "scope": "状态代理和模型内决策比较，不是实测凋亡或临床获益"})
    text = ["# 一键研究套件汇总", "", "合成工程验证" if context["synthetic"] else "冻结研究计算结果",
            "", "状态代理和模型内决策比较，不是实测凋亡或临床获益。", "", "| 任务 | 状态 | 原因 |", "| --- | --- | --- |"]
    for name, item in records.items():
        text.append(f"| {name} | {item['status']} | {str(item.get('reason') or '').replace('|', '/')} |")
    for name, item in records.items():
        text.extend(["", f"## {name} 产物", *[f"- {path}" for path in item["artifacts"]]])
    markdown.write_text("\n".join(text) + "\n", encoding="utf-8")
    return {"artifacts": [str(json_path), str(markdown)], "report": str(json_path)}
