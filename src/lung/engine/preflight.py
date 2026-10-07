"""Read-only startup prerequisites; no model fitting or test-expression reads."""

import json


def stage_readiness(config, data, pipeline):
    stage = config["stage"]
    blockers, pending = [], []
    result = {"readiness_scope": "selected_stage_startup_not_scientific_validation",
              "training_started": False, "expression_rows_read": 0}
    if not config.get("synthetic", False):
        assets = data.assets_dir
        required = ("normalization.npz", "neighbors.npz", "bandwidths.npz")
        missing = [name for name in required if not (assets / name).is_file()]
        prepared = data.manifest["status"] == "complete" and not missing
        result.update({"data_status": data.manifest["status"], "assets": str(assets),
                       "training_conditions": len(data.condition_ids("T")),
                       "validation_conditions": len(data.condition_ids("V")),
                       "first_run_preprocessing": not prepared,
                       "missing_preprocessing_assets": missing})
        if not prepared:
            pending.append("T_ONLY_PREPROCESSING_REQUIRED")
            if not config.get("auto_prepare", True):
                blockers.append("PREPROCESSING_REQUIRED: run scripts/prepare_training.py first")
        if stage in {"state", "all"}:
            fold = config.get("state_fold", "1_to_8")
            path = assets / "tidyi_split_report.json"
            report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            selected = report.get(fold, {})
            result["state_split"] = {"fold": fold, "status": selected.get("status", "missing"),
                                     "missing_classes": selected.get("missing_classes", {})}
            if selected.get("status") != "ready":
                blockers.append("STATE_SPLIT_INSUFFICIENT")
        if pipeline.get("_data_identity") not in (None, data.identity):
            blockers.append("PIPELINE_DATA_IDENTITY_MISMATCH")
    if stage not in {"response", "all"} and not pipeline.get("response"):
        blockers.append("TRAINED_RESPONSE_REQUIRED: select its frozen pipeline.json")
    if stage in {"potential", "daac", "dlpa", "all"} and config["task"] == "RESPONSE-A549-V1":
        blockers.append("TASK_REQUIRED: decisions require an explicit decision task")
    if (config["task"] == "ANNEXIN-A549-V1" and
            stage in {"endpoint", "potential", "daac", "dlpa", "all"} and
            not config.get("endpoint_manifest") and not pipeline.get("endpoint")):
        blockers.append("ENDPOINT_UNAVAILABLE")
    result.update({"pending_preparation": pending, "blockers": blockers,
                   "ready_for_selected_stage": not blockers and not pending,
                   "can_start_with_automatic_preparation": not blockers,
                   "later_stage_guards_still_apply": True})
    return result
