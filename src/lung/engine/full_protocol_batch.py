"""Serial production for the fixed joint-choice view, with data-only preflight."""
import json
from pathlib import Path

from filelock import FileLock

from ..data.joint_selection_evaluation import DEFAULT_VIEW, JointSelectionEvaluation
from . import full_protocol_response as producer
from .checkpoint import atomic_json


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def context(view_directory=DEFAULT_VIEW):
    view = JointSelectionEvaluation.load(view_directory)
    units = view.metadata_units()
    status = producer.preflight()
    requests = {row["candidate_id"]: row for row in status["candidates"]}
    ids = [row["candidate_id"] for row in units]
    require(len(ids) == len(set(ids)) == 20, "EXACT_JOINT_TWENTY_REQUIRED")
    require(all(cid in requests and requests[cid]["input_ready"] is True for cid in ids),
            "JOINT_CANDIDATE_INPUTS_PENDING")
    return ids, {"schema": "joint-full-protocol-serial-production-v1",
        "view_identity": view.identity(), "producer_bindings": status["bindings"],
        "candidate_ids": ids, "batch_source_sha256": producer.inputs.sha(__file__)}


def preflight(view_directory=DEFAULT_VIEW):
    ids, identity = context(view_directory)
    return {**identity, "candidate_count": len(ids), "PT_loads": 0, "model_forwards": 0,
        "P_outputs": 0, "functional_Y_read": False, "scientific_effect_verified": False,
        "pending": ["matching_completed_AE_P"],
        "real_execution_authorized_by_preparation": False}


def produce_all(metadata, output, *, seed, steps, microbatch, device,
                view_directory=DEFAULT_VIEW):
    """Future explicitly authorized execution; one candidate writer per batch.

    Every invocation produces a new batch in an independent output directory.
    """
    output = Path(output).resolve()
    require(output.is_relative_to(producer.ROOT / "experiments"), "BATCH_PROJECT_BOUNDARY")
    require(type(seed) is int and seed >= 0 and type(steps) is int and steps > 0
        and type(microbatch) is int and microbatch > 0,
        "BATCH_EXECUTION_ARGUMENTS")
    ids, identity = context(view_directory)
    metadata = json.loads(producer.canonical(metadata))
    producer.validate_metadata(metadata, identity["producer_bindings"])
    plan = {**identity, "model_metadata": metadata, "seed": seed, "steps": steps,
        "microbatch": microbatch, "device": device}
    require(not output.exists(), "NEW_BATCH_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    with FileLock(str(output / ".production.lock"), timeout=0):
        plan_path = output / "batch_plan.json"
        producer.write_new(plan_path, plan)
        outputs = []
        try:
            for cid in ids:
                directory = output / cid
                require(directory.is_relative_to(output), "CANDIDATE_PATH_BOUNDARY")
                path = directory / "prediction.json"
                producer.produce(cid, metadata, directory, seed=seed, steps=steps,
                    microbatch=microbatch, device=device)
                value = producer.read_prediction(path)
                require(value["candidate_id"] == cid and value["model_metadata"] == metadata
                    and value["bindings"] == plan["producer_bindings"]
                    and value["seed"] == seed and value["steps"] == steps,
                    "BATCH_PREDICTION_IDENTITY_CHANGED")
                outputs.append({"candidate_id": cid, "path": str(path),
                    "sha256": producer.inputs.sha(path), "prediction_identity": value["prediction_identity"]})
            require(context(view_directory)[1] == identity, "BATCH_SOURCE_CHANGED")
            result = {"schema": plan["schema"], "status": "all_twenty_predictions_authenticated",
                "plan_sha256": producer.inputs.sha(plan_path), "outputs": outputs,
                "new_P_outputs_in_invocation": len(outputs),
                "functional_Y_read": False, "G_outputs": 0, "Q_outputs": 0,
                "candidate_protocol_scientifically_validated": False,
                "cross_source_transfer_validated": False, "whole_project_qualified": False}
            atomic_json(output / "batch_result.json", result)
            return result
        except Exception as error:
            with (output / "failures.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"error_type": type(error).__name__, "reason": str(error),
                    "completed_candidates": [v["candidate_id"] for v in outputs]}, ensure_ascii=False) + "\n")
            raise
