"""Retrospective frozen-caller consistency evaluation, not producer certification.

Metrics are definitions proposed for this project, not an attributed paper algorithm.
Final labels cannot qualify rewards or select/tune rules. Pixel bounds are not CIs.
"""
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

from lung.data.martin_state_function import MartinStateFunctionInputs, SCALE, sha

SCHEMA = "martin-frozen-caller-consistency-v1"


def _strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("NONFINITE_JSON_CONSTANT")

    return json.loads(text, object_pairs_hook=unique, parse_constant=constant)


def _read(path):
    return _strict_json(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def number(value):
    # JSON numeric values only: no coercion of strings, bool, complex or arrays.
    if type(value) not in (int, float):
        raise ValueError("FINITE_NONNEGATIVE_REAL_JSON_NUMBER_REQUIRED")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("FINITE_NONNEGATIVE_REAL_JSON_NUMBER_REQUIRED") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError("FINITE_NONNEGATIVE_REAL_JSON_NUMBER_REQUIRED")
    return result


def preflight(source):
    """Metadata only: existing reader verifies label bytes but never parses Y."""
    return {"schema": SCHEMA, "source_manifest_sha256": sha(source.directory / "manifest.json"),
            "counts": source.manifest["counts"], "execution_certified": False,
            "producer_binding": "pending_native_P_cache_capability_check",
            "evaluation_scope": "source_compatible_consistency_only",
            "retrospective": True, "blind_chronology_certified": False,
            "reward_qualified": False, "Q_qualified": False,
            "actual_model_inference": False, "labels_parsed": False}


def _validate(source, packet):
    if packet.get("schema") != SCHEMA or packet.get("value_scale") != SCALE:
        raise ValueError("PREDICTION_SCHEMA_OR_SCALE_MISMATCH")
    if packet.get("source_manifest_sha256") != sha(source.directory / "manifest.json"):
        raise ValueError("PREDICTION_SOURCE_IDENTITY_MISMATCH")
    if packet.get("rules_tuned_with_final_Y") is not False:
        raise ValueError("FINAL_Y_RULE_TUNING_FORBIDDEN")
    rows = packet["predictions"]
    expected = {(fid, cid) for fid, f in source.folds.items() for cid in f["final_test_candidate_ids"]}
    identities = []
    for row in rows:
        fid, cid = row["fold_id"], row["candidate_id"]
        if (fid, cid) not in expected:
            raise ValueError("NOT_FINAL_LOBO_CANDIDATE")
        candidate = source.candidates[cid]
        for field in ("decision_block_id", "measurement_domain_id", "state_reference_id", "cell_background"):
            if row.get(field) != candidate[field]:
                raise ValueError("PREDICTION_CONDITION_IDENTITY_MISMATCH")
        number(row["prediction"])
        identities.append((fid, cid))
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("COMPLETE_FINAL_LOBO_BLOCKS_REQUIRED")
    for block in source.blocks.values():
        source.decision_inputs(block["candidate_ids"])
    return rows


def freeze(source, prediction_file, model_artifact, output):
    """Snapshot caller predictions after binding source and explicit artifact bytes.

    Arbitrary artifact bytes are never proof of training/native-P execution.
    This API does not call a scorer, predictor or label observation API.
    """
    prediction_file, model_artifact, output = map(lambda p: Path(p).resolve(),
                                                (prediction_file, model_artifact, output))
    # Re-open existing immutable source rather than trusting mutable caller metadata.
    source = MartinStateFunctionInputs(source.directory, project_root=source.root)
    prediction_bytes = prediction_file.read_bytes()
    packet = _strict_json(prediction_bytes.decode("utf-8-sig"))
    _validate(source, packet)
    model_hash = sha(model_artifact)
    if packet.get("model_artifact_sha256") != model_hash:
        raise ValueError("PREDICTION_MODEL_IDENTITY_MISMATCH")
    receipt = {**preflight(source), "labels_parsed": False,
               "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
               "caller_prediction_path": str(prediction_file), "model_artifact_path": str(model_artifact),
               "model_artifact_sha256": model_hash, "source_directory": str(source.directory),
               "project_root": str(source.root), "evaluation_code_sha256": sha(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.json").open("xb") as stream:
        stream.write(prediction_bytes)
    _write(output / "receipt.json", receipt)
    _write(output / "seal.json", {"receipt_sha256": sha(output / "receipt.json")})
    return receipt


def block_metrics(predictions, labels):
    """Mean-level MAE and regret; tie selections use their uniform mean Y.

    regret = max_i Y_i - mean_{i in argmax prediction} Y_i, within one block.
    Rank concordance is reported only for disjoint pixel reading intervals.
    """
    for field in ("fold_id", "measurement_domain_id", "decision_block_id", "state_reference_id", "cell_background"):
        if len({r[field] for r in predictions if field in r}) > 1:
            raise ValueError("MIXED_DECISION_BLOCK_METRIC_FORBIDDEN")
    p = {r["candidate_id"]: number(r["prediction"]) for r in predictions}
    y = {r["candidate_id"]: r for r in labels}
    if len(p) != len(predictions) or len(y) != len(labels) or set(p) != set(y) or not p:
        raise ValueError("METRIC_IDENTITIES_MISMATCH")
    for r in y.values():
        if (type(r["weight"]) is not int or r["weight"] != 1 or r["value_scale"] != SCALE
                or not number(r["reading_lower"]) <= number(r["value"]) <= number(r["reading_upper"])):
            raise ValueError("PUBLISHED_MEAN_UNIT_REQUIRED")
    selected = [cid for cid in p if p[cid] == max(p.values())]
    pairs = []
    for a, b in combinations(sorted(p), 2):
        direction = (1 if y[a]["reading_lower"] > y[b]["reading_upper"] else
                     -1 if y[b]["reading_lower"] > y[a]["reading_upper"] else 0)
        predicted = (p[a] > p[b]) - (p[a] < p[b])
        status = "uncertain" if direction == 0 else ("tie" if predicted == 0 else
                    "concordant" if predicted == direction else "discordant")
        pairs.append({"candidate_ids": [a, b], "status": status})
    resolved = [r for r in pairs if r["status"] != "uncertain"]
    mae = math.fsum(abs(p[c] - y[c]["value"]) / len(p) for c in p)
    regret = max(r["value"] for r in y.values()) - math.fsum(y[c]["value"] / len(selected) for c in selected)
    if not math.isfinite(mae) or not math.isfinite(regret):
        raise ValueError("NONFINITE_COMPUTED_METRICS")
    return {"observation_count": len(y), "MAE": mae,
            "mean_candidate_regret": regret,
            "selected_candidate_ids": selected, "rank_pairs": pairs,
            "resolved_rank_pair_concordance": (sum(r["status"] == "concordant" for r in resolved) /
                                               len(resolved) if resolved else None),
            "uncertainty_type": "pixel_bounds_not_CI", "biological_p_value": None}


def evaluate(frozen_directory):
    directory = Path(frozen_directory).resolve()
    seal, receipt = _read(directory / "seal.json"), _read(directory / "receipt.json")
    if sha(directory / "receipt.json") != seal["receipt_sha256"]:
        raise ValueError("RECEIPT_IDENTITY_CHANGED")
    if sha(__file__) != receipt["evaluation_code_sha256"]:
        raise ValueError("EVALUATION_RULES_CHANGED_AFTER_FREEZE")
    for path in (directory / "predictions.json", Path(receipt["caller_prediction_path"])):
        if sha(path) != receipt["prediction_sha256"]:
            raise ValueError("PREDICTION_IDENTITY_CHANGED")
    if sha(receipt["model_artifact_path"]) != receipt["model_artifact_sha256"]:
        raise ValueError("MODEL_IDENTITY_CHANGED")
    source = MartinStateFunctionInputs(receipt["source_directory"], project_root=receipt["project_root"])
    if sha(source.directory / "manifest.json") != receipt["source_manifest_sha256"]:
        raise ValueError("SOURCE_IDENTITY_CHANGED")
    packet = _read(directory / "predictions.json")
    rows = _validate(source, packet)
    if packet["model_artifact_sha256"] != receipt["model_artifact_sha256"]:
        raise ValueError("RECEIPT_MODEL_PREDICTION_MISMATCH")
    results = []
    # All source/model/receipt/prediction checks precede the first final-Y call.
    for fid, fold in source.folds.items():
        for domain in source.domains:
            labels = source.observation_units(fid, "final_test", domain)
            selected = [r for r in rows if r["fold_id"] == fid and r["measurement_domain_id"] == domain]
            block_ids = {r["decision_block_id"] for r in selected}
            if len(block_ids) != 1:
                raise ValueError("ONE_FINAL_DECISION_BLOCK_REQUIRED")
            results.append({"fold_id": fid, "measurement_domain_id": domain,
                            "decision_block_id": next(iter(block_ids)), **block_metrics(selected, labels)})
    return {**preflight(source), "labels_parsed": True, "blocks": results,
            "aggregation": "separate_domain_and_fold_blocks_no_global_ranking",
            "independent_biological_function_replicates_available": 0,
            "RNA_triplicates_expanded_as_Y": False, "significance_claim": False}
