"""Audited standard continuous-24h request producer, independent of endpoint Y."""
import json
import hashlib
from pathlib import Path

import numpy as np

from ..data.endpoint_protocol_inputs import request_for
from .checkpoint import atomic_json, load_checkpoint
from .config import StageUnavailable, config_hash
from .endpoint_contract import FEATURE_KEYS, feature_dependencies_v2, frozen_feature_identity_v2
from .endpoint_features import (CONTEXT_KEYS, ExternalConditionWorld, checked, record,
                               generate_feature_rows, validate_reference)
from .representation import require_finite_state, require_trained

PRODUCER_VERSION = "standardP-continuous24h-v1"
DESCRIPTOR_DEFINITION = {"version": PRODUCER_VERSION, "population": "pool_summary[d_model]",
    "bio": "current_world_missing_masked_gene_sets[27]", "context": "background_source_protocol[3*context_dim]",
    "history": "single_standard_continuous24h_action[1,d_model+3*context_dim+1]", "dose_unit": "uM",
    "duration_h": 24, "elapsed_h": 24, "horizon_h": 24,
    "state_probabilities": "diagnostic_only_excluded", "endpoint_values": "excluded"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(ok, reason):
    if not ok:
        raise StageUnavailable("STANDARD_P_" + reason)


def standard_requests(input_bridge):
    """Deduplicate P only; preserve complete candidate protocols in the bridge."""
    requests = {}
    for cid in input_bridge.candidate_ids:
        value = input_bridge.model_input(cid)
        row = value["candidate_record"]
        req = request_for(row)
        fp = np.asarray(value["chemical_fingerprint"], dtype=np.float32)
        require(fp.shape == (2048,) and np.isfinite(fp).all() and np.isin(fp, [0, 1]).all()
                and fp.any(), "FINGERPRINT_INVALID")
        require(row["p_request_id"] == req["request_id"], "REQUEST_ID_CHANGED")
        item = {"request": req, "fingerprint": fp, "source_id": row["source_id"],
                "fingerprint_record": record(row["chemical"]["fingerprint_path"])}
        previous = requests.get(req["request_id"])
        if previous:
            require(previous["request"] == req and np.array_equal(previous["fingerprint"], fp)
                    and previous["fingerprint_record"] == item["fingerprint_record"], "DUPLICATE_REQUEST_CONFLICT")
        requests[req["request_id"]] = item
    require(len(input_bridge.candidate_ids) == 28 and len(requests) == 14, "COMPLETE_28_CANDIDATES_14_REQUESTS_REQUIRED")
    return requests


def validate_weights(pipeline, data_identity, *, latent=True):
    require(all(pipeline.get(k) for k in ("response", "pool", "state")), "TRAINED_RESPONSE_POOL_STATE_REQUIRED")
    require(not latent or pipeline.get("representation"), "LATENT_REPRESENTATION_REQUIRED")
    qualifications = {}
    for stage in FEATURE_KEYS:
        if not pipeline.get(stage):
            continue
        paths = pipeline[stage] if isinstance(pipeline[stage], (list, tuple)) else [pipeline[stage]]
        qualifications[stage] = []
        for path in paths:
            if stage == "correction_gate":
                require(read(path).get("passed") is True, "CORRECTION_GATE_REQUIRED")
                qualifications[stage].append(record(path))
                continue
            saved = load_checkpoint(path)
            metadata = saved.get("metadata", {})
            expected_stages = {"representation": {"latent_ae"}, "response": {"response", "latent_cfm"}}.get(stage, {stage})
            require(type(saved.get("epoch")) is int and saved["epoch"] > 0 and bool(saved.get("model")), "TRAINED_CHECKPOINT_REQUIRED")
            require_trained(saved, "STANDARD_P_OPTIMIZER_STEP_REQUIRED")
            require(metadata.get("stage") in expected_stages and metadata.get("data_identity") == data_identity
                    and (metadata.get("synthetic") is False or (metadata.get("stage") in {"latent_ae", "latent_cfm"}
                         and "synthetic" not in metadata)), "CHECKPOINT_STAGE_DATA_OR_SYNTHETIC_MISMATCH")
            require_finite_state(saved["model"])
            qualifications[stage].append({**record(path), "epoch": saved["epoch"],
                "optimizer_step": saved["counters"]["optimizer_step"], "stage": metadata["stage"],
                "data_identity": data_identity, "synthetic": False})
    return qualifications


def validate_mapping(mappings, requests, reference):
    require(isinstance(mappings, list) and len(mappings) == 2, "TWO_SOURCE_MAPPING_REQUIRED")
    sources = {v["source_id"] for v in requests.values()}
    require({m["source_id"] for m in mappings} == sources, "MAPPING_SOURCE_COVERAGE")
    for m in mappings:
        require(set(m) == {"source_id", "cell_line", "status", "source_state_preserved", "reference_id", "target_reference_plate",
                          "target_source_id", "target_background_id", "target_protocol_id", "evidence"}
                and isinstance(m.get("evidence"), str) and bool(m["evidence"].strip())
                and m.get("status") == "explicit_assumption_unvalidated" and m.get("reference_id") == reference["reference_id"]
                and m.get("source_state_preserved") is True and m.get("cell_line") == "A549"
                and m.get("target_reference_plate") == reference["reference_plate"]
                and all(m.get("target_" + k) == reference[k] for k in CONTEXT_KEYS), "REFERENCE_MAPPING_MISMATCH")


def preflight_standard_requests(input_bridge, *, reference_path, mapping_path, data_config_path, pipeline, config):
    requests = standard_requests(input_bridge)
    reference, mappings = read(reference_path), read(mapping_path)
    cfg = read(data_config_path)
    base = Path(data_config_path).resolve().parents[1]
    manifest_path = base / cfg["split_manifest"]
    manifest = read(manifest_path)
    identity = config_hash({"signature": manifest["signature"], "files": manifest["files"]})
    require(manifest.get("status") == "complete" and manifest.get("preprocessing", {}).get("fitted_role") == "fit"
            and manifest["preprocessing"].get("gene_count") == cfg["gene_count"], "FIT_PREPROCESSING_REQUIRED")
    for name in ("genes.parquet", "normalization.npz", "neighbors.npz", "bandwidths.npz", "conditions.parquet", "control_links.parquet"):
        checked({"path": str(base / cfg["assets"] / name), "sha256": manifest["files"][name]["sha256"]}, base)
    require(config.get("synthetic") is False and config.get("history_support") == "single_exposure_only", "REAL_SINGLE_EXPOSURE_CONFIG_REQUIRED")
    require(reference.get("unique_fit_cell_indices") == 2431 and reference.get("condition_id") == "tahoe:smp_1495"
            and reference.get("reference_plate") == "plate1" and reference.get("matched_initial_state_of_endpoint_experiment") is False,
            "FROZEN_REFERENCE_SELECTION_CHANGED")
    validate_reference(reference, manifest_path, base / cfg["assets"], identity, base)
    checked(reference["cell_index"], base)
    import pandas as pd
    cells = pd.read_parquet(reference["cell_index"]["path"])
    require(len(cells) == 2431 and not cells.cell_id.duplicated().any()
            and not cells[["matrix_path", "matrix_row"]].duplicated().any()
            and cells.role.eq("fit").all() and cells.plate.eq(reference["reference_plate"]).all()
            and set(cells.condition_id) == set(reference["control_condition_ids"])
            and all(cells[k].eq(reference[k]).all() for k in CONTEXT_KEYS), "REFERENCE_CELL_METADATA_MISMATCH")
    conditions = pd.read_parquet(base / cfg["assets"] / "conditions.parquet")
    rows = conditions.loc[conditions.condition_id.eq(reference["condition_id"])]
    require(len(rows) == 1 and rows.iloc[0]["role"] == "fit" and rows.iloc[0]["plate"] == reference["reference_plate"]
            and all(rows.iloc[0][k] == reference[k] for k in CONTEXT_KEYS), "REFERENCE_CONDITION_METADATA_MISMATCH")
    validate_mapping(mappings, requests, reference)
    missing = [k for k in ("response", "pool", "state", "representation") if not pipeline.get(k)]
    qualification = {} if missing else validate_weights(pipeline, identity)
    return {"status": "pending" if missing else "ready_for_generation", "missing_stages": missing,
        "request_count": len(requests), "candidate_count": len(input_bridge.candidate_ids),
        "data_identity": identity, "reference_identity": reference["reference_id"],
        "checkpoints": qualification, "reference_cells_checked": len(cells),
        "source_records": [record(reference_path), record(mapping_path), record(data_config_path), record(manifest_path)],
        "producer_version": PRODUCER_VERSION, "descriptor_definition": DESCRIPTOR_DEFINITION,
        "model_execution_performed": False, "formal_training_ready": False}


def make_actions(requests, data, reference):
    require(not data.synthetic and reference["condition_id"] in data.condition_ids("T"), "FIT_REAL_REFERENCE_REQUIRED")
    encoded = data.action_features(reference["condition_id"])
    ref = data.reader.conditions.loc[data.reader.conditions.condition_id.eq(reference["condition_id"])]
    require(len(ref) == 1 and all(ref.iloc[0][k] == reference[k] for k in CONTEXT_KEYS)
            and ref.iloc[0]["plate"] == reference["reference_plate"], "REFERENCE_CONTEXT_MISMATCH")
    actions, provenance = {}, {}
    for rid, item in requests.items():
        req = item["request"]
        actions[rid] = {"fingerprint": item["fingerprint"].copy(), "dose": req["dose_uM"],
            "duration": 24., "group_id": req["compound_id"], "is_vehicle": False,
            **{k: encoded[k] for k in CONTEXT_KEYS}}
        provenance[rid] = {"source_id": item["source_id"], "reference_id": reference["reference_id"],
            "background_transfer_validated": False, "model_context": {k: reference[k] for k in CONTEXT_KEYS}}
    return actions, provenance


def validate_rows(rows, requests, shape, reference_identity, feature_identity):
    required = {"request_id", "compound_id", "dose_uM", "reference_protocol_arm", "reference_terminal_time_h",
                "population", "bio", "context", "history", "elapsed", "horizon", "descriptor_definition",
                "reference_identity", "feature_identity"}
    require(isinstance(rows, list) and len(rows) == 14 and {r.get("request_id") for r in rows} == set(requests), "TABLE_REQUEST_COVERAGE")
    for row in rows:
        require(set(row) == required, "TABLE_FIXED_SCHEMA_REQUIRED")
        req = requests[row["request_id"]]["request"]
        require(all(row[k] == req[k] for k in ("compound_id", "dose_uM", "reference_protocol_arm", "reference_terminal_time_h"))
                and row["elapsed"] == row["horizon"] == 24 and row["descriptor_definition"] == DESCRIPTOR_DEFINITION
                and row["reference_identity"] == reference_identity and row["feature_identity"] == feature_identity, "TABLE_IDENTITY_MISMATCH")
        for k, dimensions in {"population": (shape["d_model"],), "bio": (27,), "context": (3*shape["context_dim"],),
                "history": (1, shape["history_input_dim"])}.items():
            a = np.asarray(row[k], dtype=float)
            require(a.shape == dimensions and np.isfinite(a).all(), "TABLE_SHAPE_OR_FINITE_MISMATCH")


def producer_sources():
    root = Path(__file__).resolve().parents[3]
    return [record(p) for p in sorted((root / "src/lung").rglob("*.py"))] + [record(root / "scripts/generate_endpoint_protocol_features.py")]


def runtime_identity():
    import torch
    return {"torch_version": torch.__version__, "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def generate_standard_cache(input_bridge, *, reference_path, mapping_path, data_config_path, pipeline, config, output):
    report = preflight_standard_requests(input_bridge, reference_path=reference_path, mapping_path=mapping_path,
        data_config_path=data_config_path, pipeline=pipeline, config=config)
    require(not report["missing_stages"], "MATCHING_TRAINED_WEIGHTS_PENDING")
    output = Path(output).resolve()
    require(not output.exists(), "IMMUTABLE_OUTPUT_EXISTS")
    from .data import EngineData
    reference, requests = read(reference_path), standard_requests(input_bridge)
    cfg = read(data_config_path)
    base = Path(data_config_path).resolve().parents[1]
    assets_dir = base / cfg["assets"]
    input_dir = Path(input_bridge.directory)
    input_files = [record(input_dir / n) for n in ("manifest.json", "candidate_records.json", "p_descriptor_requests.json")]
    sources = producer_sources()
    assets = {"gene_axis": [record(assets_dir / "genes.parquet")],
        "preprocessing": [record(assets_dir / n) for n in ("manifest.json", "normalization.npz", "neighbors.npz", "bandwidths.npz")],
        "chemical_identity": list({v["fingerprint_record"]["path"]: v["fingerprint_record"] for v in requests.values()}.values()),
        "feature_generator": sources, "condition_inputs": input_files + [record(mapping_path), record(data_config_path)],
        "initial_state": [record(reference_path), reference["cell_index"], *reference["source_files"]]}
    dependencies = feature_dependencies_v2(pipeline, assets, base=base)
    generation = {"producer": PRODUCER_VERSION, "definition": DESCRIPTOR_DEFINITION, "config": config,
                  "pipeline": pipeline, "reference_identity": reference["reference_id"], "runtime": runtime_identity()}
    data = EngineData({**config, "data": str(data_config_path)}, roles=("T",), require_preprocessing=True)
    require(data.identity == report["data_identity"], "DATA_CHANGED")
    actions, provenance = make_actions(requests, data, reference)
    world = ExternalConditionWorld(data, config, pipeline, actions)
    root = world.root(reference["condition_id"], "T")
    initial, _ = world._initial(root, world.n_pop)
    population = initial.cpu().numpy()
    cells = world._initial_cache[2].get("control_cell_ids")
    require(cells is not None and len(cells) >= world.n_pop and np.isfinite(population).all(), "ACTUAL_REFERENCE_DRAW_REQUIRED")
    draw = {"root_identity": root.payload.identity, "data_identity": data.identity,
        "reference_identity": reference["reference_id"], "sampling_seed": root.payload.initial_seed,
        "sampling_draw_size": root.payload.population_size, "used_population": world.n_pop,
        "control_cell_ids": list(cells[:world.n_pop]), "population_shape": list(population.shape),
        "population_content_sha256": hashlib.sha256(population.tobytes()).hexdigest()}
    generation = {**generation, "actual_reference_draw": draw}
    raw = generate_feature_rows(world, root, provenance, seed=config["seed"], reference_id=reference["reference_id"])
    shape = {"d_model": world.shape.d_model, "context_dim": world.shape.context_dim,
             "history_input_dim": world.shape.history_input_dim}
    content = [{"request_id": r["experimental_condition_id"], **{k: r[k] for k in
               ("population", "bio", "context", "history", "elapsed", "horizon")}} for r in raw]
    generation = {**generation, "feature_content_identity": config_hash(content)}
    identity = frozen_feature_identity_v2(report["data_identity"], dependencies, generation)
    rows = [{"request_id": r["experimental_condition_id"], **{k: requests[r["experimental_condition_id"]]["request"][k]
        for k in ("compound_id", "dose_uM", "reference_protocol_arm", "reference_terminal_time_h")},
        **{k: r[k] for k in ("population", "bio", "context", "history", "elapsed", "horizon")},
        "descriptor_definition": DESCRIPTOR_DEFINITION, "reference_identity": reference["reference_id"],
        "feature_identity": identity} for r in raw]
    validate_rows(rows, requests, shape, reference["reference_id"], identity)
    require(feature_dependencies_v2(pipeline, assets, base=base) == dependencies, "ARTIFACT_CHANGED_DURING_GENERATION")
    require(generation["runtime"] == runtime_identity(), "RUNTIME_CHANGED_DURING_GENERATION")
    validate_weights(pipeline, report["data_identity"])
    output.mkdir(parents=True)
    np.save(output / "reference_population.npy", population, allow_pickle=False)
    atomic_json(output / "reference_draw.json", draw)
    atomic_json(output / "features.json", rows)
    receipt = {"producer_version": PRODUCER_VERSION, "descriptor_definition": DESCRIPTOR_DEFINITION,
        "shape": shape, "data_identity": data.identity, "reference_identity": reference["reference_id"],
        "feature_identity": identity, "feature_dependencies": dependencies, "generation": generation,
        "pipeline": pipeline, "qualifications": report["checkpoints"], "producer_sources": sources,
        "input_files": input_files, "features": record(output / "features.json"),
        "reference_population": record(output / "reference_population.npy"), "reference_draw": record(output / "reference_draw.json"),
        "measured_endpoint_values_read": False, "formal_training_ready": False}
    atomic_json(output / "manifest.json", receipt)
    return receipt


def load_standard_cache(cache_manifest_path, input_bridge):
    path = Path(cache_manifest_path).resolve()
    m = read(path)
    require(m.get("producer_version") == PRODUCER_VERSION and m.get("descriptor_definition") == DESCRIPTOR_DEFINITION
            and m.get("producer_sources") == producer_sources() and m.get("measured_endpoint_values_read") is False
            and m.get("formal_training_ready") is False, "PRODUCER_UNVERIFIED")
    for item in m["input_files"]:
        checked(item, path.parent)
    require(m["input_files"] == [record(Path(input_bridge.directory) / n) for n in
            ("manifest.json", "candidate_records.json", "p_descriptor_requests.json")], "INPUT_CHANGED")
    dependencies = feature_dependencies_v2(m["pipeline"], m["feature_dependencies"]["assets"], base=path.parent)
    require(dependencies == m["feature_dependencies"] and frozen_feature_identity_v2(m["data_identity"], dependencies,
            m["generation"]) == m["feature_identity"], "FEATURE_IDENTITY_MISMATCH")
    require(validate_weights(m["pipeline"], m["data_identity"]) == m["qualifications"], "WEIGHT_QUALIFICATION_CHANGED")
    generation = m["generation"]
    require(generation.get("producer") == PRODUCER_VERSION and generation.get("definition") == DESCRIPTOR_DEFINITION
            and generation.get("pipeline") == m["pipeline"] and generation.get("reference_identity") == m["reference_identity"],
            "GENERATION_METADATA_MISMATCH")
    assets = dependencies["assets"]
    reference_path = assets["initial_state"][0]["path"]
    mapping_path, data_config_path = [a["path"] for a in assets["condition_inputs"][-2:]]
    report = preflight_standard_requests(input_bridge, reference_path=reference_path, mapping_path=mapping_path,
        data_config_path=data_config_path, pipeline=m["pipeline"], config=generation["config"])
    require(report["data_identity"] == m["data_identity"] and report["reference_identity"] == m["reference_identity"], "CACHE_REFERENCE_DATA_MISMATCH")
    response_path = m["pipeline"]["response"]
    saved = load_checkpoint(response_path[0] if isinstance(response_path, list) else response_path)
    model_shape = saved["metadata"]["model_config"]
    from lung.models import ModelConfig
    model_config = ModelConfig(**model_shape)
    require(m["shape"] == {"d_model": model_config.d_model, "context_dim": model_config.context_dim,
                          "history_input_dim": model_config.history_input_dim}, "CACHE_MODEL_SHAPE_MISMATCH")
    draw = read(checked(m["reference_draw"], path.parent))
    population = np.load(checked(m["reference_population"], path.parent), allow_pickle=False)
    import pandas as pd
    fit_cells = pd.read_parquet(read(reference_path)["cell_index"]["path"], columns=["cell_id"])
    require(draw == generation["actual_reference_draw"] and draw["data_identity"] == m["data_identity"]
            and draw["reference_identity"] == m["reference_identity"] and list(population.shape) == draw["population_shape"]
            and population.shape == (draw["used_population"], model_config.n_genes)
            and len(draw["control_cell_ids"]) == draw["used_population"]
            and set(draw["control_cell_ids"]) <= set(fit_cells.cell_id)
            and np.isfinite(population).all() and hashlib.sha256(population.tobytes()).hexdigest() == draw["population_content_sha256"],
            "CACHE_REFERENCE_DRAW_MISMATCH")
    rows = read(checked(m["features"], path.parent))
    validate_rows(rows, standard_requests(input_bridge), m["shape"], m["reference_identity"], m["feature_identity"])
    content = [{k: r[k] for k in ("request_id", "population", "bio", "context", "history", "elapsed", "horizon")} for r in rows]
    require(config_hash(content) == generation.get("feature_content_identity"), "TABLE_CONTENT_IDENTITY_MISMATCH")
    return {r["request_id"]: r for r in rows}
