"""Full-protocol E -> P -> D contract, separate from the old native producer.

Preparation reads actual fit proxy counts and physical inputs, never weights or
models. The single-cell outlet is a declared proxy domain, not Viel RNA, bulk
RNA, raw source counts, or an Annexin prediction. Real execution needs matching
full-protocol weights; synthetic adapter checks carry no production authority.
"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from lung.data import full_protocol_request_inputs as inputs
from lung.data.preprocess import standardized
from lung.models.full_protocol import (DOSE_UNITS, FEATURE_SCHEMA, PARAMETER_IDENTITY,
    FullPhysicalProtocolBatch, FullProtocolConditioner, FullProtocolLatentAdapter)

ROOT = inputs.ROOT
VERSION = "full-protocol-response-v1"
DOMAIN = "joint_single_cell_common_gene_library10000_log1p_standardized_v1"
OUTLET = "common_gene_library10000_log1p_not_source_raw_counts"
STAGE = "full_protocol_latent_cfm_v1"
NUMERIC = ("fingerprint", "dose", "duration", "elapsed", "horizon", "vehicle",
           "background", "source", "protocol")
REFERENCE_SCHEMA = "fullP_reference_h_input_v1"
REFERENCE_VARIANTS = ("observed", "zero_h", "mean_h")
PARTIAL_PROTOCOL_COVERAGE = {"schema": "native-partial-protocol-supervision-v1",
    "nominal_event_count": 1, "event_list_verified_exhaustive": False,
    "known_physical_scalar_fields": ["dose"], "endpoint_hours_known": True,
    "unknown_physical_scalar_fields": ["start_hours", "end_hours", "exposure_hours",
        "wash_count", "prior_retention", "media_change", "recovery_hours"],
    "base_duration_semantics": "native_reported_observation_hours_not_verified_active_exposure",
    "prior_drug_history_verified": False, "washout_RNA_supervision": False,
    "recovery_RNA_supervision": False, "Viel_pulse_washout_recovery_protocol_OOD": True,
    "Viel_protocol_independently_validated": False, "functional_Y_supervision": False}


def require(ok, reason):
    if not ok:
        raise ValueError("FULL_P_" + reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def bound_inputs():
    reader = inputs.FullProtocolRequestInputs.load()
    rows, context = inputs.reconstruct()
    reference = context["reference"]
    require(reference["matched_initial_state_of_endpoint_experiment"] is False
            and reference["status"] == "cross_source_transfer_unvalidated", "PROXY_SEMANTICS_CHANGED")
    manifest = inputs.read(ROOT / inputs.JOINT / "manifest.json")
    require(reference["data_identity"] == digest({"signature": manifest["signature"],
            "files": manifest["files"]}), "REFERENCE_DATA_IDENTITY_MISMATCH")
    neighbors = ROOT / inputs.JOINT / "neighbors.npz"
    require(inputs.sha(neighbors) == manifest["files"]["neighbors.npz"]["sha256"], "NEIGHBORS_CHANGED")
    bindings = {"request_version": reader.version,
        "request_manifest_sha256": inputs.sha(inputs.OUTPUT / "manifest.json"),
        "request_rows_sha256": inputs.sha(inputs.OUTPUT / "requests.json"),
        "data_identity": reference["data_identity"], "reference_id": reference["reference_id"],
        "reference_index_sha256": reference["cell_index"]["sha256"],
        "source_reference_files": reference["source_files"],
        "ordered_gene_ids_sha256": digest(context["gene_ids"]),
        "gene_axis_sha256": inputs.PINS[inputs.JOINT + "/genes.parquet"],
        "normalization_sha256": inputs.PINS[inputs.JOINT + "/normalization.npz"],
        "neighbors_sha256": inputs.sha(neighbors), "source_pins": inputs.PINS,
        "model_context": context["model_context_indexes"],
        "context_vocabulary": context["context_vocabulary"],
        "input_domain": DOMAIN, "outlet": OUTLET, "feature_schema": FEATURE_SCHEMA,
        "parameter_identity": PARAMETER_IDENTITY,
        "duration_contract": inputs.DECLARATIONS["duration_contract"]}
    return reader, rows, context, bindings


def preflight():
    _, rows, _, bindings = bound_inputs()
    records = [{"candidate_id": r["candidate_id"], "source_id": r["source_id"],
        "source_state_id": r["source_state_id"], "observation_id": r["observation_id"],
        "role_binding": r["role_binding"], "request_status": r["request_status"],
        "input_ready": r["base_kwargs"] is not None,
        "pending": ["matching_full_protocol_AE_response_decoder_weights"]
            if r["base_kwargs"] is not None else [r["request_status"]],
        "prediction_domain": DOMAIN if r["base_kwargs"] is not None else None,
        "source_state_numerically_encoded": False,
        "proxy_transfer_validated": False} for r in rows]
    return {"schema": VERSION, "bindings": bindings, "candidates": records,
        "candidate_count": len(records), "numeric_ready": sum(r["input_ready"] for r in records),
        "PT_loads": 0, "model_forwards": 0, "P_outputs": 0, "G_outputs": 0, "Q_outputs": 0,
        "new_roles_assigned": False, "scientific_effect_verified": False,
        "real_execution_authorized_by_preparation": False}


def read_proxy_reference(context):
    """Read all pinned fit controls in their fixed row order, no target labels.

    Each source shard is SHA256-checked immediately before and after reading.
    The transform is the joint fit transform, not newly fitted preprocessing.
    """
    from scipy import sparse
    rows = pd.DataFrame(context["reference_rows"])
    reference = context["reference"]
    genes = pd.read_parquet(ROOT / inputs.JOINT / "genes.parquet")
    require(genes.ensembl_id.astype(str).tolist() == context["gene_ids"], "ORDERED_GENE_AXIS_CHANGED")
    columns = genes.tahoe_column.to_numpy(dtype=np.int64)
    require(np.all(columns >= 0) and len(set(columns.tolist())) == len(columns), "SOURCE_COLUMNS_INVALID")
    require(len(rows) == 2431 and rows.cell_id.is_unique and rows.role.eq("fit").all()
            and not rows[["matrix_path", "matrix_row"]].duplicated().any(), "FIT_REFERENCE_ROWS_CHANGED")
    require(set(rows.condition_id) == set(reference["control_condition_ids"])
            and all(rows[k].eq(reference[k]).all() for k in ("source_id", "background_id", "protocol_id")),
            "REFERENCE_CONTEXT_CHANGED")
    pinned = {str(Path(s["path"]).resolve()): s["sha256"] for s in reference["source_files"]}
    x = np.empty((len(rows), len(columns)), dtype=np.float32)
    source_records, nnz, mass = [], 0, []
    for name, group in rows.groupby("matrix_path", sort=False):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and str(path) in pinned, "UNREGISTERED_REFERENCE_SHARD")
        expected = pinned[str(path)]
        require(inputs.sha(path) == expected, "REFERENCE_SOURCE_CHANGED")
        matrix = sparse.load_npz(path)
        require(sparse.isspmatrix_csr(matrix) and columns.max() < matrix.shape[1], "REFERENCE_CSR_AXIS_INVALID")
        indices = group.matrix_row.to_numpy(dtype=np.int64)
        require(np.all(indices >= 0) and np.all(indices < matrix.shape[0]), "REFERENCE_ROW_OUT_OF_BOUNDS")
        counts = matrix[indices][:, columns]
        require(np.isfinite(counts.data).all() and (counts.data >= 0).all()
                and np.equal(counts.data, np.floor(counts.data)).all(), "SOURCE_COUNTS_INVALID")
        current_mass = np.asarray(counts.sum(1)).ravel()
        require((current_mass > 0).all(), "ZERO_COMMON_GENE_REFERENCE_LIBRARY")
        transformed = standardized(counts, context["mean"], context["scale"])
        require(np.isfinite(transformed).all(), "REFERENCE_TRANSFORM_NONFINITE")
        x[group.index] = transformed
        require(inputs.sha(path) == expected, "REFERENCE_CHANGED_DURING_READ")
        source_records.append({"path": str(path), "sha256": expected,
                               "reference_cells_read": len(group), "selected_count_nnz": counts.nnz})
        nnz += counts.nnz
        mass.extend(current_mass.tolist())
    x.setflags(write=False)
    return x, {"reference_id": reference["reference_id"], "reference_cell_ids": rows.cell_id.tolist(),
        "reference_rows_sha256": digest(context["reference_rows"]), "shape": list(x.shape),
        "float32_content_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "reference_role": "fit", "source_files_read": source_records, "selected_count_nnz": nnz,
        "common_gene_library_min": min(mass), "common_gene_library_max": max(mass),
        "transform": DOMAIN, "matched_initial_state_of_endpoint_experiment": False,
        "same_cell_fate_supervision": False, "target_expression_read": False,
        "functional_Y_read": False, "model_execution_performed": False}


def numeric_batch(reader, candidate_id, n, device="cpu"):
    kwargs, physical = reader.numeric_kwargs(candidate_id, batch_size=n)
    result = {k: torch.as_tensor(v.copy(), device=device) for k, v in kwargs.items()}
    physical = FullPhysicalProtocolBatch(**{f.name: getattr(physical, f.name).to(device)
                                           for f in fields(physical)})
    validate_batch(result, physical)
    return result, physical


def validate_batch(numeric, physical):
    """Cross-check duplicated base fields against the first active interval."""
    require(set(numeric) == set(NUMERIC), "FIXED_NUMERIC_SCHEMA_REQUIRED")
    physical.validate()
    n = len(physical.lengths)
    require(numeric["fingerprint"].shape == (n, 2048)
            and all(numeric[k].shape == (n,) for k in NUMERIC if k != "fingerprint"), "NUMERIC_SHAPE_MISMATCH")
    require(all(numeric[k].dtype == torch.long for k in ("source", "background", "protocol"))
            and numeric["vehicle"].dtype == torch.bool, "NUMERIC_DTYPE_MISMATCH")
    for k in ("fingerprint", "dose", "duration", "elapsed", "horizon"):
        require(numeric[k].is_floating_point() and torch.isfinite(numeric[k]).all(), "NUMERIC_NONFINITE_OR_NONFLOAT")
    require((physical.event_type[:, 0] == 0).all() and physical.fingerprint_known[:, 0].all()
            and physical.known[:, 0, 0].all() and physical.known[:, 0, 3].all()
            and physical.endpoint_known.all(), "ACTIVE_KNOWN_FIRST_INTERVAL_REQUIRED")
    require(torch.equal(numeric["fingerprint"], physical.fingerprint[:, 0])
            and torch.equal(numeric["vehicle"], physical.vehicle[:, 0]), "BASE_PHYSICAL_CHEMISTRY_MISMATCH")
    factor = physical.values.new_full((n,), float("nan"))
    for unit, conversion in (("uM", 1.), ("nM", .001), ("mM", 1000.)):
        factor = torch.where(physical.dose_unit[:, 0] == DOSE_UNITS.index(unit), conversion, factor)
    require(torch.isfinite(factor).all() and torch.allclose(numeric["dose"], physical.values[:, 0, 0] * factor),
            "BASE_PHYSICAL_DOSE_UNIT_MISMATCH")
    require(torch.equal(numeric["duration"], physical.values[:, 0, 3])
            and torch.equal(numeric["horizon"], physical.endpoint_hours)
            and (numeric["elapsed"] == 0).all(), "BASE_PHYSICAL_TIME_CONTRACT_MISMATCH")


def reference_contract(variant):
    require(type(variant) is str and variant in REFERENCE_VARIANTS, "REFERENCE_VARIANT_INVALID")
    return {"schema": REFERENCE_SCHEMA, "variant": variant,
        "population_scope": "same_current_allowed_model_input_reference_population",
        "population_definition_by_stage": {
            "CFM_fit": "original_post_OT_selected_control_h_ii_finite_draw_equal_selected_row_weight",
            "B4_and_selection": "original_uncoupled_control_condition_then_cell_draw_finite_row_population",
            "production_and_C_sidecar": "full_fixed_2431_fit_proxy_reference_equal_cell_weight_before_P_microbatch",
            "native_final": "full_current_allowed_final_control_draw_equal_row_weight_before_P_microbatch"},
        "same_numerical_population_mean_across_stages_claimed": False,
        "RNA_state_used_by_P": variant != "zero_h", "source_state_id_numerically_encoded": False,
        "mean_coordinate": "mean_of_frozen_E_h_not_E_of_expression_mean" if variant == "mean_h" else None}


def declared_reference_variant(metadata):
    contract = metadata.get("reference_contract", {})
    variant = contract.get("variant")
    require(contract == reference_contract(variant), "REFERENCE_CONTRACT_MISMATCH")
    return variant


def apply_reference_variant(h, variant):
    """Transform the whole permitted input population, before microbatching."""
    reference_contract(variant)
    require(isinstance(h, torch.Tensor) and h.ndim == 2 and min(h.shape) > 0
        and h.is_floating_point() and torch.isfinite(h).all(), "REFERENCE_H_POPULATION_INVALID")
    if variant == "zero_h":
        return torch.zeros_like(h)
    return h.mean(0, keepdim=True).expand_as(h) if variant == "mean_h" else h


@torch.no_grad()
def encode_reference_h(shared, reference, variant):
    require(reference.ndim == 2 and reference.shape[1] == shared.config.n_genes
        and len(reference) > 0 and reference.is_floating_point() and torch.isfinite(reference).all(),
        "REFERENCE_DOMAIN_SHAPE_INVALID")
    reference_contract(variant)
    if variant == "zero_h":
        return reference.new_zeros((len(reference), shared.config.d_model))
    return apply_reference_variant(shared.encode_h(reference)[1], variant)


@torch.no_grad()
def prepare_reference_h(shared, reference, variant, *, device="cpu", microbatch=64):
    """All draw/reference h first, then one latent summary; no per-block mean."""
    reference_contract(variant)
    require(type(microbatch) is int and microbatch > 0 and len(reference) > 0, "REFERENCE_PREPARATION_ARGUMENTS")
    if variant == "zero_h":
        require(tuple(reference.shape) == (len(reference), shared.config.n_genes), "REFERENCE_DOMAIN_SHAPE_INVALID")
        check = torch.as_tensor(reference)
        require(check.is_floating_point() and torch.isfinite(check).all(), "REFERENCE_DOMAIN_SHAPE_INVALID")
        return torch.zeros((len(reference), shared.config.d_model), device=device, dtype=torch.float32)
    values = []
    for start in range(0, len(reference), microbatch):
        part = torch.as_tensor(reference[start:start+microbatch], device=device, dtype=torch.float32)
        values.append(encode_reference_h(shared, part, "observed"))
    return apply_reference_variant(torch.cat(values), variant)


class FullProtocolResponseAdapter(nn.Module):
    """Executable single-cell proxy E/full-P/D in the declared expression coordinates."""
    def __init__(self, shared, full, mean, scale, *, input_domain=DOMAIN, reference_variant="observed"):
        super().__init__()
        require(input_domain == DOMAIN, "SOURCE_DOMAIN_NOT_SUPPORTED_BY_SINGLE_CELL_OUTLET")
        require(type(full) is FullProtocolLatentAdapter and full.member.shared is shared,
                "SHARED_E_RESPONSE_BINDING_MISMATCH")
        require(full.feature_schema == FEATURE_SCHEMA and full.parameter_identity == PARAMETER_IDENTITY,
                "FULL_PARAMETER_IDENTITY_REQUIRED")
        self.shared, self.full = shared, full
        self.reference_contract = reference_contract(reference_variant)
        for name, value in (("mean", mean), ("scale", scale)):
            array = np.asarray(value, dtype=np.float32)
            require(array.shape == (shared.config.n_genes,) and np.isfinite(array).all()
                    and (name != "scale" or (array > 0).all()), "OUTLET_COORDINATES_INVALID")
            self.register_buffer(name, torch.as_tensor(array.copy()), persistent=False)

    @torch.no_grad()
    def forward(self, reference, physical, numeric, *, noise, steps=32):
        require(type(steps) is int and steps > 0, "STEPS_INVALID")
        validate_batch(numeric, physical)
        require(reference.ndim == 2 and reference.shape == (len(physical.lengths), len(self.mean))
                and reference.is_floating_point() and torch.isfinite(reference).all(), "REFERENCE_DOMAIN_SHAPE_INVALID")
        reference_h = encode_reference_h(self.shared, reference, self.reference_contract["variant"])
        return self.predict_h(reference_h, physical, numeric, noise=noise, steps=steps)

    @torch.no_grad()
    def predict_h(self, reference_h, physical, numeric, *, noise, steps=32):
        """For a whole-population h prepared by the matching engine helper."""
        require(type(steps) is int and steps > 0, "STEPS_INVALID")
        validate_batch(numeric, physical)
        require(reference_h.shape == (len(physical.lengths), self.shared.config.d_model)
            and reference_h.is_floating_point() and torch.isfinite(reference_h).all(), "PREPARED_REFERENCE_H_SHAPE")
        variant = self.reference_contract["variant"]
        require(variant != "zero_h" or (reference_h == 0).all(), "ZERO_H_INPUT_CHANGED")
        require(variant != "mean_h" or (reference_h == reference_h[0]).all(), "MEAN_H_INPUT_CHANGED")
        latent = self.full.sample_latent(reference_h, physical, noise=noise, steps=steps, **numeric)
        raw = self.full.member.decode(latent)
        from lung.models.objectives import physical_projection
        result = physical_projection(raw, self.mean, self.scale)
        require(result.valid.all(), "INVALID_GENERATED_MASS_NO_FALLBACK")
        return result


def validate_metadata(metadata, bindings):
    """Check the declared data, model stages and AE/response pairing."""
    require(type(metadata) is dict, "MATCHING_MODEL_METADATA_PENDING")
    require(canonical(metadata.get("bindings")) == canonical(bindings), "MODEL_BINDINGS_MISMATCH")
    for name, stage in (("representation", "latent_ae"), ("response", STAGE)):
        item = metadata.get(name)
        require(type(item) is dict and item.get("stage") == stage and item.get("synthetic") is False
                and item.get("data_identity") == bindings["data_identity"], "MATCHING_STAGE_DATA_PENDING")
        require(type(item.get("optimizer_step")) is int and item["optimizer_step"] > 0,
                "MATCHING_TRAINED_WEIGHTS_PENDING")
        require(isinstance(item.get("checkpoint"), str) and bool(item["checkpoint"])
                and isinstance(item.get("checkpoint_sha256"), str) and len(item["checkpoint_sha256"]) == 64,
                "CHECKPOINT_FILE_BINDING_REQUIRED")
    require(metadata["response"].get("representation_hash") == metadata["representation"]["checkpoint_sha256"],
            "MATCHING_AE_RESPONSE_HASH_REQUIRED")
    require(metadata["response"].get("training_protocol_coverage") == PARTIAL_PROTOCOL_COVERAGE,
            "ACTUAL_PROTOCOL_SUPERVISION_COVERAGE_REQUIRED")
    declared_reference_variant(metadata["response"])


def finite_weights(state):
    require(isinstance(state, dict) and bool(state), "NONEMPTY_TENSOR_STATE_REQUIRED")
    for value in state.values():
        require(isinstance(value, torch.Tensor), "TENSOR_WEIGHTS_REQUIRED")
        data = value if value.layout == torch.strided else value.coalesce().values()
        require(torch.isfinite(data).all(), "FINITE_WEIGHTS_REQUIRED")


def equal_weight(left, right):
    if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.layout == right.layout and left.shape == right.shape and left.dtype == right.dtype):
        return False
    if left.layout == torch.strided:
        return torch.equal(left, right)
    left, right = left.coalesce(), right.coalesce()
    return torch.equal(left.indices(), right.indices()) and torch.equal(left.values(), right.values())


def load_model(metadata, bindings, context, device="cpu"):
    """Load matching AE and response weights, checking actual tensors and coordinates."""
    validate_metadata(metadata, bindings)
    from lung.models import ModelConfig
    from lung.models.latent_staged import SharedAutoencoder, LatentMember
    saved = {}
    for name in ("representation", "response"):
        item = metadata[name]
        path = (ROOT / item["checkpoint"]).resolve()
        require(path.is_relative_to(ROOT) and inputs.sha(path) == item["checkpoint_sha256"], "CHECKPOINT_CHANGED")
        value = torch.load(path, weights_only=True, map_location="cpu")
        actual = value.get("metadata", {})
        require(actual.get("stage") == item["stage"] and actual.get("data_identity") == bindings["data_identity"]
                and actual.get("synthetic") is False and canonical(actual.get("full_response_bindings")) == canonical(bindings),
                "ACTUAL_CHECKPOINT_DOMAIN_BINDINGS_MISMATCH")
        counter = value.get("counters", {}).get("optimizer_step")
        require(type(counter) is int and counter == item["optimizer_step"], "ACTUAL_TRAINED_COUNTER_MISMATCH")
        require(type(value.get("epoch")) is int and value["epoch"] > 0, "ACTUAL_TRAINED_EPOCH_REQUIRED")
        finite_weights(value.get("model"))
        if name == "response":
            finite_weights(value.get("shared"))
        saved[name] = value
    response = saved["response"]
    actual = response["metadata"]
    require(actual.get("feature_schema") == FEATURE_SCHEMA and actual.get("parameter_identity") == PARAMETER_IDENTITY
            and actual.get("representation_hash") == metadata["representation"]["checkpoint_sha256"]
            and actual.get("training_protocol_coverage") == PARTIAL_PROTOCOL_COVERAGE
            and actual.get("reference_contract") == metadata["response"]["reference_contract"],
            "ACTUAL_FULL_PROTOCOL_PARAMETERS_REQUIRED")
    shape = ModelConfig(**actual["model_config"])
    require(shape.n_genes == len(context["gene_ids"]) and shape.reference_mode == "observed"
            and shape.backbone == "compact_full_gene", "ACTUAL_MODEL_DOMAIN_SHAPE_MISMATCH")
    for field, key in (("n_sources", "source_id"), ("n_backgrounds", "background_id"), ("n_protocols", "protocol_id")):
        require(getattr(shape, field) == len(context["context_vocabulary"][key]), "ACTUAL_CONTEXT_CAPACITY_MISMATCH")
    with np.load(ROOT / inputs.JOINT / "neighbors.npz", allow_pickle=False) as archive:
        neighbors = torch.as_tensor(archive["neighbors"].copy(), dtype=torch.long)
    shared = SharedAutoencoder(shape, neighbors)
    shared.load_state_dict(saved["representation"]["model"], strict=True)
    require(set(response["shared"]) == set(shared.state_dict()) and all(equal_weight(response["shared"][k], t)
            for k, t in shared.state_dict().items()), "ACTUAL_EMBEDDED_AE_CHANGED")
    full = FullProtocolLatentAdapter(LatentMember(shape, shared), FullProtocolConditioner(shape))
    full.load_registered_state(response["model"], actual)
    return FullProtocolResponseAdapter(shared, full, context["mean"], context["scale"],
        reference_variant=declared_reference_variant(actual)).to(device).eval().requires_grad_(False)


def prepare(output):
    """Immutable real data preparation only, no model loading."""
    output = Path(output).resolve()
    require(not output.exists(), "REFUSING_EXISTING_OUTPUT")
    reader, rows, context, bindings = bound_inputs()
    report = preflight()
    x, reference_receipt = read_proxy_reference(context)
    ready = [r for r in rows if r["base_kwargs"] is not None]
    numeric, physical = [], []
    for request in ready:
        kwargs, batch = numeric_batch(reader, request["candidate_id"], 1)
        numeric.append({k: v.numpy() for k, v in kwargs.items()})
        physical.append({f.name: getattr(batch, f.name).numpy().tolist() for f in fields(batch)})
    require(bound_inputs()[3] == bindings, "INPUTS_CHANGED_DURING_PREPARATION")
    output.mkdir(parents=True)
    np.save(output / "reference_full_fit.npy", x, allow_pickle=False)
    np.savez(output / "candidate_numeric.npz", **{k: np.concatenate([r[k] for r in numeric]) for k in NUMERIC})
    write_new(output / "candidate_physical.json", [{"candidate_id": r["candidate_id"], "batch": p}
              for r, p in zip(ready, physical)])
    write_new(output / "reference_receipt.json", reference_receipt)
    write_new(output / "candidate_status.json", report["candidates"])
    report = {**report, "reference_cells_actually_read": len(x), "reference_gene_count": x.shape[1],
        "physical_numeric_candidates_actually_prepared": len(ready), "reference_receipt": reference_receipt,
        "files_sha256": {n: inputs.sha(output / n) for n in ("reference_full_fit.npy", "candidate_numeric.npz",
            "candidate_physical.json", "reference_receipt.json", "candidate_status.json")}}
    write_new(output / "manifest.json", report)
    return report


def produce(candidate_id, metadata, output, *, seed, steps=32, microbatch=64, device="cpu"):
    """Explicit future real inference, never called by preparation or training."""
    reader, rows, context, bindings = bound_inputs()
    request = next((r for r in rows if r["candidate_id"] == candidate_id), None)
    require(request is not None, "UNKNOWN_CANDIDATE")
    require(request["base_kwargs"] is not None, "SOURCE_PHYSICAL_REFERENCE_OR_OUTLET_PENDING")
    require(type(seed) is int and seed >= 0 and type(steps) is int and steps > 0
            and type(microbatch) is int and microbatch > 0, "GENERATION_ARGUMENTS_INVALID")
    output = Path(output).resolve()
    require(not output.exists(), "REFUSING_EXISTING_OUTPUT")
    metadata = json.loads(canonical(metadata))
    model = load_model(metadata, bindings, context, device)
    x, reference = read_proxy_reference(context)
    h = prepare_reference_h(model.shared, x, model.reference_contract["variant"], device=device, microbatch=microbatch)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((len(x), model.full.member.config.latent_dim), device=device, generator=generator)
    results = []
    for start in range(0, len(x), microbatch):
        end = min(start + microbatch, len(x))
        kwargs, physical = numeric_batch(reader, candidate_id, end-start, device)
        prediction = model.predict_h(h[start:end], physical, kwargs, noise=noise[start:end], steps=steps)
        results.append(prediction.y.cpu().numpy())
    y = np.concatenate(results).astype(np.float64)
    require(bound_inputs()[3] == bindings and all(inputs.sha(ROOT / metadata[k]["checkpoint"])
            == metadata[k]["checkpoint_sha256"] for k in ("representation", "response")), "GENERATION_ASSET_CHANGED")
    body = {"schema": VERSION, "candidate_id": candidate_id, "request": request, "bindings": bindings,
        "model_metadata": metadata,
        "reference": reference, "seed": seed, "steps": steps,
        "execution_kind": "matching_full_protocol_weights", "outlet": OUTLET,
        "reference_contract": model.reference_contract,
        "training_protocol_coverage": metadata["response"]["training_protocol_coverage"],
        "candidate_protocol_OOD": True, "candidate_protocol_scientifically_validated": False,
        "expression_mean": y.mean(0).tolist(), "expression_variance": y.var(0).tolist(),
        "summary": "equal_cell_weighted_full_fit_proxy_common_gene_log1p_mean_variance",
        "reference_target_cell_pairing": False, "source_state_numerically_encoded": False,
        "cross_source_transfer_validated": False, "source_target_RNA_read": False,
        "functional_Y_read": False, "G_outputs": 0, "Q_outputs": 0, "new_roles_assigned": False,
        "scientific_effect_verified": False}
    body["prediction_identity"] = digest(body)
    output.mkdir(parents=True)
    write_new(output / "prediction.json", body)
    return body


def read_prediction(path):
    """Authenticate a saved full-P outlet for downstream G/Q, without PT/ML.

    Protocol OOD and the unvalidated cross-source reference remain explicit.
    File integrity and matching weights do not establish endpoint science.
    """
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "experiments"), "PREDICTION_PROJECT_BOUNDARY")
    value = inputs.read(path)
    require(value.get("schema") == VERSION and value.get("prediction_identity")
        == digest({k: v for k, v in value.items() if k != "prediction_identity"}), "PREDICTION_CONTENT_CHANGED")
    _, rows, context, bindings = bound_inputs()
    request = next((r for r in rows if r["candidate_id"] == value.get("candidate_id")), None)
    require(request is not None and request["base_kwargs"] is not None
        and canonical(value.get("request")) == canonical(request)
        and canonical(value.get("bindings")) == canonical(bindings), "PREDICTION_REQUEST_OR_BINDINGS_CHANGED")
    metadata = value.get("model_metadata")
    validate_metadata(metadata, bindings)
    require(value.get("reference_contract") == metadata["response"]["reference_contract"], "PREDICTION_REFERENCE_VARIANT_CHANGED")
    for key in ("representation", "response"):
        asset = (ROOT / metadata[key]["checkpoint"]).resolve()
        require(asset.is_relative_to(ROOT) and inputs.sha(asset) == metadata[key]["checkpoint_sha256"],
                "PREDICTION_MODEL_ASSET_CHANGED")
    reference = value.get("reference", {})
    require(reference.get("reference_id") == bindings["reference_id"]
        and reference.get("reference_rows_sha256") == digest(context["reference_rows"])
        and reference.get("reference_cell_ids") == [r["cell_id"] for r in context["reference_rows"]]
        and reference.get("shape") == [2431, len(context["gene_ids"])]
        and reference.get("reference_role") == "fit" and reference.get("target_expression_read") is False,
        "PREDICTION_REFERENCE_CHANGED")
    require(value.get("outlet") == OUTLET and value.get("execution_kind") == "matching_full_protocol_weights"
        and value.get("training_protocol_coverage") == PARTIAL_PROTOCOL_COVERAGE
        and value.get("candidate_protocol_OOD") is True
        and value.get("candidate_protocol_scientifically_validated") is False
        and value.get("cross_source_transfer_validated") is False and value.get("functional_Y_read") is False
        and value.get("reference_target_cell_pairing") is False and value.get("source_state_numerically_encoded") is False
        and value.get("source_target_RNA_read") is False and value.get("new_roles_assigned") is False
        and type(value.get("G_outputs")) is int and value["G_outputs"] == 0
        and type(value.get("Q_outputs")) is int and value["Q_outputs"] == 0
        and value.get("summary") == "equal_cell_weighted_full_fit_proxy_common_gene_log1p_mean_variance"
        and value.get("scientific_effect_verified") is False, "PREDICTION_OUTLET_OR_SCIENCE_CHANGED")
    require(type(value.get("seed")) is int and value["seed"] >= 0
        and type(value.get("steps")) is int and value["steps"] > 0, "PREDICTION_EXECUTION_ARGUMENTS")
    mean, variance = np.asarray(value.get("expression_mean")), np.asarray(value.get("expression_variance"))
    require(mean.shape == variance.shape == (len(context["gene_ids"]),)
        and mean.dtype.kind == variance.dtype.kind == "f"
        and np.isfinite(mean).all() and np.isfinite(variance).all()
        and (mean >= 0).all() and (mean <= np.log(10001) + 1e-5).all()
        and (mean > 0).any() and (variance >= 0).all(), "PREDICTION_MOMENTS_INVALID")
    return value
