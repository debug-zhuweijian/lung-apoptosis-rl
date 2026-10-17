"""Frozen external-condition features, with explicit unpaired reference transfer.

This module never trains a model or reads measured endpoint values. Generation
uses BiologicalWorld's actual inference and feature construction. Output is a
feature table, not a ready endpoint training manifest or a feasibility verdict.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lung.data.common import data_file_bindings
from lung.rl import stream_seed
from .auxiliary import encode_vectors, verify_auxiliary_dependencies
from .checkpoint import atomic_json, load_checkpoint
from .config import StageUnavailable, config_hash, file_hash
from .endpoint_contract import FEATURE_KEYS, feature_dependencies_v2, frozen_feature_identity_v2
from .representation import require_finite_state, require_trained
from .world import BiologicalWorld, INITIAL_STATE_VERSION


FEATURE_VERSION = "external-condition-world-score-v1"
CONTEXT_KEYS = ("source_id", "background_id", "protocol_id")


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def checked(recorded, base):
    path = (Path(base) / recorded["path"]).resolve()
    if not path.is_file() or file_hash(path) != recorded["sha256"]:
        raise StageUnavailable("FEATURE_SOURCE_MISSING_OR_CHANGED: " + str(path))
    return path


def descriptor_record(path, value):
    """Bind the exact descriptor object used for inference to its source bytes."""
    path = Path(path).resolve()
    content = path.read_bytes()
    if json.loads(content.decode("utf-8-sig")) != value:
        raise StageUnavailable("FEATURE_DESCRIPTOR_CONTENT_MISMATCH: " + str(path))
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def load_condition_inputs(directory, project_root):
    """Read hashes, conditions, chemistry and protocol metadata; never Y values."""
    directory, project_root = Path(directory).resolve(), Path(project_root).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("CONDITION_INPUT_SCHEMA_UNSUPPORTED")
    files = [manifest_path]
    for name, digest in data_file_bindings(manifest["files"]).items():
        files.append(checked({"path": name, "sha256": digest}, directory))
    protocols, protocol_files = None, []
    for name, digest in data_file_bindings(manifest["inputs_sha256"]).items():
        path = checked({"path": name, "sha256": digest}, project_root)
        files.append(path)
        if path.name == "manifest.json":
            metadata = json.loads(path.read_text(encoding="utf-8"))
            items = metadata.get("files", [])
            if isinstance(items, list):
                matches = [p for p in items if p.get("file") == "protocols.json"]
                if matches:
                    if protocols is not None or len(matches) != 1:
                        raise ValueError("AMBIGUOUS_ENDPOINT_PROTOCOL_REGISTRY")
                    p = checked({"path": matches[0]["file"], "sha256": matches[0]["sha256"]}, path.parent)
                    protocols = json.loads(p.read_text(encoding="utf-8"))
                    protocol_files.append(p)
    if protocols is None:
        raise StageUnavailable("ENDPOINT_PROTOCOL_REGISTRY_UNBOUND")
    frames = {name: pd.read_parquet(directory / f"{name}.parquet")
              for name in ("condition_requests", "components", "compounds")}
    fingerprints = np.load(directory / "fingerprints.npy", allow_pickle=False)
    requests, components, compounds = (frames[name] for name in frames)
    if requests.condition_id.duplicated().any() or compounds.compound_id.duplicated().any():
        raise ValueError("DUPLICATE_CONDITION_OR_COMPOUND")
    if (fingerprints.shape != (len(compounds), 2048) or not np.isin(fingerprints, [0, 1]).all()
            or not set(components.condition_id) <= set(requests.condition_id)
            or not np.array_equal(np.sort(compounds.fingerprint_row.to_numpy()), np.arange(len(compounds)))):
        raise ValueError("CONDITION_CHEMISTRY_ALIGNMENT_INVALID")
    registry = compounds.set_index("compound_id")
    for component in components.to_dict("records"):
        row = registry.loc[component["compound_id"]]
        for key in ("fingerprint_row", "structure_eligible", "identity_id", "canonical_full_smiles"):
            a, b = component[key], row[key]
            if not ((pd.isna(a) and pd.isna(b)) or a == b):
                raise ValueError("COMPONENT_REGISTRY_IDENTITY_MISMATCH")
    files = sorted(set(files + protocol_files))
    return {**frames, "fingerprints": fingerprints, "protocols": protocols,
            "files": files, "source_records": [record(p) for p in files], "manifest": manifest}


def condition_review(bundle):
    """Eligibility reflects current single-drug 24 h Annexin feature semantics."""
    review = []
    for row in bundle["condition_requests"].to_dict("records"):
        parts = bundle["components"].loc[lambda f: f.condition_id.eq(row["condition_id"])]
        reasons = set(filter(None, str(row.get("input_contract_blockers", "")).split(";")))
        if bundle["protocols"].get(row["protocol_id"], {}).get("endpoint") != "Annexin_positive_fraction":
            reasons.add("not_Annexin_schema2_endpoint")
        if row["is_control"]:
            reasons.add("control_transfer_requires_separate_validation")
        if row["treatment_hours"] != 24:
            reasons.add("unsupported_time_not_24h")
        if row["exposure_design"] != "single_exposure" or len(parts) != 1:
            reasons.add("unsupported_combination_or_exposure_design")
        if len(parts) == 1:
            part = parts.iloc[0]
            if not part.structure_eligible or not isinstance(part.identity_id, str):
                reasons.add("unreviewed_structure")
            elif not bundle["fingerprints"][int(part.fingerprint_row)].any():
                reasons.add("invalid_empty_chemical_fingerprint")
            if part.dose_unit != "uM" or not np.isfinite(part.dose_value) or part.dose_value <= 0:
                reasons.add("unsupported_or_invalid_dose")
        if row.get("initial_state_kind") != "reference" or row.get("same_sample_expression_endpoint"):
            reasons.add("matched_initial_state_requires_separate_generator")
        review.append({"experimental_condition_id": row["condition_id"], "source_id": row["source_id"],
                       "protocol_id": row["protocol_id"], "source_background": row["cell_line"],
                       "treatment_hours": row["treatment_hours"], "eligible_condition_input": not reasons,
                       "reasons": sorted(reasons)})
    return review


def propose_reference_inputs(project_root, data_config, bundle, output):
    """Choose a fit-only control pool from metadata, independent of external Y.

    This registers a reference assumption, never a matched baseline. If the data
    manifest later gains preprocessing assets, regenerate into a new version so
    its current data identity and manifest hash are rebound explicitly.
    """
    base, output = Path(project_root).resolve(), Path(output).resolve()
    cfg_path = (base / data_config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assets = base / cfg["assets"]
    manifest_path = assets / "manifest.json"
    manifest_digest = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = [manifest_path, cfg_path]
    for name in ("conditions.parquet", "control_links.parquet", "role_condition_counts.parquet", "train.parquet"):
        if name not in manifest["files"]:
            raise StageUnavailable("REFERENCE_CONTROL_METADATA_MISSING: " + name)
        metadata.append(checked({"path": str(assets / name), "sha256": manifest["files"][name]["sha256"]}, base))
    conditions = pd.read_parquet(assets / "conditions.parquet")
    links = pd.read_parquet(assets / "control_links.parquet")
    counts = pd.read_parquet(assets / "role_condition_counts.parquet")
    fit_ids = set(counts.loc[counts.role.eq("fit") & counts.n_cells.gt(0), "condition_id"])
    candidates = conditions.loc[conditions.eligible & conditions.role.eq("fit")
        & conditions.condition_type.eq("treated") & conditions.source_id.eq("Tahoe-100M")
        & conditions.background_id.eq("CVCL_0023") & conditions.duration_h.eq(24)
        & conditions.condition_id.isin(fit_ids)].sort_values("condition_id")
    selected, control_ids = None, None
    for row in candidates.to_dict("records"):
        matched = links.loc[links.condition_id.eq(row["condition_id"]) & links.plate.eq(row["plate"])]
        ids = sorted(set(matched.control_condition_id) & fit_ids)
        controls = conditions.loc[conditions.condition_id.isin(ids)]
        if ids and len(controls) == len(ids) and controls.condition_type.eq("control").all():
            if all(controls[key].eq(row[key]).all() for key in (*CONTEXT_KEYS, "plate")):
                selected, control_ids = row, ids
                break
    if selected is None:
        raise StageUnavailable("NO_FIT_TAHOE_A549_SAME_PLATE_CONTROL_POOL")
    cells = pd.read_parquet(assets / "train.parquet", filters=[("role", "=", "fit"), ("condition_id", "in", control_ids)])
    required = {"cell_id", "condition_id", "matrix_path", "matrix_row", "role", "plate", *CONTEXT_KEYS}
    if (not required <= set(cells) or cells.empty or cells.cell_id.duplicated().any()
            or not set(cells.condition_id) == set(control_ids) or not cells.role.eq("fit").all()
            or cells.matrix_row.isna().any() or not cells.matrix_row.ge(0).all()
            or any(not cells[key].eq(selected[key]).all() for key in (*CONTEXT_KEYS, "plate"))):
        raise StageUnavailable("REFERENCE_CONTROL_CELL_INDEX_INVALID")
    source_files = [record(base / name) for name in sorted(cells.matrix_path.unique())]
    if file_hash(manifest_path) != manifest_digest:
        raise StageUnavailable("REFERENCE_DATA_MANIFEST_CHANGED_DURING_PROPOSAL")
    output.mkdir(parents=True, exist_ok=False)
    cell_path = output / "reference_fit_control_cells.parquet"
    cells.sort_values(["condition_id", "cell_id"]).to_parquet(cell_path, index=False)
    identity = config_hash({"signature": manifest["signature"], "files": manifest["files"]})
    reference_id = "Tahoe_fit_reference:" + config_hash({"data_identity": identity,
        "condition_id": selected["condition_id"], "cell_index": record(cell_path)["sha256"]})[:24]
    reference = {"kind": "reference", "status": "cross_source_transfer_unvalidated",
        "reference_id": reference_id, "condition_id": selected["condition_id"], "role": "T",
        "data_identity": identity, **{key: selected[key] for key in CONTEXT_KEYS},
        "reference_plate": selected["plate"], "control_condition_ids": control_ids,
        "unique_fit_cell_indices": len(cells), "source_csr_files": len(source_files),
        "cell_index": record(cell_path), "bound_data_manifest_status": manifest["status"],
        "selection_rule": "lexicographically_first_eligible_fit_Tahoe_A549_24h_condition_with_same_plate_fit_controls",
        "external_endpoint_values_used_for_selection": False,
        "matched_initial_state_of_endpoint_experiment": False,
        "evidence": "Frozen fit metadata/control links and recorded CSR rows; external source has no matched t0 RNA. Reference use is an unvalidated cross-source assumption.",
        "source_files": [record(p) for p in metadata] + [record(cell_path)] + source_files}
    reference_path = output / "reference.json"
    atomic_json(reference_path, reference)
    mappings = []
    contexts = bundle["condition_requests"][["source_id", "cell_line", "protocol_id"]].drop_duplicates()
    evidence_files = [record(p) for p in bundle["files"] if p.name in {"manifest.json", "protocols.json", "condition_requests.parquet"}]
    for row in contexts.sort_values(["source_id", "cell_line", "protocol_id"]).to_dict("records"):
        mappings.append({**row, **{"target_" + key: selected[key] for key in CONTEXT_KEYS},
            "target_reference_plate": selected["plate"], "reference_id": reference_id,
            "kind": "cross_source_transfer_unvalidated", "status": "explicit_assumption_unvalidated",
            "background_transfer_validated": False,
            "evidence": "Use the named frozen Tahoe fit-control domain solely as a reference model input. Original experiment background and assay protocol remain distinct; this mapping is not paired RNA or biological equivalence.",
            "source_files": evidence_files})
    mapping_path = output / "context_map.json"
    atomic_json(mapping_path, mappings)
    return reference_path, mapping_path


def preflight(project_root, data_config, pipeline, bundle, reference=None, mappings=None):
    """Collect real missing inputs without creating preprocessing or weights."""
    project_root = Path(project_root)
    issues, assets = [], []
    cfg_path = (project_root / data_config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    asset_dir = (project_root / cfg["assets"]).resolve()
    manifest_path = asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = config_hash({"signature": manifest["signature"], "files": manifest["files"]})
    assets.extend([cfg_path, manifest_path])
    if manifest.get("status") != "complete":
        issues.append("PREPROCESSING_NOT_COMPLETE: " + str(manifest.get("status")))
    for name in ("genes.parquet", "normalization.npz", "neighbors.npz", "bandwidths.npz"):
        path = asset_dir / name
        item = manifest["files"].get(name)
        if not path.is_file() or not item:
            issues.append("MISSING_FROZEN_DATA_ASSET: " + name)
        elif file_hash(path) != item["sha256"]:
            issues.append("FROZEN_DATA_ASSET_CHANGED: " + name)
        else:
            assets.append(path)
    identity_path = cfg.get("drug_identity_manifest")
    if not identity_path:
        issues.append("EXPLICIT_CHEMICAL_IDENTITY_MANIFEST_REQUIRED")
    else:
        try:
            assets.append(checked({"path": identity_path, "sha256": cfg["drug_identity_manifest_sha256"]}, project_root))
        except (KeyError, StageUnavailable) as error:
            issues.append(str(error))
    for key in ("response", "pool", "state"):
        if not pipeline.get(key):
            issues.append("MISSING_FROZEN_CHECKPOINT: " + key)
    for key in FEATURE_KEYS:
        if not pipeline.get(key):
            continue
        values = pipeline[key] if isinstance(pipeline[key], (list, tuple)) else [pipeline[key]]
        for value in values:
            path = Path(value)
            if not path.is_file():
                issues.append("MISSING_FROZEN_CHECKPOINT_FILE: " + str(path))
                continue
            assets.append(path)
            if key == "correction_gate":
                continue
            saved = load_checkpoint(path)
            if (saved.get("epoch", 0) < 1 or saved.get("counters", {}).get("optimizer_step", 0) < 1
                    or saved.get("metadata", {}).get("synthetic", False)):
                issues.append("UNTRAINED_OR_SYNTHETIC_CHECKPOINT: " + key)
            if saved.get("metadata", {}).get("data_identity") != identity:
                issues.append("CHECKPOINT_DATA_IDENTITY_MISMATCH: " + key)
            stage = saved.get("metadata", {}).get("stage")
            expected = {"response": {"response", "latent_cfm", "latent_members"},
                        "representation": {"latent_ae"}}.get(key, {key})
            if stage not in expected:
                issues.append("CHECKPOINT_STAGE_MISMATCH: " + key)
            if key == "response" and stage in {"latent_cfm", "latent_members"} and not pipeline.get("representation"):
                issues.append("MISSING_FROZEN_CHECKPOINT: representation")
            try:
                require_trained(saved, "CHECKPOINT_NO_OPTIMIZER_UPDATES")
                require_finite_state(saved["model"])
                if key in {"pool", "state", "backward", "correction"}:
                    verify_auxiliary_dependencies(path, pipeline, saved=saved)
            except (ValueError, StageUnavailable, FileNotFoundError, KeyError) as error:
                issues.append(f"CHECKPOINT_DEPENDENCY_INVALID: {key}: {error}")
    if pipeline.get("correction") and not all(pipeline.get(k) for k in ("backward", "correction_gate")):
        issues.append("CORRECTION_CHAIN_INCOMPLETE")
    if pipeline.get("correction_gate") and not pipeline.get("correction"):
        issues.append("CORRECTION_CHAIN_INCOMPLETE")
    if reference is None:
        issues.append("EXPLICIT_REFERENCE_DESCRIPTOR_REQUIRED")
    else:
        try:
            validate_reference(reference, manifest_path, asset_dir, identity, project_root)
        except (ValueError, StageUnavailable, KeyError) as error:
            issues.append(str(error))
    if not mappings:
        issues.append("EXPLICIT_SOURCE_CONTEXT_MAPPING_REQUIRED")
    review = condition_review(bundle)
    return {"status": "pending" if issues else "preflight_inputs_present",
            "formal_training_ready": False, "data_identity": identity,
            "data_manifest": record(manifest_path), "missing_or_invalid": issues,
            "condition_count": len(review),
            "eligible_condition_inputs": sum(r["eligible_condition_input"] for r in review),
            "excluded_reason_counts": dict(Counter(r for row in review for r in row["reasons"])),
            "conditions": review, "checked_assets": [record(p) for p in sorted(set(assets))]}


def validate_reference(reference, data_manifest, assets_dir, data_identity, base):
    required = ("reference_id", "reference_plate", "condition_id", "evidence", *CONTEXT_KEYS)
    if (reference.get("kind") != "reference" or reference.get("role") != "T"
            or reference.get("data_identity") != data_identity
            or any(not isinstance(reference.get(k), str) or not reference[k].strip() for k in required)):
        raise StageUnavailable("EXPLICIT_FIT_REFERENCE_IDENTITY_REQUIRED")
    paths = {checked(p, base) for p in reference.get("source_files", [])}
    if not {Path(data_manifest).resolve(), (Path(assets_dir) / "control_links.parquet").resolve()} <= paths:
        raise StageUnavailable("REFERENCE_SOURCE_AND_CONTROL_LINK_HASHES_REQUIRED")


def mapped_conditions(bundle, data, reference, mappings, *, base):
    """Map only explicitly named domains; mapping is an unvalidated assumption."""
    if any(not isinstance(reference.get(k), str) or not reference[k].strip()
           for k in ("reference_id", "reference_plate")):
        raise StageUnavailable("REFERENCE_ID_AND_PLATE_REQUIRED")
    source_rows = data.reader.conditions
    ref = source_rows.loc[source_rows.condition_id.eq(reference["condition_id"])]
    if len(ref) != 1 or reference["condition_id"] not in data.condition_ids("T"):
        raise StageUnavailable("REFERENCE_FIT_CONDITION_UNAVAILABLE")
    ref = ref.iloc[0]
    if any(reference[key] != ref[key] for key in CONTEXT_KEYS):
        raise StageUnavailable("REFERENCE_CONTEXT_MISMATCH")
    if "plate" not in ref or reference["reference_plate"] != ref["plate"]:
        raise StageUnavailable("REFERENCE_PLATE_MISMATCH")
    encoded = data.action_features(reference["condition_id"])
    mapping_index = {}
    for item in mappings:
        key = (item["source_id"], item["cell_line"], item["protocol_id"])
        if (key in mapping_index or item.get("status") != "explicit_assumption_unvalidated"
                or not item.get("evidence")):
            raise StageUnavailable("SOURCE_CONTEXT_MAPPING_UNVERIFIED")
        if any(item.get("target_" + name) != reference[name] for name in CONTEXT_KEYS):
            raise StageUnavailable("SOURCE_CONTEXT_TARGET_NOT_REFERENCE_DOMAIN")
        if item.get("reference_id") != reference["reference_id"]:
            raise StageUnavailable("SOURCE_CONTEXT_REFERENCE_ID_MISMATCH")
        if item.get("target_reference_plate") != reference["reference_plate"]:
            raise StageUnavailable("SOURCE_CONTEXT_REFERENCE_PLATE_MISMATCH")
        if not item.get("source_files"):
            raise StageUnavailable("SOURCE_CONTEXT_MAPPING_EVIDENCE_REQUIRED")
        for source in item["source_files"]:
            checked(source, base)
        mapping_index[key] = item
    review, actions, provenance = condition_review(bundle), {}, {}
    indexed = bundle["condition_requests"].set_index("condition_id")
    fitted = source_rows.loc[source_rows.role.eq("fit") & source_rows.condition_type.eq("treated")]
    for entry in review:
        if not entry["eligible_condition_input"]:
            continue
        condition = entry["experimental_condition_id"]
        source = indexed.loc[condition]
        key = (source.source_id, source.cell_line, source.protocol_id)
        if key not in mapping_index:
            entry["reasons"].append("explicit_source_context_mapping_missing")
            entry["eligible_condition_input"] = False
            continue
        part = bundle["components"].loc[lambda f: f.condition_id.eq(condition)].iloc[0]
        same_drug = fitted.loc[fitted.chemical_or_experiment_group_id.eq(part.identity_id)]
        dose = float(part.dose_value)
        actions[condition] = {"fingerprint": bundle["fingerprints"][int(part.fingerprint_row)].astype(np.float32),
            "dose": dose, "duration": 24., "group_id": part.identity_id, "is_vehicle": False,
            **{k: encoded[k] for k in CONTEXT_KEYS}}
        provenance[condition] = {"source_background": source.cell_line,
            "source_id": source.source_id, "protocol_id": source.protocol_id,
            "model_context": {k: reference[k] for k in CONTEXT_KEYS},
            "reference_id": reference["reference_id"], "reference_plate": reference["reference_plate"],
            "mapping_status": "explicit_assumption_unvalidated", "mapping_evidence": mapping_index[key]["evidence"],
            "chemical_identity_id": part.identity_id, "compound_id": part.compound_id,
            "dose_um": dose, "chemical_warning": "" if pd.isna(part.warning) else str(part.warning),
            "unseen_chemical_identity_in_fit": same_drug.empty,
            "outside_global_fit_dose_range": bool(fitted.empty or dose < fitted.dose_um.min() or dose > fitted.dose_um.max()),
            "outside_same_drug_fit_dose_range": None if same_drug.empty else bool(dose < same_drug.dose_um.min() or dose > same_drug.dose_um.max()),
            "background_transfer_validated": False}
    return actions, provenance, review


class ExternalConditionWorld(BiologicalWorld):
    """Same frozen inference/features, externally reviewed single-drug actions."""

    def __init__(self, data, config, pipeline, actions):
        self.external_actions = actions
        super().__init__(data, {**config, "task": "PROXY-A549-V1", "protocol": None}, pipeline)

    def _action_metadata(self):
        if not self.external_actions:
            raise StageUnavailable("NO_SUPPORTED_EXTERNAL_CONDITIONS")
        return self.external_actions


@torch.no_grad()
def generate_feature_rows(world, root, provenance, *, seed, reference_id):
    """One population per condition, exactly the world single-member score path."""
    rows = []
    for condition in sorted(world.actions):
        draw_seed = stream_seed(seed, FEATURE_VERSION, condition)
        member = int(np.random.default_rng(stream_seed(draw_seed, "member")).integers(len(world.members)))
        expression, context = world.propagate(root, (condition,), draw_seed, scoring_member=member)
        features, history, elapsed = world._features(expression, context, (condition,))
        d, c = world.shape.d_model, world.shape.context_dim
        z = encode_vectors(world.representation, expression, world.device,
                           batch_size=world.inference_microbatch, output_device=world.device)
        probabilities = world.classifier.probabilities(z).mean(0)
        if not torch.isfinite(features).all() or not torch.isfinite(history).all() or not torch.isfinite(probabilities).all():
            raise FloatingPointError("NONFINITE_EXTERNAL_FEATURES")
        rows.append({"experimental_condition_id": condition, "feature_kind": "predicted",
            "feature_status": "complete", "population": features[:d].cpu().tolist(),
            "bio": features[d:d+27].cpu().tolist(), "context": features[2*d+27:2*d+27+3*c].cpu().tolist(),
            "history": history.cpu().tolist(), "elapsed": elapsed / world.protocol.ticks_per_hour,
            "horizon": world.protocol.horizon_ticks / world.protocol.ticks_per_hour,
            "initial_state": {"kind": "reference", "reference": reference_id,
                              "evidence": "Explicit frozen reference population; not measured initial RNA of endpoint experiment"},
            "source_transfer": provenance[condition], "generation_seed": draw_seed,
            "scoring_member": member, "state_probabilities_diagnostic": probabilities.cpu().tolist(),
            "state_probabilities_used_by_endpoint_head": False,
            "biological_gene_set_features": "missing_masked_as_current_world"})
    return rows


def generate_to_directory(project_root, data, config, pipeline, bundle, reference, mappings,
                          output, *, reference_path, mapping_path, data_config_path):
    """Write actual frozen features and the inputs needed for a later schema2 join."""
    output, project_root = Path(output).resolve(), Path(project_root).resolve()
    if output.exists():
        raise FileExistsError("Feature output exists; choose a new immutable version")
    if config.get("history_support") != "single_exposure_only":
        raise StageUnavailable("EXTERNAL_FEATURE_GENERATOR_SINGLE_EXPOSURE_ONLY")
    pipeline = {k: v for k, v in pipeline.items() if k in {*FEATURE_KEYS, "inference_ode_steps"}}
    validate_reference(reference, data.assets_dir / "manifest.json", data.assets_dir, data.identity, project_root)
    reference_record = descriptor_record(reference_path, reference)
    mapping_record = descriptor_record(mapping_path, mappings)
    for item in bundle["source_records"]:
        checked(item, project_root)
    guard_paths = [*bundle["files"], Path(data_config_path)]
    guard_paths.extend(data.assets_dir / n for n in ("manifest.json", "genes.parquet", "normalization.npz", "neighbors.npz", "bandwidths.npz"))
    for key in FEATURE_KEYS:
        if pipeline.get(key):
            guard_paths.extend(pipeline[key] if isinstance(pipeline[key], (list, tuple)) else [pipeline[key]])
    before = [reference_record, mapping_record,
              *[record(p) for p in sorted({Path(p).resolve() for p in guard_paths})]]
    actions, provenance, review = mapped_conditions(bundle, data, reference, mappings, base=project_root)
    world = ExternalConditionWorld(data, config, pipeline, actions)
    root = world.root(reference["condition_id"], "T")
    initial, context = world._initial(root, world.n_pop)
    output.mkdir(parents=True)
    np.save(output / "reference_population.npy", initial.cpu().numpy(), allow_pickle=False)
    ids = world._initial_cache[2].get("control_cell_ids")
    if ids is None and not data.synthetic:
        raise StageUnavailable("REFERENCE_CELL_PROVENANCE_REQUIRED")
    atomic_json(output / "reference_draw.json", {"reference_id": reference["reference_id"],
        "root_identity": root.payload.identity, "data_identity": data.identity,
        "sampling_seed": root.payload.initial_seed, "sampling_draw_size": root.payload.population_size,
        "used_population": world.n_pop, "control_cell_ids": [] if ids is None else list(ids[:world.n_pop]),
        "sampling_unit": "reference_cells_not_new_biological_replicates", "source_files": reference["source_files"]})
    rows = generate_feature_rows(world, root, provenance, seed=config["seed"], reference_id=reference["reference_id"])
    data_cfg = json.loads(Path(data_config_path).read_text(encoding="utf-8"))
    generation = {"algorithm": FEATURE_VERSION, "seed": config["seed"],
        "member_rule": "stream_seed_then_single_member_as_world_score_one", "feature_pipeline_stages": sorted(k for k in FEATURE_KEYS if pipeline.get(k)),
        "reference_kind": "reference_not_matched_t0", "reference_draw": record(output / "reference_draw.json"),
        "source_context_mapping": mapping_record,
        "runtime": {"device": str(world.device), "torch_version": torch.__version__,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "world_inference": {"population": world.n_pop, "inference_ode_steps": world.steps,
            "inference_microbatch": world.inference_microbatch, "initial_state_interface": INITIAL_STATE_VERSION,
            "history_support": config["history_support"]}}
    for item in before:
        checked(item, project_root)
    feature_assets = {"gene_axis": [record(data.assets_dir / "genes.parquet")],
        "preprocessing": [record(data.assets_dir / n) for n in ("manifest.json", "normalization.npz", "neighbors.npz", "bandwidths.npz")],
        "chemical_identity": [record(project_root / data_cfg["drug_identity_manifest"]),
                              *[record(p) for p in bundle["files"] if p.name in {"compounds.parquet", "fingerprints.npy"}]],
        "condition_inputs": [record(p) for p in bundle["files"]] + [mapping_record, record(data_config_path)],
        "initial_state": [reference_record, record(output / "reference_population.npy"), record(output / "reference_draw.json"),
                          *reference["source_files"]]}
    dependencies = feature_dependencies_v2(pipeline, feature_assets, base=project_root)
    identity = frozen_feature_identity_v2(data.identity, dependencies, generation)
    rows = [{**row, "frozen_feature_identity": identity} for row in rows]
    atomic_json(output / "features.json", rows)
    manifest = {"schema_version": 1, "status": "generated_frozen_features_not_endpoint_training_manifest",
        "formal_training_ready": False, "data_identity": data.identity, "features": record(output / "features.json"),
        "generated_conditions": len(rows), "feature_dependencies": dependencies, "feature_assets": feature_assets,
        "feature_generation": generation, "frozen_feature_identity": identity,
        "condition_review": review, "measured_endpoint_values_read": False,
        "remaining_schema2_join": "Join measured conditions/protocols/observations, assign independent T/V/C/E; then make_manifest_v2"}
    atomic_json(output / "manifest.json", manifest)
    return manifest
