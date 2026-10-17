"""Independent partial-protocol full-P learning; default entry is data preflight.

Native single-cell endpoint RNA supervises the old B3 latent CFM and optional
B4 decoded endpoint MMD. A nominal drug record does not certify continuous
exposure, washout, or a complete treatment history. No functional Y is opened.
"""
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from lung.data.native_response_units import NativeResponseUnits
from lung.models import ModelConfig, entropic_ot, rbf_mmd, sample_ot_pairs
from lung.models.full_protocol import (DOSE_UNITS, FEATURE_SCHEMA, PARAMETER_IDENTITY,
    FullPhysicalProtocolBatch, FullProtocolConditioner, FullProtocolLatentAdapter)
from lung.models.latent_staged import LatentMember, SharedAutoencoder
from . import full_protocol_response as producer
from .checkpoint import atomic_json, atomic_save, load_checkpoint, seed_all
from .config import ROOT, config_hash, merge
from .data import EngineData
from .latent_pipeline import _EarlyStop, _couple, _model_config, latent_settings, run_latent_ae
from .response import condition_kwargs, projected, tensor

SCHEMA = "full-protocol-native-endpoint-learning-v1"
AE_RESULT_SCHEMA = SCHEMA + ":completed-joint-AE-result"
SETTINGS_SCHEMA = "fullP_learning_reference_variant_settings_v2"
NATIVE = ROOT / "experiments/20261011-data-alignment-round17/native-response-units-v1"
JOINT_CONFIG = "configs/data_joint_sciplex_v1.json"
COVERAGE = producer.PARTIAL_PROTOCOL_COVERAGE


def require(ok, reason):
    if not ok:
        raise ValueError("FULL_P_LEARNING_" + reason)


def joint_config(config):
    """Independent copy; existing active configs and interrupted runs stay intact."""
    return merge(config, {"data": JOINT_CONFIG, "synthetic": False, "auto_prepare": False,
        "workflow": "modular_stage", "stage": "response", "task": "RESPONSE-A549-V1",
        "response": {"model": {"n_genes": 32461, "n_sources": 2,
                              "n_backgrounds": 1, "n_protocols": 2}}})


def joint_ae_config(config):
    """A P-only ablation never changes shared E/D fitting or its namespace."""
    value = joint_config(config)
    if "research" in value:
        value["research"].pop("full_protocol_response", None)
    return value


def settings_from_config(config):
    latent, response = latent_settings(config), config["response"]
    value = {"seed": config["seed"], "device": config["device"], "epochs": config["epochs"],
        "early_stopping": config.get("early_stopping", False), "coupling_space": latent["coupling_space"],
        "refine": latent["refine"], **{k: response[k] for k in ("lr", "weight_decay", "grad_clip",
            "conditions_per_update", "ot_samples", "microbatch", "mmd_samples", "train_ode_steps",
            "inference_ode_steps")}, "numeric_mode": "fp32_explicit_independent_version"}
    value.update({"settings_version": SETTINGS_SCHEMA,
        "reference_variant": config.get("research", {}).get("full_protocol_response", {}).get("reference_variant", "observed")})
    producer.reference_contract(value["reference_variant"])
    require(all(type(value[k]) is int and value[k] > 0 for k in ("epochs", "conditions_per_update",
        "ot_samples", "microbatch", "mmd_samples", "train_ode_steps", "inference_ode_steps")), "INTEGER_SETTINGS")
    require(type(value["seed"]) is int and value["seed"] >= 0, "SEED")
    require(all(type(value[k]) in (float, int) and math.isfinite(value[k]) for k in
        ("lr", "weight_decay", "grad_clip")) and value["lr"] > 0
        and value["weight_decay"] >= 0 and value["grad_clip"] > 0, "OPTIMIZER_SETTINGS")
    return value


class NativeProtocolData:
    """Old exact role/control distribution plus authenticated partial protocol."""
    synthetic = False

    def __init__(self, config):
        self.config = joint_config(config)
        self.data = EngineData(self.config, roles=("T", "V"), allow_test=False)
        self.native = NativeResponseUnits(ROOT, NATIVE, allowed_roles=("fit", "selection"))
        self.requests = {}
        for role, code in (("fit", "T"), ("selection", "V")):
            rows = [r for r in self.native.requests(role) if r["measurement_level"] == "single_cell"]
            require({r["condition_id"] for r in rows} == set(self.data.condition_ids(code)), "NATIVE_ROLE_CONDITIONS")
            for row in rows:
                require(row["gene_axis_id"] == "joint_ensembl_32461" and row["unique_functional_Y"] is None,
                        "RNA_ONLY_NATIVE_DOMAIN")
                protocol = row["actual_protocol"]
                require(protocol["exposure_duration_unknown"] is True and protocol["exposure_duration_h"] is None
                    and protocol["prior_drug_unknown"] is True and protocol["washout_unknown"] is True
                    and protocol["withdrawal_unknown"] is True, "PARTIAL_PROTOCOL_SOURCE_CHANGED")
                feature = self.data.action_features(row["condition_id"])
                require(float(feature["dose"]) == row["dose_uM"]
                    and float(feature["duration"]) == protocol["duration_h"]
                    and all(feature[k] == self.data.reader.context_vocabulary[k].index(row[k])
                            for k in ("source_id", "background_id", "protocol_id")),
                    "SOURCE_ACTION_PROTOCOL_BINDING")
                self.requests[row["condition_id"]] = row
        self.identity = self.data.identity
        self.mu, self.scale, self.n_genes = self.data.mu, self.data.scale, self.data.n_genes
        self.neighbors, self.bandwidth = self.data.neighbors, self.data.bandwidth
        self.context_sizes = self.data.context_sizes
        self.protocol_identity = config_hash({"native_manifest": self.native.manifest,
            "requests": self.requests, "coverage": COVERAGE})

    def condition_ids(self, role):
        require(role in ("T", "V"), "CLOSED_ROLE")
        return self.data.condition_ids(role)

    def iter_response_epoch(self, role, seed):
        require(role == "T", "FIT_EPOCH_ONLY")
        return self.data.iter_response_epoch(role, seed)

    def batch(self, condition, n, device):
        request = self.requests[condition]
        numeric = condition_kwargs(self.data.action_features(condition), n, device)
        # Native duration remains the existing observation-time feature, not a
        # known exposure interval. Mask false is deliberately retained.
        values = torch.full((n, 1, 8), float("nan"), device=device)
        values[:, 0, 0] = request["dose_uM"]
        known = torch.zeros((n, 1, 8), dtype=torch.bool, device=device)
        known[:, 0, 0] = True
        vehicle = numeric["vehicle"][:, None]
        physical = FullPhysicalProtocolBatch(numeric["fingerprint"][:, None],
            torch.ones((n, 1), dtype=torch.bool, device=device), vehicle,
            vehicle.long(), torch.full((n, 1), DOSE_UNITS.index("uM"), dtype=torch.long, device=device),
            values, known, torch.ones(n, dtype=torch.long, device=device), numeric["duration"].clone(),
            torch.ones(n, dtype=torch.bool, device=device))
        physical.validate()
        return numeric, physical


def preflight(data, settings, *, ae_checkpoint=None, read_samples=False):
    _, _, _, bindings = producer.bound_inputs()
    require(data.identity == bindings["data_identity"], "JOINT_SERVING_DATA_IDENTITY")
    records = list(data.requests.values())
    result = {"schema": SCHEMA, "stage": producer.STAGE, "bindings": bindings,
        "settings": settings,
        "protocol_identity": data.protocol_identity, "coverage": COVERAGE,
        "roles": dict(Counter(r["role"] for r in records)),
        "by_source_role_endpoint": dict(Counter("|".join((r["source_id"], r["role"],
            str(r["actual_protocol"]["duration_h"]))) for r in records)),
        "joint_AE_required": {"checkpoint": str(ae_checkpoint) if ae_checkpoint else None,
            "exists": bool(ae_checkpoint and Path(ae_checkpoint).is_file()),
            "status": "actual_matching_trained_AE_not_yet_verified" if ae_checkpoint else "missing_matching_joint_AE",
            "legacy_interrupted_AE_reuse": False},
        "fit_ready": False, "real_model_forwards": 0, "PT_loads": 0, "optimizer_steps": 0,
        "final_RNA_enabled": False, "functional_Y_enabled": False, "scientific_qualified": False}
    for r in records:
        numeric, physical = data.batch(r["condition_id"], 1, "cpu")
        require(physical.known.sum().item() == 1 and not physical.known[..., 3:].any(), "UNKNOWN_MASK_CHANGED")
    result["actual_partial_protocol_batches_prepared"] = len(records)
    if read_samples:
        groups = {}
        for r in records:
            key = (r["source_id"], r["role"], r["actual_protocol"]["duration_h"])
            groups.setdefault(key, r)
        receipts = []
        for key, r in groups.items():
            raw = data.native.read(r["request_id"], role=r["role"],
                expected_gene_ids=data.native.axes[r["gene_axis_id"]], max_rows=2)
            from lung.data.preprocess import standardized
            arrays = {k: standardized(raw[k], data.mu, data.scale) for k in ("reference", "target")}
            require(all(x.shape == (2, data.n_genes) and np.isfinite(x).all() for x in arrays.values()),
                    "REAL_RNA_SAMPLE_DOMAIN")
            receipts.append({"source_id": key[0], "role": key[1], "observation_hours": key[2],
                "request_id": r["request_id"], "reference_ids": raw["reference_ids"],
                "target_ids": raw["target_ids"], "shape_per_side": [2, data.n_genes],
                "raw_units": raw["units"], "model_input_domain": producer.DOMAIN,
                "normalized_float32_sha256": {k: hashlib.sha256(x.tobytes()).hexdigest() for k, x in arrays.items()},
                "raw_count_nnz": {k: raw[k].nnz for k in ("reference", "target")},
                "raw_RNA_actually_read": True, "functional_Y": None, "model_execution": False})
        result["real_RNA_readback"] = receipts
    return result


class _AEMonitor:
    def __init__(self, output):
        self.experiment, self.weights = Path(output), Path(output) / "weights"

    def say(self, value):
        print(value, flush=True)

    def record(self, stage, epoch, value):
        with (self.experiment / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "epoch": epoch, **value}) + "\n")


def fit_matching_ae(config, output):
    """Explicit future B1, same old dual-end fit MSE in an independent folder."""
    config = joint_ae_config(config)
    data = EngineData(config, roles=("T", "V"), allow_test=False)
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / "experiments"), "REAL_AE_OUTPUT_PROJECT_BOUNDARY")
    manifest = {"schema": SCHEMA + ":joint-AE", "config": config, "data_identity": data.identity}
    _new_experiment(output, manifest)
    result = run_latent_ae(config, _AEMonitor(output), data)
    final_path = output / "weights/latent_ae/final.pt"
    final = load_checkpoint(final_path)
    result = {**result, "schema": AE_RESULT_SCHEMA, "stage": "latent_ae", "data_identity": data.identity,
        "training_completion": final.get("extra", {}).get("training_completion", {}),
        "checkpoint_sha256": producer.inputs.sha(result["weights"]),
        "final_checkpoint": str(final_path), "final_checkpoint_sha256": producer.inputs.sha(final_path)}
    verify_completed_ae_result(result, output, data.identity)
    atomic_json(output / "AE_result.json", result)
    return result


def verify_completed_ae_result(result, output, data_identity):
    require(result.get("schema") == AE_RESULT_SCHEMA and result.get("stage") == "latent_ae"
        and result.get("data_identity") == data_identity
        and result.get("training_completion", {}).get("status") == "complete", "COMPLETED_AE_RESULT_PROOF")
    output = Path(output).resolve()
    selected, final = Path(result["weights"]).resolve(), Path(result["final_checkpoint"]).resolve()
    require(selected in (output / "weights/latent_ae/best.pt", output / "weights/latent_ae/final.pt")
        and final == output / "weights/latent_ae/final.pt"
        and producer.inputs.sha(selected) == result.get("checkpoint_sha256")
        and producer.inputs.sha(final) == result.get("final_checkpoint_sha256"), "COMPLETED_AE_CHECKPOINT_CHANGED")


def _new_experiment(output, manifest):
    require(not output.exists(), "NEW_EXPERIMENT_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    atomic_json(output / "run_manifest.json", manifest)


def export_matching_ae(ae_path, data, output, bindings, config):
    """Trusted project B1 checkpoint -> safe inference weights, identical tensors.

    Context cardinalities do not occur in SharedAutoencoder tensors. They are
    expanded explicitly for the new condition/P vocabulary, not inferred from
    the old response model, and the source shape is retained in provenance.
    """
    ae_path = Path(ae_path).resolve()
    require(data.synthetic is False and data.identity == bindings["data_identity"], "REAL_JOINT_AE_EXPORT_REQUIRED")
    require(ae_path.is_relative_to(ROOT / "experiments"), "AE_SOURCE_PROJECT_BOUNDARY")
    origin = ae_path.parents[2]
    declaration = json.loads((origin / "AE_result.json").read_text(encoding="utf-8"))
    verify_completed_ae_result(declaration, origin, data.identity)
    manifest = json.loads((origin / "run_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == SCHEMA + ":joint-AE" and manifest.get("config") == joint_ae_config(config)
        and declaration.get("data_identity") == data.identity
        and Path(declaration.get("weights", "")).resolve() == ae_path
        and declaration.get("checkpoint_sha256") == producer.inputs.sha(ae_path)
        and declaration.get("training_completion", {}).get("status") == "complete", "REGISTERED_NEW_JOINT_AE_REQUIRED")
    saved = load_checkpoint(ae_path)
    require(saved.get("identity") == declaration.get("identity") and isinstance(saved.get("identity"), str)
        and bool(saved["identity"]), "AE_SOURCE_RUN_IDENTITY")
    meta, steps = saved.get("metadata", {}), saved.get("counters", {}).get("optimizer_step")
    source_shape = ModelConfig(**meta["model_config"])
    expected = _model_config(config, latent_settings(config), data.n_genes)
    require(meta.get("stage") == "latent_ae" and meta.get("data_identity") == data.identity
        and asdict(source_shape) == asdict(expected), "MATCHING_JOINT_AE_DOMAIN_SHAPE")
    require(type(steps) is int and steps > 0 and type(saved.get("epoch")) is int and saved["epoch"] > 0
        and saved.get("extra", {}).get("training_completion", {}).get("status") == "complete", "MATCHING_COMPLETED_AE_REQUIRED")
    producer.finite_weights(saved.get("model"))
    shape = replace(source_shape, **data.context_sizes)
    shared = SharedAutoencoder(shape, tensor(data.neighbors, "cpu", torch.long))
    shared.load_state_dict(saved["model"], strict=True)
    require(all(producer.equal_weight(t, saved["model"][k]) for k, t in shared.state_dict().items()), "AE_TENSORS_CHANGED")
    exported = {"schema_version": 1, "model": saved["model"], "epoch": saved["epoch"],
        "counters": {"optimizer_step": steps}, "metadata": {"stage": "latent_ae", "synthetic": False,
        "model_config": asdict(shape), "data_identity": data.identity, "full_response_bindings": bindings,
        "source_checkpoint_sha256": producer.inputs.sha(ae_path), "source_model_config": asdict(source_shape),
        "context_cardinality_expansion_changes_no_AE_tensor": True}}
    path = Path(output) / "representation_weights.pt"
    declaration_path = Path(output) / "representation_declaration.json"
    if path.exists():
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        require(producer.inputs.sha(path) == declaration["checkpoint_sha256"], "EXPORTED_AE_CHANGED")
        previous = torch.load(path, weights_only=True, map_location="cpu")
        require(previous["metadata"] == exported["metadata"] and all(
            producer.equal_weight(t, previous["model"].get(k)) for k, t in exported["model"].items()), "EXPORTED_AE_BINDING_CHANGED")
    else:
        atomic_save(path, exported)
        atomic_json(declaration_path, {"checkpoint": str(path), "checkpoint_sha256": producer.inputs.sha(path),
            "optimizer_step": steps, "data_identity": data.identity, "stage": "latent_ae", "synthetic": False})
    shared.full_response_bindings = bindings
    shared.representation_hash = producer.inputs.sha(path)
    return shared.eval().requires_grad_(False), path, steps


def _coupled(data, shared, cache, condition, n, rng, device, space):
    if cache is not None:
        return _couple(condition, data.data, cache, shared, n, rng, device, space)[:2]
    require(data.synthetic, "REAL_FROZEN_REPRESENTATION_CACHE_REQUIRED")
    drawn = data.sample_response(condition, n, rng, role="T")
    with torch.no_grad():
        h = shared.encode_h(tensor(drawn["control"], device))[1]
        target = shared.project(shared.encode_h(tensor(drawn["target"], device))[1])
        source = shared.project(h)
    coupling = entropic_ot(source, target) if space == "latent" else entropic_ot(
        tensor(drawn["control"], device), tensor(drawn["target"], device))
    ii, jj = sample_ot_pairs(coupling, n_pairs=n)
    return h[ii], target[jj].detach()


def _endpoint_batch(data, cache, shared, condition, n, rng, role, device):
    drawn = (data.data if cache is not None else data).sample_response(condition, n, rng, role=role)
    target = tensor(drawn["target"], device)
    with torch.no_grad():
        h = cache.lookup("fit" if role == "T" else "selection", np.asarray(drawn["control_cell_ids"])) \
            if cache is not None else shared.encode_h(tensor(drawn["control"], device))[1]
    return h, target


def _validation(full, data, cache, settings, device):
    seed_all(20260923 + 999)
    rng, values = np.random.default_rng(20260923 + 999), []
    generator = torch.Generator(device=device).manual_seed(20260923 + 999)
    with torch.no_grad():
        for condition in data.condition_ids("V"):
            h, target = _endpoint_batch(data, cache, full.member.shared, condition, settings["mmd_samples"], rng, "V", device)
            h = producer.apply_reference_variant(h, settings["reference_variant"])
            numeric, physical = data.batch(condition, len(h), device)
            noise = torch.randn((len(h), full.member.config.latent_dim), device=device, generator=generator)
            z = full.sample_latent(h, physical, noise=noise, steps=settings["inference_ode_steps"], **numeric)
            generated = projected(full.member.decode(z), data)
            values.append(float(rbf_mmd(generated, target, bandwidth=data.bandwidth)
                + (generated.mean(0) - target.mean(0)).square().mean()))
    require(values and all(math.isfinite(v) for v in values), "SELECTION_NONFINITE")
    return float(np.mean(values))


def fit_response(data, shared, settings, output, *, bindings, representation_path,
                 cache=None, synthetic=False):
    """Explicit new version; B3/B4 objectives and old role sampling are retained."""
    require(data.synthetic is synthetic and type(synthetic) is bool, "SYNTHETIC_IDENTITY")
    require(settings.get("settings_version", SETTINGS_SCHEMA) == SETTINGS_SCHEMA, "REFERENCE_SETTINGS_VERSION")
    settings = {**settings, "settings_version": SETTINGS_SCHEMA, "reference_variant": settings.get("reference_variant", "observed")}
    reference = producer.reference_contract(settings["reference_variant"])
    output = Path(output).resolve()
    if not synthetic:
        require(output.is_relative_to(ROOT / "experiments"), "REAL_RESPONSE_OUTPUT_PROJECT_BOUNDARY")
        require(cache is not None and getattr(shared, "full_response_bindings", None) == bindings
            and getattr(shared, "representation_hash", None) == producer.inputs.sha(representation_path),
            "MATCHING_EXPORTED_AE_AND_FROZEN_CACHE_REQUIRED")
    shape = shared.config
    metadata = {"stage": producer.STAGE, "model_config": asdict(shape), "data_identity": data.identity,
        "synthetic": synthetic, "feature_schema": FEATURE_SCHEMA, "parameter_identity": PARAMETER_IDENTITY,
        "full_response_bindings": bindings, "representation_hash": producer.inputs.sha(representation_path),
        "training_protocol_coverage": COVERAGE, "coupling_space": settings["coupling_space"],
        "numeric_mode": settings["numeric_mode"], "reference_contract": reference}
    manifest = {"schema": SCHEMA, "metadata": metadata, "protocol_identity": data.protocol_identity,
        "settings": settings,
        "cache_encoder": cache.manifest["encoder"] if cache is not None else "synthetic_only_no_cache"}
    _new_experiment(output, manifest)
    device = settings["device"]
    identity = config_hash(manifest)
    stop = _EarlyStop(settings)
    seed_all(settings["seed"])
    shared.to(device).eval().requires_grad_(False)
    full = FullProtocolLatentAdapter(LatentMember(shape, shared), FullProtocolConditioner(shape)).to(device)
    full.member.init_decoder_from_shared()
    full.member.decoder.requires_grad_(False)
    best, best_step, best_epoch = math.inf, 0, 0
    refine = settings["refine"]
    switch = int(refine.get("switch_epoch", 10**9))
    params = list(full.member.velocity_model.parameters()) + list(full.member.condition_encoder.parameters()) + list(full.full_protocol.parameters())
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=settings["weight_decay"])
    frozen = {k: v.detach().cpu().clone() for k, v in shared.state_dict().items()}
    for epoch in range(settings["epochs"]):
        active = bool(refine.get("enabled", False)) and epoch + 1 > switch
        if active and not full.member.decoder[0].weight.requires_grad:
            full.member.decoder.requires_grad_(True)
            optimizer.add_param_group({"params": list(full.member.decoder.parameters()), "lr": optimizer.param_groups[0]["lr"]})
            best = math.inf
            stop.reset()
        rng = np.random.default_rng(settings["seed"] + 200000 + epoch)
        generator = torch.Generator(device=device).manual_seed(settings["seed"] + 200000 + epoch)
        keys = list(data.iter_response_epoch("T", settings["seed"] + 200000 + epoch))
        group_size = settings["conditions_per_update"]
        full.train()
        for start in range(0, len(keys), group_size):
            group = keys[start:start + group_size]
            optimizer.zero_grad(set_to_none=True)
            for condition in group:
                h, z = _coupled(data, shared, cache, condition, settings["ot_samples"], rng, device, settings["coupling_space"])
                h = producer.apply_reference_variant(h, settings["reference_variant"])
                for low in range(0, len(z), settings["microbatch"]):
                    high = min(low + settings["microbatch"], len(z))
                    numeric, physical = data.batch(condition, high-low, device)
                    loss = full(z[low:high], h[low:high], physical, **numeric)
                    (loss * (high-low) / len(z) / len(group)).backward()
                if active:
                    from lung.models.flow import heun_integrate
                    h0, y = _endpoint_batch(data, cache, shared, condition, settings["mmd_samples"], rng, "T", device)
                    h0 = producer.apply_reference_variant(h0, settings["reference_variant"])
                    numeric, physical = data.batch(condition, len(h0), device)
                    ca = full.encode_condition(physical, **numeric)
                    noise = torch.randn((len(h0), shape.latent_dim), device=device, generator=generator)
                    latent = heun_integrate(lambda state, tau: full.member.velocity(state, tau, h0, ca),
                        noise, settings["train_ode_steps"], checkpoint=True)
                    loss = rbf_mmd(full.member.decode(latent), y, bandwidth=data.bandwidth, unbiased=False)
                    (loss * float(refine.get("weight", 1.0)) / len(group)).backward()
            torch.nn.utils.clip_grad_norm_([p for p in full.parameters() if p.requires_grad],
                settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            stop.step_done()
        full.eval()
        score = _validation(full, data, cache, settings, device)
        if stop.observe(score):
            best, best_step, best_epoch = score, stop.steps, epoch+1
            serving = {"schema_version": 1, "model": full.state_dict(), "shared": frozen,
                "metadata": metadata, "epoch": epoch+1, "counters": {"optimizer_step": stop.steps}}
            atomic_save(output / "response_weights.pt", serving)
        if stop.stopped:
            break
    atomic_save(output / "final.pt", {"schema_version": 1, "identity": identity, "model": full.state_dict(),
        "shared": frozen, "epoch": epoch+1, "metadata": metadata,
        "counters": {"optimizer_step": stop.steps}, "extra": {"training_completion": {"status": "complete"}}})
    return {"schema": SCHEMA, "weights": str(output / "response_weights.pt"), "optimizer_steps": stop.steps,
        "selected_optimizer_step": best_step, "selected_epoch": best_epoch,
        "best_selection_loss": best, "synthetic": metadata["synthetic"], "coverage": COVERAGE,
        "reference_contract": metadata["reference_contract"],
        "production_registered": False, "scientific_qualified": False}
