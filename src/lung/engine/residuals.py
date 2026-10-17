"""R-only backward and correction fits with independent G-domain selection."""

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lung.models import BackwardModel, CorrectionModel, group_expression_error
from .auxiliary import (fit_epochs, encode_vectors, load_auxiliary, pipeline_identity,
                        dependency_record)
from .checkpoint import atomic_save, atomic_json, load_checkpoint, seed_all
from .config import config_hash, file_hash, StageUnavailable
from .response import tensor, condition_kwargs, projected
from .response_loader import load_response
from .representation import load_representation, verify_response_representation


def residual_dependencies(pipeline):
    if not pipeline.get("pool"):
        raise StageUnavailable("TRAINED_POOL_REQUIRED")
    keys = ("representation", "response", "pool") if pipeline.get("representation") else ("response", "pool")
    return {key: pipeline[key] for key in keys}


def gate_for(table, batch):
    """Unknown source/background/protocol domains are exactly identity K."""
    domain = f"{batch['source_id']}|{batch['background_id']}|{batch['protocol_id']}"
    return bool(table.get("domains", {}).get(domain, {}).get("enabled", False))


def prediction_rows(role, data, config, monitor, pipeline):
    """Individual condition files avoid holding all expression matrices in RAM."""
    deps = residual_dependencies(pipeline)
    steps = pipeline.get("inference_ode_steps", config["response"]["inference_ode_steps"])
    effective_config = {**config, "response": {**config["response"], "inference_ode_steps": steps}}
    identity = pipeline_identity("residual_predictions", data, effective_config, deps)
    device = torch.device(config["device"])
    reference = load_response(pipeline["response"][0], data, device)
    representation = (load_representation(pipeline["representation"], data, device)
                      if pipeline.get("representation") else reference)
    if hasattr(reference, "representation_identity") and not pipeline.get("representation"):
        raise StageUnavailable("LATENT_REPRESENTATION_REQUIRED")
    if pipeline.get("representation"):
        verify_response_representation(reference, representation)
    pool = load_auxiliary(pipeline["pool"], data, device, pipeline=pipeline)
    rows = []
    for key in monitor.progress(data.condition_ids(role), desc=f"cache residual {role}"):
        # Short path components for Windows; full identities remain verified inside.
        filename = hashlib.sha256(str(key).encode()).hexdigest()[:24] + ".pt"
        path = monitor.experiment / "cache" / f"residual-{identity[:20]}" / role / filename
        if not path.exists():
            seed = int(hashlib.sha256((identity + str(key)).encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            batch = data.sample_response(key, config.get("population", 512), rng, role=role)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(seed)
                raw_members, means = [], []
                with torch.no_grad():
                    for member_index, checkpoint in enumerate(pipeline["response"]):
                        member = reference if member_index == 0 else load_response(checkpoint, data, device)
                        if pipeline.get("representation"):
                            verify_response_representation(member, representation)
                        generated = []
                        condition = member.encode_condition(**condition_kwargs(batch, 1, device))
                        for cell in range(len(batch["control"])):
                            source = tensor(batch["control"][cell:cell+1], device)
                            raw = member.sample(source, condition, steps=steps, checkpoint=False)
                            generated.append(raw.cpu())
                        raw = torch.cat(generated)
                        means.append(projected(raw.to(device), data).mean(0).cpu())
                        raw_members.append(raw)
                        if member_index:
                            del member
                    raw = torch.cat(raw_members)
                    valid = projected(raw.to(device), data)
                    vectors = encode_vectors(representation, valid, device, output_device=device)
                    observed = tensor(batch["target"], device)
                    observed_vectors = encode_vectors(representation, observed, device, output_device=device)
                    record = {"raw": raw, "observed": observed.cpu(),
                              "population": pool(vectors).cpu(), "observed_population": pool(observed_vectors).cpu(),
                              "condition": reference.encode_condition(**condition_kwargs(batch, 1, device))[0].cpu(),
                              "reference_mean": tensor(batch["control"], device).mean(0).cpu(),
                              "ensemble_variance": torch.stack(means).var(0, correction=0).mean(),
                              "condition_id": key, "group_id": str(batch["group_id"]),
                              "domain": f"{batch['source_id']}|{batch['background_id']}|{batch['protocol_id']}",
                              "role": role, "cache_identity": identity}
                atomic_save(path, record)
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record["role"] != role or record["cache_identity"] != identity or record["condition_id"] != key:
            raise ValueError("RESIDUAL_CACHE_ROLE_OR_IDENTITY_MISMATCH")
        compact_path = compact_prediction(path, record)
        rows.append({"path": str(path), "compact_path": str(compact_path),
                     "group_id": record["group_id"], "domain": record["domain"], "condition_id": key})
    del reference, representation, pool
    return rows


BACKWARD_FIELDS = ("observed_population", "condition", "reference_mean")


def compact_prediction(path, record):
    """Atomically derive B-only fields, bound to the full immutable source bytes."""
    path = Path(path)
    compact_path = path.with_suffix(".backward.pt")
    source_hash = file_hash(path)
    if compact_path.exists():
        saved = torch.load(compact_path, map_location="cpu", weights_only=False)
        if saved.get("source_sha256") == source_hash:
            return compact_path
    atomic_save(compact_path, {"source_sha256": source_hash,
                              "fields": {key: record[key].clone() for key in BACKWARD_FIELDS}})
    return compact_path


class BackwardPredictionCache:
    """Fit-local bounded CPU cache; expression populations never enter this cache."""
    def __init__(self, max_bytes=64 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.nbytes = 0

    def read(self, descriptor, device):
        path = Path(descriptor.get("compact_path", descriptor["path"]))
        stat = path.stat()
        source_stat = Path(descriptor["path"]).stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
               source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ctime_ns)
        if key in self.entries:
            fields, size = self.entries.pop(key)
            self.entries[key] = (fields, size)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if "compact_path" in descriptor and saved.get("source_sha256") != file_hash(descriptor["path"]):
                raise ValueError("RESIDUAL_COMPACT_SOURCE_MISMATCH")
            source = saved["fields"] if "compact_path" in descriptor else saved
            fields = {name: source[name].clone() for name in BACKWARD_FIELDS}
            size = sum(value.numel() * value.element_size() for value in fields.values())
            while self.entries and self.nbytes + size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.nbytes -= removed
            if size <= self.max_bytes:
                self.entries[key] = (fields, size)
                self.nbytes += size
        return {name: value.to(device) for name, value in fields.items()}


def read_prediction(row, device):
    result = torch.load(row["path"], map_location="cpu", weights_only=False)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in result.items()}


def fit_backward(rows, data, config, monitor, pipeline):
    stage = "backward"
    device = torch.device(config["device"])
    shape = load_checkpoint(pipeline["response"][0])["metadata"]["model_config"]
    identity = config_hash({"base": pipeline_identity(stage, data, config, residual_dependencies(pipeline)),
                            "groups": sorted({row["group_id"] for row in rows})})
    seed_all(config.get("seed", 20260923))
    model = BackwardModel(shape["n_genes"], shape["d_model"]).to(device)
    cache = BackwardPredictionCache()

    def loss(network, descriptor):
        row = cache.read(descriptor, device)
        return F.mse_loss(network(row["observed_population"], row["condition"]), row["reference_mean"])

    path = fit_epochs(stage, model, rows, loss, data, config, monitor, identity,
                      {"width": shape["d_model"], "n_genes": shape["n_genes"], "train_role": "R",
                       "reference_response_hash": file_hash(pipeline["response"][0]), "pool_hash": file_hash(pipeline["pool"]),
                       "dependencies": dependency_record(pipeline, tuple(residual_dependencies(pipeline))),
                       "fit_groups": sorted({row["group_id"] for row in rows})})
    return model.eval().requires_grad_(False), path


def fit_correction(rows, backward, backward_path, data, config, monitor, pipeline):
    stage = "correction"
    device = torch.device(config["device"])
    shape = load_checkpoint(pipeline["response"][0])["metadata"]["model_config"]
    deps = {**residual_dependencies(pipeline), "backward": backward_path}
    identity = config_hash({"base": pipeline_identity(stage, data, config, deps),
                            "groups": sorted({row["group_id"] for row in rows})})
    seed_all(config.get("seed", 20260923))
    model = CorrectionModel(shape["n_genes"], shape["d_model"]).to(device)

    def loss(network, descriptor):
        row = read_prediction(descriptor, device)
        e_b = backward.diagnostic(row["population"], row["condition"], row["reference_mean"])
        corrected = network(row["raw"], row["population"], row["condition"], e_b,
                            row["ensemble_variance"], training_candidate=True)
        return group_expression_error(projected(corrected, data), row["observed"], data.bandwidth)

    path = fit_epochs(stage, model, rows, loss, data, config, monitor, identity,
                      {"width": shape["d_model"], "n_genes": shape["n_genes"], "train_role": "R",
                       "reference_response_hash": file_hash(pipeline["response"][0]), "pool_hash": file_hash(pipeline["pool"]),
                       "dependencies": dependency_record({**pipeline, "backward": backward_path},
                                                         (*residual_dependencies(pipeline), "backward")),
                       "fit_groups": sorted({row["group_id"] for row in rows}), "gate_during_fit": 1})
    return model.eval().requires_grad_(False), path


@torch.no_grad()
def correction_error(row, backward, correction, gate, data):
    e_b = backward.diagnostic(row["population"], row["condition"], row["reference_mean"])
    output = correction(row["raw"], row["population"], row["condition"], e_b,
                        row["ensemble_variance"], gate=gate)
    output = projected(output, data)
    return group_expression_error(output, row["observed"], data.bandwidth)


def domain_gate(rows, backward, correction, data, config):
    device = torch.device(config["device"])
    domains = {}
    for descriptor in rows:
        row = read_prediction(descriptor, device)
        before = float(correction_error(row, backward, correction, False, data))
        after = float(correction_error(row, backward, correction, True, data))
        domains.setdefault(row["domain"], {}).setdefault(row["group_id"], []).append((before, after))
    results = {}
    for domain, groups in domains.items():
        values = np.asarray([np.mean(groups[key], axis=0) for key in sorted(groups)])
        difference = values[:, 1] - values[:, 0]
        if len(groups) < 20:
            results[domain] = {"enabled": False, "groups": len(groups), "reason": "FEWER_THAN_20_GROUPS"}
            continue
        rng = np.random.default_rng(20260923)
        bootstrap = np.asarray([rng.choice(difference, len(difference), replace=True).mean() for _ in range(10000)])
        upper = float(np.quantile(bootstrap, .975, method="linear"))
        enabled = upper < 0 and difference.mean() <= -.01 * values[:, 0].mean()
        results[domain] = {"enabled": bool(enabled), "groups": len(groups), "ci_upper": upper,
                           "mean_difference": float(difference.mean()), "reason": "PASSED" if enabled else "NO_VALIDATED_IMPROVEMENT"}
    return results


def train_residual(stage, data, config, monitor, pipeline):
    rows = prediction_rows("R", data, config, monitor, pipeline)
    if stage == "backward":
        _, path = fit_backward(rows, data, config, monitor, pipeline)
        return {**pipeline, "backward": path}
    if not pipeline.get("backward"):
        raise StageUnavailable("TRAINED_BACKWARD_REQUIRED")
    backward = load_auxiliary(pipeline["backward"], data, config["device"])
    correction, path = fit_correction(rows, backward, pipeline["backward"], data, config, monitor, pipeline)
    selection = prediction_rows("G", data, config, monitor, pipeline)
    gate = {"schema_version": 1, "domains": domain_gate(selection, backward, correction, data, config),
            "data_identity": data.identity, "response_hashes": [file_hash(p) for p in pipeline["response"]],
            "pool_hash": file_hash(pipeline["pool"]), "backward_hash": file_hash(pipeline["backward"]),
            "correction_hash": file_hash(path), "selection_role": "G", "bootstrap_repeats": 10000,
            "population": config.get("population", 512),
            "inference_ode_steps": pipeline.get("inference_ode_steps", config["response"]["inference_ode_steps"])}
    gate_path = monitor.weights / "correction" / "domain_gate.json"
    atomic_json(gate_path, gate)
    return {**pipeline, "correction": path, "correction_gate": str(gate_path)}
