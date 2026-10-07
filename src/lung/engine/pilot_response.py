"""Frozen five-member response ensembles for the V-only development report."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lung.models import ModelConfig, physical_projection
from lung.models.objectives import kernel_sum, shared_control_mmd
from .checkpoint import load_checkpoint
from .config import ROOT, config_hash, file_hash, read_yaml
from .pilot import METRICS, require_v_only, iter_plan_batches, distribution_metrics
from .response import RESPONSE_RNG_POLICY, TrainingDetailProgress, condition_kwargs, load_response, tensor
from .inference_noise import response_noise


METHOD_WEIGHTS = {"P3_cfm_mmd": 1., "P4_cfm_only": 0.}
NOISE_POLICY = "shared_control_block_noise_v1"


def inspect_response_set(paths, data, method, *, root=ROOT):
    """Validate real checkpoint content and its registered run, not file labels."""
    require_v_only(data)
    if method not in METHOD_WEIGHTS or len(paths) != 5:
        raise ValueError("FIVE_RESPONSE_MEMBERS_REQUIRED")
    paths = [Path(path).resolve() for path in paths]
    if len(set(paths)) != 5:
        raise ValueError("FIVE_UNIQUE_CHECKPOINT_PATHS_REQUIRED")
    run_directories = {path.parents[2] for path in paths}
    if len(run_directories) != 1:
        raise ValueError("RESPONSE_MEMBERS_MUST_SHARE_RUN")
    run_directory = next(iter(run_directories))
    if run_directory.parent != (Path(root) / "checkpoints").resolve():
        raise ValueError("REGISTERED_CHECKPOINT_RUN_REQUIRED")
    experiment = Path(root) / "experiments" / run_directory.name
    run = json.loads((experiment / "run_manifest.json").read_text(encoding="utf-8"))
    binding = json.loads((experiment / "data_manifest.json").read_text(encoding="utf-8"))
    config_path = experiment / "active_config.yaml"
    config = read_yaml(config_path if config_path.exists() else experiment / "resolved_config.yaml")
    if (run.get("run_id") != run_directory.name or binding.get("identity") != data.identity
            or run.get("synthetic", False) != data.synthetic
            or config.get("synthetic", False) != data.synthetic):
        raise ValueError("RESPONSE_RUN_DATA_BINDING_MISMATCH")
    if run.get("config_hash") != config_hash(config):
        raise ValueError("RESPONSE_RUN_CONFIG_HASH_MISMATCH")
    members, common_identity, common_cfg = [], None, None
    for path in paths:
        digest = file_hash(path)
        saved = load_checkpoint(path)
        metadata = saved["metadata"]
        member, cfg = metadata.get("member"), metadata.get("response_config", {})
        if (type(member) is not int or member not in range(5)
                or path.parent.name != f"member_{member+1}" or path.parent.parent.name != "response"):
            raise ValueError("RESPONSE_MEMBER_INDEX_MISMATCH")
        if (metadata.get("stage") != "response" or saved.get("epoch", 0) < 1
                or metadata.get("train_role") != "T" or metadata.get("selection_role") != "V"
                or metadata.get("data_identity") != data.identity or metadata.get("synthetic", False) != data.synthetic):
            raise ValueError("TRAINED_T_V_RESPONSE_REQUIRED")
        if (metadata.get("rng_policy") != RESPONSE_RNG_POLICY
                or cfg.get("mmd_weight") != METHOD_WEIGHTS[method]
                or metadata.get("mmd_training") != ("computed" if METHOD_WEIGHTS[method] else "not_computed")):
            raise ValueError("RESPONSE_OBJECTIVE_OR_RNG_POLICY_MISMATCH")
        if cfg.get("members") != 5 or cfg.get("model", {}).get("n_genes") != data.n_genes:
            raise ValueError("RESPONSE_FULL_GENE_FIVE_MEMBER_MODEL_REQUIRED")
        if metadata.get("model_config") != asdict(ModelConfig(**cfg["model"])):
            raise ValueError("RESPONSE_MODEL_CONFIG_MISMATCH")
        expected = config_hash({"data": data.identity, "response": cfg, "task": "RESPONSE-A549-V1",
                                "rng_policy": RESPONSE_RNG_POLICY})
        run_cfg = {**config["response"], "mmd_weight": float(config["response"].get("mmd_weight", 1.))}
        if saved.get("identity") != expected or cfg != run_cfg or (
                common_identity is not None and expected != common_identity):
            raise ValueError("RESPONSE_CHECKPOINT_IDENTITY_MISMATCH")
        for name, value in saved["model"].items():
            if name.endswith("neighbors") and not np.array_equal(value.cpu().numpy(), data.neighbors):
                raise ValueError("RESPONSE_FROZEN_GENE_GRAPH_MISMATCH")
        if file_hash(path) != digest:
            raise ValueError("RESPONSE_CHECKPOINT_CHANGED_DURING_READ")
        common_identity, common_cfg = expected, cfg
        members.append({"member": member, "path": str(path), "sha256": digest,
                        "epoch": saved["epoch"], "initialization_seed": cfg["seed"] + member})
        del saved
    if sorted(item["member"] for item in members) != list(range(5)):
        raise ValueError("RESPONSE_MEMBERS_MUST_BE_ZERO_TO_FOUR")
    steps = common_cfg["inference_ode_steps"]
    micro = common_cfg.get("inference_microbatch", common_cfg.get("mmd_microbatch", 1))
    if type(steps) is not int or steps < (1 if data.synthetic else 128) or type(micro) is not int or micro < 1:
        raise ValueError("FULL_HEUN_PRECISION_AND_POSITIVE_MICROBATCH_REQUIRED")
    ordered = sorted(members, key=lambda item: item["member"])
    audit_path = experiment / "ode_audit.json"
    audit = {"status": "not_attached_checkpoint_precision_only"}
    if audit_path.exists():
        record = json.loads(audit_path.read_text(encoding="utf-8"))
        expected = config_hash({"response": [member["sha256"] for member in ordered], "data": data.identity})
        if record.get("identity") != expected or record.get("passed") is not True:
            raise ValueError("RESPONSE_ODE_AUDIT_IDENTITY_OR_PASS_MISMATCH")
        if type(record.get("selected_steps")) is not int or record["selected_steps"] < 1:
            raise ValueError("INVALID_ODE_AUDIT_STEPS")
        steps = max(steps, record["selected_steps"])
        audit = {"status": "matching_V_audit_passed", "path": str(audit_path), "sha256": file_hash(audit_path)}
    return {"method": method, "run_id": run_directory.name, "checkpoint_identity": common_identity,
            "data_identity": data.identity, "gene_axis_sha256": config_hash(data.reader.gene_ids.tolist()),
            "response_config": common_cfg, "members": ordered, "inference_steps": steps,
            "inference_microbatch": micro, "ode_audit": audit,
            "run_manifest_sha256": file_hash(experiment / "run_manifest.json")}


def shared_inference_steps(sets):
    if len(sets) > 1:
        reference = {key: value for key, value in sets[0]["response_config"].items() if key != "mmd_weight"}
        for selected in sets[1:]:
            if {key: value for key, value in selected["response_config"].items() if key != "mmd_weight"} != reference:
                raise ValueError("PAIRED_CFM_CONFIG_MISMATCH: only MMD weight may differ")
    return max(selected["inference_steps"] for selected in sets) if sets else None


def noise_seed(sample):
    return int(config_hash([NOISE_POLICY, sample["condition_id"], sample["seed"]])[:16], 16) % (2**63 - 1)


@torch.no_grad()
def mixture_metrics(raw_members, control, target, mean, scale, bandwidth, *, device="cpu", block_size=128, progress=None,
                    population_dependent=False):
    """Equal-member distribution; U kernel excludes every same-control block pair."""
    if len(raw_members) != 5 or any(tuple(value.shape) != tuple(control.shape) for value in raw_members):
        raise ValueError("FIVE_EQUAL_SIZE_FULL_GENE_POPULATIONS_REQUIRED")
    result = {**dict.fromkeys(METRICS), "not_applicable": ["delta_correlation", "direction_agreement", "variance_rmse"],
              "mmd_estimator": "equal_member_shared_control_block_U", "na_reasons": {},
              "output_validity_definition": "per_cell_physical_projection"}
    n, m = len(control), len(target)
    if not n or not m:
        return {**result, "status": "missing_population", "na_reasons": {
            key: "missing_population" for key in ("mean_rmse", "mmd_u", "output_valid_rate")}}
    y, mu, sd = (torch.as_tensor(value, dtype=torch.float64, device=device) for value in (target, mean, scale))
    projected = [physical_projection(torch.as_tensor(value, dtype=torch.float64, device=device), mu, sd)
                 for value in raw_members]
    result["output_valid_rate"] = float(torch.cat([part.valid for part in projected]).double().mean())
    if not torch.isfinite(y).all() or not all(bool(part.valid.all()) for part in projected):
        return {**result, "status": "invalid_population", "na_reasons": {
            key: "invalid_member_or_observation_no_dropping" for key in ("mean_rmse", "mmd_u")}}
    populations = [part.x for part in projected]
    population_mean = torch.stack([part.mean(0) for part in populations]).mean(0)
    result["mean_rmse"] = float((population_mean - y.mean(0)).square().mean().sqrt())
    if population_dependent:
        pooled = torch.cat(populations)
        options = {"bandwidth": bandwidth, "block_size": block_size}
        empirical = (kernel_sum(pooled, pooled, **options) / len(pooled)**2
                     + kernel_sum(y, y, **options) / m**2
                     - 2*kernel_sum(pooled, y, **options) / (len(pooled)*m))
        return {**result, "mmd_estimator": "empirical_population_V",
                "empirical_population_mmd_v": float(empirical), "status": "partial_metrics",
                "na_reasons": {"mmd_u": "population_dependent_reference"}}
    if min(n, m) < 2:
        return {**result, "status": "partial_metrics", "na_reasons": {"mmd_u": "fewer_than_two_unique_control_or_observed_blocks"}}
    result["mmd_u"] = float(shared_control_mmd(populations, y, bandwidth,
                                              block_size=block_size, progress=progress))
    return {**result, "status": "ok"}


@torch.no_grad()
def evaluate_response_sets(data, plan, sets, *, device="cpu", block_size=128, progress=None):
    require_v_only(data)
    steps = shared_inference_steps(sets)
    mixtures, member_rows = [], []
    for selected in sets:
        models = []
        for member in selected["members"]:
            if file_hash(member["path"]) != member["sha256"]:
                raise ValueError("RESPONSE_CHECKPOINT_CHANGED")
            models.append(load_response(member["path"], data, "cpu"))
            if file_hash(member["path"]) != member["sha256"]:
                raise ValueError("RESPONSE_CHECKPOINT_CHANGED")
        batches = iter_plan_batches(data, plan)
        bar = progress(batches, total=len(plan["samples"]), desc=f"V {selected['method']}") if progress else batches
        detail = TrainingDetailProgress(bar) if progress else None
        for sample, control, target in bar:
            seed = noise_seed(sample)
            if sample.get("response_noise_seed", seed) != seed:
                raise ValueError("RESPONSE_NOISE_PLAN_MISMATCH")
            noise = response_noise(models[0], torch.as_tensor(control, dtype=torch.float32),
                                   torch.Generator().manual_seed(seed))
            base = {key: sample[key] for key in ("condition_id", "group_id", "seed", "role",
                                                "target_unique_n", "control_unique_n", "replacement")}
            base = {**base, "method": selected["method"], "run_id": selected["run_id"],
                    "noise_seed": seed, "noise_policy": NOISE_POLICY, "noise_dimensions": noise.shape[-1],
                    "inference_steps": steps,
                    "control_summary_scope": "sampled_V_controls", "n_blocks": len(control)}
            predictions = []
            population_dependent = selected["response_config"]["model"].get("reference_mode", "observed") == "population_mean"
            features = data.action_features(sample["condition_id"])
            for record, model in zip(selected["members"], models):
                chunks = []
                label = f"{sample['condition_id']} seed={sample['seed']} member={record['member']}"
                if detail:
                    detail(label + " load", 0, 1)
                try:
                    model.to(device)
                    micro = selected["inference_microbatch"]
                    reference = tensor(control, device)
                    if hasattr(model, "condition_reference"):
                        reference = model.condition_reference(reference)
                    for start in range(0, len(control), micro):
                        x = reference[start:start+micro]
                        condition = model.encode_condition(**condition_kwargs(features, len(x), device))
                        raw = model.sample(x, condition, noise=noise[start:start+micro].to(device),
                                           steps=steps, checkpoint=False,
                                           progress=(lambda current, total: detail(
                                               label + f" cells={start+1}:{start+len(x)} ODE", current, total)) if detail else None)
                        chunks.append(raw.cpu())
                finally:
                    model.cpu()
                prediction = torch.cat(chunks) if chunks else torch.empty(control.shape)
                predictions.append(prediction)
                metrics = distribution_metrics(prediction, control, target, data.mu, data.scale, data.bandwidth, device=device)
                if population_dependent:
                    metrics = {**metrics, "mmd_u": None, "mmd_estimator": "population_dependent_reference",
                               "na_reasons": {**metrics.get("na_reasons", {}), "mmd_u": "population_dependent_reference"}}
                member_rows.append({**base, "member": record["member"], "checkpoint_sha256": record["sha256"],
                    "checkpoint_epoch": record["epoch"], "initialization_seed": record["initialization_seed"],
                    "generated_n": len(control), "mmd_estimator": "single_member_standard_U",
                    **metrics})
            mixtures.append({**base, "member_ids": [record["member"] for record in selected["members"]],
                "member_weights": [.2]*5, "generated_n": 5*len(control),
                "correlated_generated_outputs": 5*len(control),
                "mixture": "equal_member_distributions_not_mean_cells", "kernel_block_size": block_size,
                **mixture_metrics(predictions, control, target, data.mu, data.scale, data.bandwidth,
                                  device=device, block_size=block_size, progress=detail,
                                  population_dependent=population_dependent)})
        if progress:
            bar.close()
        del models
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        for member in selected["members"]:
            if file_hash(member["path"]) != member["sha256"]:
                raise ValueError("RESPONSE_CHECKPOINT_CHANGED")
    return mixtures, member_rows
