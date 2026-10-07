"""Prespecified V-only numerical precision checks, before C calibration and RL."""

import json
import gc
from pathlib import Path

import numpy as np
import torch

from lung.models import physical_projection
from lung.rl import stream_seed
from .checkpoint import atomic_json, atomic_save
from .config import config_hash, file_hash, StageUnavailable
from .response import tensor, condition_kwargs
from .response_loader import load_response
from .inference_noise import response_noise


def _audit_stop(monitor, stage, step, path):
    check = getattr(monitor, "stop_check", None)
    if callable(check) and check(step, 0):
        from .modular_budget import StagePaused
        raise StagePaused("budget_exhausted", str(path), stage=stage, step=step, cursor=step)


def resolve_ode_steps(data, config, monitor, pipeline):
    identity = config_hash({"response": [file_hash(p) for p in pipeline["response"]], "data": data.identity})
    if pipeline.get("ode_audit"):
        old = json.loads(Path(pipeline["ode_audit"]).read_text(encoding="utf-8"))
        if old.get("identity") == identity and old.get("passed"):
            return {**pipeline, "inference_ode_steps": old["selected_steps"]}
    if config.get("synthetic"):
        path = monitor.experiment / "ode_audit_SYNTHETIC.json"
        atomic_json(path, {"identity": identity, "passed": True, "synthetic_test_only": True,
                           "selected_steps": config["response"]["inference_ode_steps"],
                           "biological_numerical_resolution_verified": False})
        return {**pipeline, "ode_audit": str(path), "inference_ode_steps": config["response"]["inference_ode_steps"]}
    ids = data.condition_ids("V")
    if not ids:
        raise StageUnavailable("NUMERICAL_PRECISION_UNRESOLVED: no V conditions")
    rows, selected = [], 128
    cache_root = monitor.experiment / f"ode_progress-{identity[:20]}"
    for member, path in enumerate(pipeline["response"]):
        model = load_response(path, data, config["device"])
        samples = []
        rng = np.random.default_rng(20260923 + 404)
        for index in range(256):
            batch = data.sample_controls(ids[index % len(ids)], 1, rng, "V")
            samples.append(batch)
        previous = None
        for steps in (128, 256, 512, 1024):
            differences, current = [], []
            for index, batch in enumerate(monitor.progress(samples, desc=f"ODE V member {member+1}, steps {steps}")):
                _audit_stop(monitor, "ode_audit", index, cache_root)
                device = config["device"]
                cache_path = cache_root / f"member-{member}" / f"steps-{steps}" / f"sample-{index}.pt"
                if cache_path.is_file():
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if cached.get("identity") != identity or cached.get("index") != index or cached.get("steps") != steps:
                        raise StageUnavailable("ODE_AUDIT_CACHE_IDENTITY_MISMATCH")
                    output = cached["expression"]
                    if output.shape != (1, data.n_genes) or not torch.isfinite(output).all():
                        raise StageUnavailable("ODE_AUDIT_CACHE_INVALID")
                else:
                    gen = torch.Generator(device=device).manual_seed(stream_seed(20260923, "ODE", member, index))
                    noise = response_noise(model, tensor(batch["control"], device), gen)
                    with torch.no_grad():
                        raw = model.sample(tensor(batch["control"], device), model.encode_condition(
                             **condition_kwargs(batch, 1, device)), noise=noise, steps=steps, checkpoint=False)
                        result = physical_projection(raw, tensor(data.mu, device), tensor(data.scale, device))
                        if not result.valid.all():
                            raise StageUnavailable("INVALID_GENERATED_MASS in V ODE diagnostic")
                        output = result.y.cpu()
                    atomic_save(cache_path, {"identity": identity, "index": index, "steps": steps,
                                             "member": member, "expression": output})
                current.append(output)
                if previous is not None:
                    differences.append(float((output - previous[index]).abs().mean()))
            if previous is not None:
                difference = float(np.mean(differences))
                rows.append({"member": member+1, "lower": steps//2, "upper": steps, "log_expression_mae": difference})
                if difference <= 1e-3:
                    selected = max(selected, steps//2)
                    break
                if steps == 1024:
                    atomic_json(monitor.experiment / "ode_audit_failed.json", {"identity": identity, "passed": False, "rows": rows})
                    raise StageUnavailable("NUMERICAL_PRECISION_UNRESOLVED: 512/1024 V difference exceeds 1e-3")
            previous = current
        del model
    path = monitor.experiment / "ode_audit.json"
    atomic_json(path, {"identity": identity, "passed": True, "selected_steps": selected,
                       "n_fixed_V_noise": 256, "tolerance": .001, "rows": rows})
    return {**pipeline, "ode_audit": str(path), "inference_ode_steps": selected}


def population_diagnostic(world, monitor, pipeline):
    from .auxiliary import calibration_audit_binding
    if pipeline.get("freeze_certificate") and world.adapter.population_resolution_passed:
        return pipeline
    if pipeline.get("numerical_audit"):
        existing = json.loads(Path(pipeline["numerical_audit"]).read_text(encoding="utf-8"))
        if existing.get("biological_identity") == world.biological_identity and existing.get("passed"):
            return {**pipeline, "precalibration_identity": world.biological_identity}
    path = monitor.experiment / "population_resolution.json"
    if world.config.get("synthetic"):
        atomic_json(path, {"passed": True, "biological_identity": world.biological_identity,
                           "calibration_binding": calibration_audit_binding(world.data, world.config, pipeline),
                           "synthetic_test_only": True, "biological_numerical_resolution_verified": False})
        return {**pipeline, "numerical_audit": str(path), "precalibration_identity": world.biological_identity}
    roots = sorted(world.roots("V"), key=lambda r: config_hash(["population-v1", r.root_id]))[:8]
    if not roots:
        raise StageUnavailable("POPULATION_RESOLUTION_UNRESOLVED: no V roots")
    # Registration is metadata-only and happens before observing any scores.
    plans = []
    for root in roots:
        rng = np.random.default_rng(stream_seed(20260923, "prespecified", root.root_id))
        prefix = ()
        while not world.protocol.position(world.registry, prefix)[2]:
            ids = torch.nonzero(world.protocol.legal_mask(world.registry, prefix), as_tuple=False)[:, 0].tolist()
            if not ids:
                raise StageUnavailable("UNSUPPORTED_PROTOCOL in prespecified V plan")
            prefix = (*prefix, world.registry.candidates[int(rng.choice(ids))].action_id)
        plans.append((root, prefix))
    registration = {"biological_identity": world.biological_identity, "repeats": 1024,
                    "calibration_binding": calibration_audit_binding(world.data, world.config, pipeline),
                    "sizes": [512, 1024, 2048], "adjacent_tolerance": .01,
                    "plans": [{"root": r.root_id, "sequence": p} for r, p in plans]}
    registration_path = monitor.experiment / "population_preregistration.json"
    if registration_path.exists():
        if json.loads(registration_path.read_text(encoding="utf-8")) != json.loads(json.dumps(registration)):
            raise StageUnavailable("NUMERICAL_AUDIT_REGISTRATION_CHANGED")
    else:
        atomic_json(registration_path, registration)
    results = []
    for root, sequence in plans:
        cache = monitor.experiment / f"population-{config_hash(root.root_id)[:16]}.json"
        values = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
        for repeat in monitor.progress(range(len(values), 1024), desc=f"V population {root.root_id}"):
            _audit_stop(monitor, "assemble", repeat, cache)
            seed = stream_seed(20260923, "population-size", root.root_id, repeat)
            values.append([world.score_one(root, sequence, seed, n=n) for n in registration["sizes"]])
            atomic_json(cache, values)
        means = np.mean(values, axis=0)
        differences = np.abs(np.diff(means))
        results.append({"root": root.root_id, "means": means.tolist(), "differences": differences.tolist(),
                        "passed": bool((differences <= .01).all())})
    report = {**registration, "passed": all(row["passed"] for row in results), "results": results}
    atomic_json(path, report)
    if not report["passed"]:
        raise StageUnavailable("POPULATION_RESOLUTION_UNRESOLVED: V fixed-size diagnostic failed; do not fit Phi or RL")
    return {**pipeline, "numerical_audit": str(path), "precalibration_identity": world.biological_identity}


def prepare_frozen_environment(data, config, monitor, pipeline):
    from .world import BiologicalWorld
    pipeline = resolve_ode_steps(data, config, monitor, pipeline)
    world = BiologicalWorld(data, config, pipeline, require_resolution=False)
    pipeline = population_diagnostic(world, monitor, pipeline)
    # Calibrate C only after the registered numerical checks. Readers are explicitly permissioned.
    from .data import open_data
    calibration_data = open_data(config, "all", prepare=False)
    if pipeline.get("uncertainty") and not pipeline.get("uncertainty_calibration"):
        from .residuals import finalize_calibration
        pipeline = finalize_calibration(calibration_data, config, monitor, pipeline)
    if pipeline.get("endpoint") and config["task"] == "ANNEXIN-A549-V1":
        from .endpoints import finalize_calibration
        pipeline = finalize_calibration(calibration_data, config, monitor, pipeline)
    # Pre-C precision evidence and the final calibrated function are different versions.
    # Bind them explicitly; never run the tuning diagnostic again on C-calibrated scores.
    audit = json.loads(Path(pipeline["numerical_audit"]).read_text(encoding="utf-8"))
    del world
    gc.collect()  # The adapter's bound callbacks form a reference cycle with its world.
    if str(config["device"]).startswith("cuda"):
        torch.cuda.empty_cache()
    final = BiologicalWorld(data, config, pipeline, require_resolution=False)
    certificate_path = monitor.experiment / "frozen_environment.json"
    atomic_json(certificate_path, {"audit_sha256": file_hash(pipeline["numerical_audit"]),
        "precalibration_identity": audit["biological_identity"],
        "frozen_identity": final.biological_identity, "calibration_after_diagnostic": True,
        "population": config["population"], "synthetic_test_only": config.get("synthetic", False)})
    pipeline = {**pipeline, "freeze_certificate": str(certificate_path)}
    final.adapter.population_resolution_passed = True
    return final, pipeline
