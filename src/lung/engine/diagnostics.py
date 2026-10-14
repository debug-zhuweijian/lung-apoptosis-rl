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


def _audit_batch_width(path, binding, requested, legacy):
    """Keep one execution layout, including legacy scalar caches, for the audit."""
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != 1 or any(record.get(key) != value for key, value in binding.items()):
            raise StageUnavailable("NUMERICAL_AUDIT_BATCH_LAYOUT_IDENTITY_MISMATCH")
        width = record.get("batch_size")
        if (isinstance(width, bool) or not isinstance(width, int) or not 0 < width <= requested):
            raise StageUnavailable("NUMERICAL_AUDIT_BATCH_LAYOUT_INVALID")
        return width
    width = 1 if legacy else requested
    atomic_json(path, {"schema_version": 1, **binding, "batch_size": width})
    return width


@torch.no_grad()
def ode_audit_batch(model, data, indexed_samples, *, device, member, steps, batch_size=1):
    """Bounded fixed-noise generation in request order, independent of batch width."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("ODE_AUDIT_BATCH_SIZE_INVALID")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("ODE_AUDIT_STEPS_INVALID")
    if getattr(getattr(model, "config", None), "reference_mode", "observed") == "population_mean":
        batch_size = 1  # Each registered observation remains its own reference population.
    samples, outputs = list(indexed_samples), []
    for start in range(0, len(samples), batch_size):
        group = samples[start:start + batch_size]
        references, noises, conditions = [], [], []
        for index, batch in group:
            reference = tensor(batch["control"], device)
            if reference.shape != (1, data.n_genes):
                raise ValueError("ODE_AUDIT_REQUIRES_SINGLE_FIXED_OBSERVATIONS")
            generator = torch.Generator(device=device).manual_seed(stream_seed(20260923, "ODE", member, index))
            references.append(reference)
            noises.append(response_noise(model, reference, generator))
            conditions.append(condition_kwargs(batch, 1, device))
        condition = model.encode_condition(**{key: torch.cat([row[key] for row in conditions])
                                              for key in conditions[0]})
        raw = model.sample(torch.cat(references), condition, noise=torch.cat(noises),
                           steps=steps, checkpoint=False)
        result = physical_projection(raw, tensor(data.mu, device), tensor(data.scale, device))
        if (result.y.shape != (len(group), data.n_genes) or not result.valid.all()
                or not torch.isfinite(result.y).all()):
            raise StageUnavailable("INVALID_GENERATED_MASS in V ODE diagnostic")
        # Independent storage keeps each existing sample checkpoint one row wide.
        outputs.extend(row.clone() for row in result.y.cpu().split(1))
    return outputs


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
    batch_size = config["response"].get("inference_microbatch", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("ODE_AUDIT_BATCH_SIZE_INVALID")
    cache_root = monitor.experiment / f"ode_progress-{identity[:20]}"
    for member, path in enumerate(pipeline["response"]):
        model = load_response(path, data, config["device"])
        folder = cache_root / f"member-{member}"
        requested = (1 if getattr(getattr(model, "config", None), "reference_mode", "observed") == "population_mean"
                     else batch_size)
        width = _audit_batch_width(folder / "batch_layout.json", {"identity": identity, "member": member},
                                  requested, any(folder.glob("steps-*/sample-*.pt")))
        samples = []
        rng = np.random.default_rng(20260923 + 404)
        for index in range(256):
            batch = data.sample_controls(ids[index % len(ids)], 1, rng, "V")
            samples.append(batch)
        previous = None
        for steps in (128, 256, 512, 1024):
            differences, current = [], []
            for start in monitor.progress(range(0, len(samples), width),
                                          desc=f"ODE V member {member+1}, steps {steps}"):
                _audit_stop(monitor, "ode_audit", start, cache_root)
                indices = list(range(start, min(start + width, len(samples))))
                outputs, missing, paths = {}, [], {}
                for index in indices:
                    cache_path = cache_root / f"member-{member}" / f"steps-{steps}" / f"sample-{index}.pt"
                    paths[index] = cache_path
                    if cache_path.is_file():
                        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                        if cached.get("identity") != identity or cached.get("index") != index or cached.get("steps") != steps:
                            raise StageUnavailable("ODE_AUDIT_CACHE_IDENTITY_MISMATCH")
                        if (cached.get("batch_size", 1) != width
                                or cached.get("group_start", index) != start):
                            raise StageUnavailable("ODE_AUDIT_CACHE_BATCH_LAYOUT_MISMATCH")
                        output = cached["expression"]
                        if output.shape != (1, data.n_genes) or not torch.isfinite(output).all():
                            raise StageUnavailable("ODE_AUDIT_CACHE_INVALID")
                        outputs[index] = output
                    else:
                        missing.append(index)
                if missing:
                    # Recompute the entire fixed chunk on partial resume, so its
                    # missing outputs use exactly the original floating-point layout.
                    generated = ode_audit_batch(model, data, [(index, samples[index]) for index in indices],
                        device=config["device"], member=member, steps=steps, batch_size=width)
                    outputs.update((index, output) for index, output in zip(indices, generated) if index in missing)
                for index in indices:
                    if index != start:
                        _audit_stop(monitor, "ode_audit", index, cache_root)
                    output = outputs[index]
                    if index in missing:
                        atomic_save(paths[index], {"identity": identity, "index": index, "steps": steps,
                                                  "member": member, "expression": output,
                                                  "batch_size": width, "group_start": start})
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
    endpoint_schema2 = getattr(getattr(world, "endpoint", None), "endpoint_schema_version", 1) == 2
    if endpoint_schema2:
        registration["endpoint_resolution_diagnostic"] = {
            "role": "V", "purpose": "numerical_sensitivity_only_not_formal_reward",
            "formal_scope": world.endpoint.feature_generation["world_inference"],
            "evaluated_population_sizes": registration["sizes"],
            "evaluated_ode_steps": world.steps}
    registration_path = monitor.experiment / "population_preregistration.json"
    if registration_path.exists():
        if json.loads(registration_path.read_text(encoding="utf-8")) != json.loads(json.dumps(registration)):
            raise StageUnavailable("NUMERICAL_AUDIT_REGISTRATION_CHANGED")
    else:
        atomic_json(registration_path, registration)
    results = []
    batch_size = getattr(world, "score_batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("POPULATION_AUDIT_BATCH_SIZE_INVALID")
    batch_size = _audit_batch_width(monitor.experiment / "population_batch_layout.json",
        {"biological_identity": world.biological_identity}, batch_size,
        any((monitor.experiment / f"population-{config_hash(root.root_id)[:16]}.json").is_file()
            for root, _ in plans))
    for root, sequence in plans:
        cache = monitor.experiment / f"population-{config_hash(root.root_id)[:16]}.json"
        values = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
        first_group = (len(values) // batch_size) * batch_size
        for start in monitor.progress(range(first_group, 1024, batch_size), desc=f"V population {root.root_id}"):
            next_repeat = len(values)
            _audit_stop(monitor, "assemble", next_repeat, cache)
            seeds = [stream_seed(20260923, "population-size", root.root_id, repeat)
                     for repeat in range(start, min(start + batch_size, 1024))]
            if callable(getattr(world, "score", None)):
                scorer = world.score_diagnostic if endpoint_schema2 else world.score
                by_size = [scorer(root, sequence, seeds, n=n).tolist() for n in registration["sizes"]]
            else:
                by_size = [[world.score_one(root, sequence, seed, n=n) for seed in seeds]
                           for n in registration["sizes"]]
            for offset, row in enumerate(zip(*by_size)):
                repeat = start + offset
                if repeat < next_repeat:
                    continue
                if repeat != next_repeat:
                    _audit_stop(monitor, "assemble", repeat, cache)
                values.append(list(row))
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
