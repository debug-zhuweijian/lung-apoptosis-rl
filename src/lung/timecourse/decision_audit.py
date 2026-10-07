"""Preregistered D-only population-resolution check, with fixed predictor ODE settings."""
from pathlib import Path
import json
import numpy as np
import torch

from lung.engine.checkpoint import atomic_json
from lung.engine.config import file_hash
from lung.rl import stream_seed


def audit_population(world, config, output, progress=None):
    output = Path(output)
    registration_path, report_path = output/"registration.json", output/"report.json"
    audit = config.get("population_audit",{})
    repeats = int(audit.get("repeats",1024))
    tolerance = float(audit.get("tolerance",.01))
    if repeats < 2 or not 0 < tolerance <= 1:
        raise ValueError("INVALID_POPULATION_AUDIT_REGISTRATION")
    if not config.get("synthetic") and repeats < 1024:
        raise ValueError("REAL_POPULATION_AUDIT_REQUIRES_1024_REPEATS")
    plans = [(sorted(world.ids)[0],),("STOP",)]
    sizes = [world.n,2*world.n,4*world.n]
    registration = {"environment":world.identity,"proxy":world.proxy.identity,"board":list(world.board),
        "root_id":world.root.root_id,
        "population":world.n,"sizes":sizes,"repeats":repeats,"tolerance":tolerance,
        "candidates":world.registry.version,"plans":plans,"seed":int(audit.get("seed",1907)),
        "scope":"D_only_fixed_empirical_state_MC_replication_not_independent_cells_not_E_guarantee_not_ODE_convergence",
        "synthetic_engineering_only":bool(config.get("synthetic",False))}
    # JSON-normalize tuples before checking an existing immutable registration.
    registration = json.loads(json.dumps(registration))
    if registration_path.exists() and json.loads(registration_path.read_text()) != registration:
        raise ValueError("POPULATION_AUDIT_REGISTRATION_CHANGED")
    atomic_json(registration_path,registration)
    if report_path.exists():
        world.enable_population_audit(registration_path,report_path)
        return str(registration_path.resolve()),str(report_path.resolve())
    rows = []
    for plan in plans:
        seeds = tuple(stream_seed(registration["seed"],"population-audit",plan,i) for i in range(repeats))
        values = []
        for size in sizes:
            scores = world.raw_score(world.root,plan,seeds,size)
            if not torch.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
                raise ValueError("INVALID_POPULATION_AUDIT_SCORES")
            values.append(scores.numpy())
            if progress:
                progress("time_population_audit",len(rows)*len(sizes)+len(values),len(plans)*len(sizes))
        means = [float(v.mean()) for v in values]
        differences = [abs(means[i]-means[j]) for i,j in ((0,1),(0,2),(1,2))]
        rows.append({"plan":list(plan),"sizes":sizes,"means":means,"max_mean_difference":max(differences),
                     "standard_errors":[float(v.std(ddof=1)/np.sqrt(repeats)) for v in values]})
    passed = all(row["max_mean_difference"] <= tolerance for row in rows)
    atomic_json(report_path,{"registration_sha256":file_hash(registration_path),"passed":passed,
        "rows":rows,"scope":registration["scope"],"no_threshold_adjustment":True})
    world.enable_population_audit(registration_path,report_path)
    return str(registration_path.resolve()),str(report_path.resolve())
