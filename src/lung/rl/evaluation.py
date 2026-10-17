"""Fixed-sequence numerical re-evaluation; no policy callback exists in this interface."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Mapping, Sequence

import torch

from .contracts import RootContext
from .environment import SimulationAdapter, stream_seed


@dataclass(frozen=True)
class FixedEvaluation:
    means: dict[tuple[str, ...], float]
    standard_errors: dict[tuple[str, ...], float | None]
    repeats: int
    metadata: dict


def precision_repeats(pool_size: int, eta: float = .01, delta: float = .05) -> int:
    if pool_size < 1 or not 0 < eta <= 1 or not 0 < delta < 1:
        raise ValueError("Invalid fixed-pool precision parameters")
    return max(1024, math.ceil(math.log(2 * pool_size / delta) / (2 * eta**2)))


def reevaluate_fixed_sequences(adapter: SimulationAdapter, root: RootContext,
                               sequences: Sequence[tuple[str, ...]], seed: int = 0,
                               repeats: int | None = None, eta: float = .01,
                               delta: float = .05, block_size: int = 256,
                               progress: Callable[[dict], None] | None = None,
                               namespace: str = "final-reevaluation") -> FixedEvaluation:
    adapter.validate_ready()
    pool = tuple(sorted(set(tuple(s) for s in sequences)))
    required = precision_repeats(len(pool), eta, delta)
    count = required if repeats is None else repeats
    if count < 1 or block_size < 1:
        raise ValueError("Evaluation repetitions/block size must be positive")
    means, errors, digest = {}, {}, hashlib.sha256()
    for plan_index, sequence in enumerate(pool):
        adapter.protocol.validate_complete(adapter.registry, sequence)
        total = square = 0.0
        for start in range(0, count, block_size):
            seeds = tuple(stream_seed(seed, namespace, sequence, j)
                          for j in range(start, min(count, start + block_size)))
            digest.update(repr(seeds).encode("ascii"))
            values = adapter.score(root, sequence, seeds)
            total += float(values.sum())
            square += float(values.square().sum())
            if progress is not None:
                progress({"plan": plan_index + 1, "plans": len(pool),
                          "replicates": start + len(seeds), "target_replicates": count})
        means[sequence] = total / count
        variance = max(0.0, (square - total**2 / count) / (count - 1)) if count > 1 else None
        errors[sequence] = math.sqrt(variance / count) if variance is not None else None
    return FixedEvaluation(means, errors, count,
                           {"versions": adapter.versions, "n_pop": adapter.n_pop,
                            "task_id": adapter.task_id, "evidence": dict(adapter.evidence),
                            "root_id": root.root_id, "root_role": root.role,
                            "pool_size": len(pool), "seed_digest": digest.hexdigest(),
                            "namespace": namespace, "required_repeats": required,
                            "fixed_pool_bound_met": count >= required,
                            "eta": eta, "delta": delta, "actor_calls": 0,
                            "uncertainty_scope": "numerical_frozen_model_only"})


def audit_winners(adapter: SimulationAdapter, root: RootContext, evaluation: FixedEvaluation,
                  proposed_by: Mapping[str, Sequence[tuple[str, ...]]], seed: int,
                  repeats: int = 4096) -> dict:
    if (evaluation.metadata.get("versions") != adapter.versions
            or evaluation.metadata.get("root_id") != root.root_id
            or evaluation.metadata.get("root_role") != root.role
            or evaluation.metadata.get("n_pop") != adapter.n_pop
            or evaluation.metadata.get("task_id") != adapter.task_id):
        raise ValueError("Winner audit must use the same root and frozen scientific environment")
    winners = {}
    for algorithm, candidates in proposed_by.items():
        unique = set(tuple(s) for s in candidates)
        if not unique or not unique <= evaluation.means.keys():
            raise ValueError("Every algorithm must select only from its own evaluated candidates")
        winners[algorithm] = min(unique, key=lambda u: (-evaluation.means[u], u))
    audit = reevaluate_fixed_sequences(adapter, root, tuple(winners.values()), seed=seed,
                                       repeats=repeats, namespace="report-audit")
    return {"winners": winners, "audit": audit, "reselected_after_audit": False}
