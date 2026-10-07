"""Discrete prefix-tree DLPA extension; local search does not mutate proposal weights."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor

from .contracts import DecisionState, RootContext, VersionStamp
from .environment import SimulationAdapter, stream_seed
from .protocol import CandidateRegistry


@dataclass(frozen=True)
class DLPAConfig:
    rounds: int = 6
    candidates_per_round: int = 256
    elite_count: int = 64
    score_repeats: int = 32
    temperature: float = .1
    update_weight: float = .8
    uniform_mix: float = .02

    def __post_init__(self):
        if min(self.rounds, self.candidates_per_round, self.elite_count, self.score_repeats) < 1:
            raise ValueError("Planning counts must be positive")
        if self.temperature <= 0 or not 0 < self.update_weight <= 1 or not 0 <= self.uniform_mix <= 1:
            raise ValueError("Invalid local proposal update parameters")


class ConditionalTable:
    """A joint vector is a lossless storage of the three conditional prefix tables."""

    def __init__(self, registry: CandidateRegistry, mask: Tensor, probabilities: Tensor):
        self.registry = registry
        self.mask = mask.detach().cpu().bool().clone()
        self.probabilities = probabilities.detach().cpu().double().clone()
        self._check()

    def _check(self):
        p = self.probabilities
        if not self.mask.any() or (p < 0).any() or not torch.isfinite(p).all():
            raise ValueError("Invalid conditional probability table")
        if abs(float(p.sum()) - 1) > 1e-5 or p[~self.mask].abs().sum() != 0:
            raise ValueError("Local probabilities must be a masked complete-action simplex")

    @staticmethod
    def _conditional(values: Tensor, mask: Tensor) -> Tensor:
        total = values.sum()
        if total > 0:
            return values / total
        return mask.double() / mask.sum().clamp_min(1)

    def update(self, counts: Tensor, amount: float) -> None:
        """Conditional MLE per prefix, retaining each unvisited conditional distribution."""
        r, old = self.registry, self.probabilities
        counts = counts.detach().cpu().double()
        if counts.shape != old.shape or (counts < 0).any() or counts[~self.mask].sum() != 0:
            raise ValueError("Elite counts include invalid actions")
        if counts.sum() == 0:
            return
        pd_old = torch.tensor([old[list(ids)].sum() for ids in r.drug_indices.values()])
        pd_new = torch.tensor([counts[list(ids)].sum() for ids in r.drug_indices.values()])
        pd = (1 - amount) * pd_old + amount * pd_new / pd_new.sum()
        result = torch.zeros_like(old)
        for di, d in enumerate(r.drugs):
            ids = list(r.drug_indices[d])
            if r.candidates[ids[0]].is_stop:
                result[ids[0]] = pd[di]
                continue
            qgroups = [list(r.group_indices[d, q]) for q in r.dose_groups[d]]
            qmask = torch.tensor([bool(self.mask[g].any()) for g in qgroups])
            qold = self._conditional(torch.stack([old[g].sum() for g in qgroups]), qmask)
            qcounts = torch.stack([counts[g].sum() for g in qgroups])
            pq = (1 - amount) * qold + amount * qcounts / qcounts.sum() if qcounts.sum() > 0 else qold
            for qi, group in enumerate(qgroups):
                told = self._conditional(old[group], self.mask[group])
                tc = counts[group]
                pt = (1 - amount) * told + amount * tc / tc.sum() if tc.sum() > 0 else told
                result[group] = pd[di] * pq[qi] * pt
        self.probabilities = result
        self._check()


@dataclass(frozen=True)
class TeacherNode:
    state: DecisionState
    mask: Tensor
    probabilities: Tensor
    versions: VersionStamp
    run_id: str


@dataclass(frozen=True)
class PlanResult:
    root_id: str
    role: str
    sequences: tuple[tuple[str, ...], ...]
    search_scores: dict[tuple[str, ...], float]
    teachers: tuple[TeacherNode, ...]
    run_id: str
    metadata: dict


class DLPAPlanner:
    def __init__(self, adapter: SimulationAdapter, proposal,
                 config: DLPAConfig = DLPAConfig()):
        self.adapter, self.proposal, self.config = adapter, proposal, config
        self.root_count = self.plan_iteration = self.evaluated_sequences = 0
        self._active = False

    @torch.no_grad()
    def plan(self, root: RootContext, seed: int = 0,
             progress: Callable[[dict], None] | None = None) -> PlanResult:
        self.adapter.validate_ready()
        if self._active:
            raise RuntimeError("A planner instance runs one root at a time")
        self._active = True
        committed = self.plan_iteration, self.evaluated_sequences
        try:
            result = self._plan(root, seed, progress)
        except BaseException:
            self.plan_iteration, self.evaluated_sequences = committed
            raise
        finally:
            self._active = False
        self.root_count += 1
        return result

    def _plan(self, root: RootContext, seed: int, progress) -> PlanResult:
        a, c = self.adapter, self.config
        run_id = f"{root.root_id}:{self.root_count}:{stream_seed(seed, 'run')}"
        planning_seed = stream_seed(seed, "dlpa-planning", self.root_count)
        generator = torch.Generator().manual_seed(stream_seed(seed, "dlpa-actions", self.root_count))
        nodes: dict[tuple[str, ...], tuple[DecisionState, ConditionalTable]] = {}
        archive, round_counts = {}, []

        def node(prefix):
            if prefix not in nodes:
                state = a.observe(root, prefix, planning_seed)
                mask = a.protocol.legal_mask(a.registry, prefix)
                p = self.proposal([state], mask[None])[0]
                nodes[prefix] = state, ConditionalTable(a.registry, mask, p)
            return nodes[prefix]

        for iteration in range(c.rounds):
            candidates = set()
            for _ in range(c.candidates_per_round):
                prefix = ()
                while not a.protocol.position(a.registry, prefix)[2]:
                    _, table = node(prefix)
                    uniform = table.mask.double() / table.mask.sum()
                    p = (1 - c.uniform_mix) * table.probabilities + c.uniform_mix * uniform
                    action_index = int(torch.multinomial(p, 1, generator=generator))
                    prefix = (*prefix, a.registry.candidates[action_index].action_id)
                candidates.add(prefix)
            requests = tuple((sequence, tuple(
                stream_seed(seed, "dlpa-search", self.root_count, iteration, sequence, j)
                for j in range(c.score_repeats))) for sequence in sorted(candidates))
            scores = {sequence: float(values.mean()) for (sequence, _), values in
                      zip(requests, a.score_many(root, requests))}
            # Every unique plan is scored and enters the elite set once per round.
            archive.update(scores)
            elite = sorted(scores, key=lambda u: (-scores[u], u))[:c.elite_count]
            means = torch.tensor([scores[u] for u in elite], dtype=torch.float64)
            weights = torch.softmax((means - means.max()) / c.temperature, 0)
            counts = {}
            for sequence, weight in zip(elite, weights):
                for k, aid in enumerate(sequence):
                    prefix = sequence[:k]
                    counts.setdefault(prefix, torch.zeros(len(a.registry), dtype=torch.float64))
                    counts[prefix][a.registry.by_id[aid]] += weight
            for prefix, counts_here in counts.items():
                nodes[prefix][1].update(counts_here, c.update_weight)
            self.plan_iteration += 1
            self.evaluated_sequences += len(scores)
            round_counts.append(len(scores))
            if progress is not None:
                progress({"plan_iteration": self.plan_iteration, "round": iteration + 1,
                          "unique_candidates": len(scores), "elite_count": len(elite),
                          "search_score": max(scores.values())})
        teachers = tuple(TeacherNode(state, table.mask.clone(), table.probabilities.clone(),
                                     a.versions, run_id) for state, table in nodes.values())
        return PlanResult(root.root_id, root.role, tuple(sorted(archive)), archive, teachers, run_id,
                          {"planning_seed": planning_seed, "n_pop": a.n_pop,
                           "versions": a.versions, "round_unique_counts": round_counts,
                           "scoring_repeats": c.score_repeats, "role": "search_not_report",
                           "node_key_fields": (run_id, root.root_id, "full_prefix", a.versions,
                                               planning_seed, a.n_pop)})

    def state_dict(self) -> dict:
        if self._active:
            raise RuntimeError("Save DLPA only at completed root boundaries")
        return {"schema": "DLPA-PLAN-V1", "config": asdict(self.config),
                "versions": self.adapter.versions, "root_count": self.root_count,
                "plan_iteration": self.plan_iteration, "evaluated_sequences": self.evaluated_sequences}

    def load_state_dict(self, state: dict) -> None:
        if (self._active or state["schema"] != "DLPA-PLAN-V1"
                or state["config"] != asdict(self.config) or state["versions"] != self.adapter.versions):
            raise ValueError("Incompatible planner resume")
        for key in ("root_count", "plan_iteration", "evaluated_sequences"):
            setattr(self, key, state[key])
