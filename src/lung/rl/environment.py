"""Strict callback boundary: visible planning summaries never contain scoring particles."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from .contracts import DecisionState, RLUnavailable, RootContext, Transition, VersionStamp
from .protocol import CandidateRegistry, ProtocolSpec


def stream_seed(seed: int, *parts: Any) -> int:
    text = repr((int(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & ((1 << 63) - 1)


def shape_reward(base: float, phi_current: float, phi_next: float, terminated: bool) -> float:
    if terminated and phi_next != 0:
        raise ValueError("Absorbing-state potential must be zero")
    if not terminated and base != 0:
        raise ValueError("Only one terminal transition receives a base outcome")
    return base + phi_next - phi_current


@dataclass
class SimulationAdapter:
    registry: CandidateRegistry
    protocol: ProtocolSpec | None
    versions: VersionStamp
    observe_fn: Callable[[RootContext, tuple[str, ...], int], DecisionState]
    score_fn: Callable[[RootContext, tuple[str, ...], tuple[int, ...]], Tensor]
    potential_fn: Callable[[DecisionState], float] | None = None
    task_id: str = "PROXY-A549-V1"
    endpoint_available: bool = False
    population_resolution_passed: bool = False
    n_pop: int = 512
    evidence: dict[str, Any] = field(default_factory=dict)

    score_many_fn: Callable | None = None

    def validate_ready(self, require_potential: bool = False) -> None:
        if self.protocol is None:
            raise RLUnavailable("PROTOCOL_INCOMPLETE", "Register endpoint and switching/tail rules first")
        if self.versions.protocol != self.protocol.version or self.versions.candidates != self.registry.version:
            raise RLUnavailable("VERSION_MISMATCH", "Frozen protocol/candidate identities disagree")
        if not self.endpoint_available:
            raise RLUnavailable("ENDPOINT_UNAVAILABLE", f"No trained, supported endpoint for {self.task_id}")
        if require_potential and self.potential_fn is None:
            raise RLUnavailable("POTENTIAL_UNAVAILABLE", "The requested shaping requires a fitted potential")
        if self.task_id not in {"PROXY-A549-V1", "ANNEXIN-A549-V1"}:
            if not (self.task_id.startswith("TEST-") and self.evidence.get("synthetic_test") is True):
                raise RLUnavailable("UNSUPPORTED_TASK", self.task_id)
        if self.n_pop <= 0:
            raise ValueError("n_pop must be positive")

    @torch.no_grad()
    def observe(self, root: RootContext, prefix: tuple[str, ...], planning_seed: int) -> DecisionState:
        self.validate_ready()
        _, elapsed, done = self.protocol.position(self.registry, prefix)
        if done:
            raise ValueError("Do not construct a terminal policy observation")
        state = self.observe_fn(root, prefix, planning_seed)
        if (state.prefix != prefix or state.root_id != root.root_id or state.role != root.role
                or state.elapsed_ticks != elapsed):
            raise ValueError("Adapter observation identity or biological clock mismatch")
        return state

    @torch.no_grad()
    def score(self, root: RootContext, sequence: tuple[str, ...], seeds: tuple[int, ...]) -> Tensor:
        return self.score_many(root, ((sequence, seeds),))[0]

    @torch.no_grad()
    def score_many(self, root, requests):
        """Validate every request before executing; duplicates never become new replicates."""
        self.validate_ready()
        requests = tuple((tuple(sequence), tuple(seeds)) for sequence, seeds in requests)
        for sequence, seeds in requests:
            self.protocol.validate_complete(self.registry, sequence)
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("Each population scoring replicate requires a distinct seed")
        if not requests:
            return ()
        rows = (self.score_many_fn(root, requests) if self.score_many_fn is not None else
                tuple(self.score_fn(root, sequence, seeds) for sequence, seeds in requests))
        if len(rows) != len(requests):
            raise RLUnavailable("INVALID_ENDPOINT_OUTPUT", "Expected one result per request")
        results = []
        for row, (_, seeds) in zip(rows, requests):
            values = torch.as_tensor(row).detach().cpu().double()
            if values.shape != (len(seeds),) or not torch.isfinite(values).all():
                raise RLUnavailable("INVALID_ENDPOINT_OUTPUT", "Expected one finite scalar per whole-population replicate")
            if (values < 0).any() or (values > 1).any():
                raise RLUnavailable("INVALID_ENDPOINT_OUTPUT", "Endpoint scores must lie in [0, 1]")
            results.append(values)
        return tuple(results)

    @torch.no_grad()
    def potential(self, state: DecisionState) -> float:
        self.validate_ready()
        if self.potential_fn is None:
            return 0.0
        value = float(self.potential_fn(state))
        if not 0 <= value <= 1:
            raise RLUnavailable("INVALID_POTENTIAL", "A frozen potential must return a finite [0, 1] scalar")
        return value


class ModelRollout:
    """One complete-action transition; hidden scores only appear at final settlement."""

    def __init__(self, adapter: SimulationAdapter, root: RootContext, seed: int,
                 score_repeats: int = 32, episode_id: str | None = None):
        adapter.validate_ready()
        if root.role != "T":
            raise ValueError("Training rollouts require T; evaluate V/E with fixed sequences")
        if score_repeats < 1:
            raise ValueError("At least one independent population scoring repeat is required")
        self.adapter, self.root = adapter, root
        self.episode_id = episode_id or f"{root.root_id}:{seed}"
        self.planning_seed = stream_seed(seed, "planning")
        self.scoring_seeds = tuple(stream_seed(seed, "training-score", i) for i in range(score_repeats))
        self.prefix: tuple[str, ...] = ()
        self.state = adapter.observe(root, (), self.planning_seed)
        self.phi = adapter.potential(self.state)
        self.done = self.incomplete = False

    @property
    def legal_mask(self) -> Tensor:
        mask = self.adapter.protocol.legal_mask(self.adapter.registry, self.prefix)
        if not self.done and not mask.any():
            raise RLUnavailable("UNSUPPORTED_PROTOCOL", "No action with a reachable common endpoint")
        return mask

    def step(self, action_index: int, behavior_probabilities: Tensor,
             policy_step: int = 0) -> Transition:
        if self.done or self.incomplete:
            raise ValueError("A settled/incomplete episode cannot be stepped again")
        mask = self.legal_mask
        if not 0 <= action_index < len(mask) or not mask[action_index]:
            raise RLUnavailable("UNSUPPORTED_PROTOCOL", "Requested action is masked")
        action = self.adapter.registry.candidates[action_index]
        prefix = (*self.prefix, action.action_id)
        try:
            _, _, done = self.adapter.protocol.position(self.adapter.registry, prefix)
            if done:
                base = float(self.adapter.score(self.root, prefix, self.scoring_seeds).mean())
                next_state, next_mask, next_phi = None, None, 0.0
            else:
                base = 0.0
                next_state = self.adapter.observe(self.root, prefix, self.planning_seed)
                next_mask = self.adapter.protocol.legal_mask(self.adapter.registry, prefix)
                next_phi = self.adapter.potential(next_state)
            transition = Transition(
                state=self.state, legal_mask=mask, behavior_probabilities=behavior_probabilities,
                action_index=action_index, base_reward=base, phi_current=self.phi, phi_next=next_phi,
                reward=shape_reward(base, self.phi, next_phi, done), next_state=next_state,
                next_mask=next_mask, terminated=done, versions=self.adapter.versions,
                episode_id=self.episode_id, task_id=self.adapter.task_id,
                planning_seed=self.planning_seed, scoring_seeds=self.scoring_seeds,
                policy_step=policy_step, evidence=dict(self.adapter.evidence))
        except Exception:
            self.incomplete = True
            raise
        self.prefix, self.state, self.phi, self.done = prefix, next_state, next_phi, done
        return transition

@torch.no_grad()
def generate_fixed_sequences(actor, adapter: SimulationAdapter, root: RootContext,
                             count: int = 256, seed: int = 0) -> tuple[tuple[str, ...], ...]:
    """Policy reads planning observations only. Returned candidates are immutable plans."""
    adapter.validate_ready()
    generator = torch.Generator().manual_seed(stream_seed(seed, "propose-actions"))
    planning_seed = stream_seed(seed, "propose-planning")
    states = {}
    sequences = set()
    for _ in range(count):
        prefix: tuple[str, ...] = ()
        while not adapter.protocol.position(adapter.registry, prefix)[2]:
            if prefix not in states:
                states[prefix] = adapter.observe(root, prefix, planning_seed)
            mask = adapter.protocol.legal_mask(adapter.registry, prefix)
            p = actor([states[prefix]], mask[None])[0].detach().cpu()
            index = int(torch.multinomial(p, 1, generator=generator))
            prefix = (*prefix, adapter.registry.candidates[index].action_id)
        adapter.protocol.validate_complete(adapter.registry, prefix)
        sequences.add(prefix)
    return tuple(sorted(sequences))
