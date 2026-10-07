"""Finite joint actions and explicit, integer-time experimental protocols."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from math import lcm
from fractions import Fraction
from typing import Mapping, Sequence

import torch

from .contracts import Candidate, NetworkSpec, RLUnavailable


def exact_time_ticks(hours: Sequence[str | int]) -> tuple[int, tuple[int, ...]]:
    """Use decimal strings rather than binary floating-point durations."""
    if any(isinstance(t, float) for t in hours):
        raise ValueError("Pass decimal hour strings, not binary floats")
    values = tuple(Fraction(t) for t in hours)
    if any(t <= 0 for t in values):
        raise ValueError("Biological durations must be positive")
    scale = lcm(*(t.denominator for t in values))
    return scale, tuple(int(t * scale) for t in values)


class CandidateRegistry:
    def __init__(self, candidates: Sequence[Candidate], spec: NetworkSpec = NetworkSpec()):
        self.candidates, self.spec = tuple(candidates), spec
        if not candidates or len({a.action_id for a in candidates}) != len(candidates):
            raise ValueError("The candidate registry must be nonempty with unique IDs")
        if sum(a.is_stop for a in candidates) > 1:
            raise ValueError("Only one STOP candidate is allowed")
        if any(a.embedding.numel() != spec.action_dim for a in candidates):
            raise ValueError("Action embeddings do not match the network specification")
        self.by_id = {a.action_id: i for i, a in enumerate(candidates)}
        self.drugs = tuple(dict.fromkeys(a.drug_id for a in candidates))
        self.drug_indices = {d: tuple(i for i, a in enumerate(candidates) if a.drug_id == d)
                             for d in self.drugs}
        self.dose_groups = {
            d: tuple(dict.fromkeys(a.dose_id for a in candidates if a.drug_id == d))
            for d in self.drugs}
        self.group_indices = {
            (d, q): tuple(i for i in self.drug_indices[d] if candidates[i].dose_id == q)
            for d in self.drugs for q in self.dose_groups[d]}
        for d, ids in self.drug_indices.items():
            first = candidates[ids[0]]
            if first.is_stop and len(ids) != 1:
                raise ValueError("STOP must have its own drug-layer token")
            for i in ids:
                if not torch.equal(first.embedding[:spec.drug_dim],
                                   candidates[i].embedding[:spec.drug_dim]):
                    raise ValueError("A drug prefix must have one consistent frozen embedding")
            for q in self.dose_groups[d]:
                qids = self.group_indices[d, q]
                if len({candidates[i].time_id for i in qids}) != len(qids):
                    raise ValueError("Duplicate complete drug/dose/time action")
                end = spec.drug_dim + spec.dose_dim
                if any(not torch.equal(candidates[qids[0]].embedding[:end],
                                       candidates[i].embedding[:end]) for i in qids):
                    raise ValueError("A dose prefix must have a consistent embedding")
        digest = hashlib.sha256()
        for action in candidates:
            fields = (action.action_id, action.drug_id, action.dose_id, action.time_id,
                      action.duration_ticks, action.is_stop)
            digest.update(json.dumps(fields, ensure_ascii=False).encode("utf-8"))
            digest.update(action.embedding.float().numpy().tobytes())
        self.version = digest.hexdigest()

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, action_id: str) -> Candidate:
        return self.candidates[self.by_id[action_id]]


@dataclass(frozen=True)
class ProtocolSpec:
    """The state machine includes the actual switch/tail semantics, not inferred biology."""

    horizon_ticks: int
    initial_state: str
    terminal_states: frozenset[str]
    transitions: Mapping[tuple[str, str], str]
    stop_tails: Mapping[tuple[str, int], str]
    version: str
    ticks_per_hour: int = 1

    def __post_init__(self):
        if (not isinstance(self.horizon_ticks, int) or self.horizon_ticks <= 0
                or not self.version or not self.initial_state or not self.terminal_states
                or not isinstance(self.ticks_per_hour, int) or self.ticks_per_hour <= 0):
            raise RLUnavailable("PROTOCOL_INCOMPLETE", "Endpoint, integer time scale and protocol IDs required")
        if any(not name or not 0 <= t < self.horizon_ticks
               for (_, t), name in self.stop_tails.items()):
            raise RLUnavailable("PROTOCOL_INCOMPLETE", "Every STOP tail needs an explicit valid rule")
        object.__setattr__(self, "transitions", dict(self.transitions))
        object.__setattr__(self, "stop_tails", dict(self.stop_tails))

    def position(self, registry: CandidateRegistry, prefix: Sequence[str]) -> tuple[str, int, bool]:
        state, ticks, done = self.initial_state, 0, False
        for aid in prefix:
            if done or aid not in registry.by_id:
                raise RLUnavailable("UNSUPPORTED_PROTOCOL", "Unknown action or action after endpoint")
            action = registry[aid]
            if action.is_stop:
                if (state, ticks) not in self.stop_tails:
                    raise RLUnavailable("UNSUPPORTED_PROTOCOL", "No registered STOP continuation")
                ticks, done = self.horizon_ticks, True
                continue
            next_state = self.transitions.get((state, aid))
            if next_state is None or ticks + action.duration_ticks > self.horizon_ticks:
                raise RLUnavailable("UNSUPPORTED_PROTOCOL", "Unregistered switch or duration past endpoint")
            state, ticks = next_state, ticks + action.duration_ticks
            done = ticks == self.horizon_ticks
            if done and state not in self.terminal_states:
                raise RLUnavailable("UNSUPPORTED_PROTOCOL", "Endpoint reached in an invalid protocol state")
        return state, ticks, done

    def legal_mask(self, registry: CandidateRegistry, prefix: Sequence[str]) -> torch.Tensor:
        state, ticks, done = self.position(registry, prefix)
        mask = torch.zeros(len(registry), dtype=torch.bool)
        if done:
            return mask
        stop_present = any(a.is_stop for a in registry.candidates)

        @lru_cache(maxsize=None)
        def reachable(s: str, t: int) -> bool:
            if t == self.horizon_ticks:
                return s in self.terminal_states
            if stop_present and (s, t) in self.stop_tails:
                return True
            for a in registry.candidates:
                ns = self.transitions.get((s, a.action_id))
                nt = t + a.duration_ticks
                if not a.is_stop and ns is not None and nt <= self.horizon_ticks and reachable(ns, nt):
                    return True
            return False

        for index, action in enumerate(registry.candidates):
            if action.is_stop:
                mask[index] = (state, ticks) in self.stop_tails
            else:
                nxt = self.transitions.get((state, action.action_id))
                future = ticks + action.duration_ticks
                mask[index] = (nxt is not None and future <= self.horizon_ticks
                               and reachable(nxt, future))
        return mask

    def validate_complete(self, registry: CandidateRegistry, sequence: Sequence[str]) -> None:
        for k, aid in enumerate(sequence):
            mask = self.legal_mask(registry, sequence[:k])
            if aid not in registry.by_id or not mask[registry.by_id[aid]]:
                raise RLUnavailable("UNSUPPORTED_PROTOCOL", "Complete sequence contains an illegal prefix")
        if not self.position(registry, sequence)[2]:
            raise RLUnavailable("INCOMPLETE_ROLLOUT", "Sequence does not settle the common endpoint")
