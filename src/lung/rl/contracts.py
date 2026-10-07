"""Explicit data contracts for model-only decision learning (architecture V1 §§9–14)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch import Tensor


class RLUnavailable(RuntimeError):
    """A missing scientific precondition is not a zero reward or a terminal state."""

    def __init__(self, code: str, details: str):
        self.code, self.details = code, details
        super().__init__(f"{code}: {details}")

    def as_dict(self) -> dict[str, str]:
        return {"status": "unavailable", "code": self.code, "details": self.details}


@dataclass(frozen=True)
class NetworkSpec:
    state_dim: int = 637
    drug_dim: int = 256
    dose_dim: int = 32
    time_dim: int = 32
    history_dim: int = 256
    history_start: int = 283
    protocol_dim: int = 32
    encoder_hidden: tuple[int, ...] = (512, 256)
    scorer_hidden: tuple[int, ...] = (256, 128)
    critic_hidden: tuple[int, ...] = (512, 256)

    def __post_init__(self):
        if self.history_start < 0 or self.history_start + self.history_dim > self.state_dim:
            raise ValueError("History slot is outside the state vector")
        if min(self.state_dim, self.drug_dim, self.dose_dim, self.time_dim,
               self.history_dim, self.protocol_dim, *self.encoder_hidden,
               *self.scorer_hidden, *self.critic_hidden) < 1:
            raise ValueError("Network dimensions must be positive")

    @property
    def action_dim(self) -> int:
        return self.drug_dim + self.dose_dim + self.time_dim

    @property
    def history_input_dim(self) -> int:
        return self.action_dim + 1 + self.protocol_dim


@dataclass(frozen=True)
class Candidate:
    action_id: str
    drug_id: str
    dose_id: str
    time_id: str
    duration_ticks: int
    embedding: Tensor
    is_stop: bool = False

    def __post_init__(self):
        if not self.action_id or not self.drug_id:
            raise ValueError("Candidate and drug identifiers are required")
        if not isinstance(self.duration_ticks, int) or isinstance(self.duration_ticks, bool):
            raise ValueError("Durations must be exact integer ticks")
        if (self.is_stop and self.duration_ticks != 0) or (
                not self.is_stop and self.duration_ticks <= 0):
            raise ValueError("STOP has zero duration; ordinary actions have positive duration")
        if self.embedding.ndim != 1 or not torch.isfinite(self.embedding).all():
            raise ValueError("An action embedding must be a finite vector")
        object.__setattr__(self, "embedding", self.embedding.detach().cpu().clone())


@dataclass(frozen=True)
class RootContext:
    root_id: str
    role: str
    payload: Any = None


@dataclass(frozen=True)
class DecisionState:
    """Only frozen observable summaries; mutable private GRU outputs are never stored."""

    features: Tensor
    history: Tensor
    prefix: tuple[str, ...]
    elapsed_ticks: int
    root_id: str
    role: str

    def __post_init__(self):
        if self.features.ndim != 1 or self.history.ndim != 2:
            raise ValueError("Expected a feature vector and a raw history matrix")
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.history).all():
            raise ValueError("Non-finite decision inputs")
        if len(self.prefix) != len(self.history):
            raise ValueError("History contains exactly one raw row per completed ordinary action")
        object.__setattr__(self, "features", self.features.detach().cpu().clone())
        object.__setattr__(self, "history", self.history.detach().cpu().clone())
        object.__setattr__(self, "prefix", tuple(self.prefix))


@dataclass(frozen=True)
class VersionStamp:
    environment: str
    protocol: str
    reward: str
    candidates: str

    def __post_init__(self):
        if not all(asdict(self).values()):
            raise ValueError("All environment/protocol/reward/candidate versions are required")


@dataclass(frozen=True)
class Transition:
    state: DecisionState
    legal_mask: Tensor
    behavior_probabilities: Tensor
    action_index: int
    base_reward: float
    phi_current: float
    phi_next: float
    reward: float
    next_state: DecisionState | None
    next_mask: Tensor | None
    terminated: bool
    versions: VersionStamp
    episode_id: str
    task_id: str
    planning_seed: int
    scoring_seeds: tuple[int, ...]
    policy_step: int
    incomplete: bool = False
    rollout_source: str = "model_rollout"
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("legal_mask", "behavior_probabilities", "next_mask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.detach().cpu().clone())
        if self.incomplete:
            raise RLUnavailable("INCOMPLETE_ROLLOUT", "Do not add interrupted/failed transitions")
        if self.rollout_source != "model_rollout":
            raise ValueError("This replay only accepts identified model rollouts")
        if self.state.role != "T":
            raise ValueError("Policy training replay must use T roots only")
        if not 0 <= self.base_reward <= 1:
            raise ValueError("The base terminal outcome must be in [0, 1]")
        if not all(math.isfinite(v) for v in (self.reward, self.phi_current, self.phi_next)):
            raise ValueError("Stored rewards and potentials must be finite")
        if not 0 <= self.phi_current <= 1 or not 0 <= self.phi_next <= 1:
            raise ValueError("Frozen potentials must lie in [0, 1]")
        if abs(self.reward - (self.base_reward + self.phi_next - self.phi_current)) > 1e-6:
            raise ValueError("The stored reward does not satisfy potential shaping")
        if self.terminated:
            if self.phi_next != 0 or self.next_state is not None or self.next_mask is not None:
                raise ValueError("A terminal transition has zero potential and no next softmax")
        else:
            if self.next_state is None or self.next_mask is None or self.base_reward != 0:
                raise ValueError("A nonterminal transition needs a next observation and zero base reward")
            if (self.next_state.role != "T" or self.next_state.root_id != self.state.root_id
                    or not self.next_mask.any()):
                raise ValueError("Next observation must retain training root identity and legal suffixes")
        p, mask = self.behavior_probabilities, self.legal_mask
        if p.ndim != 1 or mask.shape != p.shape or mask.dtype != torch.bool:
            raise ValueError("Behavior distribution and legal mask must align")
        if not torch.isfinite(p).all() or (p < 0).any() or abs(float(p.sum()) - 1) > 1e-5:
            raise ValueError("Behavior probabilities must be a finite simplex")
        if not 0 <= self.action_index < len(p):
            raise ValueError("Executed action index is outside the registry")
        if p[~mask].abs().sum() != 0 or not mask[self.action_index] or p[self.action_index] <= 0:
            raise ValueError("Executed action must have positive legal behavior mass")
