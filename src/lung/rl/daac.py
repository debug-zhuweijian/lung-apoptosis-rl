"""MA-DAAC-Affine-V1 and the separately named ICL / single-Actor controls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .contracts import DecisionState, Transition, VersionStamp
from .networks import AffineCritic, ConditionalActor, DistributionCritic, JointActor
from .protocol import CandidateRegistry
from .replay import ReplayBuffer, restore_rng, rng_state


@dataclass(frozen=True)
class DAACConfig:
    capacity: int = 250000
    batch_size: int = 256
    warmup_transitions: int = 1000
    actor_delay: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    polyak: float = .005
    epsilon_start: float = .20
    epsilon_end: float = .02
    epsilon_steps: int = 50000
    grad_clip: float = 1.0
    critic_type: str = "affine"
    actor_type: str = "conditional"

    def __post_init__(self):
        if not 0 < self.batch_size <= self.capacity or self.actor_delay < 1 or self.epsilon_steps < 1:
            raise ValueError("Invalid batch/capacity/update schedule")
        if self.warmup_transitions < self.batch_size - 1:
            raise ValueError("Warmup must supply a full batch before the first update")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or not 0 < self.polyak <= 1:
            raise ValueError("Invalid epsilon or target interpolation")
        if self.critic_type not in {"affine", "icl_control"} or self.actor_type not in {"conditional", "joint_control"}:
            raise ValueError("Unknown architecture variant")


def min_expected_target(probabilities: Tensor, q1: Tensor, q2: Tensor) -> Tensor:
    """The minimum is OUTSIDE each complete-action expectation, as specified."""
    return torch.minimum((probabilities * q1).sum(-1), (probabilities * q2).sum(-1))


class DAACTrainer:
    def __init__(self, registry: CandidateRegistry, versions: VersionStamp,
                 config: DAACConfig = DAACConfig(), device: str = "cpu", seed: int = 0):
        if versions.candidates != registry.version:
            raise ValueError("Candidate version must be the frozen registry digest")
        self.registry, self.versions, self.config = registry, versions, config
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            actor_cls = ConditionalActor if config.actor_type == "conditional" else JointActor
            critic_cls = AffineCritic if config.critic_type == "affine" else DistributionCritic
            self.actor = actor_cls(registry).to(self.device)
            self.critics = nn.ModuleList([critic_cls(registry), critic_cls(registry)]).to(self.device)
        self.targets = deepcopy(self.critics).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.capacity, versions, seed + 1)
        self.generator = torch.Generator().manual_seed(seed + 2)
        self.total_transitions = self.critic_step = self.actor_step = self.episodes = 0

    @property
    def epsilon(self) -> float:
        if self.total_transitions < self.config.warmup_transitions:
            return 1.0
        c = self.config
        amount = min(1.0, self.critic_step / c.epsilon_steps)
        return c.epsilon_start + amount * (c.epsilon_end - c.epsilon_start)

    @torch.no_grad()
    def select(self, state: DecisionState, legal_mask: Tensor,
               explore: bool = True) -> tuple[int, Tensor]:
        probabilities = self.actor([state], legal_mask[None])[0].cpu()
        if explore:
            uniform = legal_mask.float() / legal_mask.sum()
            probabilities = (1 - self.epsilon) * probabilities + self.epsilon * uniform
        index = int(torch.multinomial(probabilities, 1, generator=self.generator))
        return index, probabilities

    def observe(self, transition: Transition) -> dict[str, float | int]:
        self.replay.append(transition)
        self.total_transitions += 1
        self.episodes += int(transition.terminated)
        metrics = {"env_step": self.total_transitions, "episodes": self.episodes,
                   "epsilon": self.epsilon, "critic_step": self.critic_step,
                   "actor_step": self.actor_step}
        if self.total_transitions <= self.config.warmup_transitions:
            metrics["warmup"] = 1
            return metrics
        metrics.update(self.update())
        return metrics

    @torch.no_grad()
    def _targets(self, batch: list[Transition]) -> Tensor:
        y = torch.tensor([r.reward for r in batch], device=self.device, dtype=torch.float32)
        active = [i for i, r in enumerate(batch) if not r.terminated]
        if active:
            states = [batch[i].next_state for i in active]
            masks = torch.stack([batch[i].next_mask for i in active])
            probabilities = self.actor(states, masks)
            if self.config.critic_type == "affine":
                continuation = min_expected_target(probabilities, self.targets[0](states), self.targets[1](states))
            else:
                continuation = torch.minimum(*(q.distribution_value(states, probabilities) for q in self.targets))
            y[active] += continuation
        return y

    def update(self) -> dict[str, float | int]:
        batch = self.replay.sample(self.config.batch_size)
        states = [r.state for r in batch]
        masks = torch.stack([r.legal_mask for r in batch])
        index = torch.tensor([r.action_index for r in batch], device=self.device)
        target = self._targets(batch)
        if self.config.critic_type == "affine":
            predictions = [q(states).gather(1, index[:, None]).squeeze(1) for q in self.critics]
        else:
            behavior = torch.stack([r.behavior_probabilities for r in batch]).to(self.device)
            onehot = nn.functional.one_hot(index, len(self.registry)).to(behavior.dtype)
            w = torch.rand((len(batch), 1), generator=self.generator).to(self.device)
            interpolation = w * behavior + (1 - w) * onehot
            predictions = [q.distribution_value(states, interpolation) for q in self.critics]
        losses = [nn.functional.mse_loss(q, target) for q in predictions]
        if not all(torch.isfinite(loss) for loss in losses):
            raise FloatingPointError("NONFINITE_CRITIC_LOSS")
        self.critic_optimizer.zero_grad(set_to_none=True)
        sum(losses).backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.grad_clip, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic_step += 1
        metrics = {"critic_loss_1": float(losses[0].detach()), "critic_loss_2": float(losses[1].detach()),
                   "target_mean": float(target.mean()), "critic_step": self.critic_step}
        if self.critic_step % self.config.actor_delay == 0:
            self.critics.requires_grad_(False)
            try:
                probabilities = self.actor(states, masks)
                if self.config.critic_type == "affine":
                    with torch.no_grad():
                        q1 = self.critics[0](states)
                    loss_actor = -(probabilities * q1).sum(-1).mean()
                else:
                    loss_actor = -self.critics[0].distribution_value(states, probabilities).mean()
                if not torch.isfinite(loss_actor):
                    raise FloatingPointError("NONFINITE_ACTOR_LOSS")
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss_actor.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip, error_if_nonfinite=True)
                self.actor_optimizer.step()
            finally:
                self.critics.requires_grad_(True)
            with torch.no_grad():
                for target_param, source_param in zip(self.targets.parameters(), self.critics.parameters()):
                    target_param.lerp_(source_param, self.config.polyak)
            self.actor_step += 1
            metrics.update(actor_loss=float(loss_actor.detach()), actor_step=self.actor_step)
        return metrics

    def state_dict(self) -> dict:
        return {"schema": "DAAC-V1", "config": asdict(self.config), "versions": self.versions,
                "spec": asdict(self.registry.spec), "actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(), "targets": self.targets.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(), "replay": self.replay.state_dict(),
                "generator": self.generator.get_state(), "global_rng": rng_state(),
                "counters": {k: getattr(self, k) for k in ("total_transitions", "critic_step", "actor_step", "episodes")}}

    def load_state_dict(self, state: dict) -> None:
        if (state["schema"] != "DAAC-V1" or state["versions"] != self.versions
                or state["config"] != asdict(self.config) or state["spec"] != asdict(self.registry.spec)):
            raise ValueError("Cannot resume a different DA-AC architecture or scientific environment")
        for key in ("actor", "critics", "targets", "actor_optimizer", "critic_optimizer", "replay"):
            getattr(self, key).load_state_dict(state[key])
        self.generator.set_state(state["generator"].cpu())
        for key, value in state["counters"].items():
            setattr(self, key, value)
        restore_rng(state["global_rng"])
