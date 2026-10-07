"""Root-balanced joint KL, with all private proposal history recomputed per update."""
from __future__ import annotations

from dataclasses import asdict
import random

import torch
from torch import Tensor, nn

from .contracts import NetworkSpec, VersionStamp
from .dlpa import PlanResult, TeacherNode
from .networks import ConditionalActor, PrivateHistory, mlp
from .protocol import CandidateRegistry
from .replay import restore_rng, rng_state


def joint_kl(teacher: Tensor, student: Tensor) -> Tensor:
    if teacher.shape != student.shape:
        raise ValueError("Teacher/student joint distributions must use the same candidates")
    teacher = teacher.detach().to(student)
    tiny = torch.finfo(student.dtype).tiny
    terms = torch.where(teacher > 0, teacher * (teacher.clamp_min(tiny).log()
                        - student.clamp_min(tiny).log()), torch.zeros_like(student))
    return terms.sum(-1)


class ProposalDistiller:
    def __init__(self, registry: CandidateRegistry, versions: VersionStamp,
                 device: str = "cpu", seed: int = 0, lr: float = 1e-4,
                 grad_clip: float = 1.0):
        if versions.candidates != registry.version:
            raise ValueError("Candidate version mismatch")
        self.registry, self.versions = registry, versions
        self.lr, self.grad_clip = lr, grad_clip
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.proposal = ConditionalActor(registry).to(device)
        self.optimizer = torch.optim.Adam(self.proposal.parameters(), lr=lr)
        self.teachers: dict[str, list[TeacherNode]] = {}
        self.random = random.Random(seed + 7)
        self.distill_step = 0

    def begin_epoch(self) -> None:
        """Each epoch collects its own T roots before the configured 200 updates."""
        self.teachers.clear()

    def add(self, result: PlanResult) -> None:
        if result.role != "T" or not result.teachers:
            raise ValueError("Only nonempty T-root planning output can teach the proposal")
        for node in result.teachers:
            p, m = node.probabilities, node.mask
            if (node.versions != self.versions or node.state.role != "T"
                    or node.state.root_id != result.root_id):
                raise ValueError("Teacher identity/version mismatch")
            if (p.shape != m.shape or not torch.isfinite(p).all() or (p < 0).any()
                    or abs(float(p.sum()) - 1) > 1e-5 or p[~m].abs().sum() != 0):
                raise ValueError("Teacher distribution is not a masked simplex")
        # Every collected root invocation, including repeats of a biological root, is one unit.
        self.teachers[result.run_id] = list(result.teachers)

    def update(self, batch_size: int = 256) -> dict[str, float | int]:
        if not self.teachers or batch_size < 1:
            raise ValueError("Collect training roots before distillation")
        keys = tuple(self.teachers)
        nodes = [self.random.choice(self.teachers[self.random.choice(keys)]) for _ in range(batch_size)]
        masks = torch.stack([n.mask for n in nodes])
        student = self.proposal([n.state for n in nodes], masks)
        teacher = torch.stack([n.probabilities for n in nodes]).to(student)
        loss = joint_kl(teacher, student).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_DISTILLATION_LOSS")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.proposal.parameters(), self.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.distill_step += 1
        return {"distill_kl": float(loss.detach()), "distill_step": self.distill_step,
                "teacher_roots": len(self.teachers)}

    def state_dict(self) -> dict:
        return {"schema": "DLPA-DISTILL-V1", "versions": self.versions,
                "spec": asdict(self.registry.spec), "lr": self.lr, "grad_clip": self.grad_clip,
                "proposal": self.proposal.state_dict(), "optimizer": self.optimizer.state_dict(),
                "teachers": dict(self.teachers), "distill_step": self.distill_step,
                "random": self.random.getstate(), "global_rng": rng_state()}

    def load_state_dict(self, state: dict) -> None:
        if (state["schema"] != "DLPA-DISTILL-V1" or state["versions"] != self.versions
                or state["spec"] != asdict(self.registry.spec) or state["lr"] != self.lr
                or state["grad_clip"] != self.grad_clip):
            raise ValueError("Incompatible proposal checkpoint")
        self.proposal.load_state_dict(state["proposal"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.teachers = dict(state["teachers"])
        self.distill_step = state["distill_step"]
        self.random.setstate(state["random"])
        restore_rng(state["global_rng"])


class PotentialNetwork(nn.Module):
    """Independent history for Phi; train on reference completion targets then freeze."""

    def __init__(self, registry: CandidateRegistry | NetworkSpec, hidden: tuple[int, ...] = (256, 64)):
        super().__init__()
        spec = registry if isinstance(registry, NetworkSpec) else registry.spec
        self.history = PrivateHistory(spec)
        self.value = nn.Sequential(mlp((spec.state_dim, *hidden, 1)), nn.Sigmoid())

    def forward(self, states) -> Tensor:
        return self.value(self.history(states)).squeeze(-1)
