"""Version-bound replay and reproducible model-learning random state."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from .contracts import Transition, VersionStamp


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class ReplayBuffer:
    def __init__(self, capacity: int, versions: VersionStamp, seed: int = 0):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.versions = capacity, versions
        self.records: list[Transition] = []
        self.position = 0
        self.random = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def append(self, transition: Transition) -> None:
        if transition.versions != self.versions:
            raise ValueError("Replay version mismatch: environment/protocol/reward/candidates changed")
        if len(self.records) < self.capacity:
            self.records.append(transition)
        else:
            self.records[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, size: int) -> list[Transition]:
        if not 0 < size <= len(self):
            raise ValueError("Insufficient replay entries for a without-replacement sample")
        return self.random.sample(self.records, size)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "versions": self.versions,
                "records": list(self.records), "position": self.position,
                "random": self.random.getstate()}

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity or state["versions"] != self.versions:
            raise ValueError("Cannot load a replay for different capacity or scientific versions")
        if any(t.versions != self.versions or t.incomplete for t in state["records"]):
            raise ValueError("Corrupt/incompatible replay record")
        self.records = list(state["records"])
        self.position = state["position"]
        self.random.setstate(state["random"])
