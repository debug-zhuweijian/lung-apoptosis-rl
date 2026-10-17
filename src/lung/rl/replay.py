"""Version-bound online replay for the current training run."""
from __future__ import annotations

import random

from .contracts import Transition, VersionStamp


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
