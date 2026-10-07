"""Actual-update quotas for one conditional Actor and one persistent optimizer."""
from copy import deepcopy

import torch


class ActorSchedule:
    stages = ("drug", "dose", "joint")

    def __init__(self, registry, quotas, legal_mask=None):
        if not isinstance(quotas, dict) or set(quotas) - set(self.stages):
            raise ValueError("ACTOR_SCHEDULE_EXPECTS_DRUG_DOSE_JOINT")
        self.requested = {stage: quotas.get(stage, 0) for stage in self.stages}
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
               for n in self.requested.values()) or sum(self.requested.values()) < 1:
            raise ValueError("ACTOR_SCHEDULE_REQUIRES_POSITIVE_TOTAL_INTEGER_QUOTA")
        mask = torch.ones(len(registry), dtype=torch.bool) if legal_mask is None else legal_mask.cpu()
        if mask.shape != (len(registry),) or mask.dtype != torch.bool or not mask.any():
            raise ValueError("ACTOR_SCHEDULE_LEGAL_MASK_INVALID")
        actions = [a for a, legal in zip(registry.candidates, mask.tolist()) if legal]
        drugs = {a.drug_id for a in actions}
        self.choices = {"drug": len(drugs) > 1,
            "dose": any(len({a.dose_id for a in actions if a.drug_id == d and not a.is_stop}) > 1
                        for d in drugs),
            "time": any(len({a.time_id for a in actions if (a.drug_id, a.dose_id) == key}) > 1
                        for key in {(a.drug_id, a.dose_id) for a in actions if not a.is_stop})}
        self.quotas = dict(self.requested)
        for stage in ("drug", "dose"):
            if not self.choices[stage]:
                self.quotas["joint"] += self.quotas[stage]
                self.quotas[stage] = 0
        self.completed = {stage: 0 for stage in self.stages}

    @property
    def stage(self):
        return next((stage for stage in self.stages
                     if self.completed[stage] < self.quotas[stage]), "joint")

    @property
    def complete(self):
        return self.completed == self.quotas

    @property
    def total_updates(self):
        return sum(self.completed.values())

    def advance(self):
        if self.complete:
            raise ValueError("ACTOR_SCHEDULE_ALREADY_COMPLETE")
        self.completed[self.stage] += 1

    def state_dict(self):
        return deepcopy({"schema_version": 1, "requested": self.requested,
            "quotas": self.quotas, "choices": self.choices, "completed": self.completed,
            "stage": self.stage, "stage_updates": self.completed[self.stage]})

    def load_state_dict(self, state):
        if (state.get("schema_version") != 1 or state.get("requested") != self.requested
                or state.get("quotas") != self.quotas or state.get("choices") != self.choices):
            raise ValueError("ACTOR_SCHEDULE_RESUME_MISMATCH")
        counts = state.get("completed", {})
        if set(counts) != set(self.stages) or any(isinstance(n, bool) or not isinstance(n, int)
                or not 0 <= n <= self.quotas[stage] for stage, n in counts.items()):
            raise ValueError("ACTOR_SCHEDULE_COUNTER_INVALID")
        unfinished = False
        for stage in self.stages:
            if unfinished and counts[stage]:
                raise ValueError("ACTOR_SCHEDULE_COUNTER_ORDER_INVALID")
            unfinished = unfinished or counts[stage] < self.quotas[stage]
        self.completed = dict(counts)
        if state.get("stage") != self.stage or state.get("stage_updates") != counts[self.stage]:
            raise ValueError("ACTOR_SCHEDULE_STAGE_MISMATCH")
