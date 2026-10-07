"""Three conditional learners and private-history value models, V1 §§10–13."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .contracts import DecisionState, NetworkSpec, RLUnavailable
from .protocol import CandidateRegistry


def mlp(dims: Sequence[int], final_activation: bool = False) -> nn.Sequential:
    layers = []
    for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
        linear = nn.Linear(left, right)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if index < len(dims) - 2 or final_activation:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Zero mass for absent internal prefixes without ever softmaxing all -inf."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    nonempty = mask.any(-1, keepdim=True)
    safe = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(nonempty, safe, torch.zeros_like(safe))
    return torch.softmax(safe, -1) * mask.to(logits.dtype)


class PrivateHistory(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        self.spec = spec
        self.project = nn.Linear(spec.history_input_dim, spec.history_dim)
        self.gru = nn.GRU(spec.history_dim, spec.history_dim, batch_first=True)
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        for name, param in self.gru.named_parameters():
            nn.init.zeros_(param) if "bias" in name else nn.init.xavier_uniform_(param)

    def forward(self, states: Sequence[DecisionState]) -> Tensor:
        device, dtype = self.project.weight.device, self.project.weight.dtype
        rows = []
        s = self.spec
        for state in states:
            if state.features.numel() != s.state_dim or state.history.shape[1] != s.history_input_dim:
                raise ValueError("State or raw history dimension differs from NetworkSpec")
            raw = state.features.detach().to(device=device, dtype=dtype)
            if len(state.history):
                h = state.history.detach().to(device=device, dtype=dtype)
                _, hidden = self.gru(self.project(h).unsqueeze(0))
                history = hidden[0, 0]
            else:
                history = raw.new_zeros(s.history_dim)
            rows.append(torch.cat((raw[:s.history_start], history,
                                   raw[s.history_start + s.history_dim:])))
        return torch.stack(rows)


class ActionNetwork(nn.Module):
    def __init__(self, registry: CandidateRegistry):
        super().__init__()
        self.registry, self.spec = registry, registry.spec
        self.history = PrivateHistory(self.spec)
        self.register_buffer("action_features", torch.stack([a.embedding for a in registry.candidates]).float())
        self.register_buffer("stop_mask", torch.tensor([a.is_stop for a in registry.candidates]))
        self.stop_token = nn.Parameter(torch.randn(self.spec.drug_dim) * 0.02)

    def actions(self) -> Tensor:
        stop = torch.cat((self.stop_token, self.stop_token.new_zeros(
            self.spec.dose_dim + self.spec.time_dim)))
        return torch.where(self.stop_mask[:, None], stop[None], self.action_features)


class ConditionalActor(ActionNetwork):
    """Shared private context, three separate role scorers, exact complete-action chain."""

    def __init__(self, registry: CandidateRegistry):
        super().__init__(registry)
        s = self.spec
        latent = s.encoder_hidden[-1]
        self.encoder = mlp((s.state_dim, *s.encoder_hidden), final_activation=True)
        self.drug = mlp((latent + s.drug_dim, *s.scorer_hidden, 1))
        self.dose = mlp((latent + s.drug_dim + s.dose_dim, *s.scorer_hidden, 1))
        self.time = mlp((latent + s.action_dim, *s.scorer_hidden, 1))

    def forward(self, states: Sequence[DecisionState], masks: Tensor) -> Tensor:
        z, actions = self.encoder(self.history(states)), self.actions()
        masks = masks.to(device=z.device, dtype=torch.bool)
        if masks.shape != (len(states), len(self.registry)) or not masks.any(-1).all():
            raise RLUnavailable("UNSUPPORTED_PROTOCOL", "Every nonterminal state needs legal candidates")
        r, s = self.registry, self.spec
        first = [r.drug_indices[d][0] for d in r.drugs]
        drug_features = actions[first, :s.drug_dim]
        dz = torch.cat((z[:, None].expand(-1, len(first), -1),
                        drug_features[None].expand(len(z), -1, -1)), -1)
        drug_masks = torch.stack([masks[:, r.drug_indices[d]].any(-1) for d in r.drugs], -1)
        pd = masked_softmax(self.drug(dz).squeeze(-1), drug_masks)
        columns = [None] * len(r)
        for di, d in enumerate(r.drugs):
            ids = r.drug_indices[d]
            if r.candidates[ids[0]].is_stop:
                columns[ids[0]] = pd[:, di]
                continue
            dose_ids = [r.group_indices[d, q][0] for q in r.dose_groups[d]]
            end = s.drug_dim + s.dose_dim
            dq = torch.cat((z[:, None].expand(-1, len(dose_ids), -1),
                            actions[dose_ids, :end][None].expand(len(z), -1, -1)), -1)
            qm = torch.stack([masks[:, r.group_indices[d, q]].any(-1) for q in r.dose_groups[d]], -1)
            pq = masked_softmax(self.dose(dq).squeeze(-1), qm)
            for qi, q in enumerate(r.dose_groups[d]):
                tids = r.group_indices[d, q]
                dt = torch.cat((z[:, None].expand(-1, len(tids), -1),
                                actions[list(tids)][None].expand(len(z), -1, -1)), -1)
                pt = masked_softmax(self.time(dt).squeeze(-1), masks[:, tids])
                for ti, ai in enumerate(tids):
                    columns[ai] = pd[:, di] * pq[:, qi] * pt[:, ti]
        return torch.stack(columns, -1)


class JointActor(ActionNetwork):
    """Architecture §16 control: one complete-action scorer with identical information."""

    def __init__(self, registry: CandidateRegistry):
        super().__init__(registry)
        s = self.spec
        self.scorer = mlp((s.state_dim + s.action_dim, *s.critic_hidden, 1))

    def forward(self, states: Sequence[DecisionState], masks: Tensor) -> Tensor:
        state, actions = self.history(states), self.actions()
        if not masks.any(-1).all():
            raise RLUnavailable("UNSUPPORTED_PROTOCOL", "A nonterminal state has no legal action")
        pair = torch.cat((state[:, None].expand(-1, len(actions), -1),
                          actions[None].expand(len(state), -1, -1)), -1)
        return masked_softmax(self.scorer(pair).squeeze(-1), masks)


class AffineCritic(ActionNetwork):
    def __init__(self, registry: CandidateRegistry):
        super().__init__(registry)
        s = self.spec
        self.value = mlp((s.state_dim + s.action_dim, *s.critic_hidden, 1))

    def forward(self, states: Sequence[DecisionState]) -> Tensor:
        x, actions = self.history(states), self.actions()
        pair = torch.cat((x[:, None].expand(-1, len(actions), -1),
                          actions[None].expand(len(x), -1, -1)), -1)
        return self.value(pair).squeeze(-1)

    def distribution_value(self, states: Sequence[DecisionState], probabilities: Tensor) -> Tensor:
        return (self(states) * probabilities).sum(-1)


class DistributionCritic(nn.Module):
    """Explicitly biased ICL control; not used by the affine main model."""

    def __init__(self, registry: CandidateRegistry):
        super().__init__()
        self.history = PrivateHistory(registry.spec)
        self.value = mlp((registry.spec.state_dim + len(registry), *registry.spec.critic_hidden, 1))

    def distribution_value(self, states: Sequence[DecisionState], probabilities: Tensor) -> Tensor:
        return self.value(torch.cat((self.history(states), probabilities), -1)).squeeze(-1)


def prefix_advantages(registry: CandidateRegistry, probabilities: Tensor,
                      values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Diagnostic only: terms telescope to Q - V; never a second Actor loss."""
    v0 = (probabilities * values).sum(-1, keepdim=True)
    ad, aq, at = (torch.zeros_like(values) for _ in range(3))
    for d, ids in registry.drug_indices.items():
        mass_d = probabilities[..., ids].sum(-1, keepdim=True)
        v1 = (probabilities[..., ids] * values[..., ids]).sum(-1, keepdim=True) / mass_d.clamp_min(1e-30)
        ad[..., ids] = v1 - v0
        if registry.candidates[ids[0]].is_stop:
            ad[..., ids] = values[..., ids] - v0
            continue
        for q in registry.dose_groups[d]:
            qids = registry.group_indices[d, q]
            mass_q = probabilities[..., qids].sum(-1, keepdim=True)
            v2 = (probabilities[..., qids] * values[..., qids]).sum(-1, keepdim=True) / mass_q.clamp_min(1e-30)
            aq[..., qids], at[..., qids] = v2 - v1, values[..., qids] - v2
    return ad, aq, at
