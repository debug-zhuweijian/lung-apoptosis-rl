"""Population, state, measured-endpoint and frozen-diagnostic heads (§§7–9)."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import mlp


class PopulationPool(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.encoder = mlp([2 * width, 2 * width, width])
        self.reconstruction = mlp([width, 2 * width, 2 * width])

    def statistics(self, cell_vectors):
        if cell_vectors.ndim not in (2, 3) or cell_vectors.shape[-1] != self.width or cell_vectors.shape[-2] < 1:
            raise ValueError("Pool expects [cells,width] or [groups,cells,width].")
        return torch.cat((cell_vectors.mean(-2), cell_vectors.std(-2, correction=0)), dim=-1)

    def forward(self, cell_vectors):
        return self.encoder(self.statistics(cell_vectors))

    def reconstruction_loss(self, cell_vectors):
        statistics = self.statistics(cell_vectors.detach())
        return F.mse_loss(self.reconstruction(self.encoder(statistics)), statistics)


class StateClassifier(nn.Module):
    LABELS = ("PI-APC-", "PI-APC+", "PI+APC+")

    def __init__(self, width=256):
        super().__init__()
        self.network = mlp([width, max(1, width // 2), 3])
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(self, cell_vectors):
        return self.network(cell_vectors)

    def probabilities(self, cell_vectors):
        return (self(cell_vectors) / self.temperature).softmax(-1)

    def population_proxy(self, cell_vectors):
        """Score cells separately; this is NOT a calibrated apoptosis rate."""
        return self.probabilities(cell_vectors)[..., 1:].sum(-1).mean(-1)

    @staticmethod
    def class_weights(labels):
        if labels.ndim != 1 or labels.dtype != torch.long or (labels < 0).any() or (labels > 2).any():
            raise ValueError("State labels must be integer class ids 0,1,2.")
        counts = torch.bincount(labels, minlength=3)
        if (counts == 0).any():
            raise ValueError("STATE_SUPERVISION_UNAVAILABLE: at least one training class missing.")
        return len(labels) / (3 * counts.float())

    def set_temperature(self, value):
        value = torch.as_tensor(value, device=self.temperature.device, dtype=self.temperature.dtype)
        if value.numel() != 1 or not torch.isfinite(value) or value <= 1e-3:
            raise ValueError("Invalid fitted state temperature.")
        self.temperature.copy_(value.detach())


class TemperatureCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(1 - 1e-3))))

    @property
    def temperature(self):
        return F.softplus(self.raw_temperature) + 1e-3

    def forward(self, detached_logits):
        return detached_logits.detach() / self.temperature


class EndpointHead(nn.Module):
    """One sample is an experimental population; call only with matched labels."""

    def __init__(self, state_dim=637, width=256):
        super().__init__()
        self.network = mlp([state_dim, width, max(1, width // 4), 1], nn.SiLU)

    def forward(self, features):
        return self.network(features).squeeze(-1).sigmoid()


class MonotoneEndpointCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.intercept = nn.Parameter(torch.zeros(()))

    def forward(self, probability):
        logits = torch.logit(probability.detach().clamp(1e-6, 1 - 1e-6))
        return (F.softplus(self.raw_slope) * logits + self.intercept).sigmoid()


class BackwardModel(nn.Module):
    def __init__(self, n_genes=32461, width=256):
        super().__init__()
        self.network = mlp([2 * width, 2 * width, n_genes])

    def forward(self, population, condition):
        return self.network(torch.cat((population, condition), dim=-1))

    def diagnostic(self, population, condition, reference_mean=None):
        if reference_mean is None:
            return None
        return (self(population, condition) - reference_mean).square().mean(-1)


def diagnostic_features(population, condition, backward_error, ensemble_variance):
    if backward_error is None or ensemble_variance is None:
        raise ValueError("DIAGNOSTIC_UNAVAILABLE: backward and ensemble features are required.")
    if not torch.isfinite(backward_error).all() or not torch.isfinite(ensemble_variance).all():
        raise ValueError("DIAGNOSTIC_UNAVAILABLE: non-finite diagnostic features.")
    if (backward_error < 0).any() or (ensemble_variance < 0).any():
        raise ValueError("DIAGNOSTIC_UNAVAILABLE: discrepancy and variance must be nonnegative.")
    return torch.cat((population, condition, backward_error[..., None], ensemble_variance[..., None]), dim=-1)


class CorrectionModel(nn.Module):
    def __init__(self, n_genes=32461, width=256):
        super().__init__()
        self.network = mlp([2 * width + 2, 2 * width, n_genes])
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def residual(self, population, condition, backward_error, ensemble_variance):
        return self.network(diagnostic_features(population, condition, backward_error, ensemble_variance)).tanh()

    def forward(self, raw, population, condition, backward_error=None, ensemble_variance=None,
                gate=False, training_candidate=False):
        """Gate supplied by a frozen domain table; candidate fitting explicitly forces 1."""
        gate_values = torch.as_tensor(gate)
        if not ((gate_values == 0) | (gate_values == 1)).all():
            raise ValueError("Correction gate must be a binary frozen-domain decision.")
        if not training_candidate and not torch.as_tensor(gate).bool().any():
            return raw
        if backward_error is None or ensemble_variance is None:
            if training_candidate:
                raise ValueError("Candidate K fitting requires valid B and ensemble features.")
            return raw
        delta = self.residual(population, condition, backward_error, ensemble_variance)
        multiplier = torch.ones_like(delta[..., :1]) if training_candidate else torch.as_tensor(gate, device=raw.device, dtype=raw.dtype)
        if multiplier.ndim == delta.ndim - 1:
            multiplier = multiplier[..., None]
        shift = multiplier * delta
        # [group,G] residual is shared over all unpaired cells in each group.
        if raw.ndim == shift.ndim + 1:
            shift = shift.unsqueeze(-2)
        return raw + shift
