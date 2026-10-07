"""Cell, molecular, physical-condition and ordered-history encoders."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .sparse import EncoderBlock, validate_neighbors


def mlp(dimensions, activation=nn.GELU):
    layers = []
    for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(dimensions) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class CellEncoder(nn.Module):
    def __init__(self, config, neighbors):
        super().__init__()
        self.config = config
        self.register_buffer("neighbors", validate_neighbors(neighbors, config.n_genes))
        self.gene_embedding = nn.Embedding(config.n_genes, config.d_model)
        self.value_embedding = mlp([1, max(1, config.d_model // 2), config.d_model])
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_layers)])

    def tokenize(self, expression):
        if expression.ndim != 2 or expression.shape[1] != self.config.n_genes:
            raise ValueError("Expression must be [batch, n_genes] in the frozen gene order.")
        return self.value_embedding(expression[..., None]) + self.gene_embedding.weight[None]

    def forward(self, expression):
        tokens = self.tokenize(expression)
        for block in self.blocks:
            if self.config.checkpoint_blocks and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, self.neighbors, use_reentrant=False)
            else:
                tokens = block(tokens, self.neighbors)
        return tokens, tokens.mean(dim=1)


class DrugEncoder(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.network = nn.Sequential(mlp([2048, 2 * width, width]), nn.LayerNorm(width))
        self.vehicle_token = nn.Parameter(torch.zeros(width))

    def forward(self, fingerprint, vehicle=None):
        if fingerprint.ndim != 2 or fingerprint.shape[-1] != 2048:
            raise ValueError("Morgan fingerprints must have shape [batch, 2048].")
        if not torch.isfinite(fingerprint).all():
            raise ValueError("Missing or non-finite molecular fingerprint.")
        encoded = self.network(fingerprint)
        if vehicle is not None:
            vehicle = torch.as_tensor(vehicle, device=fingerprint.device, dtype=torch.bool)
            if vehicle.shape != fingerprint.shape[:1]:
                raise ValueError("Vehicle mask must be [batch].")
            encoded = torch.where(vehicle[:, None], self.vehicle_token[None], encoded)
        return encoded

    def forward_unchecked(self, fingerprint, vehicle=None):
        """Dense path without value-driven checks; caller validated (vmap, S03)."""
        encoded = self.network(fingerprint)
        if vehicle is not None:
            vehicle = torch.as_tensor(vehicle, device=fingerprint.device, dtype=torch.bool)
            encoded = torch.where(vehicle[:, None], self.vehicle_token[None], encoded)
        return encoded


class DoseEncoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.network = mlp([2, width, width])

    def forward(self, dose_um, vehicle=None):
        if dose_um.ndim != 1:
            raise ValueError("Dose must be [batch], expressed in micromolar.")
        vehicle = torch.zeros_like(dose_um, dtype=torch.bool) if vehicle is None else vehicle.bool()
        if vehicle.shape != dose_um.shape:
            raise ValueError("Dose and vehicle masks differ.")
        if ((~torch.isfinite(dose_um) | (dose_um < 0)) & ~vehicle).any():
            raise ValueError("INVALID_DOSE: ordinary drugs require finite nonnegative dose.")
        dose = torch.where(vehicle, torch.zeros_like(dose_um), dose_um)
        return self.network(torch.stack((dose.log1p(), vehicle.to(dose.dtype)), dim=-1))

    def forward_unchecked(self, dose_um, vehicle=None):
        """Dense path without value-driven checks; caller validated (vmap, S03)."""
        vehicle = torch.zeros_like(dose_um, dtype=torch.bool) if vehicle is None else vehicle.bool()
        dose = torch.where(vehicle, torch.zeros_like(dose_um), dose_um)
        return self.network(torch.stack((dose.log1p(), vehicle.to(dose.dtype)), dim=-1))


class DurationEncoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.network = mlp([1, width, width])

    def forward(self, duration_hours):
        if duration_hours.ndim != 1 or not torch.isfinite(duration_hours).all() or (duration_hours <= 0).any():
            raise ValueError("INVALID_DURATION: finite positive hours required.")
        return self.network(duration_hours.log1p()[:, None])

    def forward_unchecked(self, duration_hours):
        """Dense path without value-driven checks; caller validated (vmap, S03)."""
        return self.network(duration_hours.log1p()[:, None])


class HistoryEncoder(nn.Module):
    """A separate instance is required for each private trainable history path."""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config.history_input_dim
        self.width = config.d_model
        self.input_projection = nn.Linear(self.input_dim, self.width)
        self.gru = nn.GRU(self.width, self.width, batch_first=True)

    def forward(self, history, lengths=None):
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError("History must be [batch, segments, history_input_dim].")
        batch, segments, _ = history.shape
        if segments == 0:
            return history.new_zeros(batch, self.width)
        if lengths is None:
            lengths = torch.full((batch,), segments, device=history.device, dtype=torch.long)
        if lengths.shape != (batch,) or (lengths < 0).any() or (lengths > segments).any():
            raise ValueError("Invalid history lengths.")
        sequence, _ = self.gru(self.input_projection(history))
        index = (lengths - 1).clamp_min(0)
        selected = sequence[torch.arange(batch, device=history.device), index]
        return torch.where((lengths > 0)[:, None], selected, torch.zeros_like(selected))


class ConditionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, c = config.d_model, config.context_dim
        self.drug, self.dose, self.duration = DrugEncoder(d), DoseEncoder(c), DurationEncoder(c)
        self.background = nn.Embedding(config.n_backgrounds, c)
        self.source = nn.Embedding(config.n_sources, c)
        self.protocol = nn.Embedding(config.n_protocols, c)
        self.fusion = mlp([config.condition_input_dim, 2 * d, d])

    def forward_unchecked(self, fingerprint, dose, duration, background, source, protocol,
                          elapsed, horizon, vehicle=None, history=None):
        """Dense path with NO data-dependent Python branches (vmap domain, S03).

        The caller has already validated every input tensor outside vmap; this
        method only runs the pure dense computation. Semantics match forward().
        """
        batch = fingerprint.shape[0]
        if history is None:
            history = fingerprint.new_zeros(batch, self.config.d_model)
        parts = (self.drug.forward_unchecked(fingerprint, vehicle),
                 self.dose.forward_unchecked(dose, vehicle),
                 self.duration.forward_unchecked(duration),
                 self.background(background), self.source(source), self.protocol(protocol),
                 history, (elapsed / horizon)[:, None], (duration / horizon)[:, None])
        return self.fusion(torch.cat(parts, dim=-1))

    @staticmethod
    def _context(embedding, index):
        if index.ndim != 1 or index.dtype != torch.long:
            raise ValueError("Context ids must be integer [batch] tensors.")
        if (index < 0).any() or (index >= embedding.num_embeddings).any():
            raise ValueError("UNSUPPORTED_CONTEXT: id not in the frozen context registry.")
        return embedding(index)

    def forward(self, fingerprint, dose, duration, background, source, protocol,
                elapsed, horizon, vehicle=None, history=None):
        batch = fingerprint.shape[0]
        if elapsed.shape != (batch,) or horizon.shape != (batch,):
            raise ValueError("elapsed and horizon must be [batch] in hours.")
        if not torch.isfinite(elapsed).all() or not torch.isfinite(horizon).all():
            raise ValueError("Non-finite biological time.")
        if (horizon <= 0).any() or (elapsed < 0).any() or (elapsed > horizon).any():
            raise ValueError("Invalid biological horizon.")
        if history is None:
            history = fingerprint.new_zeros(batch, self.config.d_model)
        if history.shape != (batch, self.config.d_model):
            raise ValueError("Biological history must match [batch, d_model].")
        parts = (self.drug(fingerprint, vehicle), self.dose(dose, vehicle), self.duration(duration),
                 self._context(self.background, background), self._context(self.source, source),
                 self._context(self.protocol, protocol), history,
                 (elapsed / horizon)[:, None], (duration / horizon)[:, None])
        return self.fusion(torch.cat(parts, dim=-1))


class GenerativeTimeEncoder(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(32, dtype=torch.float64))
        self.network = mlp([64, width, width])

    def forward(self, tau):
        if tau.ndim != 1 or not torch.isfinite(tau).all() or (tau < 0).any() or (tau > 1).any():
            raise ValueError("Generative time tau must be [batch] in [0, 1].")
        return self._from_valid_tau(tau)

    def _from_valid_tau(self, tau):
        """Encode internally constructed [batch] times already known to be in [0, 1]."""
        # Compute high-frequency phase in float64 before casting back to model dtype.
        phase = (2 * math.pi * tau.double()[:, None] * self.frequencies[None])
        features = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1).to(tau.dtype)
        return self.network(features)
