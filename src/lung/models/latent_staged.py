"""Staged latent response model (implementation task W3).

Split of the existing ``latent_flow`` joint objective into explicit stages:

- B1  ``SharedAutoencoder``: shared cell encoder + latent projection + full-gene
  decoder, pretrained on T/fit with the same dual-end full-gene MSE definition
  as ``LatentResponseModel.loss_components`` (control and treated ends).
- B2  freeze encoder/projection, cache h = encode_cells(x)[1] per cell.
- B3  ``LatentMember``: latent-space velocity network conditioned on the cached
  reference h and the drug condition; latent CFM with no ODE backprop.
- B4  refinement: each member owns a decoder initialised from the shared
  pretrained decoder; it stays frozen during CFM and may train in a refine
  stage while the encoder/projection (and the h cache) remain frozen.

Latent-space noise here is NOT gene-space additive noise; latent sigma follows
the registered latent path convention. ``population_mean`` ablation requires
encoding the per-condition control mean of raw expression (E(mean(x))); this
module only provides the ``condition_only`` constant h of zero expression.
"""

from dataclasses import replace

import torch
from torch import nn

from .compact import CompactCellEncoder, CompactVelocity
from .encoders import ConditionEncoder


class SharedAutoencoder(nn.Module):
    """B1/B2 module: everything shared across members, frozen after pretraining."""

    def __init__(self, config, neighbors):
        super().__init__()
        self.config = config
        self.cell_encoder = CompactCellEncoder(config, neighbors)
        self.latent_projection = nn.Linear(config.d_model, config.latent_dim)
        self.decoder = nn.Sequential(nn.Linear(config.latent_dim, config.d_model), nn.GELU(),
                                     nn.Linear(config.d_model, config.n_genes))

    def encode_h(self, expression):
        """Return (reference token [B,1,D], shared representation h [B,D])."""
        return self.cell_encoder(expression)

    def project(self, h):
        return self.latent_projection(h)

    def decode(self, z):
        return self.decoder(z)

    def reconstruction_loss(self, expression):
        """Dual-end full-gene MSE on the rows the caller provides (T only)."""
        tokens, h = self.encode_h(expression)
        reconstructed = self.decode(self.project(h))
        return (reconstructed - expression).square().mean()

    def zero_expression_h(self, n, device):
        """condition_only ablation: frozen representation of zero expression."""
        with torch.no_grad():
            _, h = self.encode_h(torch.zeros(n, self.config.n_genes, device=device))
        return h


class LatentMember(nn.Module):
    """B3/B4 module: one member's latent velocity net, condition encoder, decoder.

    The shared autoencoder is referenced WITHOUT registering it as a submodule
    (``object.__setattr__``), so member state dicts, parameter iterations and
    member-batched stacking never include the frozen shared weights.
    """

    def __init__(self, config, shared):
        super().__init__()
        self.config = config
        object.__setattr__(self, "_shared_ref", shared)
        self.velocity_model = CompactVelocity(replace(config, n_genes=config.latent_dim))
        self.condition_encoder = ConditionEncoder(config)
        self.decoder = nn.Sequential(nn.Linear(config.latent_dim, config.d_model), nn.GELU(),
                                     nn.Linear(config.d_model, config.n_genes))

    @property
    def shared(self):
        return self._shared_ref

    def init_decoder_from_shared(self):
        with torch.no_grad():
            target, source = self.decoder, self.shared.decoder
            for layer_t, layer_s in zip(target, source):
                if isinstance(layer_t, nn.Linear):
                    layer_t.weight.copy_(layer_s.weight)
                    layer_t.bias.copy_(layer_s.bias)

    def forward(self, latent_target, reference_h, fingerprint, dose, duration, background,
                source, protocol, elapsed, horizon, vehicle, noise=None, tau=None,
                validate=True):
        """Batched-trainable entry: condition encoding INSIDE the functional graph.

        Returns this member's scalar latent-CFM loss; ``functional_call``/``vmap``
        dispatch here, so velocity net and condition encoder gradients both land
        on the (stacked) member parameters (rework R1.1/R1.2). ``validate=False``
        skips input checks: callers running under vmap must validate the stacked
        inputs outside the batched call instead (rework S03).
        """
        if validate:
            condition = self.condition_encoder(fingerprint, dose, duration, background,
                                               source, protocol, elapsed, horizon, vehicle)
        else:
            condition = self.condition_encoder.forward_unchecked(
                fingerprint, dose, duration, background, source, protocol, elapsed,
                horizon, vehicle)
        return self.cfm_loss(latent_target, reference_h, condition, noise=noise, tau=tau)

    def velocity(self, state, tau, reference_h, condition):
        time_embedding = self.velocity_model.time_encoder._from_valid_tau(tau)
        return self.velocity_model(state, time_embedding, reference_h, condition)

    def cfm_loss(self, latent_target, reference_h, condition, *, noise=None, tau=None,
                 reduction="mean"):
        """Latent CFM regression; latent_target must already be detached (frozen E)."""
        if reduction not in {"mean", "none"}:
            raise ValueError("CFM reduction must be mean or none")
        if latent_target.ndim != 2 or latent_target.shape[1] != self.config.latent_dim:
            raise ValueError("Latent targets must be [batch, latent_dim].")
        noise = (torch.randn(latent_target.shape, device=latent_target.device,
                             dtype=latent_target.dtype) if noise is None else noise)
        tau = (torch.rand(len(latent_target), device=latent_target.device,
                          dtype=latent_target.dtype) if tau is None else tau)
        fraction = tau[:, None]
        state = (1 - (1 - self.config.sigma_min) * fraction) * noise + fraction * latent_target
        objective = latent_target - (1 - self.config.sigma_min) * noise
        estimate = self.velocity(state, tau, reference_h, condition)
        errors = (estimate - objective).square()
        return errors.mean() if reduction == "mean" else errors.mean(dim=1)

    @torch.no_grad()
    def sample_latent(self, reference_h, condition, *, steps=32, noise=None, generator=None):
        """Heun-integrate in latent space; caller decodes exactly once (task W3-B4)."""
        from .flow import heun_integrate
        if noise is None:
            noise = torch.randn((len(reference_h), self.config.latent_dim),
                                device=reference_h.device, dtype=reference_h.dtype,
                                generator=generator)
        if noise.shape != (len(reference_h), self.config.latent_dim):
            raise ValueError("Latent generation noise must match [batch, latent_dim].")
        return heun_integrate(
            lambda state, tau: self.velocity(state, tau, reference_h, condition),
            noise, steps, checkpoint=False)

    def decode(self, z):
        return self.decoder(z)
