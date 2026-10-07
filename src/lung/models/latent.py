"""Optional full-gene autoencoder with latent flow; a distinct research objective.

L = latent CFM(stopgrad(E(y))) + reconstruction_weight * full-gene AE MSE.
Population MMD compares decoded generated endpoints with observed gene expression.
Latent sigma is NOT gene-space additive noise. Accuracy requires independent V/E.
"""

from dataclasses import replace
import torch
from torch import nn

from .compact import CompactCellEncoder, CompactVelocity
from .encoders import ConditionEncoder
from .flow import ResponseModel, heun_integrate


class LatentResponseModel(ResponseModel):
    def __init__(self, config, neighbors):
        nn.Module.__init__(self)
        self.config = config
        self.cell_encoder = CompactCellEncoder(config, neighbors)
        self.condition_encoder = ConditionEncoder(config)
        self.latent_projection = nn.Linear(config.d_model, config.latent_dim)
        self.decoder = nn.Sequential(nn.Linear(config.latent_dim, config.d_model), nn.GELU(),
                                     nn.Linear(config.d_model, config.n_genes))
        self.velocity_model = CompactVelocity(replace(config, n_genes=config.latent_dim))

    def encode_latent(self, expression):
        return self.latent_projection(self.encode_cells(expression)[1])

    def sample_noise(self, reference, generator=None):
        return torch.randn((len(reference), self.config.latent_dim), device=reference.device,
                           dtype=reference.dtype, generator=generator)

    def velocity(self, state, tau, tokens, condition):
        if state.shape != (len(condition), self.config.latent_dim):
            raise ValueError("Latent velocity requires [batch, latent_dim].")
        return self.velocity_model(state, self.velocity_model.time_encoder(tau), tokens[:, 0], condition)

    def _velocity_with_valid_tau(self, state, tau, tokens, condition, reference_keys=None):
        return self.velocity_model(state, self.velocity_model.time_encoder._from_valid_tau(tau),
                                   tokens[:, 0], condition)

    def loss_components(self, reference, target, condition, noise=None, tau=None):
        latent = self.encode_latent(target)
        latent_target = latent.detach()
        noise = self.sample_noise(target) if noise is None else noise
        tau = torch.rand(len(target), device=target.device, dtype=target.dtype) if tau is None else tau
        if noise.shape != latent.shape or tau.shape != target.shape[:1]:
            raise ValueError("Latent CFM noise or time shape mismatch.")
        fraction = tau[:, None]
        state = (1 - (1 - self.config.sigma_min) * fraction) * noise + fraction * latent_target
        objective = latent_target - (1 - self.config.sigma_min) * noise
        tokens, _ = self.encode_cells(self.condition_reference(reference))
        cfm = (self.velocity(state, tau, tokens, condition) - objective).square().mean()
        reconstruction = .5 * ((self.decoder(latent) - target).square().mean()
            + (self.decoder(self.encode_latent(reference)) - reference).square().mean())
        return {"latent_cfm_loss": cfm, "reconstruction_loss": reconstruction}

    def cfm_loss(self, reference, target, condition, noise=None, tau=None):
        losses = self.loss_components(reference, target, condition, noise=noise, tau=tau)
        return losses["latent_cfm_loss"] + self.config.reconstruction_weight * losses["reconstruction_loss"]

    def one_step_endpoints(self, reference, target, condition, noise=None, tau=None):
        raise ValueError("LATENT_ONE_STEP_UNSUPPORTED: latent_flow requires full training schedule")

    def sample(self, reference, condition, noise=None, steps=32, checkpoint=True, progress=None,
               cache_reference=None):
        noise = self.sample_noise(reference) if noise is None else noise
        if noise.shape != (len(reference), self.config.latent_dim):
            raise ValueError("Generation noise must match [batch, latent_dim].")
        tokens, _ = self.encode_cells(self.condition_reference(reference))
        latent = heun_integrate(lambda state, tau: self._velocity_with_valid_tau(
            state, tau, tokens, condition), noise, steps, checkpoint, progress)
        return self.decoder(latent)
