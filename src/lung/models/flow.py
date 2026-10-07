"""Differentiable, full-interval conditional flow matching, architecture §6."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as recompute

from .encoders import CellEncoder, ConditionEncoder, GenerativeTimeEncoder
from .sparse import PADBlock


def heun_integrate(velocity, initial, steps, checkpoint=True, progress=None):
    """Integrate from 0 to 1; no teacher endpoint enters this computation."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("Heun integration requires a positive integer step count.")
    state = initial
    step_size = 1.0 / steps
    for index in range(steps):
        start = initial.new_full((initial.shape[0],), index / steps)
        end = initial.new_full((initial.shape[0],), (index + 1) / steps)

        def advance(current, tau_start, tau_end):
            first = velocity(current, tau_start)
            second = velocity(current + step_size * first, tau_end)
            return current + (step_size / 2) * (first + second)

        if checkpoint and torch.is_grad_enabled():
            state = recompute(advance, state, start, end, use_reentrant=False)
        else:
            state = advance(state, start, end)
        if progress is not None:
            progress(index + 1, steps)
    if not torch.isfinite(state).all():
        raise FloatingPointError("NONFINITE_ODE: complete generated endpoint is invalid.")
    return state


class PADVelocity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_encoder = GenerativeTimeEncoder(config.d_model)
        self.blocks = nn.ModuleList([PADBlock(config) for _ in range(config.pad_layers)])
        self.decoder = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))

    def prepare_reference(self, reference):
        """Trajectory-local projections, with the original reference gradient retained."""
        projected = []
        for block in self.blocks:
            if self.config.checkpoint_blocks and torch.is_grad_enabled():
                projected.append(recompute(block.prepare_reference, reference, use_reentrant=False))
            else:
                projected.append(block.prepare_reference(reference))
        return tuple(projected)

    def forward(self, tokens, tau, reference, condition, neighbors, reference_keys=None):
        time_embedding = self.time_encoder(tau)
        return self._from_time_embedding(tokens, time_embedding, reference, condition, neighbors, reference_keys)

    def _with_valid_tau(self, tokens, tau, reference, condition, neighbors, reference_keys=None):
        time_embedding = self.time_encoder._from_valid_tau(tau)
        return self._from_time_embedding(tokens, time_embedding, reference, condition, neighbors, reference_keys)

    def _from_time_embedding(self, tokens, time_embedding, reference, condition, neighbors, reference_keys=None):
        for index, block in enumerate(self.blocks):
            cached = None if reference_keys is None else reference_keys[index]
            if self.config.checkpoint_blocks and torch.is_grad_enabled():
                tokens = recompute(block, tokens, reference, condition, time_embedding,
                                   neighbors, reference_keys=cached, use_reentrant=False)
            else:
                tokens = block(tokens, reference, condition, time_embedding, neighbors, reference_keys=cached)
        return self.decoder(tokens).squeeze(-1)


class ResponseModel(nn.Module):
    """One ensemble member; preprocessing and ensemble management live outside."""

    def __init__(self, config, neighbors):
        super().__init__()
        self.config = config
        self.cell_encoder = CellEncoder(config, neighbors)
        self.condition_encoder = ConditionEncoder(config)
        self.velocity_model = PADVelocity(config)

    def encode_cells(self, expression):
        return self.cell_encoder(expression)

    def encode_condition(self, *args, **kwargs):
        return self.condition_encoder(*args, **kwargs)

    def sample_noise(self, reference, generator=None):
        return torch.randn(reference.shape, device=reference.device, dtype=reference.dtype,
                           generator=generator)

    def one_step_endpoints(self, reference, target, condition, noise=None, tau=None):
        """Teacher-path endpoints; this is not free-generation accuracy."""
        noise = torch.randn_like(target) if noise is None else noise
        tau = torch.rand(len(target), device=target.device, dtype=target.dtype) if tau is None else tau
        if noise.shape != target.shape or tau.shape != target.shape[:1]:
            raise ValueError("One-step noise or time shape mismatch.")
        fraction = tau[:, None]
        state = (1 - (1 - self.config.sigma_min) * fraction) * noise + fraction * target
        tokens, _ = self.encode_cells(self.condition_reference(reference))
        estimate = state + (1 - fraction) * self.velocity(state, tau, tokens, condition)
        endpoint = target + self.config.sigma_min * noise
        return estimate, endpoint

    def one_step_loss(self, reference, target, condition, *, bandwidth, noise=None, tau=None):
        from .objectives import rbf_mmd
        estimate, endpoint = self.one_step_endpoints(reference, target, condition, noise=noise, tau=tau)
        return rbf_mmd(estimate, endpoint, bandwidth=bandwidth, unbiased=False)

    def condition_reference(self, reference):
        """Apply a registered input ablation before splitting a population into microbatches.

        This operation is idempotent. Population means use only the observed control
        population, never treated labels; OT pairing still uses the original controls.
        """
        if reference.ndim != 2 or reference.shape[1] != self.config.n_genes or not len(reference):
            raise ValueError("Expected a nonempty reference population on the full gene axis.")
        if self.config.reference_mode == "population_mean":
            return reference.mean(0, keepdim=True).expand_as(reference)
        if self.config.reference_mode == "condition_only":
            return torch.zeros_like(reference)
        return reference

    def velocity(self, state, tau, tokens, condition):
        if condition.shape != (state.shape[0], self.config.d_model):
            raise ValueError("Condition batch must match the expression batch.")
        return self.velocity_model(self.cell_encoder.tokenize(state), tau, tokens,
                                   condition, self.cell_encoder.neighbors)

    def _velocity_with_valid_tau(self, state, tau, tokens, condition, reference_keys=None):
        """Only for time values generated by Heun integration or torch.rand below."""
        if condition.shape != (state.shape[0], self.config.d_model):
            raise ValueError("Condition batch must match the expression batch.")
        return self.velocity_model._with_valid_tau(self.cell_encoder.tokenize(state), tau,
                                                  tokens, condition, self.cell_encoder.neighbors, reference_keys)

    def cfm_loss(self, reference, target, condition, noise=None, tau=None, reduction="mean"):
        if reduction not in {"mean", "none"}:
            raise ValueError("CFM reduction must be mean or none")
        if reference.shape != target.shape or reference.ndim != 2:
            raise ValueError("CFM query and coupled target must have matching [batch, genes] shape.")
        noise = torch.randn_like(target) if noise is None else noise
        sampled_tau = tau is None
        tau = torch.rand(target.shape[0], device=target.device, dtype=target.dtype) if tau is None else tau
        if noise.shape != target.shape or tau.shape != target.shape[:1]:
            raise ValueError("CFM noise or generative-time shape mismatch.")
        fraction = tau[:, None]
        state = (1 - (1 - self.config.sigma_min) * fraction) * noise + fraction * target
        objective = target - (1 - self.config.sigma_min) * noise
        tokens, _ = self.encode_cells(self.condition_reference(reference))
        velocity = self._velocity_with_valid_tau if sampled_tau else self.velocity
        estimate = velocity(state, tau, tokens, condition)
        errors = (estimate - objective).square()
        return errors.mean() if reduction == "mean" else errors.mean(dim=1)

    def sample(self, reference, condition, noise=None, steps=32, checkpoint=True, progress=None,
               cache_reference=None):
        """Return raw standardized endpoints; call physical_projection next."""
        noise = torch.randn_like(reference) if noise is None else noise
        if noise.shape != reference.shape:
            raise ValueError("Generation noise must match the query expression shape.")
        tokens, _ = self.encode_cells(self.condition_reference(reference))
        # Keeping all projected keys in a differentiable trajectory raises the
        # measured full-gene replay peak; default reuse is inference/generation only.
        if cache_reference is None:
            cache_reference = not torch.is_grad_enabled()
        reference_keys = self.velocity_model.prepare_reference(tokens) if cache_reference else None
        return heun_integrate(lambda state, tau: self._velocity_with_valid_tau(
                                  state, tau, tokens, condition, reference_keys),
                              noise, steps, checkpoint, progress)

    def forward(self, state, tau, reference, condition):
        tokens, _ = self.encode_cells(self.condition_reference(reference))
        return self.velocity(state, tau, tokens, condition)


FlowModel = ResponseModel
