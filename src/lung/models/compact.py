"""Full-gene compact candidate; not functionally equivalent to gene-token PAD."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as recompute

from .encoders import ConditionEncoder, GenerativeTimeEncoder
from .flow import ResponseModel, heun_integrate
from .sparse import validate_neighbors


class CompactCellEncoder(nn.Module):
    """Project expression and fixed graph means without gene-width token tensors."""

    def __init__(self, config, neighbors):
        super().__init__()
        self.config = config
        neighbors = validate_neighbors(neighbors, config.n_genes)
        rows = torch.arange(config.n_genes, device=neighbors.device)[:, None].expand_as(neighbors)
        indices = torch.stack((rows.flatten(), neighbors.flatten()))
        weights = torch.full((neighbors.numel(),), 1.0 / neighbors.shape[1], device=neighbors.device)
        self.register_buffer("graph", torch.sparse_coo_tensor(
            indices, weights, (config.n_genes, config.n_genes), check_invariants=True).coalesce())
        self.expression_projection = nn.Linear(config.n_genes, config.d_model)
        self.graph_projection = nn.Linear(config.n_genes, config.d_model, bias=False)
        self.output = nn.Sequential(nn.LayerNorm(config.d_model), nn.GELU())

    def graph_mean(self, expression):
        # CUDA sparse matmul does not support BF16. Keep this fixed graph
        # reduction in its stored dtype; dense projections still use autocast.
        with torch.autocast(device_type=expression.device.type, enabled=False):
            return torch.sparse.mm(
                self.graph, expression.to(self.graph.dtype).transpose(0, 1)).transpose(0, 1)

    def forward(self, expression):
        if expression.ndim != 2 or expression.shape[1] != self.config.n_genes:
            raise ValueError("Expression must be [batch, n_genes] in the frozen gene order.")
        vector = self.output(self.expression_projection(expression)
                             + self.graph_projection(self.graph_mean(expression)))
        return vector[:, None, :], vector


class CompactModulatedBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(width, 2 * width)
        self.network = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                     nn.Linear(2 * width, width))

    def forward(self, hidden, context):
        scale, shift = self.modulation(context).chunk(2, dim=-1)
        return hidden + self.network(self.norm(hidden) * (1 + scale.tanh()) + shift)


class CompactVelocity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_projection = nn.Linear(config.n_genes, config.d_model)
        self.time_encoder = GenerativeTimeEncoder(config.d_model)
        self.context_projection = nn.Linear(3 * config.d_model, config.d_model)
        self.blocks = nn.ModuleList([CompactModulatedBlock(config.d_model)
                                     for _ in range(config.pad_layers)])
        self.decoder = nn.Sequential(nn.LayerNorm(config.d_model),
                                     nn.Linear(config.d_model, 2 * config.n_genes))

    def forward(self, state, time_embedding, reference, condition):
        context = self.context_projection(torch.cat((reference, condition, time_embedding), dim=-1))
        hidden = self.state_projection(state) + context
        for block in self.blocks:
            if self.config.checkpoint_blocks and torch.is_grad_enabled():
                hidden = recompute(block, hidden, context, use_reentrant=False)
            else:
                hidden = block(hidden, context)
        scale, shift = self.decoder(hidden).chunk(2, dim=-1)
        return scale.tanh() * state + shift


class CompactResponseModel(ResponseModel):
    """Same full-gene CFM and full-interval ODE; a different velocity function family.

    encode_cells returns one summary token [B, 1, D] and a vector [B, D].
    Every gene enters the learned projections and every gene has a velocity output.
    No pretrained latent objective or additional supervision is introduced.
    """

    def __init__(self, config, neighbors):
        nn.Module.__init__(self)
        self.config = config
        self.cell_encoder = CompactCellEncoder(config, neighbors)
        self.condition_encoder = ConditionEncoder(config)
        self.velocity_model = CompactVelocity(config)

    def _estimate(self, state, time_embedding, tokens, condition):
        if state.ndim != 2 or state.shape[1] != self.config.n_genes:
            raise ValueError("State must have the full gene axis.")
        if condition.shape != (state.shape[0], self.config.d_model):
            raise ValueError("Condition batch must match the expression batch.")
        if tokens.shape != (state.shape[0], 1, self.config.d_model):
            raise ValueError("Reference summary must match the expression batch.")
        return self.velocity_model(state, time_embedding, tokens[:, 0], condition)

    def velocity(self, state, tau, tokens, condition):
        if tau.shape != state.shape[:1]:
            raise ValueError("Generative time must match the expression batch.")
        return self._estimate(state, self.velocity_model.time_encoder(tau), tokens, condition)

    def _velocity_with_valid_tau(self, state, tau, tokens, condition, reference_keys=None):
        return self._estimate(state, self.velocity_model.time_encoder._from_valid_tau(tau), tokens, condition)

    def sample(self, reference, condition, noise=None, steps=32, checkpoint=True, progress=None,
               cache_reference=None):
        noise = torch.randn_like(reference) if noise is None else noise
        if noise.shape != reference.shape:
            raise ValueError("Generation noise must match the query expression shape.")
        tokens, _ = self.encode_cells(self.condition_reference(reference))
        return heun_integrate(lambda state, tau: self._velocity_with_valid_tau(
            state, tau, tokens, condition), noise, steps, checkpoint, progress)

