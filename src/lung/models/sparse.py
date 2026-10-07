"""Indexed-neighbour attention; never materializes a gene-by-gene matrix."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def validate_neighbors(neighbors: torch.Tensor, n_genes: int) -> torch.Tensor:
    neighbors = torch.as_tensor(neighbors, dtype=torch.long)
    if neighbors.ndim != 2 or neighbors.shape[0] != n_genes or not neighbors.shape[1]:
        raise ValueError("neighbors must have shape [n_genes, nonzero_degree].")
    if (neighbors < 0).any() or (neighbors >= n_genes).any():
        raise ValueError("Neighbor index outside the gene panel.")
    ordered = neighbors.sort(dim=1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError("Repeated neighbour indices change the attention weights.")
    own = torch.arange(n_genes, device=neighbors.device)[:, None]
    if not (neighbors == own).any(dim=1).all():
        raise ValueError("Every neighbourhood must include its own gene.")
    return neighbors.clone()


class SparseAttention(nn.Module):
    """Standard or differential attention on a fixed directed neighbourhood."""

    def __init__(self, width, heads, chunk=256, differential=False,
                 checkpoint_chunks=True):
        super().__init__()
        if width % heads or chunk < 1:
            raise ValueError("Invalid attention head width or query chunk.")
        self.width, self.heads, self.chunk = width, heads, chunk
        self.head_dim = width // heads
        self.differential = differential
        self.checkpoint_chunks = checkpoint_chunks
        self.q1, self.k1, self.value = (nn.Linear(width, width) for _ in range(3))
        self.output = nn.Linear(width, width)
        if differential:
            self.q2, self.k2 = nn.Linear(width, width), nn.Linear(width, width)
            self.lambda_logit = nn.Parameter(torch.zeros(()))

    def _project(self, layer, tokens):
        batch, genes, _ = tokens.shape
        return layer(tokens).reshape(batch, genes, self.heads, self.head_dim).transpose(1, 2).contiguous()

    def project_keys(self, keys):
        """Return caller-owned key/value projections without detaching autograd."""
        return (self._project(self.k1, keys), self._project(self.value, keys),
                self._project(self.k2, keys) if self.differential else None)

    def _chunk(self, q1, k1, value, indices, q2=None, k2=None):
        batch, heads, queries, dims = q1.shape
        degree = indices.shape[1]

        def gather(matrix):
            return matrix.index_select(2, indices.reshape(-1)).reshape(batch, heads, queries, degree, dims)

        score = torch.matmul(q1.unsqueeze(-2), gather(k1).transpose(-1, -2)).squeeze(-2) / math.sqrt(dims)
        weights = score.softmax(dim=-1)
        if self.differential:
            score2 = torch.matmul(q2.unsqueeze(-2), gather(k2).transpose(-1, -2)).squeeze(-2) / math.sqrt(dims)
            weights = weights - 2 * self.lambda_logit.sigmoid() * score2.softmax(dim=-1)
        result = torch.matmul(weights.unsqueeze(-2), gather(value)).squeeze(-2)
        return result.transpose(1, 2).reshape(batch, queries, self.width)

    def forward_projected(self, query, projected, neighbors):
        """Attend to local projections produced by this module's project_keys."""
        k1, value, k2 = projected
        if query.ndim != 3 or k1.shape != (query.shape[0], self.heads, query.shape[1], self.head_dim):
            raise ValueError("Attention expects matching [batch, genes, width] axes.")
        genes = query.shape[1]
        q1 = self._project(self.q1, query)
        q2 = self._project(self.q2, query) if self.differential else None
        outputs = []
        for start in range(0, genes, self.chunk):
            end = min(start + self.chunk, genes)
            args = (q1[:, :, start:end], k1, value, neighbors[start:end],
                    q2[:, :, start:end] if q2 is not None else None, k2)
            if self.checkpoint_chunks and torch.is_grad_enabled():
                outputs.append(checkpoint(self._chunk, *args, use_reentrant=False))
            else:
                outputs.append(self._chunk(*args))
        return self.output(torch.cat(outputs, dim=1))

    def forward(self, query, keys, neighbors):
        if query.ndim != 3 or keys.shape != query.shape:
            raise ValueError("Attention expects matching [batch, genes, width] axes.")
        return self.forward_projected(query, self.project_keys(keys), neighbors)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = SparseAttention(width, config.n_heads, config.query_chunk,
                                        checkpoint_chunks=config.checkpoint_attention)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, tokens, neighbors):
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(normalized, normalized, neighbors)
        return tokens + self.ffn(self.norm2(tokens))


class PADBlock(nn.Module):
    """V1 conditioned differential self/cross attention, not upstream scDFM code."""

    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.injection_norm = nn.LayerNorm(width)
        self.injection = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.cross_norm = nn.LayerNorm(width)
        self.self_attention = SparseAttention(width, config.n_heads, config.query_chunk, True, config.checkpoint_attention)
        self.cross_attention = SparseAttention(width, config.n_heads, config.query_chunk, True, config.checkpoint_attention)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.modulations = nn.ModuleList()
        for _ in range(3):
            modulation = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, 3 * width))
            nn.init.zeros_(modulation[-1].weight)
            nn.init.zeros_(modulation[-1].bias)
            self.modulations.append(modulation)

    def prepare_reference(self, reference):
        """Prepare reference projections for reuse within one caller-owned graph."""
        return self.cross_attention.project_keys(self.cross_norm(reference))

    def forward(self, tokens, reference, condition, time_embedding, neighbors, reference_keys=None):
        width = tokens.shape[-1]
        first = self.injection[0]
        injected = (F.linear(self.injection_norm(tokens), first.weight[:, :width])
                    + F.linear(condition, first.weight[:, width:], first.bias)[:, None])
        tokens = tokens + self.injection[2](self.injection[1](injected))
        conditioning = torch.cat((condition, time_embedding), dim=-1)
        for index in range(3):
            alpha, beta, gate = self.modulations[index](conditioning).chunk(3, dim=-1)
            value = (1 + alpha[:, None]) * self.norms[index](tokens) + beta[:, None]
            if index == 0:
                update = self.self_attention(value, value, neighbors)
            elif index == 1:
                if reference_keys is None:
                    reference_keys = self.prepare_reference(reference)
                update = self.cross_attention.forward_projected(value, reference_keys, neighbors)
            else:
                update = self.ffn(value)
            tokens = tokens + gate[:, None].tanh() * update
        return tokens
