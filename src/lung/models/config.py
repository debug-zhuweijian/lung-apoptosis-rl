"""Shape contract for the thesis V1 biological models (architecture §§4–9)."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    n_genes: int = 32461
    d_model: int = 256
    n_heads: int = 4
    encoder_layers: int = 2
    pad_layers: int = 6
    query_chunk: int = 256
    checkpoint_blocks: bool = True
    checkpoint_attention: bool = True
    n_backgrounds: int = 1
    n_sources: int = 2
    n_protocols: int = 1
    context_dim: int = 32
    sigma_min: float = 1e-3
    reference_mode: str = "observed"
    backbone: str = "gene_token"
    latent_dim: int = 64
    reconstruction_weight: float = 1.0

    def __post_init__(self):
        if self.backbone not in {"gene_token", "compact_full_gene", "latent_flow"}:
            raise ValueError("Unknown response backbone.")
        positive = (self.n_genes, self.d_model, self.n_heads, self.encoder_layers,
                    self.pad_layers, self.query_chunk, self.n_backgrounds,
                    self.n_sources, self.n_protocols, self.context_dim, self.latent_dim)
        if any(value < 1 for value in positive):
            raise ValueError("Model dimensions and layer counts must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 < self.sigma_min < 1:
            raise ValueError("sigma_min must lie strictly between zero and one.")
        if self.reference_mode not in {"observed", "population_mean", "condition_only"}:
            raise ValueError("Unknown response reference ablation.")
        if not math.isfinite(self.reconstruction_weight) or self.reconstruction_weight <= 0:
            raise ValueError("reconstruction_weight must be finite and positive.")

    @property
    def condition_input_dim(self):
        return 2 * self.d_model + 5 * self.context_dim + 2

    @property
    def history_input_dim(self):
        return self.d_model + 3 * self.context_dim + 1

    @property
    def state_dim(self):
        return 2 * self.d_model + 3 * self.context_dim + 29

