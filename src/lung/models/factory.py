"""Explicit backbone selection, with legacy configs defaulting to gene tokens."""

from .compact import CompactResponseModel
from .flow import ResponseModel
from .latent import LatentResponseModel


def build_response_model(config, neighbors):
    if config.backbone == "gene_token":
        return ResponseModel(config, neighbors)
    if config.backbone == "compact_full_gene":
        return CompactResponseModel(config, neighbors)
    if config.backbone == "latent_flow":
        return LatentResponseModel(config, neighbors)
    raise ValueError(f"Unknown response backbone: {config.backbone}")
