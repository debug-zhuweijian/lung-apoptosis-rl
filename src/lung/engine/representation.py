"""Frozen, independently trained shared representation for auxiliary stages."""

import torch

from lung.models import ModelConfig
from lung.models.latent_staged import SharedAutoencoder
from .checkpoint import load_checkpoint
from .config import file_hash
from .representation_cache import encoder_identity
from .response import tensor


def latent_shape(metadata, data):
    """Validate the saved coordinate system before constructing a latent module."""
    if metadata.get("data_identity") != data.identity:
        raise ValueError("MODEL_DATA_IDENTITY_MISMATCH")
    shape = ModelConfig(**metadata["model_config"])
    if shape.n_genes != data.n_genes:
        raise ValueError("MODEL_GENE_AXIS_MISMATCH")
    if shape.backbone != "compact_full_gene" or shape.reference_mode != "observed":
        raise ValueError("LATENT_MODEL_CONFIG_MISMATCH")
    return shape


def require_trained(saved, error):
    steps = saved.get("counters", {}).get("optimizer_step", 0)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(error)


def require_finite_state(state):
    for value in state.values():
        if isinstance(value, torch.Tensor):
            values = value.coalesce().values() if value.is_sparse else value
            if not torch.isfinite(values).all():
                raise ValueError("NONFINITE_MODEL_WEIGHTS")


def verify_response_representation(member, reference):
    """Old members bind embedded AE tensors; new members also bind AE file bytes."""
    identity = getattr(member, "representation_identity", None)
    if identity is None:
        return
    current = {key: value for key, value in identity.items() if key != "checkpoint_sha256"}
    expected = {key: value for key, value in reference.identity.items() if key != "checkpoint_sha256"}
    if (current != expected or (member.representation_hash is not None
                               and member.representation_hash != reference.checkpoint_hash)):
        raise ValueError("LATENT_REPRESENTATION_IDENTITY_MISMATCH")


class SharedRepresentation:
    """The public auxiliary interface returns h, never projected response z."""

    def __init__(self, shared, checkpoint_hash):
        self.shared, self.config = shared, shared.config
        self.checkpoint_hash = checkpoint_hash
        self.identity = encoder_identity(shared, shared.config)

    def encode_cells(self, expression):
        return self.shared.encode_h(expression)


def load_representation(path, data, device):
    """Load an updated AE checkpoint without requiring a response member."""
    saved = load_checkpoint(path)
    metadata = saved.get("metadata", {})
    if metadata.get("stage") != "latent_ae":
        raise ValueError("TRAINED_REPRESENTATION_REQUIRED: expected latent_ae")
    require_trained(saved, "TRAINED_REPRESENTATION_REQUIRED: no optimizer updates")
    shape = latent_shape(metadata, data)
    require_finite_state(saved["model"])
    shared = SharedAutoencoder(shape, tensor(data.neighbors, "cpu", torch.long))
    shared.load_state_dict(saved["model"])
    shared.to(device).eval().requires_grad_(False)
    shared.representation_hash = file_hash(path)
    reference = SharedRepresentation(shared, shared.representation_hash)
    # Training entry points carry these values into the response member metadata.
    shared.representation_identity = reference.identity
    return reference
