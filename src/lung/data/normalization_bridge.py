"""Affine standardization coordinates only; no model or identity compatibility."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class NormalizationCoordinates:
    gene_ids: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    base_transform: str
    units: str
    source_sha256: str
    clipping: str = "none"

    def __post_init__(self):
        genes = tuple(self.gene_ids)
        if np.iscomplexobj(self.mean) or np.iscomplexobj(self.scale):
            raise ValueError("Real normalization coordinates required")
        mean, scale = (np.array(x, dtype=np.float64, copy=True) for x in (self.mean, self.scale))
        if not genes or len(set(genes)) != len(genes) or not all(isinstance(x, str) and x for x in genes):
            raise ValueError("Unique ordered gene IDs required")
        if mean.shape != (len(genes),) or scale.shape != mean.shape:
            raise ValueError("Normalization axis shape mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Finite mean and positive finite scales required")
        if not self.base_transform or not self.units or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("Explicit transform, units and source SHA256 required")
        if self.clipping != "none":
            raise ValueError("Clipped coordinates are not invertible")
        mean.flags.writeable = scale.flags.writeable = False
        object.__setattr__(self, "gene_ids", genes)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def load(cls, path, *, expected_sha256, base_transform, units, clipping="none"):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("Normalization source SHA256 mismatch")
        import io
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            return cls(tuple(data["gene_ids"].astype(str)), data["mean"], data["scale"],
                       base_transform, units, expected_sha256, clipping)


def bridge_standardized(values, source, target):
    """Return float32 target coordinates; float32 source rounding remains."""
    if source.gene_ids != target.gene_ids:
        raise ValueError("Ordered gene axis mismatch")
    if (source.base_transform, source.units) != (target.base_transform, target.units):
        raise ValueError("Base transform or units mismatch")
    if source.clipping != "none" or target.clipping != "none":
        raise ValueError("Clipped coordinates are not invertible")
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise ValueError("Real expression coordinates required")
    if values.ndim < 1 or values.shape[-1] != len(source.gene_ids) or not np.isfinite(values).all():
        raise ValueError("Finite values on declared ordered gene axis required")
    result = (values.astype(np.float64) * (source.scale / target.scale)
              + (source.mean - target.mean) / target.scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Coordinate conversion overflow")
    return result
