"""Frozen Tahoe classifier applied through an explicit, unvalidated coordinate bridge."""
import numpy as np
import torch

from lung.engine.auxiliary import encode_vectors, load_auxiliary, verify_auxiliary_dependencies
from lung.engine.checkpoint import load_checkpoint
from lung.engine.config import config_hash, file_hash
from lung.engine.data import open_data
from lung.engine.response import load_response


def convert_coordinates(x, time_mu, time_scale, tahoe_mu, tahoe_scale):
    if isinstance(x, torch.Tensor):
        arrays = [torch.as_tensor(a, dtype=torch.float64, device=x.device)
                  for a in (time_mu, time_scale, tahoe_mu, tahoe_scale)]
        if any(a.shape != arrays[0].shape for a in arrays) or (arrays[1] <= 0).any() or (arrays[3] <= 0).any():
            raise ValueError("INVALID_COORDINATE_BRIDGE")
        result = (x * arrays[1] + arrays[0] - arrays[2]) / arrays[3]
        if not torch.isfinite(result).all():
            raise ValueError("NONFINITE_COORDINATE_BRIDGE")
        return result.float()
    arrays = [np.asarray(a) for a in (time_mu, time_scale, tahoe_mu, tahoe_scale)]
    if any(a.shape != arrays[0].shape for a in arrays) or np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0):
        raise ValueError("INVALID_COORDINATE_BRIDGE")
    result = (np.asarray(x) * arrays[1] + arrays[0] - arrays[2]) / arrays[3]
    if not np.isfinite(result).all():
        raise ValueError("NONFINITE_COORDINATE_BRIDGE")
    return result.astype(np.float32)


class FrozenStateProxy:
    def __init__(self, time_data, pipeline, device="cpu"):
        saved = load_checkpoint(pipeline["state"])
        self.original_config = pipeline.get("original_config") or saved["metadata"]["config"]
        self.data = open_data(self.original_config, "state", prepare=False)
        self.time_data, self.device = time_data, device
        if self.data.n_genes != time_data.n_genes:
            raise ValueError("GENE_COORDINATE_MISMATCH")
        if not self.original_config.get("synthetic", False):
            genes = time_data.genes
            if hasattr(genes, "columns"):
                genes = genes["ensembl_id"].to_numpy()
            if not np.array_equal(np.asarray(genes).astype(str), self.data.reader.gene_ids):
                raise ValueError("GENE_ORDER_MISMATCH")
        verify_auxiliary_dependencies(pipeline["state"], pipeline)
        self.reference = load_response(pipeline["response"][0], self.data, device)
        self.classifier = load_auxiliary(pipeline["state"], self.data, device)
        self.identity = config_hash({"original_data": self.data.identity,
            "time_coordinates": time_data.coordinate_identity,
            "state": file_hash(pipeline["state"]), "reference": file_hash(pipeline["response"][0])})

    @torch.no_grad()
    def __call__(self, x):
        converted = convert_coordinates(x, self.time_data.mu, self.time_data.scale,
                                        self.data.mu, self.data.scale)
        z = encode_vectors(self.reference, torch.as_tensor(converted, device=self.device),
                           self.device, batch_size=1, output_device=self.device)
        return float(self.classifier.population_proxy(z))
