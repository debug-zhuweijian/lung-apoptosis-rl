"""Small array fixtures for timecourse tests; never a production data source."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from lung.data.preprocess import weighted_moments, exact_neighbor_graph, log_normalize
from lung.engine.config import config_hash
from lung.timecourse.data import TimeData
from lung.timecourse.preparation import kernel_bandwidth
from lung.timecourse.protocol import GROUPS, VEHICLE


class ArrayRows:
    def __init__(self, counts):
        self.counts = counts

    def read(self, rows):
        return self.counts[rows.matrix_row.to_numpy(int)]


class ArrayTimeData(TimeData):
    @classmethod
    def from_arrays(cls, counts, cells, genes, fingerprints, *, fit_drugs, allowed_target_drugs,
                    role="fit", synthetic=False, drug_groups=None, neighbors=2):
        if synthetic is not True:
            raise ValueError("ARRAY_FIXTURE_REQUIRES_EXPLICIT_SYNTHETIC")
        obj = cls.__new__(cls)
        obj.synthetic, obj.root, obj.config = True, Path("."), {"synthetic": True}
        obj.assets_dir = None
        obj._array = sparse.csr_matrix(counts)
        obj._cells = pd.DataFrame(cells).reset_index(drop=True).copy()
        obj._cells["array_row"] = np.arange(len(obj._cells))
        obj._cells["shard_id"] = 0
        obj._cells["row_in_shard"] = obj._cells.array_row
        obj._matrix_paths = {0: "fixture"}
        obj._sparse_rows = ArrayRows(obj._array)
        obj.genes = np.asarray(genes).astype(str)
        obj.n_genes = len(obj.genes)
        if len(set(obj.genes)) != obj.n_genes or obj._array.shape != (len(obj._cells), obj.n_genes):
            raise ValueError("GENE_AXIS_MISMATCH")
        obj._fingerprints = {key: np.asarray(value).tolist() for key, value in fingerprints.items()}
        obj.drug_groups = copy.deepcopy(drug_groups or GROUPS)
        registered = sum(obj.drug_groups.values(), []) + [VEHICLE]
        known = obj._cells.time_point.eq(24) & obj._cells.author_hash_qc.eq(True) & obj._cells.catalog_number.isin(registered)
        values, valid = log_normalize(obj._array[np.flatnonzero(known)])
        if not valid.all() or len(valid) < 2:
            raise ValueError("INVALID_KNOWN24")
        def factory():
            yield values, np.full(len(valid), 1 / len(valid))
        obj.mu, obj.scale, sd, _ = weighted_moments(factory(), obj.n_genes)
        obj.neighbors = exact_neighbor_graph(factory, obj.mu, sd, obj.genes, neighbors=min(neighbors, obj.n_genes - 1))
        obj.bandwidth = kernel_bandwidth((values[:128].toarray() - obj.mu) / obj.scale)
        obj.coordinate_identity = config_hash({"genes": obj.genes.tolist(), "mu": obj.mu.tolist(),
            "scale": obj.scale.tolist(), "neighbors": obj.neighbors.tolist(),
            "bandwidth": obj.bandwidth, "synthetic": True})
        obj.identity = config_hash({"coordinate": obj.coordinate_identity, "cells": obj._cells.to_dict("records"),
            "counts": obj._array.toarray().tolist(), "fingerprints": obj._fingerprints})
        obj.manifest = {"identity": obj.identity, "synthetic": True}
        obj._initialize(fit_drugs, allowed_target_drugs, role)
        return obj
