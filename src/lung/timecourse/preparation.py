"""Independent full-axis assets; fitted exclusively to registered 24 h cells."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from lung.data.preprocess import log_normalize, weighted_moments, exact_neighbor_graph
from lung.engine.checkpoint import atomic_json
from lung.engine.config import config_hash, file_hash
from .protocol import GROUPS, VEHICLE, RULES, settings


def kernel_bandwidth(standardized):
    """Positive pair median in engine.rbf_kernel's mean-squared units.

    Duplicate pairs are excluded; a wholly constant population has no estimable
    positive scale and is rejected instead of receiving an arbitrary fallback.
    """
    from scipy.spatial.distance import pdist
    values = np.asarray(standardized, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("INVALID_KNOWN24_BANDWIDTH_INPUT")
    distances = pdist(values, metric="sqeuclidean") / values.shape[1]
    positive = distances[distances > 0]
    if not len(positive) or not np.isfinite(distances).all():
        raise ValueError("DEGENERATE_KNOWN24_BANDWIDTH")
    bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("DEGENERATE_KNOWN24_BANDWIDTH")
    return bandwidth


def source_paths(root, config):
    cfg = settings(config)
    root = Path(root)
    source = root / cfg.get("source", "08-datasets/sciPlex3/GSM4150378/extracted/v1")
    assets = root / cfg.get("assets", "08-datasets/sciPlex3/GSM4150378/timecourse-v1")
    return source.resolve(), assets.resolve()


def fingerprints(source, drugs):
    from lung.data.chemistry import build_chemical_catalog
    path = source.parent.parent / "sources/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz"
    frame = pd.read_csv(path, sep="\t")
    rows = []
    for drug in drugs:
        smiles = frame.loc[frame.catalog_number.eq(drug), "SMILES"].dropna().unique()
        if drug != VEHICLE and len(smiles) != 1:
            raise ValueError("AMBIGUOUS_SOURCE_SMILES: " + drug)
        rows.append({"drug_id": drug, "drug_name": drug,
                     "canonical_smiles": smiles[0] if len(smiles) else None,
                     "component_role": "vehicle" if drug == VEHICLE else "drug"})
    catalog, values, _ = build_chemical_catalog(pd.DataFrame(rows))
    if catalog.identity_status.eq("DRUG_IDENTITY_UNRESOLVED").any():
        raise ValueError("UNRESOLVED_FINGERPRINT")
    return {row.drug_id: values[row.fingerprint_row].tolist() for row in catalog.itertuples()}, path


def prepare_time_data(root, config, progress=None):
    source, assets = source_paths(root, config)
    cfg = settings(config)
    names = ["a549_cells.parquet", "common_genes.parquet", "extraction_status.json"]
    hashes = {name: file_hash(source / name) for name in names}
    extracted = json.loads((source / names[2]).read_text(encoding="utf-8"))
    if extracted["status"] != "complete_expression_extraction_not_training_ready":
        raise ValueError("SOURCE_EXTRACTION_INCOMPLETE")
    genes = pd.read_parquet(source / names[1])
    from lung.research.tidyi_diagnostics import validate_gene_axis, require_gene_order
    validate_gene_axis(genes)
    canonical = Path(root) / "08-datasets/LungCancer_A549_Integration/prepared/v1/gene_map.parquet"
    reference = pd.read_parquet(canonical)
    validate_gene_axis(reference)
    require_gene_order(genes.ensembl_id, reference.ensembl_id)
    require_gene_order(genes.ensembl_key, reference.ensembl_id)
    hashes["canonical_gene_axis"] = file_hash(canonical)
    metadata = pd.read_parquet(source / names[0])
    drugs = [drug for group in GROUPS.values() for drug in group] + [VEHICLE]
    fp, fp_path = fingerprints(source, drugs)
    hashes["chemical_metadata"] = file_hash(fp_path)
    identity = config_hash({"sources": hashes, "rules": RULES, "neighbors": cfg.get("neighbors", 32)})
    manifest_path = assets / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["status"] == "complete":
            if previous["sources"] != hashes or previous["rules"] != RULES:
                raise ValueError("TIME_ASSET_IDENTITY_CHANGED: use a new assets directory")
            with np.load(assets / "preprocessing.npz", allow_pickle=False) as saved:
                if saved["neighbors"].shape[1] != cfg.get("neighbors", 32):
                    raise ValueError("TIME_ASSET_NEIGHBORS_CHANGED: use a new assets directory")
            for name, digest in previous["files"].items():
                if file_hash(assets / name) != digest:
                    raise ValueError("TIME_ASSET_HASH_MISMATCH")
            return manifest_path
        if previous["identity"] != identity:
            raise ValueError("TIME_ASSET_IDENTITY_CHANGED: use a new assets directory")
    assets.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, {"status": "preparing", "identity": identity})
    selected = metadata.loc[metadata.catalog_number.isin(drugs) & metadata.time_point.isin([24, 72])].copy()
    if selected.author_hash_qc.isna().any() or not selected.author_hash_qc.isin([True, False]).all():
        raise ValueError("INVALID_HASH_QC")
    selected.to_parquet(assets / "cells.parquet", index=False)
    genes.to_parquet(assets / "genes.parquet", index=False)
    atomic_json(assets / "fingerprints.json", fp)
    known = selected.loc[selected.author_hash_qc.eq(True) & selected.time_point.eq(24)]
    if len(known) < 2:
        raise ValueError("INSUFFICIENT_KNOWN24")
    cache = assets / "known24"
    cache.mkdir(exist_ok=True)
    registry = {item["shard_id"]: item["common"] for item in extracted["shards"]}
    parts = []
    for number, (shard, rows) in enumerate(known.groupby("shard_id"), 1):
        specification = registry[int(shard)]
        path = source / specification["file"]
        if file_hash(path) != specification["sha256"]:
            raise ValueError("SOURCE_SHARD_HASH_MISMATCH")
        matrix = sparse.load_npz(path)[rows.row_in_shard.to_numpy(int)]
        if matrix.shape[1] != 32461 or not np.array_equal(np.asarray(matrix.sum(1)).ravel(), rows.common_totals_umi):
            raise ValueError("SOURCE_AXIS_OR_TOTAL_MISMATCH")
        normalized, valid = log_normalize(matrix)
        if not valid.all():
            raise ValueError("ZERO_KNOWN_LIBRARY")
        destination = cache / f"{int(shard):05d}.npz"
        sparse.save_npz(destination, normalized)
        parts.append(destination)
        if progress:
            progress("time_known24_cache", number, known.shard_id.nunique())
    def factory():
        for path in parts:
            matrix = sparse.load_npz(path)
            yield matrix, np.full(matrix.shape[0], 1 / len(known))
    mu, scale, sd, _ = weighted_moments(factory(), 32461, progress, len(parts))
    neighbors = exact_neighbor_graph(factory, mu, sd, genes.ensembl_id,
        neighbors=cfg.get("neighbors", 32), output_dir=assets / "graph_parts",
        device=config.get("device", "cuda:0"), progress=progress, total_parts=len(parts), input_identity=identity)
    # Fixed bandwidth estimated only from known24 standardized rows.
    sample = np.concatenate([sparse.load_npz(path)[:16].toarray() for path in parts[:8]])
    standardized = (sample - mu) / scale
    bandwidth = kernel_bandwidth(standardized)
    np.savez(assets / "preprocessing.npz", mu=mu, scale=scale, neighbors=neighbors, bandwidth=bandwidth)
    atomic_json(assets / "protocol.json", RULES)
    file_names = ["cells.parquet", "genes.parquet", "fingerprints.json", "preprocessing.npz", "protocol.json"]
    files = {name: file_hash(assets / name) for name in file_names}
    coordinate = config_hash({"genes": genes.ensembl_id.tolist(), "preprocessing": files["preprocessing.npz"]})
    atomic_json(manifest_path, {"status": "complete", "identity": identity, "coordinate_identity": coordinate,
        "sources": hashes, "source": str(source), "files": files, "known24_cells": len(known),
        "fit72_cells": 0, "synthetic": False, "rules": RULES, "shards": registry})
    return manifest_path
