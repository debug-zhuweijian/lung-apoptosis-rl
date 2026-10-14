"""Eight frozen Tahoe fit-control rows; no model loading or execution."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.normalization_bridge import NormalizationCoordinates, bridge_standardized
from lung.data.preprocess import standardized

BASE = ROOT / "08-datasets/LungCancer_A549_Integration"
OUT = ROOT / "experiments/20261011-data-alignment-round16/normalization_bridge"
TRANSFORM = "float64:log1p(10000*common_counts/sum(common_counts));zero_library=reject"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "validation.json").exists():
        raise FileExistsError("Validation exists; retain its frozen evidence")
    reference_path = ROOT / "experiments/20261010-endpoint-feature-preflight-joint/final-v3/reference_inputs/reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    cell_path = Path(reference["cell_index"]["path"])
    assert sha(cell_path) == reference["cell_index"]["sha256"]
    pool = pd.read_parquet(cell_path)
    assert len(pool) == 2431 and pool.cell_id.is_unique
    old, new = BASE / "training-v1", BASE / "training-joint-sciplex-v1"
    old_rows = pd.read_parquet(old / "train.parquet", filters=[("condition_id", "in", reference["control_condition_ids"]), ("role", "==", "fit")])
    assert set(old_rows.cell_id) == set(pool.cell_id)
    rows = old_rows.sort_values("cell_id").iloc[:8].copy()
    matched = pool.set_index("cell_id").loc[rows.cell_id]
    for key in ("matrix_path", "matrix_row", "condition_id"):
        assert np.array_equal(rows[key].to_numpy(), matched[key].to_numpy())
    assert rows.role.eq("fit").all() and rows.source_id.eq("Tahoe-100M").all()
    assets, coordinates, genes = {}, [], []
    for directory in (old, new):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name in ("genes.parquet", "normalization.npz", "train.parquet"):
            path = directory / name
            actual = sha(path)
            assert actual == manifest["files"][name]["sha256"]
            assets[str(path.relative_to(ROOT))] = actual
        gene = pd.read_parquet(directory / "genes.parquet")
        genes.append(gene)
        norm = NormalizationCoordinates.load(directory / "normalization.npz",
            expected_sha256=assets[str((directory / "normalization.npz").relative_to(ROOT))],
            base_transform=TRANSFORM, units="same_Tahoe_source_raw_count_units_unknown")
        assert tuple(gene.ensembl_id) == norm.gene_ids
        coordinates.append(norm)
    assert len(genes[0]) == 32461
    assert np.array_equal(genes[0].tahoe_column, genes[1].tahoe_column)
    raw_parts, order = [], []
    for name, part in rows.reset_index(drop=True).groupby("matrix_path", sort=False):
        path = ROOT / name
        before = sha(path)
        registry = pd.read_parquet(new / "source_shards.parquet").set_index("matrix_path")
        entry = registry.loc[name]
        assert entry.gene_axis == "tahoe"
        # Source identity is independently bound by the joint source registry.
        hash_fields = [key for key in entry.index if "sha256" in key]
        assert any(str(entry[key]) == before for key in hash_fields)
        matrix = sparse.load_npz(path).tocsr()
        raw_parts.append(matrix[part.matrix_row.to_numpy(dtype=np.int64)][:, genes[0].tahoe_column.to_numpy(dtype=np.int64)])
        order.extend(part.index.tolist())
        assert sha(path) == before
        assets[name] = before
    counts = sparse.vstack(raw_parts, format="csr")[np.argsort(order)]
    assert np.isfinite(counts.data).all() and (counts.data >= 0).all()
    old_norm, new_norm = coordinates
    direct = standardized(counts, old_norm.mean, old_norm.scale)
    newer = standardized(counts, new_norm.mean, new_norm.scale)
    bridged = bridge_standardized(newer, new_norm, old_norm)
    reverse = bridge_standardized(direct, old_norm, new_norm)
    np.savez(OUT / "eight_cells_coordinates.npz", direct_old=direct, new=newer, new_to_old=bridged, old_to_new=reverse)
    with np.load(OUT / "eight_cells_coordinates.npz", allow_pickle=False) as saved:
        assert all(np.array_equal(saved[key], value) for key, value in
                   (("direct_old", direct), ("new", newer), ("new_to_old", bridged), ("old_to_new", reverse)))
    error = np.abs(bridged.astype(np.float64) - direct)
    reverse_error = np.abs(reverse.astype(np.float64) - newer)
    atol, rtol = 2e-6, 2e-6
    assert np.isfinite(bridged).all() and np.allclose(bridged, direct, atol=atol, rtol=rtol)
    assert np.allclose(reverse, newer, atol=atol, rtol=rtol)
    rows.to_parquet(OUT / "selected_cells.parquet", index=False)
    for name in ("src/lung/data/preprocess.py", "src/lung/data/reader.py", "src/lung/engine/data.py", "src/lung/data/normalization_bridge.py", "scripts/verify_normalization_bridge.py"):
        assets[name] = sha(ROOT / name)
    assets[str(cell_path.relative_to(ROOT))] = sha(cell_path)
    result = {"scope": "standardization_coordinates_only_no_model_certification", "cells": rows.cell_id.tolist(),
        "pool_count": 2431, "shape": list(direct.shape), "base_transform": TRANSFORM,
        "clipping": "none_in_standardized_or_reader_row_conversion", "output_dtype": "float32",
        "source_units": "same_Tahoe_counts; biological measurement units unknown; no cross-source equivalence",
        "max_abs_error": float(error.max()), "max_relative_error_floor_1": float((error / np.maximum(np.abs(direct), 1.)).max()),
        "reverse_max_abs_error": float(reverse_error.max()), "atol": atol, "rtol": rtol, "finite": True,
        "bitwise_equal": bool(np.array_equal(direct, bridged)), "asset_sha256": assets,
        "artifacts_sha256": {name: sha(OUT / name) for name in ("selected_cells.parquet", "eight_cells_coordinates.npz")},
        "protocol_mapping": "Tahoe continuous24h old ID=0 joint ID=1; semantic mapping only, encoding unchanged",
        "uncertified": ["weights", "AE-response binding", "pool/state", "graph compatibility", "data identity migration", "model forward", "drug effect", "cross-source transfer"]}
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("asset_sha256", "cells")}, indent=2))


if __name__ == "__main__":
    main()
