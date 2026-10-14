"""Recover the published TIDYI/Tahoe identity axis and measured A549 state rows.

This creates a NEW source-bound reference, not a replacement for frozen training
assets. No abundance-based gene selection, normalization, fitting or role changes.
The resulting genes.parquet is consumed by recover_public_expression --gene-axis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def qc_difference_signature(counts: pd.DataFrame, metadata: pd.DataFrame) -> str:
    """Pin the exact per-cell discrepancy, rather than invent a tolerance."""
    rows = [[str(cell), float(total), float(detected)] for cell, total, detected in zip(
        counts.index, pd.to_numeric(metadata.total_counts).to_numpy() - counts.to_numpy(dtype=np.float64).sum(axis=1),
        pd.to_numeric(metadata.n_genes_by_counts).to_numpy() - (counts.to_numpy(dtype=np.float64) > 0).sum(axis=1))]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_reference(genes: pd.DataFrame, counts: pd.DataFrame, metadata: pd.DataFrame,
                    *, expected_qc_difference_sha256: str | None = None):
    """Validate native tables and return identity axis, confirmed cells and counts."""
    required_genes = {"ensembl_id", "gene_symbol", "token_id"}
    required_meta = {"Cell_line", "Plate", "Well", "cell_state", "total_counts", "n_genes_by_counts"}
    if not required_genes.issubset(genes) or not required_meta.issubset(metadata):
        raise ValueError("Missing native source columns")
    if (genes.empty or counts.empty or not genes.columns.is_unique or not metadata.columns.is_unique
            or not genes.ensembl_id.is_unique or not genes.token_id.is_unique
            or genes.ensembl_id.isna().any()
            or not genes.ensembl_id.astype(str).str.fullmatch(r"ENSG\d{11}").all()
            or not counts.columns.is_unique or not counts.index.is_unique or not metadata.index.is_unique
            or counts.index.isna().any() or metadata.index.isna().any()
            or not counts.index.equals(metadata.index)
            or not counts.columns.astype(str).str.fullmatch(r"ENSG\d{11}").all()):
        raise ValueError("Invalid or misaligned native gene/cell identities")
    token = pd.to_numeric(genes.token_id, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(token).all() or np.any(token < 3) or np.any(token != np.floor(token)):
        raise ValueError("Invalid source gene tokens")
    values = counts.to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values > np.iinfo(np.int32).max) or np.any(values != np.floor(values))):
        raise ValueError("Native TIDYI counts are not finite nonnegative int32-compatible counts")
    source_qc = metadata[["total_counts", "n_genes_by_counts"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(source_qc).all() or np.any(source_qc < 0) or np.any(source_qc != np.floor(source_qc)):
        raise ValueError("Invalid original QC values")
    difference = source_qc - np.column_stack([values.sum(axis=1), (values > 0).sum(axis=1)])
    if expected_qc_difference_sha256 is not None:
        if qc_difference_signature(counts, metadata) != expected_qc_difference_sha256:
            raise ValueError("Original QC discrepancy differs from the explicit source contract")
    elif np.any(difference != 0):
        raise ValueError("Original QC and exported matrix disagree; an explicit source discrepancy contract is required")
    metadata = metadata.copy()
    metadata["export_total_counts"] = values.sum(axis=1)
    metadata["export_detected_genes"] = (values > 0).sum(axis=1)
    metadata["source_total_minus_export"] = difference[:, 0]
    metadata["source_genes_minus_export"] = difference[:, 1]
    tahoe = genes.sort_values("token_id", kind="stable").reset_index(drop=True).copy()
    tahoe["tahoe_column"] = np.arange(len(tahoe), dtype=np.int64)
    axis = tahoe.loc[tahoe.ensembl_id.isin(counts.columns)].sort_values("ensembl_id", kind="stable").reset_index(drop=True)
    if axis.empty:
        raise ValueError("No exact shared Ensembl identities")
    axis = axis.rename(columns={"token_id": "tahoe_token_id"})
    axis.insert(0, "common_gene_index", np.arange(len(axis), dtype=np.int64))
    axis["tidyi_column"] = counts.columns.get_indexer(axis.ensembl_id)
    confirmed = metadata.Cell_line.eq("A549").fillna(False).to_numpy(dtype=bool)
    if not confirmed.any():
        raise ValueError("No source-confirmed A549 cells")
    cells = metadata.loc[confirmed].copy()
    if cells[["Plate", "Well", "cell_state"]].isna().any().any():
        raise ValueError("Confirmed cells lack original plate/well/state annotations")
    cells.insert(0, "source_cell_id", cells.index.astype(str))
    cells = cells.reset_index(drop=True).rename(columns={"cell_state": "state_label_original"})
    cells.insert(0, "matrix_row", np.arange(len(cells), dtype=np.int64))
    cells["data_role"] = "unassigned_external_source"
    cells["label_scope"] = "original_sorted_state_not_natural_apoptosis_rate"
    exclusions = metadata.loc[~confirmed].copy()
    exclusions.insert(0, "source_cell_id", exclusions.index.astype(str))
    exclusions["exclusion_reason"] = "unconfirmed_A549_identity"
    exclusions = exclusions.reset_index(drop=True)
    projected = values[confirmed][:, axis.tidyi_column].astype(np.int32)
    return axis, cells, projected, exclusions


def recover_reference(gene_path: Path, count_path: Path, metadata_path: Path, output: Path,
                      source_contract: Path | None = None) -> dict:
    """Produce a source-bound, round-trip-checked reference in a new directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; frozen assets are never overwritten")
    paths = {"genes": Path(gene_path), "counts": Path(count_path), "metadata": Path(metadata_path)}
    sources = {name: {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
               for name, path in paths.items()}
    contract = None
    if source_contract is not None:
        contract = json.loads(Path(source_contract).read_text(encoding="utf-8"))
        for name, source in sources.items():
            expected = contract["sources"][name]
            if source["sha256"] != expected["sha256"] or source["bytes"] != expected["bytes"]:
                raise ValueError(f"Pinned source identity mismatch: {name}")
            source.update(url=expected["url"], source_version=expected["source_version"])
    genes = pd.read_parquet(gene_path)
    counts = pd.read_csv(count_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    axis, cells, values, exclusions = build_reference(genes, counts, metadata,
        expected_qc_difference_sha256=contract.get("expected_qc_difference_sha256") if contract else None)
    # This content hash is independent of Parquet encoding and library versions.
    axis_hash = hashlib.sha256(("\n".join(axis.ensembl_id) + "\n").encode("ascii")).hexdigest()
    report = {"source_rows": len(counts), "confirmed_a549_cells": len(cells), "excluded_identity_rows": len(exclusions),
              "tahoe_genes": len(genes), "tidyi_genes": counts.shape[1], "shared_genes": len(axis),
              "source_gene_axis_only_tahoe": int((~genes.ensembl_id.isin(counts.columns)).sum()),
              "source_gene_axis_only_tidyi": int((~counts.columns.isin(genes.ensembl_id)).sum()),
              "plate_counts": {str(k): int(v) for k, v in cells.Plate.value_counts().items()},
              "state_counts": {str(k): int(v) for k, v in cells.state_label_original.value_counts().items()},
              "ordered_ensembl_newline_sha256": axis_hash,
              "source_qc_difference_sha256": qc_difference_signature(counts, metadata),
              "source_qc_matches_export": bool(np.array_equal(counts.sum(axis=1), metadata.total_counts)
                                               and np.array_equal((counts > 0).sum(axis=1), metadata.n_genes_by_counts)),
              "common_axis_measured_zeros": int((values == 0).sum()),
              "common_axis_unobserved_genes": 0, "expected_source_hashes_verified": contract is not None,
              "frozen_asset_identity_verified": False, "normalization_performed": False,
              "training_performed": False, "whole_project_qualified": False}
    if contract is not None:
        for name, expected in contract.get("expected", {}).items():
            if report.get(name) != expected:
                raise ValueError(f"Pinned source scope mismatch: {name}")
    output.mkdir(parents=True)
    axis.to_parquet(output / "genes.parquet", index=False)
    cells.to_parquet(output / "tidyi_cells.parquet", index=False)
    exclusions.to_csv(output / "identity_exclusions.csv", index=False)
    np.savez_compressed(output / "tidyi_common_counts.npz", expression=values,
                        observed_mask=np.ones(values.shape, dtype=bool),
                        sample_ids=cells.source_cell_id.to_numpy(dtype=str),
                        ensembl_ids=axis.ensembl_id.to_numpy(dtype=str))
    pd.testing.assert_frame_equal(pd.read_parquet(output / "genes.parquet"), axis)
    with np.load(output / "tidyi_common_counts.npz", allow_pickle=False) as restored:
        np.testing.assert_array_equal(restored["expression"], values)
        np.testing.assert_array_equal(restored["sample_ids"], cells.source_cell_id.to_numpy(dtype=str))
        np.testing.assert_array_equal(restored["ensembl_ids"], axis.ensembl_id.to_numpy(dtype=str))
        if not restored["observed_mask"].all():
            raise ValueError("Measured common gene mask changed during serialization")
    for name, path in paths.items():
        if digest(path) != sources[name]["sha256"]:
            raise ValueError(f"Source changed during recovery: {name}")
    report["roundtrip_exact"] = True
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "validation": report, "sources": sources, "script_sha256": digest(Path(__file__)),
                "source_contract_sha256": digest(source_contract) if source_contract else None,
                "outputs": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in output.iterdir()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-table", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, default=ROOT / "configs/public_reference_sources_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover_reference(args.gene_table, args.counts, args.metadata, args.output, args.source_contract), indent=2))


if __name__ == "__main__":
    main()
