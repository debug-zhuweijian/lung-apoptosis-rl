"""Audit the shared gene vocabulary; descriptive summaries are not fitted features."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAHOE_VERSION = "2dc57900b7981cfcf5e211527169a0b006546a95"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tidyi = ROOT / "08-datasets/TIDYI/E-MTAB-16354"
    gene_path = (ROOT / "08-datasets/Tahoe-100M" / TAHOE_VERSION
                 / "metadata/gene_metadata.parquet")
    expression_path = tidyi / "SS3_Human_A549_counts.csv"
    metadata_path = tidyi / "SS3_Human_A549_metadata.csv"
    output = ROOT / "08-datasets/LungCancer_A549_Integration/2026-09-18"
    output.mkdir(parents=True, exist_ok=True)

    genes = pd.read_parquet(gene_path)
    expression = pd.read_csv(expression_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    assert expression.index.is_unique and expression.columns.is_unique
    assert metadata.index.is_unique and genes.ensembl_id.is_unique
    assert expression.index.equals(metadata.index)
    assert np.isfinite(expression.to_numpy()).all()
    assert (expression.to_numpy() >= 0).all()
    assert (expression.to_numpy() % 1 == 0).all()

    a549 = expression.loc[metadata.Cell_line.eq("A549")]
    panel = genes.loc[genes.ensembl_id.isin(expression.columns)].copy()
    panel = panel.sort_values("ensembl_id").reset_index(drop=True)
    panel.to_csv(output / "shared_gene_vocabulary.csv", index=False)

    # These whole-source summaries must not be used to select model features.
    stats = panel.copy()
    stats["tidyi_detected_cells_descriptive"] = (
        (a549 > 0).sum().reindex(stats.ensembl_id).to_numpy())
    stats["tidyi_mean_counts_descriptive"] = (
        a549.mean().reindex(stats.ensembl_id).to_numpy())
    author_path = tidyi / "audit_sources/A549_author_collection_counts.npz"
    author_summary = {"status": "not_available"}
    if author_path.exists():
        author = np.load(author_path, allow_pickle=False)
        assert len(set(author["cell_ids"])) == len(author["cell_ids"])
        assert set(author["gene_ids"]) == set(expression.columns)
        ax = pd.DataFrame(author["expression"], index=author["cell_ids"],
                          columns=author["gene_ids"])
        assert np.array_equal(ax.loc[a549.index, a549.columns].to_numpy(),
                              a549.to_numpy())
        stats["tidyi_author_two_plate_detected_cells_descriptive"] = (
            (ax > 0).sum().reindex(stats.ensembl_id).to_numpy())
        author_summary = {
            "status": "author_training_collection_two_plates",
            "cells": len(ax), "overlap_with_original_strict_a549": len(a549),
            "plate_counts": {str(k): int(v) for k, v in
                             pd.Series(author["plates"]).value_counts().items()},
            "shared_genes_detected": int(
                (stats.tidyi_author_two_plate_detected_cells_descriptive > 0).sum()),
            "overlap_counts_identical": True,
            "sha256": digest(author_path),
        }
    smoke_path = (ROOT / "08-datasets/Tahoe-100M" / TAHOE_VERSION
                  / "derived/a549_expression_smoke.parquet")
    smoke_summary = {"status": "not_available"}
    if smoke_path.exists():
        smoke = pd.read_parquet(smoke_path)
        detected = pd.Series(0, index=genes.token_id.to_numpy(), dtype="int64")
        for tokens, values in zip(smoke.genes, smoke.expressions):
            assert len(tokens) == len(values)
            assert tokens[0] == 1 and values[0] == -2
            tokens, values = np.asarray(tokens[1:]), np.asarray(values[1:])
            assert len(tokens) == len(set(tokens))
            assert np.isfinite(values).all() and (values >= 0).all()
            assert (values % 1 == 0).all()
            assert set(tokens).issubset(detected.index)
            detected.loc[tokens[values > 0]] += 1
        stats["tahoe_smoke_detected_cells_descriptive"] = (
            detected.reindex(stats.token_id).to_numpy())
        smoke_summary = {
            "status": "two_shard_technical_sample_only",
            "cells": len(smoke),
            "shared_genes_detected": int(
                (stats.tahoe_smoke_detected_cells_descriptive > 0).sum()),
            "warning": "Not a full-source expression audit or a fitted training panel.",
            "sha256": digest(smoke_path),
        }
    stats.to_csv(output / "shared_gene_descriptive_statistics.csv", index=False)
    missing = metadata.loc[~metadata.Cell_line.eq("A549"),
                           ["Well", "Cell_line", "species", "cell_state"]]
    missing.to_csv(output / "unconfirmed_cell_identity.csv", index_label="cell_id")

    summary = {
        "date": "2026-09-18", "tahoe_commit": TAHOE_VERSION,
        "tidyi_all_rows": len(expression), "tidyi_confirmed_a549_rows": len(a549),
        "tidyi_genes": expression.shape[1], "tahoe_genes": len(genes),
        "shared_gene_ids": len(panel),
        "shared_genes_detected_in_at_least_one_tidyi_a549": int(
            (stats.tidyi_detected_cells_descriptive > 0).sum()),
        "shared_genes_zero_in_all_tidyi_a549": int(
            (stats.tidyi_detected_cells_descriptive == 0).sum()),
        "metadata_count_max_absolute_difference": float(
            (expression.sum(axis=1) - metadata.total_counts).abs().max()),
        "metadata_ngene_max_absolute_difference": float(
            ((expression > 0).sum(axis=1) - metadata.n_genes_by_counts).abs().max()),
        "feature_policy": "Fixed identity intersection only. Fit detection/variance/scaling inside training folds.",
        "tahoe_a549_detection_panel_status": "Not inferred from the global gene vocabulary.",
        "tahoe_expression_smoke": smoke_summary,
        "tidyi_author_collection": author_summary,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                   for p in [gene_path, expression_path, metadata_path]],
    }
    (output / "panel_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
