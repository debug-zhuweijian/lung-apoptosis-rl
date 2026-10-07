"""Download the official A549 table and audit it without changing source files."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08-datasets" / "TIDYI" / "E-MTAB-16354"
BASE = "https://www.ebi.ac.uk/biostudies/files/E-MTAB-16354/"
SOURCES = {
    "SS3_Human_A549_metadata.csv": BASE + "SS3_Human_A549_metadata.csv",
    "SS3_Human_A549_counts.csv": BASE + "SS3_Human_A549_counts.csv",
    "E-MTAB-16354.study.json": "https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-16354",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "download_manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {}
    for filename, url in SOURCES.items():
        path = OUT / filename
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
                content_length = response.headers.get("Content-Length")
                if content_length and not response.headers.get("Content-Encoding"):
                    assert partial.stat().st_size == int(content_length), filename
            partial.replace(path)
            retrieved = datetime.now(timezone.utc).isoformat()
        else:
            retrieved = prior.get(filename, {}).get("retrieved_utc", "existing_file_first_audited_now")
        checksum = sha256(path)
        if filename in prior:
            assert checksum == prior[filename]["sha256"], f"Existing source changed: {filename}"
        manifest[filename] = {
            "url": url, "bytes": path.stat().st_size, "sha256": checksum,
            "retrieved_utc": retrieved, "audited_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Source available: {filename}: {path.stat().st_size:,} bytes", flush=True)


def audit() -> dict:
    meta = pd.read_csv(OUT / "SS3_Human_A549_metadata.csv", index_col=0)
    path = OUT / "SS3_Human_A549_counts.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        original_header = next(csv.reader(handle))
    counts = pd.read_csv(path, index_col=0)
    assert counts.index.is_unique and meta.index.is_unique, "Duplicate cell IDs"
    assert len(set(original_header[1:])) == len(original_header) - 1, "Duplicate gene IDs"
    assert set(meta.index) == set(counts.index), "Metadata/count cell IDs differ"
    counts = counts.loc[meta.index]
    values = counts.to_numpy(dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all(), "Invalid expression values"
    totals = values.sum(axis=1)
    detected = (values > 0).sum(axis=1)
    known_a549 = meta["Cell_line"].eq("A549")
    known_label = meta["cell_state"].isin(["PI-APC-", "PI-APC+", "PI+APC+"])
    eligible = known_a549 & known_label & (totals > 0)
    cell_qc = pd.DataFrame({
        "cell_id": meta.index,
        "cell_line_original": meta["Cell_line"].to_numpy(),
        "cell_state_original": meta["cell_state"].to_numpy(),
        "plate_original": meta["Plate"].to_numpy(),
        "expression_sum": totals, "detected_genes": detected,
        "strict_a549_nonempty_eligible": eligible.to_numpy(),
    })
    for name in ["Well", "pct_counts_mt", "doublet_scores", "predicted_doublets", "total_counts", "n_genes_by_counts"]:
        if name in meta:
            cell_qc[f"source_{name}"] = meta[name].to_numpy()
    cell_qc.to_csv(OUT / "A549_cell_qc.csv", index=False)
    well_columns = meta["Well"].str.extract(r"(\d+)", expand=False).astype(int)
    pd.crosstab(well_columns, meta["cell_state"]).to_csv(OUT / "A549_well_column_by_state.csv")
    gene_qc = pd.DataFrame({
        "ensembl_gene_id": counts.columns, "cells_detected": (values > 0).sum(axis=0),
        "expression_sum": values.sum(axis=0),
    })
    gene_qc.to_csv(OUT / "A549_gene_qc.csv", index=False)
    pd.DataFrame({"ensembl_gene_id": counts.columns}).to_csv(OUT / "A549_gene_ids.csv", index=False)
    reasons = np.where(~known_a549, "missing_or_non_A549_cell_line", np.where(~known_label, "missing_or_unrecognized_state", np.where(totals <= 0, "empty_expression", "")))
    pd.DataFrame({"cell_id": meta.index, "exclusion_reason": reasons}).loc[~eligible.to_numpy()].to_csv(OUT / "A549_exclusions.csv", index=False)
    # Keep all valid measured states. Do not remove dying cells using healthy-cell QC thresholds.
    np.savez_compressed(OUT / "A549_expression_audited.npz", expression=values.astype(np.float32), cell_ids=meta.index.to_numpy(dtype=str), gene_ids=counts.columns.to_numpy(dtype=str), eligible=eligible.to_numpy())
    expression_summary = pd.DataFrame({"sum": totals, "detected": detected, "state": meta["cell_state"].to_numpy()}).groupby("state").agg(["count", "min", "median", "max"])
    expression_summary.to_csv(OUT / "A549_state_expression_summary.csv")
    fractional = np.abs(values - np.rint(values)) > 1e-6
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_shape": list(meta.shape), "expression_shape": list(counts.shape),
        "all_cell_ids_match": True, "duplicate_cell_ids": 0, "duplicate_gene_ids": 0,
        "nonfinite_values": 0, "negative_values": 0,
        "noninteger_values": int(fractional.sum()), "maximum": float(values.max()),
        "zero_expression_cells": int((totals == 0).sum()),
        "all_zero_genes": int(((values > 0).sum(axis=0) == 0).sum()),
        "nonzero_fraction": float((values > 0).mean()),
        "identical_expression_rows": int(counts.duplicated().sum()),
        "cell_line_counts_including_missing": {str(k): int(v) for k, v in meta["Cell_line"].value_counts(dropna=False).items()},
        "state_counts_all": meta["cell_state"].value_counts().to_dict(),
        "state_counts_strict_a549": meta.loc[eligible, "cell_state"].value_counts().to_dict(),
        "strict_a549_eligible_cells": int(eligible.sum()),
        "plate_values": sorted(meta["Plate"].dropna().astype(str).unique().tolist()),
        "well_columns": sorted(well_columns.unique().tolist()),
        "missing_metadata_by_field": meta.isna().sum().loc[lambda s: s > 0].to_dict(),
        "sum_quantiles": pd.Series(totals).quantile([0, .25, .5, .75, 1]).to_dict(),
        "detected_quantiles": pd.Series(detected).quantile([0, .25, .5, .75, 1]).to_dict(),
    }
    for name, recomputed in [("total_counts", totals), ("n_genes_by_counts", detected)]:
        if name in meta:
            src = pd.to_numeric(meta[name], errors="coerce").to_numpy()
            good = np.isfinite(src)
            report[f"source_{name}_max_abs_difference"] = float(np.abs(src[good] - recomputed[good]).max())
    discrepancy = pd.DataFrame({
        "cell_id": meta.index,
        "source_total_minus_export": pd.to_numeric(meta["total_counts"]).to_numpy() - totals,
        "source_genes_minus_export": pd.to_numeric(meta["n_genes_by_counts"]).to_numpy() - detected,
    })
    discrepancy.to_csv(OUT / "A549_source_qc_discrepancies.csv", index=False)
    report["cells_with_source_qc_discrepancy"] = int((discrepancy.iloc[:, 1:] != 0).any(axis=1).sum())
    (OUT / "A549_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "missing_metadata_by_field"}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    download()
    audit()
