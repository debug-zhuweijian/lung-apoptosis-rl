"""Audit the complete official count stream and preserve A549 in sparse shards.

Raw human and unchanged common-panel counts are both retained. No normalization,
training split, model fitting, or merging of PAR_Y annotations is performed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "08-datasets/sciPlex3/GSM4150378"
PREFIX = "GSM4150378_sciPlex3_"


def sha(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(path)


def extract_stream(path, output, selected_cells, human_genes, common_genes, n_cells,
                   *, chunk_rows=5_000_000, checkpoint=None, progress=None):
    """Coordinates are zero-based arrays except the official source triples.

    Restarting scans the source again and checks existing committed shards instead
    of overwriting them. Boundaries never split a cell between output shards.
    """
    output.mkdir(parents=True, exist_ok=True)
    n_genes = max(max(human_genes, default=-1), max(common_genes, default=-1)) + 1
    # The caller passes the full annotation length separately in checkpoint.
    n_genes = (checkpoint or {}).get("source_gene_count", n_genes)
    human_map = np.full(n_genes, -1, dtype=np.int32)
    human_map[human_genes] = np.arange(len(human_genes))
    common_map = np.full(n_genes, -1, dtype=np.int32)
    common_map[common_genes] = np.arange(len(common_genes))
    selected = np.asarray(selected_cells, dtype=np.int64)
    if not np.all(np.diff(selected) > 0) or not np.all((selected >= 0) & (selected < n_cells)):
        raise ValueError("Selected cell coordinates must be unique and increasing")
    totals = np.zeros(n_cells, dtype=np.int64)
    human_totals = np.zeros(n_cells, dtype=np.int64)
    common_totals = np.zeros(n_cells, dtype=np.int64)
    par_y_totals = np.zeros(n_cells, dtype=np.int64)
    par_y = np.zeros(n_genes, dtype=bool)
    par_y[(checkpoint or {}).get("par_y_indices", [])] = True
    rows_written, source_records, prior_key, shard_id = 0, 0, -1, 0
    shard_list, row_links = [], []
    previous = {s["shard_id"]: s for s in (checkpoint or {}).get("shards", [])}

    def process(block):
        nonlocal prior_key, source_records, rows_written, shard_id
        if not len(block):
            return
        gene, cell, count = (block[:, j] for j in range(3))
        if not ((gene >= 1) & (gene <= n_genes) & (cell >= 1) & (cell <= n_cells)
                & (count > 0) & (count <= np.iinfo(np.int32).max)).all():
            raise ValueError("Invalid integer coordinate or count")
        key = cell * (n_genes + 1) + gene
        if key[0] <= prior_key or not np.all(np.diff(key) > 0):
            raise ValueError("Unsorted or duplicate sparse coordinate")
        prior_key = int(key[-1])
        gene, cell = gene - 1, cell - 1
        source_records += len(block)
        np.add.at(totals, cell, count)
        for mask, target in [(human_map[gene] >= 0, human_totals),
                             (common_map[gene] >= 0, common_totals), (par_y[gene], par_y_totals)]:
            np.add.at(target, cell[mask], count[mask])
        lo, hi = np.searchsorted(selected, [cell[0], cell[-1] + 1])
        chosen = selected[lo:hi]
        if not len(chosen):
            return
        locations = np.searchsorted(chosen, cell)
        in_selected = (locations < len(chosen))
        in_selected[in_selected] &= chosen[locations[in_selected]] == cell[in_selected]
        record = {"shard_id": shard_id, "cells": len(chosen), "a549_row_start": int(lo),
                  "a549_row_stop": int(hi), "source_cell_start": int(chosen[0]), "source_cell_stop": int(chosen[-1] + 1)}
        for name, mapping, width in [("human", human_map, len(human_genes)), ("common", common_map, len(common_genes))]:
            name_file = f"shard_{shard_id:05d}_{name}_counts.npz"
            target = output / name_file
            keep = in_selected & (mapping[gene] >= 0)
            matrix = sparse.csr_matrix((count[keep].astype(np.int32),
                       (locations[keep], mapping[gene[keep]])), shape=(len(chosen), width), dtype=np.int32)
            matrix.sort_indices()
            expected = previous.get(shard_id, {}).get(name, {})
            if target.exists():
                if expected.get("sha256") != sha(target):
                    raise RuntimeError("Existing shard without matching checkpoint; preserved for inspection")
                old = sparse.load_npz(target)
                if old.shape != matrix.shape or (old != matrix).nnz:
                    raise RuntimeError("Restart source-to-shard mismatch")
            else:
                temp = output / (name_file + ".tmp.npz")
                sparse.save_npz(temp, matrix, compressed=False)
                temp.rename(target)
            record[name] = {"file": name_file, "rows": len(chosen), "columns": width,
                            "nnz": matrix.nnz, "bytes": target.stat().st_size, "sha256": sha(target)}
        shard_list.append(record)
        row_links.append(pd.DataFrame({"source_cell_index": chosen, "shard_id": shard_id,
                                      "row_in_shard": np.arange(len(chosen))}))
        rows_written += len(chosen)
        shard_id += 1
        if progress:
            progress(source_records, rows_written, shard_list)

    carry = np.empty((0, 3), dtype=np.int64)
    with gzip.open(path, "rb") as gz:
        for frame in pd.read_csv(gz, sep="\t", header=None, dtype=np.int64, chunksize=chunk_rows):
            if frame.shape[1] != 3:
                raise ValueError("Expected exactly three sparse coordinate columns")
            block = frame.to_numpy()
            if len(carry):
                block = np.concatenate([carry, block])
            cut = int(np.searchsorted(block[:, 1], block[-1, 1], side="left"))
            process(block[:cut])
            carry = block[cut:].copy()
        process(carry)
        if gz.read(1):
            raise RuntimeError("Unexpected bytes after parser exhaustion")
    if rows_written != len(selected):
        raise ValueError("Some selected cells have no complete expression block")
    links = pd.concat(row_links, ignore_index=True)
    if links.source_cell_index.duplicated().any() or not np.array_equal(links.source_cell_index, selected):
        raise ValueError("Selected cell coverage mismatch")
    return {"source_records": source_records, "shards": shard_list, "links": links,
            "totals": totals, "human_totals": human_totals, "common_totals": common_totals,
            "par_y_totals": par_y_totals, "gzip_eof_and_crc_passed": True}


def main():
    output = DATA / "extracted/v1"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "extraction_status.json"
    source = DATA / "sources" / (PREFIX + "A549_MCF7_K562_screen_UMI.count.matrix.gz")
    download = json.loads((DATA / "expression_download_status.json").read_text(encoding="utf8"))
    if download.get("status") not in {"downloaded_pending_content_audit", "complete_content_audited"}:
        raise RuntimeError("Full expression download is not ready")
    if source.stat().st_size != download["source_bytes"] or sha(source) != download["sha256"]:
        raise RuntimeError("Source file identity mismatch")
    code_hash = sha(Path(__file__))
    prior = json.loads(status_path.read_text(encoding="utf8")) if status_path.exists() else {}
    if prior and (prior.get("source_sha256") != download["sha256"] or prior.get("code_sha256") != code_hash):
        raise RuntimeError("Extraction identity changed; use a new version")
    if prior.get("status") == "complete_expression_extraction_not_training_ready":
        print("Existing extraction complete; no overwrite.")
        return
    annotation = pd.read_csv(DATA / "sources" / (PREFIX + "A549_MCF7_K562_screen_cell.annotations.txt.gz"),
                             sep="\t", header=None, names=["cell", "sample"])
    annotation["source_cell_index"] = np.arange(len(annotation))
    meta = pd.read_csv(DATA / "sources" / (PREFIX + "pData.txt.gz"), sep=r"\s+", index_col=0)
    meta = annotation.merge(meta.reset_index(drop=True), on=["cell", "sample"], validate="one_to_one", how="left")
    if meta["n.umi"].isna().any():
        raise ValueError("Missing metadata count")
    genes = pd.read_csv(DATA / "sources" / (PREFIX + "A549_MCF7_K562_screen_gene.annotations.txt.gz"), sep=r"\s+")
    human = np.flatnonzero(genes.id.str.startswith("ENSG"))
    gene_map = pd.read_csv(DATA / "audit/shared_gene_axis.csv")
    common = gene_map.matrix_gene_index_1based.to_numpy(dtype=np.int64) - 1
    selected = meta.loc[meta.cell_type.eq("A549"), "source_cell_index"].to_numpy()
    status = {"status": "extracting_and_auditing", "source_sha256": download["sha256"],
              "source_bytes": source.stat().st_size, "code_sha256": code_hash,
              "source_gene_count": len(genes), "par_y_indices": np.flatnonzero(genes.id.str.endswith("_PAR_Y")).tolist(),
              "shards": prior.get("shards", []), "source_records": 0, "a549_cells_written": 0}
    def progress(records, cells, shards):
        status.update(source_records=records, a549_cells_written=cells, shards=shards,
                      updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, status)
        print(f"Matrix audit: {records:,} triples; A549 {cells:,}/{len(selected):,}", flush=True)
    atomic_json(status_path, status)
    try:
        result = extract_stream(source, output, selected, human, common, len(meta), checkpoint=status, progress=progress)
        expected = meta["n.umi"].to_numpy(dtype=np.int64)
        mismatches = np.flatnonzero(result["totals"] != expected)
        if len(mismatches):
            raise ValueError(f"UMI total mismatch for {len(mismatches)} source cells")
        a549 = pd.read_parquet(DATA / "audit/a549_cells_metadata.parquet")
        enriched = meta.loc[meta.cell_type.eq("A549"), ["cell", "sample", "source_cell_index"]].merge(
            a549, on=["cell", "sample"], validate="one_to_one")
        enriched = enriched.merge(result["links"], on="source_cell_index", validate="one_to_one").sort_values("source_cell_index")
        for key in ["totals", "human_totals", "common_totals", "par_y_totals"]:
            enriched[key + "_umi"] = result[key][selected]
        enriched.to_parquet(output / "a549_cells.parquet", index=False)
        genes.iloc[human].assign(source_gene_index=human).to_parquet(output / "human_genes.parquet", index=False)
        gene_map.to_parquet(output / "common_genes.parquet", index=False)
        np.savez(output / "all_source_cell_umi_audit.npz", actual=result["totals"], expected=expected)
        status.update(status="complete_expression_extraction_not_training_ready", gzip_eof_and_crc_passed=True,
                      source_records=result["source_records"], source_cells=len(meta), source_count_mismatches=0,
                      a549_cells_written=len(enriched), shards=result["shards"], human_gene_count=len(human),
                      common_gene_count=len(common), source_counts_preserved=True, normalization_applied=False,
                      qc_filter_applied_to_storage=False, train_val_test_split_created=False,
                      par_y_policy="preserved_in_human_panel_not_collapsed_into_common_panel",
                      updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, status)
        download.update(status="complete_content_audited", content_audit_passed=True,
                        audit_file=str(status_path.relative_to(DATA)))
        atomic_json(DATA / "expression_download_status.json", download)
        print(json.dumps({k: v for k, v in status.items() if k not in {"shards", "par_y_indices"}}, indent=2), flush=True)
    except Exception as error:
        status.update(status="failed_incomplete", error_type=type(error).__name__, message=str(error))
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
