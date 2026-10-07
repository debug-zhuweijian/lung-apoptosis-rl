"""Independent read-back audit of every extracted A549 sparse shard."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "08-datasets/sciPlex3/GSM4150378"


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def verify(data=DATA):
    out = data / "extracted/v1"
    manifest = json.loads((out / "extraction_status.json").read_text(encoding="utf8"))
    if manifest.get("status") != "complete_expression_extraction_not_training_ready":
        raise ValueError("EXTRACTION_NOT_COMPLETE")
    cells = pd.read_parquet(out / "a549_cells.parquet")
    human = pd.read_parquet(out / "human_genes.parquet")
    common = pd.read_parquet(out / "common_genes.parquet")
    if cells.duplicated(["sample", "cell"]).any() or cells.source_cell_index.duplicated().any():
        raise ValueError("DUPLICATE_CELL")
    human_to_common = human.reset_index(names="human_row").merge(
        common[["common_gene_index", "matrix_gene_index_1based"]].assign(
            source_gene_index=lambda x: x.matrix_gene_index_1based - 1),
        on="source_gene_index", how="inner", validate="one_to_one").sort_values("common_gene_index")
    if len(human_to_common) != len(common):
        raise ValueError("GENE_AXIS_MAPPING_FAILED")
    verified, rows, total_nnz = [], 0, {"human": 0, "common": 0}
    for record in manifest["shards"]:
        part = cells.loc[cells.shard_id.eq(record["shard_id"])].sort_values("row_in_shard")
        np.testing.assert_array_equal(part.row_in_shard, np.arange(len(part)))
        matrices = {}
        for panel in ("human", "common"):
            info = record[panel]
            path = (out / info["file"]).resolve()
            if not path.is_relative_to(out.resolve()) or digest(path) != info["sha256"]:
                raise ValueError("SHARD_HASH_OR_PATH_MISMATCH")
            matrix = sparse.load_npz(path)
            expected_columns = len(human) if panel == "human" else len(common)
            if (not sparse.isspmatrix_csr(matrix) or matrix.shape != (len(part), expected_columns)
                    or matrix.dtype != np.int32 or (matrix.data <= 0).any()
                    or not matrix.has_canonical_format or matrix.nnz != info["nnz"]):
                raise ValueError("SHARD_FORMAT_OR_VALUES_INVALID")
            np.testing.assert_array_equal(np.asarray(matrix.sum(axis=1)).ravel(), part[panel + "_totals_umi"])
            total_nnz[panel] += matrix.nnz
            matrices[panel] = matrix
        extracted = matrices["human"][:, human_to_common.human_row.to_numpy()]
        if (extracted != matrices["common"]).nnz:
            raise ValueError("HUMAN_COMMON_COUNTS_DIFFER")
        rows += len(part)
        verified.append(record["shard_id"])
        print(f"Read-back audit: {len(verified)}/{len(manifest['shards'])} shards; {rows:,} cells", flush=True)
    if rows != len(cells) or len(verified) != len(set(verified)):
        raise ValueError("SHARD_COVERAGE_INVALID")
    with np.load(out / "all_source_cell_umi_audit.npz", allow_pickle=False) as counts:
        np.testing.assert_array_equal(counts["actual"], counts["expected"])
        if len(counts["actual"]) != manifest["source_cells"]:
            raise ValueError("SOURCE_CELL_AUDIT_LENGTH")
    artifacts = [out / "a549_cells.parquet", out / "human_genes.parquet", out / "common_genes.parquet",
                 out / "all_source_cell_umi_audit.npz", out / "extraction_status.json"]
    report = {"status": "passed", "scope": "all_extracted_shards_and_source_cell_totals",
              "cells": len(cells), "verified_shards": len(verified), "human_genes": len(human),
              "common_genes": len(common), "nonzeros": total_nnz,
              "human_common_exact_count_agreement": True, "all_source_umi_agreement": True,
              "gzip_eof_and_crc_passed": manifest["gzip_eof_and_crc_passed"],
              "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in artifacts},
              "training_ready": False, "reason": "no normalization, model adapter or train/val/test split in this source version"}
    (out / "readback_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    verify()
