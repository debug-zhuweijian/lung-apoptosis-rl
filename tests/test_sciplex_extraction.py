"""Synthetic raw-coordinate checks; never read or fit the real expression data."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/extract_sciplex_a549.py"
spec = importlib.util.spec_from_file_location("sciplex_extract", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_source(path, text):
    with gzip.open(path, "wt") as dst:
        dst.write(text)


def test_full_counts_panels_cell_boundaries_and_restart(tmp_path):
    source = tmp_path / "counts.gz"
    write_source(source, "1\t1\t3\n2\t1\t4\n5\t1\t1\n1\t2\t2\n3\t3\t8\n4\t3\t6\n2\t4\t9\n5\t4\t2\n")
    config = {"source_gene_count": 5, "par_y_indices": [3]}
    args = (source, tmp_path / "extracted", np.array([0, 2, 3]), np.array([0, 1, 2, 3]), np.array([2, 0, 1]), 4)
    result = module.extract_stream(*args, chunk_rows=2, checkpoint=config)
    np.testing.assert_array_equal(result["totals"], [8, 2, 14, 11])
    np.testing.assert_array_equal(result["human_totals"], [7, 2, 14, 9])
    np.testing.assert_array_equal(result["common_totals"], [7, 2, 8, 9])
    np.testing.assert_array_equal(result["par_y_totals"], [0, 0, 6, 0])
    np.testing.assert_array_equal(result["links"].source_cell_index, [0, 2, 3])
    matrices = [sparse.load_npz(tmp_path / "extracted" / x["common"]["file"]) for x in result["shards"]]
    np.testing.assert_array_equal(sparse.vstack(matrices).toarray(), [[0, 3, 4], [8, 0, 0], [0, 0, 9]])
    resumed = module.extract_stream(*args, chunk_rows=2, checkpoint={**config, "shards": result["shards"]})
    assert resumed["shards"] == result["shards"]
    assert resumed["gzip_eof_and_crc_passed"]


@pytest.mark.parametrize("text", ["1\t1\t3\n1\t1\t2\n", "2\t1\t3\n1\t1\t2\n", "1\t1\t-3\n", "6\t1\t2\n"])
def test_invalid_source_never_marked_complete(tmp_path, text):
    source = tmp_path / "counts.gz"
    write_source(source, text)
    with pytest.raises(ValueError):
        module.extract_stream(source, tmp_path / "out", np.array([0]), np.arange(5), np.arange(3), 1,
                              chunk_rows=2, checkpoint={"source_gene_count": 5})


def test_corrupted_gzip_rejected(tmp_path):
    source = tmp_path / "counts.gz"
    write_source(source, "1\t1\t3\n2\t1\t4\n")
    data = bytearray(source.read_bytes())
    data[-8] ^= 0xFF
    source.write_bytes(data)
    with pytest.raises((gzip.BadGzipFile, EOFError)):
        module.extract_stream(source, tmp_path / "out", np.array([0]), np.arange(2), np.arange(2), 1,
                              checkpoint={"source_gene_count": 2})


def test_independent_readback_confirms_rows_and_gene_order(tmp_path):
    import pandas as pd
    verifier_spec = importlib.util.spec_from_file_location("sciplex_verify", SCRIPT.with_name("verify_sciplex_extraction.py"))
    verifier = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)
    source, out = tmp_path / "counts.gz", tmp_path / "extracted/v1"
    write_source(source, "1\t1\t3\n2\t1\t4\n3\t2\t8\n4\t2\t6\n")
    result = module.extract_stream(source, out, np.array([0, 1]), np.arange(4), np.array([2, 0, 1]), 2,
                                   chunk_rows=2, checkpoint={"source_gene_count": 4})
    cells = result["links"].copy()
    cells["cell"], cells["sample"] = ["c1", "c2"], "test"
    cells["human_totals_umi"], cells["common_totals_umi"] = [7, 14], [7, 8]
    cells.to_parquet(out / "a549_cells.parquet", index=False)
    pd.DataFrame({"source_gene_index": np.arange(4)}).to_parquet(out / "human_genes.parquet", index=False)
    common = pd.DataFrame({"common_gene_index": [0, 1, 2], "matrix_gene_index_1based": [3, 1, 2]})
    common.to_parquet(out / "common_genes.parquet", index=False)
    np.savez(out / "all_source_cell_umi_audit.npz", actual=[7, 14], expected=[7, 14])
    (out / "extraction_status.json").write_text(json.dumps({
        "status": "complete_expression_extraction_not_training_ready", "source_cells": 2,
        "shards": result["shards"], "gzip_eof_and_crc_passed": True}), encoding="utf8")
    assert verifier.verify(tmp_path)["human_common_exact_count_agreement"]
    # Swapping a saved gene coordinate must not silently pass the independent audit.
    common["matrix_gene_index_1based"] = [1, 3, 2]
    common.to_parquet(out / "common_genes.parquet", index=False)
    with pytest.raises(ValueError, match="COUNTS_DIFFER"):
        verifier.verify(tmp_path)
