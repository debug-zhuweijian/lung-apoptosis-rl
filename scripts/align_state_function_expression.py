"""Align public bulk state-expression data; preserve scale, samples and missingness."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.endpoint_alignment import file_hash
from lung.data.external_gene_axis import align_gene_axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-round3", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = ROOT / "08-datasets/LungCancer_A549_Integration"
    source_dir = base / "endpoint-alignment-v1/source_reviews/state_function"
    version = "state-expression-v2" if args.include_round3 else "state-expression-v1"
    output = args.output or base / "endpoint-alignment-v1" / version
    if not output.is_absolute():
        output = ROOT / output
    if (output / "manifest.json").exists():
        raise FileExistsError("State expression alignment already frozen")
    output.mkdir(parents=True, exist_ok=True)
    axis_path = base / "prepared/v1/gene_map.parquet"
    axis = pd.read_parquet(axis_path)
    reports = []
    jobs = [
        ("Jochems2021", "Jochems2021_A549_state_log2CPM.csv", "ensembl_id", "bulk_log2CPM"),
        ("GSE133029", "GSE133029_gene_symbol_counts.csv.gz", "gene_symbol", "bulk_raw_counts"),
    ]
    if args.include_round3:
        jobs.append(("GSE246690", "GSE246690_A549_raw_counts.csv.gz", "ensembl_id", "bulk_raw_counts"))
    for name, filename, identifier, scale in jobs:
        current_source_dir = (base / "endpoint-alignment-v1/source_reviews/state_function_round3"
                              if name == "GSE246690" else source_dir)
        source = current_source_dir / filename
        frame = pd.read_csv(source)
        if name == "GSE246690":
            frame = frame.rename(columns={"gene_id": "ensembl_id"})
        samples = [c for c in frame if c.startswith("A549_") or c.startswith("GSM")]
        extra = {}
        if name == "GSE246690":
            metadata_path = current_source_dir / "GSE246690_samples.csv"
            metadata = pd.read_csv(metadata_path)
            metadata = metadata.loc[metadata.cell_line.eq("A549")].copy()
            if (metadata.geo_accession.duplicated().any()
                    or metadata.matrix_column_id.duplicated().any()
                    or metadata.geo_accession.tolist() != samples
                    or not metadata.cell_line.eq("A549").all()):
                raise ValueError("GSE246690 sample identity does not match A549 matrix columns")
            primary_files = [metadata_path, current_source_dir / "GSE246690_family.soft.gz",
                             current_source_dir / "GSE246690_vramponi_202207_rnaseq_raw_counts.xlsx"]
            extra = {
                "primary_files": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in primary_files},
                "state_sample_counts": {k: int(v) for k, v in metadata.state_label.value_counts().items()},
                "independent_replicates_status": "unresolved_2_and_2-2_both_labelled_replicate_2",
                "endpoint_condition_link_status": "no_verified_apoptosis_rechallenge_pairing",
                "reference_scope": "day7_bulk_states_not_measured_t0_of_external_endpoint_experiments",
            }
        aligned, mask, ledger = align_gene_axis(frame, axis, identifier=identifier, samples=samples, scale=scale)
        array_path = output / (name + ".npz")
        ledger_path = output / (name + "_gene_mapping.parquet")
        np.savez_compressed(array_path, expression=aligned, observed_mask=mask,
                            sample_ids=np.asarray(samples, dtype=str),
                            ensembl_ids=axis.sort_values("common_gene_index").ensembl_id.to_numpy(dtype=str))
        ledger.to_parquet(ledger_path, index=False)
        # Read back: negative log2CPM values and zero measured counts stay unchanged.
        saved = np.load(array_path, allow_pickle=False)
        np.testing.assert_array_equal(saved["observed_mask"], mask)
        np.testing.assert_allclose(saved["expression"], aligned, rtol=0, atol=0, equal_nan=True)
        reports.append({
            "source": name, "source_file": source.relative_to(ROOT).as_posix(),
            "source_sha256": file_hash(source), "sample_ids": samples,
            "source_genes": len(frame), "common_axis_genes": len(axis),
            "matched_genes": int(ledger.mapping_status.eq("matched").sum()),
            "mapping_counts": {k: int(v) for k, v in ledger.mapping_status.value_counts().items()},
            "measurement_scale": scale, "same_cell_pre_post_pairing": False,
            "missing_gene_policy": "NaN plus explicit observed mask; never zero-imputed",
            "preprocessing_status": "original_bulk_scale_not_single_cell_normalized",
            "endpoint_condition_link_status": "protocol_discrepancies_require_resolution",
            "files": {array_path.name: file_hash(array_path), ledger_path.name: file_hash(ledger_path)},
            **extra,
        })
    manifest = {"schema_version": 1, "gene_axis_file": axis_path.relative_to(ROOT).as_posix(),
                "gene_axis_sha256": file_hash(axis_path), "datasets": reports,
                "status": "gene_coordinate_alignment_verified_not_endpoint_pairing",
                "formal_training_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: row[k] for k in ("source", "matched_genes", "mapping_counts")} for row in reports], ensure_ascii=False))


if __name__ == "__main__":
    main()
