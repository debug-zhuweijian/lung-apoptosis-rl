"""Align six deposited GSE279184 count columns without inventing endpoint pairs."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.external_gene_axis import align_gene_axis

BASE = ROOT / "08-datasets/LungCancer_A549_Integration"
REVIEWS = BASE / "endpoint-alignment-v1/source_reviews"
SOURCE = REVIEWS / "state_function_round7/induced"
COUNTS = SOURCE / "GSE279184_Gene_counts_replicates_UT_and_salinomycin_treated.csv.gz"
HGNC = REVIEWS / "state_gene_round4/hgnc_complete_set_2026-10-10.txt"
SAMPLES = ["ut_r1", "ut_r2", "ut_r3", "tr_r1", "tr_r2", "tr_r3"]
SCALE = "author_deposited_processed_count_table"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def values(text, field):
    prefix = "!" + field + " = "
    return [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]


def sample_record(text, accession):
    if values(text, "Sample_geo_accession") != [accession] or values(text, "Sample_series_id") != ["GSE279184"]:
        raise ValueError("GEO sample/series identity mismatch")
    title = values(text, "Sample_title")[0]
    match = re.fullmatch(r"(Untreated|Salinomycin|Saliomycin)_Sen_A549_R1_biological replicate ([123])", title)
    if not match or values(text, "Sample_source_name_ch1") != ["A549"]:
        raise ValueError("Unknown sample title/group/biological replicate")
    if not any("de-duplicated gene counts" in item and "Ensembl release 102" in item for item in values(text, "Sample_data_processing")):
        raise ValueError("Sample lacks the declared deduplicated-count processing evidence")
    treated = match.group(1) != "Untreated"
    return {"sample_id": accession, "author_title": title, "source_title_spelling_preserved": True,
            "deposited_count_column": ("tr_r" if treated else "ut_r") + match.group(2),
            "biological_replicate_author_label": int(match.group(2)),
            "column_mapping_basis": "GEO title treatment and final biological replicate suffix; no per-GSM count table to independently match values",
            "cell_line": "A549", "state": "alisertib_induced_senescent", "induction_compound": "alisertib",
            "induction_uM": 0.5, "induction_days": 7,
            "reference_role": "post_salinomycin_expression" if treated else "untreated_senescent_parallel_reference_not_t0",
            "salinomycin_uM": 2.5 if treated else 0, "salinomycin_treatment_h": 48 if treated else None,
            "time_provenance": "GSE series design; PMC11670111 s14 and Fig2C/3C" if treated else "UT reference; separate collection duration not explicitly stated per GSM",
            "author_annotation": "Ensembl release 102 transcript GTF", "measurement_scale": SCALE,
            "author_scale_description": "bulk_deduplicated_gene_counts_as_reported_in_methods",
            "raw_unmodified_count_matrix_verified": False, "transform_applied": "none",
            "observed_t0": False, "RNA_apoptosis_pairing": False, "same_replicate_function_pairing": False}


def symbol_ledger(symbols, axis, hgnc):
    maps = {name: defaultdict(set) for name in ["symbol", "prev_symbol", "alias_symbol"]}
    hgnc_ids = defaultdict(set)
    for row in hgnc.to_dict("records"):
        if row["status"] != "Approved":
            continue
        gene_ids = set(re.findall(r"ENSG\d+", row["ensembl_gene_id"]))
        for field in maps:
            for label in row[field].split("|"):
                if label:
                    maps[field][label].update(gene_ids)
                    hgnc_ids[label].add(row["hgnc_id"])
    axis_map = defaultdict(set)
    for row in axis.to_dict("records"):
        axis_map[row["gene_symbol"]].add(row["ensembl_id"])
    in_axis = set(axis.ensembl_id)
    duplicate_symbols = set(symbols[symbols.duplicated(keep=False)])
    rows = []
    for i, symbol in enumerate(symbols):
        direct = maps["symbol"].get(symbol, set()) | axis_map.get(symbol, set())
        old = maps["prev_symbol"].get(symbol, set()) | maps["alias_symbol"].get(symbol, set())
        candidates = direct if direct else old
        basis = "exact_axis_or_HGNC_approved_symbol" if direct else "unique_HGNC_previous_or_alias_symbol"
        gene = next(iter(candidates)) if len(candidates) == 1 else ""
        if not symbol or symbol == "NO_GENE_NAME":
            status = "missing_or_placeholder_source_symbol"
        elif symbol in duplicate_symbols:
            status = "duplicate_source_symbol"
        elif len(candidates) > 1:
            status = "ambiguous_or_conflicting_gene_identity"
        elif not candidates:
            status = "no_supported_gene_identity"
        elif gene not in in_axis:
            status = "resolved_gene_outside_common_axis"
        else:
            status = "resolved_candidate"
        rows.append({"source_row": i, "source_symbol": symbol, "hgnc_ids": "|".join(sorted(hgnc_ids.get(symbol, set()))),
                     "direct_gene_candidates": "|".join(sorted(direct)), "previous_or_alias_candidates": "|".join(sorted(old)),
                     "resolved_ensembl_id": gene, "mapping_basis": basis, "identity_status": status})
    return pd.DataFrame(rows)


def align(frame, axis, hgnc):
    identity = symbol_ledger(frame.GeneSymbol, axis, hgnc)
    resolved = identity.resolved_ensembl_id.where(identity.identity_status.eq("resolved_candidate"), None)
    input_frame = frame[SAMPLES].assign(ensembl_id=resolved)
    matrix, mask, coordinate_ledger = align_gene_axis(input_frame, axis, identifier="ensembl_id", samples=SAMPLES, scale="bulk_raw_counts")
    identity["common_gene_index"] = coordinate_ledger.common_gene_index
    identity["mapping_status"] = np.where(identity.identity_status.eq("resolved_candidate"), coordinate_ledger.mapping_status, identity.identity_status)
    identity.loc[identity.mapping_status.eq("duplicate_source_identifier"), "mapping_status"] = "duplicate_resolved_gene_not_aggregated"
    return matrix, mask, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REVIEWS / "state_function_round7/GSE279184_common_axis_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Keep the existing version; choose a new output")
    source_note = COUNTS.with_name(COUNTS.name + ".source.json")
    if sha(COUNTS) != json.loads(source_note.read_text())["sha256"]:
        raise ValueError("Deposited source hash changed")
    annotation_manifest = HGNC.parent / "annotation_download_manifest.json"
    annotation_entry = next(row for row in json.loads(annotation_manifest.read_text()) if row["file"] == HGNC.name)
    if sha(HGNC) != annotation_entry["sha256"]:
        raise ValueError("Frozen official HGNC snapshot changed")
    axis_path = BASE / "prepared/v1/gene_map.parquet"
    helper_path = ROOT / "src/lung/data/external_gene_axis.py"
    samples = [sample_record(path.read_text(encoding="utf-8"), path.stem) for path in sorted(SOURCE.glob("GSM*.soft"))]
    metadata = pd.DataFrame(samples).set_index("deposited_count_column").loc[SAMPLES].reset_index()
    if len(metadata) != 6 or not metadata.sample_id.is_unique:
        raise ValueError("Sample columns must match six unique GEO samples")
    series = (SOURCE / "GSE279184_series.soft").read_text(encoding="utf-8")
    if set(values(series, "Series_sample_id")) != set(metadata.sample_id):
        raise ValueError("Series membership differs from mapped sample columns")
    inputs = [COUNTS, source_note, HGNC, annotation_manifest, axis_path, helper_path,
              SOURCE / "GSE279184_series.soft", SOURCE / "PMC11670111.xml", SOURCE / "PMC11670111.source.json"]
    inputs += [SOURCE / (accession + ".soft") for accession in metadata.sample_id]
    before = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    frame = pd.read_csv(COUNTS, dtype=str, keep_default_na=False)
    if len(frame) != 16670 or frame.columns[:7].tolist() != ["GeneSymbol", *SAMPLES]:
        raise ValueError("Unexpected deposited gene/column scope")
    axis = pd.read_parquet(axis_path).sort_values("common_gene_index").reset_index(drop=True)
    if len(axis) != 32461:
        raise ValueError("Expected current full 32,461 gene axis")
    hgnc = pd.read_csv(HGNC, sep="\t", dtype=str, keep_default_na=False,
                       usecols=["hgnc_id", "symbol", "status", "ensembl_gene_id", "prev_symbol", "alias_symbol"])
    matrix, mask, ledger = align(frame, axis, hgnc)
    counts = frame[SAMPLES].to_numpy(dtype=np.int64)
    # A separate CSV parser re-reads all original numeric cells and row identities.
    with gzip.open(COUNTS, "rt", encoding="utf-8-sig", newline="") as stream:
        reread = list(csv.DictReader(stream))
    if [row["GeneSymbol"] for row in reread] != frame.GeneSymbol.tolist():
        raise ValueError("Original gene row identities changed on reread")
    np.testing.assert_array_equal(np.array([[int(row[column]) for column in SAMPLES] for row in reread], dtype=np.int64), counts)
    matched = ledger.mapping_status.eq("matched")
    columns = ledger.loc[matched, "common_gene_index"].to_numpy(dtype=int)
    np.testing.assert_array_equal(matrix[:, columns], counts[matched].T)
    if not np.isnan(matrix[~mask]).all():
        raise ValueError("Unobserved genes must remain NaN")
    out.mkdir(parents=True)
    source_rows = np.full(len(axis), -1, dtype=np.int64)
    source_rows[columns] = ledger.loc[matched, "source_row"].to_numpy()
    np.savez_compressed(out / "aligned_counts.npz", expression=matrix, observed_mask=mask,
                        ensembl_ids=axis.ensembl_id.to_numpy(dtype=str), sample_ids=metadata.sample_id.to_numpy(dtype=str),
                        source_columns=np.asarray(SAMPLES), source_row_for_common_gene=source_rows,
                        measurement_scale=SCALE, dataset_scope="author_deposited_processed_table",
                        raw_unmodified_count_matrix_verified=False, transform_applied="none", joint_single_cell_input_ready=False)
    np.savez_compressed(out / "original_six_column_counts.npz", expression=counts.T,
                        source_symbols=frame.GeneSymbol.to_numpy(dtype=str), sample_ids=metadata.sample_id.to_numpy(dtype=str), source_columns=np.asarray(SAMPLES),
                        measurement_scale=SCALE, raw_unmodified_count_matrix_verified=False, transform_applied="none", joint_single_cell_input_ready=False)
    with np.load(out / "aligned_counts.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["expression"], matrix)
        np.testing.assert_array_equal(saved["observed_mask"], mask)
        np.testing.assert_array_equal(saved["expression"][:, columns], counts[matched].T)
    with np.load(out / "original_six_column_counts.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["expression"].T, counts)
    ledger.to_csv(out / "gene_mapping.csv.gz", index=False)
    ledger.to_parquet(out / "gene_mapping.parquet", index=False)
    metadata.to_csv(out / "samples.csv", index=False)
    statistics = [column for column in frame if column not in ["GeneSymbol", *SAMPLES]]
    frame[["GeneSymbol", *statistics]].to_csv(out / "author_statistics_not_counts.csv.gz", index=False)
    (out / "external_gene_axis_snapshot.py").write_bytes(helper_path.read_bytes())
    (out / "align_gse279184_expression_snapshot.py").write_bytes(Path(__file__).read_bytes())
    validation = {"source_rows": len(frame), "source_numeric_count_values": int(counts.size), "samples": len(SAMPLES),
                  "common_axis_genes": len(axis), "matched_genes": int(matched.sum()),
                  "unobserved_common_genes": int(len(axis) - matched.sum()), "mapping_status_counts": ledger.mapping_status.value_counts().to_dict(),
                  "matched_mapping_basis_counts": ledger.loc[matched, "mapping_basis"].value_counts().to_dict(),
                  "all_100020_original_values_reread_equal": True, "all_aligned_measured_values_equal_original": True,
                  "full_npz_roundtrip_equal": True, "source_count_min": int(counts.min()), "source_count_max": int(counts.max()),
                  "author_statistic_columns_not_counts": statistics, "gene_list_scope": "16670_author_deposited_processed_table_rows; filtering_rule_or_full_unfiltered_matrix_not_established",
                  "measurement_scale": SCALE, "raw_unmodified_count_matrix_verified": False, "transform_applied": "none",
                  "joint_single_cell_input_ready": False, "source_annotation": "Ensembl_102", "identity_bridge": "exact_common_axis_or_approved_HGNC_symbol_else_unique_HGNC_previous_alias; snapshot_2026-10-10_not_Ensembl102_annotation",
                  "UT_is_parental_A549": False, "UT_is_observed_t0": False, "RNA_apoptosis_pairing_created": False,
                  "RNA_salinomycin_uM": 2.5, "Fig3B_WB_salinomycin_uM": 2, "normalization_or_training_performed": False}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if any(sha(ROOT / path) != expected for path, expected in before.items()):
        raise ValueError("Input changed during mapping")
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "processed_bulk_count_gene_coordinate_alignment_not_endpoint_pairing", "inputs_sha256": before,
                "measurement_scale": SCALE, "raw_unmodified_count_matrix_verified": False, "transform_applied": "none", "joint_single_cell_input_ready": False,
                "outputs_sha256": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
                "validation": validation, "formal_training_ready": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
