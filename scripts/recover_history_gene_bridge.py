"""Recover real GSE96779 FPKM on a public Ensembl axis, with explicit exclusions.

Uses the existing source-native parser. Does not train, allocate frozen roles,
create RNA/functional pseudo-pairs, or certify a missing local gene axis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.recover_public_expression import digest, history_expression, save_native
from lung.data.history_gene_axis import align_history_fpkm, read_grch37_genes


def history_links(samples: pd.DataFrame) -> dict:
    """Condition-level membership, not longitudinal cells or paired replicates."""
    expected = {"Parental", "MSR", "Withdrawn", "JQ1"}
    if set(samples.state_id) != expected or not samples.geo_accession.is_unique:
        raise ValueError("Expected the four distinct GSE96779 RNA conditions")
    if not samples.cell_line.eq("A549").all():
        raise ValueError("History source includes an unverified non-A549 background")
    groups = {}
    for state, group in samples.groupby("state_id", sort=True):
        if len(group) != 3 or set(group.replicate_label.astype(str)) != {"1", "2", "3"}:
            raise ValueError("Expected three source-labelled RNA replicates per condition")
        groups[state] = group.geo_accession.tolist()
    return {
        "source": "GSE96779", "condition_members": groups,
        "comparisons": [
            {"target": "MSR", "reference": "Parental", "meaning": "selected_background_plus_ongoing_treatment"},
            {"target": "Withdrawn", "reference": "MSR", "meaning": "10_day_withdrawal_population_contrast"},
            {"target": "JQ1", "reference": "MSR", "meaning": "24h_JQ1_with_continued_momelotinib_selumetinib"},
        ],
        "interventions": {
            "Parental": [{"name": "DMSO", "vehicle_concentration": None}],
            "MSR": [{"name": "momelotinib", "dose_value": 5, "dose_unit": "uM", "exposure_h": None},
                    {"name": "selumetinib", "dose_value": 1, "dose_unit": "uM", "exposure_h": None}],
            "Withdrawn": [{"name": "DMSO", "vehicle_concentration": None, "withdrawal_h": 240}],
            "JQ1": [{"name": "momelotinib", "dose_value": 5, "dose_unit": "uM", "exposure_h": None},
                    {"name": "selumetinib", "dose_value": 1, "dose_unit": "uM", "exposure_h": None},
                    {"name": "JQ1", "dose_value": 200, "dose_unit": "nM", "dose_uM": 0.2, "exposure_h": 24}],
        },
        "statistical_unit": "source_RNA_sample_with_reported_replicate_label",
        "independent_culture_lineage_verified": False,
        "paired_replicates_or_cells": False, "initial_cell_state_observed": False,
        "molecular_structures_verified_in_this_package": False,
        "functional_endpoint_values_in_this_package": 0,
        "matching_Annexin_supervision_verified": False,
        "arbitrary_sequential_intervention_support": False,
        "role": "unassigned_external_source",
    }


def recover(args) -> dict:
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing assets are immutable")
    sources = [args.matrix, args.soft, args.gtf, args.hgnc, args.axis_npz]
    before = {str(p.resolve()): digest(p) for p in sources}
    frame, samples, records = history_expression(args.matrix, args.soft)
    for accession, metadata, _ in records:
        if (metadata.get("taxid_ch1") != ["9606"]
                or metadata.get("organism_ch1") != ["Homo sapiens"]
                or metadata.get("molecule_ch1") != ["total RNA"]
                or metadata.get("geo_accession") != [accession]
                or "GSE96779" not in metadata.get("series_id", [])):
            raise ValueError(f"Unverified history source identity/background: {accession}")
        growth = " ".join(metadata.get("growth_protocol_ch1", [])).strip()
        if growth != "A549 and MSR-A549 cells were cultured in DMEM supplemented with 10% fetal bovine serum.":
            raise ValueError(f"Unverified history source growth background: {accession}")
        treatment = " ".join(metadata.get("treatment_protocol_ch1", []))
        clauses = ("5 μM MMB and 1 μM selumetinib", "alternately for 2 months",
                   "maintained with 5 μM momelotinib and 1 μM selumetinib continuously",
                   "without MMB and selumetinib treatment for 10 days")
        if not all(clause in treatment for clause in clauses):
            raise ValueError(f"Unverified selection/maintenance history: {accession}")
        processing = " ".join(metadata.get("data_processing", []))
        if "hg19" not in processing or "RefGene" not in processing or "Cufflinks 2.2.1" not in processing:
            raise ValueError(f"Unverified history genome/quantification metadata: {accession}")
    # Read identities only, never expression values from the donor axis package.
    with np.load(args.axis_npz, allow_pickle=False) as axis_source:
        axis = axis_source["ensembl_ids"].astype(str).tolist()
    genes = read_grch37_genes(args.gtf)
    hgnc = pd.read_csv(args.hgnc, sep="\t", dtype=str, keep_default_na=False)
    aligned, mask, ledger = align_history_fpkm(
        frame, genes, hgnc, axis, samples=samples.matrix_column_id.tolist(), source_build="hg19")
    links = history_links(samples)
    samples["data_role"] = "unassigned_external_source"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=args.output.name + ".partial-", dir=args.output.parent))
    native = save_native(stage, frame, samples, "source_feature_id", "bulk_FPKM")
    frame[["source_feature_id", "gene_id", "locus"]].to_csv(stage / "source_features.csv.gz", index=False)
    np.savez_compressed(stage / "common_axis_expression.npz", expression=aligned, observed_mask=mask,
                        ensembl_ids=np.asarray(axis), sample_ids=samples.geo_accession.to_numpy(dtype=str),
                        value_scale="bulk_FPKM")
    ledger.to_csv(stage / "gene_mapping.csv.gz", index=False)
    pd.DataFrame({"common_gene_index": np.arange(len(axis)), "ensembl_id": axis}).to_csv(stage / "genes.csv", index=False)
    (stage / "condition_links.json").write_text(json.dumps(links, indent=2), encoding="utf-8")
    (stage / "source_metadata.json").write_text(json.dumps({a: m for a, m, _ in records}, indent=2), encoding="utf-8")

    with np.load(stage / "common_axis_expression.npz", allow_pickle=False) as saved:
        # NPZ members are decompressed on every access: read each member once.
        saved_expression = saved["expression"]
        np.testing.assert_array_equal(saved_expression, aligned)
        np.testing.assert_array_equal(saved["observed_mask"], mask)
        np.testing.assert_array_equal(saved["ensembl_ids"], axis)
        np.testing.assert_array_equal(saved["sample_ids"], samples.geo_accession.to_numpy(dtype=str))
        matched = ledger.loc[ledger.mapping_status.eq("matched")]
        source_rows = matched.source_row.to_numpy(dtype=np.int64)
        target_columns = matched.common_gene_index.to_numpy(dtype=np.int64)
        source_values = frame[samples.matrix_column_id.tolist()].to_numpy(dtype=np.float64).T
        np.testing.assert_array_equal(saved_expression[:, target_columns], source_values[:, source_rows])
        if not np.isnan(saved_expression[~mask]).all():
            raise ValueError("Unobserved genes were turned into measured values")
    after = {str(p.resolve()): digest(p) for p in sources}
    if before != after:
        raise ValueError("Input changed during recovery; partial output not published")
    report = {
        "source_native": native, "samples": len(samples), "source_features": len(frame),
        "target_genes": len(axis), "observed_genes": int(mask.all(axis=0).sum()),
        "unobserved_genes": int((~mask.any(axis=0)).sum()),
        "observed_values_exactly_rechecked": int(mask.sum()),
        "observed_zero_values": int(np.count_nonzero(aligned[mask] == 0)),
        "excluded_source_rows": int(ledger.mapping_status.ne("matched").sum()),
        "mapping_status_counts": {k: int(v) for k, v in ledger.mapping_status.value_counts().items()},
        "common_axis_identity": {"algorithm": "ordered_ensembl_ids_utf8_lf_v1", "gene_count": len(axis),
                                 "ordered_ensembl_sha256": hashlib.sha256(("\n".join(axis) + "\n").encode()).hexdigest()},
        "source_projection_verified": True, "source_bytes_unchanged": True,
        "reference_axis_match_verified": False, "model_compatibility_verified": False,
        "different_transcript_quantification_equivalence_verified": False,
        "normalization_performed": False, "training_performed": False,
        "functional_endpoint_supervision_verified": False, "whole_project_qualified": False,
    }
    manifest = {"schema_version": 1, "source": "GSE96779", "created_utc": datetime.now(timezone.utc).isoformat(),
                "validation": report,
                "mapping_rule": "symbol_and_same_chr_locus_unique_in_full_annotation_before_axis_filter_no_row_aggregation",
                "coordinate_rule": "candidate_set_invariant_to_source_0based_vs_1based_start",
                "source_reference": "hg19_UCSC_RefGene_Feb2016", "bridge_reference": "GRCh37_Ensembl75_and_pinned_HGNC",
                "sources": [{"path": str(p.resolve()), "sha256": before[str(p.resolve())], "bytes": p.stat().st_size} for p in sources],
                "implementation": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / "src/lung/data/history_gene_axis.py", ROOT / "scripts/recover_public_expression.py", ROOT / "src/lung/data/__init__.py"]},
                "outputs": {p.name: digest(p) for p in stage.iterdir() if p.is_file()}}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    # The user-visible directory is published only after all checks pass.
    os.rename(stage, args.output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("matrix", "soft", "gtf", "hgnc", "axis-npz", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    print(json.dumps(recover(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
