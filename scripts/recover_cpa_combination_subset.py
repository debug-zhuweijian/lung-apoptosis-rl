"""Audit the PHAROS processed GSE206741 subset against original GEO identities.

Recover real sparse counts and well pseudobulks; do not treat PHAROS interface
placeholders as biological labels, infer sequences, or resolve dose conflicts.
No model fitting or reassignment of frozen training/evaluation roles occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.external_gene_axis import align_gene_axis


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_column(group, key):
    item = group[key]
    if isinstance(item, h5py.Group):
        codes = item["codes"][:]
        categories = item["categories"].asstr()[:]
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError(f"Missing or invalid categorical identity: {key}")
        return categories[codes]
    return item.asstr()[:] if h5py.check_string_dtype(item.dtype) else item[:]


def bind_metadata(observed, original):
    """Cell IDs are identities, never positional row matches or inferred pairs."""
    for frame in (observed, original):
        if frame.Cell.isna().any() or not frame.Cell.is_unique:
            raise ValueError("Missing or duplicate cell identity")
    if not observed.Cell.isin(original.Cell).all():
        raise ValueError("Subset contains cells absent from original GEO metadata")
    ref = original.set_index("Cell").loc[observed.Cell].reset_index()
    for key in ["Drug1", "Drug2", "Well", "RT_well", "sample", "n.umi"]:
        if not np.array_equal(observed[key].to_numpy(), ref[key].to_numpy()):
            raise ValueError(f"GEO identity conflict: {key}")
    result = observed.copy()
    if set(zip(result.Drug1, result.Drug2)) != {
        ("DMSO", "DMSO"), ("panobinostat", "crizotinib")
    }:
        raise ValueError("Unexpected treatment in the selected PHAROS subset")
    expected = result.Drug1 + "_" + result.Drug2
    if not expected.equals(result.cell_type):
        raise ValueError("PHAROS cell_type is not the expected treatment label")
    result = result.rename(columns={"cell_type": "author_treatment_named_cell_type",
                                    "drugname_drugconc": "author_interface_drug_placeholder"})
    result["cell_line"] = "A549"
    result["condition_id"] = expected
    result["exposure_mode"] = "simultaneous_24h"
    result["exposure_h"] = 24
    # Primary CPA methods: 1 uM each. GEO treatment_protocol: 10uM.
    # These provenance claims must not become a fabricated resolved numeric dose.
    result["resolved_dose_uM"] = np.nan
    result["dose_status"] = "unresolved_GEO_10uM_vs_CPA_methods_1uM"
    result["evaluation_role"] = "unassigned_expression_reference_dose_quarantine"
    result["same_cell_pre_post_pair"] = False
    result["observed_initial_state"] = False
    result["reference_role"] = np.where(result.Drug1.eq("DMSO"),
        "parallel_24h_vehicle", "treated_24h_endpoint")
    result["measured_apoptosis_label"] = False
    return result


def load_source(h5ad, cell_metadata, gene_metadata):
    original = pd.read_csv(cell_metadata, sep="\t")
    genes = pd.read_csv(gene_metadata, sep="\t", dtype=str, keep_default_na=False)
    if not genes.id.is_unique:
        raise ValueError("Duplicate GEO gene identity")
    with h5py.File(h5ad, "r") as source:
        keys = ["Cell", "Drug1", "Drug2", "Well", "RT_well", "sample",
                "n.umi", "cell_type", "drugname_drugconc"]
        observed = pd.DataFrame({key: read_column(source["obs"], key) for key in keys})
        if not np.array_equal(read_column(source["obs"], "_index"), observed.Cell):
            raise ValueError("AnnData row index differs from cell identity")
        annotation = pd.DataFrame({"ensembl_id": read_column(source["var"], "id"),
                                   "gene_symbol": read_column(source["var"], "gene_short_name")})
        if not annotation.ensembl_id.is_unique or not annotation.ensembl_id.isin(genes.id).all():
            raise ValueError("Missing or duplicate source gene identity")
        ref = genes.set_index("id").loc[annotation.ensembl_id]
        if not np.array_equal(ref.gene_short_name.to_numpy(), annotation.gene_symbol.to_numpy()):
            raise ValueError("GEO gene annotation conflict")
        group = source["layers/counts"]
        if group.attrs["encoding-type"] != "csr_matrix":
            raise ValueError("Expected explicit CSR count layer")
        counts = sparse.csr_matrix((group["data"][:], group["indices"][:], group["indptr"][:]),
                                   shape=tuple(group.attrs["shape"]))
        counts.check_format(full_check=True)
        if not counts.has_canonical_format:
            raise ValueError("Count CSR contains duplicate or unsorted entries")
        values = counts.data
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError("Source count layer is not finite nonnegative integer counts")
        if counts.shape != (len(observed), len(annotation)):
            raise ValueError("Count dimensions do not match cell and gene identities")
        # Confirm X semantics, rather than mistaking log-normalized values for counts.
        x = source["X"]
        if x.attrs["encoding-type"] != "csr_matrix" or tuple(x.attrs["shape"]) != counts.shape or not np.array_equal(x["indices"][:], counts.indices) or not np.array_equal(x["indptr"][:], counts.indptr):
            raise ValueError("Normalized X sparsity differs from the count layer")
        totals = np.asarray(counts.sum(axis=1)).ravel()
        if np.any(totals <= 0):
            raise ValueError("Empty cell in processed subset")
        expected = np.log1p(counts.data * np.repeat(10000 / totals, np.diff(counts.indptr)))
        if not np.allclose(x["data"][:], expected, rtol=2e-6, atol=2e-6):
            raise ValueError("X is not the documented log1p CP10k transform")
    return counts.astype(np.int64), bind_metadata(observed, original), annotation


def recover(h5ad, cell_metadata, gene_metadata, gene_axis, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory; existing results are not overwritten")
    counts, cells, genes = load_source(h5ad, cell_metadata, gene_metadata)
    # Wells are retained sampling units, not asserted independent biological replicates.
    wells = cells[["Well", "RT_well", "condition_id"]].drop_duplicates().sort_values("Well")
    if not wells.Well.is_unique:
        raise ValueError("One well maps to multiple treatment or RT identities")
    profiles, well_rows = [], []
    for row in wells.itertuples(index=False):
        selection = cells.Well.eq(row.Well).to_numpy()
        profiles.append(np.asarray(counts[selection].sum(axis=0)).ravel())
        well_rows.append({"well_id": row.Well, "rt_well": row.RT_well,
                          "condition_id": row.condition_id, "n_cells": int(selection.sum()),
                          "biological_repeat_independence": "not_established",
                          "dose_status": "unresolved_GEO_10uM_vs_CPA_methods_1uM"})
    well_frame = pd.DataFrame(well_rows)
    expression = genes[["ensembl_id"]].copy()
    for row, values in zip(well_rows, profiles):
        expression[row["well_id"]] = values
    aligned, mask, ledger = align_gene_axis(expression, pd.read_csv(gene_axis),
        identifier="ensembl_id", samples=well_frame.well_id.tolist(), scale="bulk_raw_counts")
    output.mkdir(parents=True)
    sparse.save_npz(output / "source_cell_counts.npz", counts)
    cells.to_csv(output / "cells.csv", index=False)
    genes.to_csv(output / "source_genes.csv", index=False)
    well_frame.to_csv(output / "wells.csv", index=False)
    ledger.to_csv(output / "gene_mapping.csv", index=False)
    np.savez_compressed(output / "well_pseudobulk_common_axis.npz", values=aligned, observed_mask=mask)
    manifest = {
        "source": "GSE206741_PHAROS_pano_criz_processed_subset",
        "synthetic": False, "n_cells": len(cells), "n_source_genes": len(genes),
        "n_wells": len(well_frame), "counts_nnz": counts.nnz,
        "n_common_axis_observed": int(mask[0].sum()),
        "n_common_axis_missing": int((~mask[0]).sum()),
        "condition_counts": cells.condition_id.value_counts().to_dict(),
        "author_placeholder_counts": cells.author_interface_drug_placeholder.value_counts().to_dict(),
        "gene_axis_scope": "provided_public_axis_not_unavailable_local_frozen_axis",
        "dose_status": "unresolved_GEO_10uM_vs_CPA_methods_1uM",
        "protocol_sources": {
            "GEO": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206741",
            "CPA_primary_methods": "https://doi.org/10.15252/msb.202211517",
            "processed_subset": "https://zenodo.org/records/21925263",
            "GEO_dose_claim_uM": 10, "CPA_per_compound_dose_claim_uM": 1,
            "resolved_drug_dose_uM": None,
        },
        "training_eligible": False, "dose_evaluation_eligible": False,
        "true_sequence_evaluation_eligible": False, "apoptosis_evaluation_eligible": False,
        "observed_timepoints": [24], "pre_post_cell_pairing": False,
        "single_agent_crizotinib_in_subset": False,
        "biological_repeat_independence": "not_established",
        "normalization_verified": "X=log1p(counts/cell_count_sum*10000)",
        "count_scope": "author_QC_filtered_cells_and_genes_not_full_GEO",
        "inputs": [{"name": Path(p).name, "sha256": digest(p)} for p in
                   [h5ad, cell_metadata, gene_metadata, gene_axis]],
        "outputs": {p.name: digest(p) for p in sorted(output.iterdir())},
        "implementation_sha256": digest(__file__),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["h5ad", "cell-metadata", "gene-metadata", "gene-axis", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.h5ad, args.cell_metadata, args.gene_metadata,
                             args.gene_axis, args.output), indent=2))
