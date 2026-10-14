"""Map external expression to the frozen gene axis without inventing missing genes."""

import numpy as np
import pandas as pd


def align_gene_axis(frame, gene_axis, *, identifier, samples, scale):
    if scale not in {"bulk_raw_counts", "bulk_estimated_counts", "bulk_log2CPM", "bulk_TPM", "bulk_FPKM", "population_mean_CP10k", "microarray_log2_RMA"}:
        raise ValueError("External expression scale must be explicit")
    axis = gene_axis.sort_values("common_gene_index")
    if not np.array_equal(axis.common_gene_index, np.arange(len(axis))):
        raise ValueError("Common gene axis is not contiguous")
    if axis.ensembl_id.duplicated().any():
        raise ValueError("Duplicate Ensembl identity in common axis")
    target_key = "ensembl_id" if identifier == "ensembl_id" else "gene_symbol"
    if identifier not in {"ensembl_id", "gene_symbol"}:
        raise ValueError("Unrecognized gene identifier type")
    ids = frame[identifier].astype("string")
    if identifier == "ensembl_id":
        ids = ids.str.replace(r"\.\d+$", "", regex=True)
    # Ambiguous symbols or multiple source rows cannot silently choose one gene.
    target = axis[target_key].astype("string")
    ambiguous = set(target[target.duplicated(keep=False)].dropna())
    mapping = dict(zip(target[~target.duplicated(keep=False)],
                       axis.loc[~target.duplicated(keep=False), "common_gene_index"]))
    duplicates = ids.duplicated(keep=False)
    values = frame[samples].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measured expression value")
    if scale == "bulk_raw_counts" and (np.any(values < 0) or np.any(values != np.floor(values))):
        raise ValueError("Raw count source contains negative or noninteger values")
    if scale in {"bulk_estimated_counts", "bulk_TPM", "bulk_FPKM", "population_mean_CP10k"} and np.any(values < 0):
        raise ValueError("Nonnegative abundance source contains negative values")
    aligned = np.full((len(samples), len(axis)), np.nan, dtype=np.float64)
    ledger = []
    for i, key in enumerate(ids):
        column = mapping.get(key) if pd.notna(key) else None
        if pd.isna(key):
            status = "missing_source_identifier"
        elif key in ambiguous:
            status = "ambiguous_target_symbol"
        elif duplicates.iloc[i]:
            status = "duplicate_source_identifier"
        elif column is None:
            status = "not_in_common_axis"
        else:
            status = "matched"
            aligned[:, column] = values[i]
        ledger.append({"source_row": i, "source_identifier": key,
                       "common_gene_index": column if status == "matched" else None,
                       "mapping_status": status})
    mask = np.isfinite(aligned)
    return aligned, mask, pd.DataFrame(ledger)
