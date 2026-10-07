"""Development-only 72 h plate cycle diagnostics, checkpoint section 4.3."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .tidyi_diagnostics import normalize_counts, require_gene_order, validate_gene_axis

D28 = tuple("S1007 S1030 S1048 S1090 S1095 S1096 S1098 S1107 S1122 S1147 S1200 S1233 S1327 S1396 S1451 S1454 S1463 S1484 S1573 S1703 S2012 S7110 S7144 S7165 S7207 S7239 S8481 S8496".split())
C9 = tuple("S1103 S1134 S1216 S1249 S1529 S1541 S7792 S8502 S8567".split())
E10 = tuple("S1085 S1087 S1100 S1133 S1378 S1515 S1782 S7128 S7259 S8323".split())
KEYS = ["catalog_number", "dose_uM", "replicate", "culture_plate"]
LAYOUT = {("rep1", .01): "plate50", ("rep1", .1): "plate50",
          ("rep1", 1.): "plate49", ("rep1", 10.): "plate49",
          ("rep2", .01): "plate52", ("rep2", .1): "plate51",
          ("rep2", 1.): "plate51", ("rep2", 10.): "plate52"}
# (c2,1)-(c1,1)+(c3,2)-(c2,2)+(c4,1)-(c3,1)+(c1,2)-(c4,2)
CYCLE = [(1., "rep1", -1), (10., "rep1", 1), (.01, "rep1", -1),
         (.1, "rep1", 1), (.1, "rep2", -1), (1., "rep2", 1),
         (.01, "rep2", 1), (10., "rep2", -1)]


def development_rows(metadata):
    if metadata.author_hash_qc.isna().any() or not metadata.author_hash_qc.isin([True, False]).all():
        raise ValueError("Invalid author_hash_qc")
    return metadata.loc[metadata.catalog_number.isin(D28) & metadata.time_point.eq(72)
                        & metadata.author_hash_qc.eq(True)].copy()


def accumulate_development(metadata, shards, directory, tracker, width=32461, loader=sparse.load_npz):
    """Only selected D72 rows enter normalization, sums or sums of squares.

    Compressed source shards physically contain other rows. Deserializing a shared
    NPZ is not a claim of physically reading only D bytes; those other rows never
    enter any numeric diagnostic and their expressions are not inspected.
    """
    selected = development_rows(metadata)
    if selected[["shard_id", "row_in_shard"]].duplicated().any():
        raise ValueError("Duplicate selected expression row")
    records, rows_used = {}, []
    by_id = {int(item["shard_id"]): item for item in shards}
    if len(by_id) != len(shards):
        raise ValueError("Duplicate shard ID")
    for shard_id, chunk in selected.groupby("shard_id", observed=True, sort=True):
        specification = by_id[int(shard_id)]["common"]
        filename = specification["file"]
        if Path(filename).name != filename:
            raise ValueError("Shard must be a local basename")
        matrix = loader(tracker(directory / filename, specification.get("sha256")))
        indices = chunk.row_in_shard.to_numpy(dtype=np.int64)
        if (matrix.shape[1] != width or (indices < 0).any() or (indices >= matrix.shape[0]).any()
                or matrix.shape[0] != specification["rows"]):
            raise ValueError("Shard shape or row index mismatch")
        chosen = matrix[indices].tocsr()
        del matrix
        if not np.array_equal(np.asarray(chosen.sum(axis=1)).ravel(), chunk.common_totals_umi):
            raise ValueError("Selected common library totals mismatch")
        normalized = normalize_counts(chosen)
        # Group positional rows rather than assuming metadata index continuity.
        for key, positions in chunk.reset_index(drop=True).groupby(KEYS, observed=True).indices.items():
            values = normalized[positions]
            totals = np.asarray(values.sum(axis=0)).ravel()
            squares = np.asarray(values.power(2).sum(axis=0)).ravel()
            previous = records.get(key)
            records[key] = (len(positions), totals, squares) if previous is None else (
                previous[0] + len(positions), previous[1] + totals, previous[2] + squares)
        rows_used.append(chunk[["shard_id", "row_in_shard", "catalog_number", "time_point", "author_hash_qc"]])
    if not rows_used:
        raise ValueError("No selected D72 cells")
    return records, pd.concat(rows_used, ignore_index=True)


def cycle_diagnostics(records, drugs=D28, width=32461):
    wells, cycles = [], []
    for drug in drugs:
        cycle, noise = np.zeros(width), np.zeros(width)
        complete, variance_available = True, True
        for dose, replicate, sign in CYCLE:
            plate = LAYOUT[(replicate, dose)]
            n, total, squares = records.get((drug, dose, replicate, plate), (0, None, None))
            wells.append({"catalog_number": drug, "dose_uM": dose, "replicate": replicate,
                          "culture_plate": plate, "hash_qc_cell_count": n,
                          "status": "empty" if n == 0 else "one_cell" if n == 1 else "variance_available"})
            if not n:
                complete, variance_available = False, False
                continue
            cycle += sign * total / n
            if n < 2:
                variance_available = False
            else:
                # Unbiased within-well cell variance / n. Not biological variance.
                noise += np.maximum(squares - total * total / n, 0) / (n * (n - 1))
        rms = float(np.sqrt(np.mean(cycle ** 2))) if complete else None
        noise_rms = float(np.sqrt(noise.mean())) if variance_available else None
        cycles.append({"catalog_number": drug, "complete_cycle": complete,
                       "all_wells_at_least_two": variance_available, "cycle_rms": rms,
                       "cell_independent_noise_rms": noise_rms,
                       "rms_to_cell_noise_ratio": rms / noise_rms if rms is not None and noise_rms else None})
    return pd.DataFrame(wells), pd.DataFrame(cycles)


def run_sciplex(root, tracker):
    directory = root / "08-datasets/sciPlex3/GSM4150378/extracted/v1"
    metadata = pd.read_parquet(tracker(directory / "a549_cells.parquet"))
    common = pd.read_parquet(tracker(directory / "common_genes.parquet"))
    reference = pd.read_parquet(tracker(root / "08-datasets/LungCancer_A549_Integration/prepared/v1/gene_map.parquet"))
    validate_gene_axis(common)
    validate_gene_axis(reference)
    require_gene_order(common.ensembl_id, reference.ensembl_id)
    require_gene_order(common.ensembl_key, reference.ensembl_id)
    manifest = json.loads(tracker(directory / "extraction_status.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete_expression_extraction_not_training_ready":
        raise ValueError("Incomplete source extraction")
    development_metadata = metadata[metadata.catalog_number.isin(D28) & metadata.time_point.eq(72)]
    observed_keys = set(development_metadata.loc[:, KEYS].itertuples(index=False, name=None))
    expected_keys = {(drug, dose, rep, plate) for drug in D28 for (rep, dose), plate in LAYOUT.items()}
    if observed_keys != expected_keys:
        raise ValueError("D72 metadata does not match the registered 224-well layout")
    records, selected = accumulate_development(metadata, manifest["shards"], directory, tracker)
    wells, cycles = cycle_diagnostics(records)
    counts = {role: int((metadata.catalog_number.isin(catalogs) & metadata.time_point.eq(72)
                        & metadata.author_hash_qc.eq(True)).sum())
              for role, catalogs in [("D", D28), ("C", C9), ("E", E10)]}
    result = {"experiment_definition": "D28 development-only 72h cycle diagnostic, checkpoint 2026-09-26 section 4.3",
              "candidate_roles": {"D": list(D28), "C": list(C9), "E": list(E10)},
              "role_hash_qc_cell_counts_metadata_only": counts,
              "gene_count": len(common), "expression_cells_used": len(selected),
              "expression_rows_used_by_role": {"D": len(selected), "C": 0, "E": 0},
              "physical_storage_boundary": "shared compressed NPZ shards are deserialized; only D72 hash-QC rows enter numerical computation",
              "normalization": "float64 log1p(10000 * counts / full_common_gene_sum)",
              "hash_qc": "existing author_hash_qc == True; not refitted or changed",
              "expected_wells": len(wells), "empty_wells": int(wells.hash_qc_cell_count.eq(0).sum()),
              "one_cell_wells": int(wells.hash_qc_cell_count.eq(1).sum()),
              "complete_cycles": int(cycles.complete_cycle.sum()),
              "cycles_with_variance": int(cycles.all_wells_at_least_two.sum()),
              "cycle_rms_median": float(cycles.cycle_rms.median()),
              "noise_ratio_median": float(cycles.rms_to_cell_noise_ratio.median()),
              "noise_ratio_max": float(cycles.rms_to_cell_noise_ratio.max()),
              "cycle_formula": "(mu_0.1,rep1-mu_0.01,rep1)+(mu_1,rep2-mu_0.1,rep2)+(mu_10,rep1-mu_1,rep1)+(mu_0.01,rep2-mu_10,rep2)",
              "noise_formula": "sqrt(mean_genes(sum_wells(sample_cell_variance / n_cells)))",
              "biological_replicates": "two replicate labels, four plates; cell-independent noise is not biological-replicate significance",
              "assumptions": "additive plate effects and replicate-stable dose effects imply zero cycle; not established by this diagnostic",
              "exploratory": True, "no_model_fit": True}
    return result, {"sciplex_wells": wells, "sciplex_cycles": cycles, "sciplex_expression_rows_used": selected}
