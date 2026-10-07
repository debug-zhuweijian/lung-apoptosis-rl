"""Read-only audit of downloaded sci-Plex metadata against the existing gene/drug axes.

Writes only new audit products; never modifies the prepared/training-v1 datasets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "08-datasets/sciPlex3/GSM4150378"
SRC = DATA / "sources"
OUT = DATA / "audit"
PREFIX = "GSM4150378_sciPlex3_"
TRAIN = ROOT / "08-datasets/LungCancer_A549_Integration/training-v1"
QC_SOURCE = "https://github.com/cole-trapnell-lab/sci-plex/blob/079639c50811dd43a206a779ab2f0199a147c98f/large_screen/Notebook3_preprocess_data.R#L49-L66"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chemical_keys(smiles: str) -> tuple[str, str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid official source SMILES; needs explicit resolution")
    full = Chem.MolToSmiles(molecule, isomericSmiles=True)
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    largest = max(fragments, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    return full, Chem.MolToSmiles(largest, isomericSmiles=True), Chem.MolToSmiles(largest, isomericSmiles=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # R write.table includes a row-name field with no header. It is NOT tab-separated.
    meta = pd.read_csv(SRC / (PREFIX + "pData.txt.gz"), sep=r"\s+", index_col=0)
    assert (meta.index == meta.cell).all()
    assert not meta.duplicated(["sample", "cell"]).any()
    qc = (
        meta.top_oligo_P.notna() & meta.top_oligo_W.notna()
        & (meta.hash_umis_P >= 5) & (meta.hash_umis_W >= 5)
        & (meta.top_to_second_best_ratio_P >= 5)
        & (meta.top_to_second_best_ratio_W >= 5) & (meta.rt_well != 1)
    )
    meta["author_hash_qc"] = qc
    a549 = meta.loc[meta.cell_type.eq("A549")].copy()
    a549["dose_nM"] = a549.dose
    a549["dose_uM"] = a549.dose / 1000.0
    a549.to_parquet(OUT / "a549_cells_metadata.parquet", index=False)
    conditions = a549.groupby(["catalog_number", "product_name", "dose_nM", "dose_uM", "time_point", "replicate"], dropna=False).agg(
        cells_pre_qc=("cell", "size"), cells_author_hash_qc=("author_hash_qc", "sum"),
        culture_plates=("culture_plate", lambda x: "|".join(sorted(x.dropna().unique()))),
        sequencing_samples=("sample", lambda x: "|".join(sorted(x.dropna().unique()))),
        wells=("Combo", "nunique"),
    ).reset_index()
    conditions.to_csv(OUT / "a549_condition_coverage.csv", index=False, encoding="utf-8-sig")
    catalog_72 = set(a549.loc[(a549.time_point == 72) & ~a549.vehicle, "catalog_number"])
    paired_counts = conditions.loc[conditions.catalog_number.isin(catalog_72)].pivot_table(
        index=["catalog_number", "dose_nM"], columns=["time_point", "replicate"],
        values="cells_author_hash_qc", aggfunc="sum",
    ).reindex(columns=pd.MultiIndex.from_product([[24, 72], ["rep1", "rep2"]])).fillna(0)
    paired_counts.to_csv(OUT / "shared_dose_time_rep_counts.csv", encoding="utf-8-sig")
    summary = []
    for time_point, rows in a549.groupby("time_point"):
        good = rows.loc[rows.author_hash_qc]
        summary.append({"time_h": int(time_point), "pre_qc_cells": len(rows), "author_hash_qc_cells": len(good),
                        "drug_labels_excluding_vehicle": int(rows.loc[~rows.vehicle, "product_name"].nunique()),
                        "conditions_including_vehicle": len(rows[["catalog_number", "dose"]].drop_duplicates()),
                        "conditions_after_qc": len(good[["catalog_number", "dose"]].drop_duplicates()),
                        "replicates": sorted(rows.replicate.unique().tolist()),
                        "plate_count": int(rows.culture_plate.nunique()),
                        "samples": sorted(rows["sample"].unique().tolist())})
    cross = pd.crosstab(a549.culture_plate, a549.time_point)
    cross.to_csv(OUT / "plate_time_cross_tab.csv", encoding="utf-8-sig")
    pd.crosstab(a549["sample"], a549.time_point).to_csv(OUT / "sample_time_cross_tab.csv", encoding="utf-8-sig")
    technical_crossings = {}
    for column in ["rt_plate", "lig_plate", "pcr_plate", "replicate"]:
        tab = pd.crosstab(a549[column], a549.time_point)
        tab.to_csv(OUT / f"{column}_time_cross_tab.csv", encoding="utf-8-sig")
        technical_crossings[column] = {"levels": len(tab), "all_levels_contain_both_times": bool((tab > 0).all(axis=1).all())}

    genes = pd.read_csv(SRC / (PREFIX + "A549_MCF7_K562_screen_gene.annotations.txt.gz"), sep=r"\s+")
    genes["matrix_gene_index_1based"] = np.arange(1, len(genes) + 1)
    # Preserve PAR_Y suffix. Blind split('.') would collapse distinct annotated rows.
    genes["ensembl_key"] = genes.id.str.replace(r"\.\d+(?=_PAR_Y$|$)", "", regex=True)
    common = pd.read_parquet(TRAIN / "genes.parquet")
    mapping = common.merge(genes, left_on="ensembl_id", right_on="ensembl_key", how="left", validate="one_to_one")
    mapping.to_csv(OUT / "shared_gene_axis.csv", index=False, encoding="utf-8-sig")
    par_y = genes.loc[genes.id.str.endswith("_PAR_Y")].copy()
    par_y["base_without_par_y"] = par_y.ensembl_key.str.replace("_PAR_Y", "", regex=False)
    par_y["base_in_project_axis"] = par_y.base_without_par_y.isin(common.ensembl_id)
    par_y.to_csv(OUT / "par_y_annotation_rows.csv", index=False, encoding="utf-8-sig")

    hashes = pd.read_csv(SRC / (PREFIX + "A549_MCF7_K562_hashTable_metadata.txt.gz"), sep="\t")
    assert not hashes.Combo.duplicated().any()
    compounds = hashes.loc[hashes.cell_type.eq("A549") & ~hashes.vehicle, ["catalog_number", "name", "SMILES"]].drop_duplicates()
    assert not compounds.catalog_number.duplicated().any()
    tahoe = pd.read_parquet(TRAIN / "chemicals.parquet")
    tahoe = tahoe.loc[~tahoe.is_vehicle & tahoe.source_smiles.notna()].copy()
    levels = ["full", "largest_fragment", "largest_fragment_no_stereo"]
    for frame, key in [(compounds, "SMILES"), (tahoe, "source_smiles")]:
        frame[levels] = [chemical_keys(s) for s in frame[key]]
    matched_rows = []
    for _, row in compounds.iterrows():
        rec = {"catalog_number": row.catalog_number, "source_name": row["name"],
               "source_smiles": row.SMILES, "source_full_encoding": row["full"],
               "has_72h": bool((hashes.loc[hashes.catalog_number.eq(row.catalog_number), "time_point"] == 72).any())}
        for key in levels:
            matched = tahoe.loc[tahoe[key].eq(row[key])]
            rec[key + "_matching_names"] = "|".join(matched.drug_name)
            rec[key + "_matching_drug_ids"] = "|".join(matched.drug_id)
            rec[key + "_match_count"] = len(matched)
        rec["identity_conclusion"] = "same_recorded_full_smiles_not_independent_reagent_verification" if rec["full_match_count"] else "candidate_or_no_match_not_confirmed"
        matched_rows.append(rec)
    matches = pd.DataFrame(matched_rows)
    matches.to_csv(OUT / "tahoe_chemical_encoding_overlap.csv", index=False, encoding="utf-8-sig")

    # Confirm the sparse triplet orientation using complete leading cells and n.umi.
    annotations = pd.read_csv(SRC / (PREFIX + "A549_MCF7_K562_screen_cell.annotations.txt.gz"), sep="\t", header=None, names=["cell", "sample"])
    annotations["matrix_cell_index_1based"] = np.arange(1, len(annotations) + 1)
    assert not annotations.duplicated(["sample", "cell"]).any()
    joined = annotations.merge(meta.reset_index(drop=True), on=["sample", "cell"], how="outer", validate="one_to_one", indicator=True)
    triplets = pd.read_csv(OUT / "expression_prefix_200000.tsv", sep="\t", header=None, names=["gene_i", "cell_i", "umi"])
    assert (triplets.umi > 0).all() and (triplets.umi % 1 == 0).all()
    assert triplets.cell_i.is_monotonic_increasing
    assert triplets.gene_i.between(1, len(genes)).all()
    assert triplets.cell_i.between(1, len(annotations)).all()
    assert not triplets.duplicated(["gene_i", "cell_i"]).any()
    last_partial_cell = int(triplets.cell_i.max())
    complete = triplets.loc[triplets.cell_i < last_partial_cell]
    counts = complete.groupby("cell_i").umi.sum().rename("prefix_umi_sum").reset_index()
    checked = counts.merge(joined, left_on="cell_i", right_on="matrix_cell_index_1based", validate="one_to_one")
    checked["umi_matches_metadata"] = checked.prefix_umi_sum.eq(checked["n.umi"])
    keep = ["cell_i", "cell", "sample", "cell_type", "time_point", "dose", "product_name", "prefix_umi_sum", "n.umi", "umi_matches_metadata", "author_hash_qc"]
    checked[keep].to_csv(OUT / "expression_prefix_cell_checks.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "metadata_and_expression_prefix_audited_not_full_training_ready", "rdkit_version": rdBase.rdkitVersion,
        "source_files": {f.name: {"bytes": f.stat().st_size, "sha256": digest(f)} for f in SRC.glob("*.gz")},
        "project_axes": {str(f.relative_to(ROOT)): digest(f) for f in [TRAIN / "genes.parquet", TRAIN / "chemicals.parquet"]},
        "metadata_rows_all_cell_types": len(meta), "a549_rows_pre_qc": len(a549), "qc_source": QC_SOURCE,
        "qc_definition": "both oligos not NA; both hash_umis >=5; both top_to_second_best_ratio >=5; rt_well !=1",
        "a549_time_summary": summary, "cell_identity_join_counts": {str(k): int(v) for k, v in joined._merge.value_counts().items()},
        "time_is_unique_within_culture_plate": bool(((cross > 0).sum(axis=1) == 1).all()),
        "technical_batch_crossings": technical_crossings,
        "shared_time_drug_dose_pairs": len(paired_counts),
        "pairs_at_min_cells_per_time_per_rep": {str(k): int(paired_counts.ge(k).all(axis=1).sum()) for k in [1, 2, 20, 50]},
        "threshold_note": "coverage sensitivity only; not a frozen cell-filter rule or a statistical-power guarantee",
        "paper_72h_cell_count": 64440,
        "paper_vs_recomputed_count": "paper 64440 versus Notebook3 hash-QC on GEO 67443 remains unresolved; no filtering to force agreement",
        "genes": {"source_rows": len(genes), "human_rows": int(genes.id.str.startswith("ENSG").sum()),
                  "project_gene_axis": len(common), "mapped_without_par_y_collapsing": int(mapping.id.notna().sum()),
                  "par_y_extra_rows_linking_to_project_base": int(par_y.base_in_project_axis.sum()),
                  "par_y_count_merge": "not_applied_requires_reference_alignment_decision"},
        "chemical_overlap": {"a549_source_catalog_entries": len(compounds),
             "same_full_smiles_entries": int((matches.full_match_count > 0).sum()),
             "same_full_smiles_72h_entries": int((matches.has_72h & (matches.full_match_count > 0)).sum()),
             "same_largest_fragment_entries_including_full_matches": int((matches.largest_fragment_match_count > 0).sum()),
             "same_largest_fragment_no_stereo_entries_including_stricter_matches": int((matches.largest_fragment_no_stereo_match_count > 0).sum()),
             "warning": "source encodings can omit stereo/salt/mixture details; relaxed matches are candidate links only"},
        "expression_preview": {"triples": len(triplets), "complete_leading_cells": len(checked),
              "all_complete_cell_umi_sums_match_pdata": bool(checked.umi_matches_metadata.all()),
              "complete_a549_by_time": {str(k): int(v) for k, v in checked.loc[checked.cell_type.eq("A549"), "time_point"].value_counts().items()},
              "last_partial_cell_excluded": last_partial_cell, "population_estimates_valid": False,
              "whole_matrix_gzip_verified": False, "sampling": "file_order_prefix_not_random"},
    }
    (OUT / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["source_files", "project_axes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
