"""Small, source-native sciPlex2 dose diagnostic, independent of frozen suites.

No cell pairing; no inference that recovered nuclei or CellTiter-Glo are Annexin.
Source dose labels differ from micromolar concentrations. The author's CTG table
and original Fig2 support the explicit mapping used here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.recover_public_expression import digest

PREFIX = "GSM4150377_sciPlex2_A549_Transcription_Modulators_"
FACTORS = {"BMS": 10.0, "Dex": 0.5, "Nutlin": 2.5, "SAHA": 10.0}


def cell_metadata(pdata, cells, hashes):
    if not pdata.index.is_unique or not pdata.Cell.is_unique or not np.array_equal(pdata.index, pdata.Cell):
        raise ValueError("Ambiguous pData cell identities")
    if pdata.Cell.tolist() != cells.iloc[:, 0].tolist():
        raise ValueError("Expression cell order differs from metadata")
    if not hashes.iloc[:, 0].is_unique:
        raise ValueError("Repeated hash identity")
    keep = pdata.top_to_second_best_ratio.gt(10) & pdata.qval.lt(.01) & pdata.hash_umis.gt(30)
    selected = pdata.loc[keep].copy()
    if not selected.top_oligo.isin(hashes.iloc[:, 0]).all():
        raise ValueError("Unknown treatment hash")
    parsed = selected.top_oligo.str.split("_", expand=True)
    if parsed.shape[1] != 3 or not parsed[0].isin(FACTORS).all():
        raise ValueError("Unknown intervention label")
    selected["drug"] = parsed[0]
    selected["dose_source_label"] = pd.to_numeric(parsed[1], errors="raise")
    selected["dose_uM"] = selected.dose_source_label * selected.drug.map(FACTORS)
    selected["source_well_label"] = parsed[2]
    # E09 is malformed in the released source. Retain it; don't invent plate A.
    valid_well = parsed[2].str.fullmatch(r"[AB][A-H](0[1-9]|1[0-2])")
    selected["plate_prefix"] = parsed[2].str[0].where(valid_well)
    selected["well_identity_status"] = np.where(valid_well, "source_label", "malformed_source_label")
    selected["solvent"] = np.where(selected.drug.eq("Dex"), "ethanol", "DMSO")
    selected["exposure_h"] = 24
    return selected, keep.to_numpy()


def verify_ctg(ctg):
    required = {"CTG", "replicate", "cell_type", "treatment", "dose", "real_dose"}
    if not required <= set(ctg) or not ctg.cell_type.eq("A549").all():
        raise ValueError("Unexpected CTG schema/background")
    if not ctg.treatment.isin(FACTORS).all():
        raise ValueError("Unknown CTG compound")
    if not np.allclose(ctg.real_dose, ctg.dose * ctg.treatment.map(FACTORS)):
        raise ValueError("Source dose mapping disagrees with Fig2-supported factors")
    if ctg.duplicated(["treatment", "dose", "replicate"]).any():
        raise ValueError("Duplicate CTG source replicate")
    if not np.isfinite(ctg.CTG).all():
        raise ValueError("Nonfinite CTG readout")


def aggregate_triplets(matrix, n_genes, n_cells, cell_group, n_groups):
    counts = np.zeros((n_groups, n_genes), dtype=np.int64)
    cell_totals = np.zeros(n_cells, dtype=np.int64)
    entries = 0
    # GEO's file is headerless 1-based triplets, despite its generic matrix label.
    for chunk in pd.read_csv(matrix, sep="\t", header=None, names=["gene", "cell", "count"],
                             dtype=np.int64, chunksize=500000):
        g, c, v = chunk.gene.to_numpy()-1, chunk.cell.to_numpy()-1, chunk["count"].to_numpy()
        if (g < 0).any() or (g >= n_genes).any() or (c < 0).any() or (c >= n_cells).any() or (v < 0).any():
            raise ValueError("Invalid matrix coordinate or count")
        np.add.at(cell_totals, c, v)
        groups = cell_group[c]
        use = groups >= 0
        np.add.at(counts, (groups[use], g[use]), v[use])
        entries += len(chunk)
    return counts, cell_totals, entries


def evaluate_dose(group_counts, groups, n_features=500):
    """Exploratory held-dose diagnostics. Each condition gets equal weight."""
    if np.any(group_counts.sum(axis=1) <= 0):
        raise ValueError("Empty condition")
    y = np.log1p(10000 * group_counts / group_counts.sum(axis=1, keepdims=True))
    results, protocols = [], []
    for held_label, kind in [(5.0, "interpolation"), (100.0, "upper_extrapolation")]:
        test = groups.dose_source_label.eq(held_label).to_numpy()
        train = ~test
        # Only training conditions select the evaluation gene panel.
        chosen = np.argsort(np.var(y[train], axis=0), kind="stable")[-min(n_features, y.shape[1]):]
        protocols.append(dict(held_source_label=held_label, type=kind, train_conditions=groups.loc[train].to_dict("records"),
                              test_conditions=groups.loc[test].to_dict("records"), gene_indices=chosen.tolist(),
                              gene_selection="variance_in_training_conditions_only", ridge_lambda=.1))
        for drug in sorted(FACTORS):
            tr = train & groups.drug.eq(drug).to_numpy()
            te = test & groups.drug.eq(drug).to_numpy()
            if tr.sum() < 3 or te.sum() != 1:
                raise ValueError("Dose diagnostic lacks complete drug support")
            z = np.log1p(groups.loc[tr, "dose_uM"].to_numpy())
            ztest = np.log1p(groups.loc[te, "dose_uM"].to_numpy())
            target = y[te][:, chosen]
            source = y[tr][:, chosen]
            design = np.column_stack([np.ones(len(z)), z])
            beta = np.linalg.solve(design.T @ design + np.diag([0., .1]), design.T @ source)
            order = np.argsort(z)
            preds = {"control_only": y[groups.drug.eq(drug) & groups.dose_source_label.eq(0)][:, chosen],
                     "drug_mean": source.mean(axis=0, keepdims=True),
                     "nearest_dose": source[[int(np.argmin(abs(z-ztest[0]))) ]],
                     "logdose_ridge": np.column_stack([np.ones(len(ztest)), ztest]) @ beta,
                     "logdose_interpolation_clamped": np.array([[np.interp(ztest[0], z[order], source[order, j]) for j in range(len(chosen))]])}
            for method, pred in preds.items():
                results.append(dict(drug=drug, held_source_label=held_label, held_dose_uM=float(groups.loc[te,"dose_uM"].iloc[0]),
                                    kind=kind, method=method, mse=float(np.mean((pred-target)**2)),
                                    evaluation_unit="condition_pseudobulk", genes=len(chosen)))
    return pd.DataFrame(results), protocols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("New versioned output required")
    p = args.source
    paths = [p/(PREFIX+"UMI.count.matrix.gz"), p/(PREFIX+"cell.annotations.txt.gz"),
             p/(PREFIX+"gene.annotations.txt.gz"), p/"GSM4150377_sciPlex2_pData.txt.gz",
             p/"GSM4150377_sciPlex2_hashSampleSheet.txt.gz", p/"sciPlex2_CTG_results.csv"]
    cells, genes = [pd.read_csv(x, sep="\t", header=None) for x in paths[1:3]]
    if not genes.iloc[:,0].is_unique:
        raise ValueError("Duplicate source gene ID")
    pdata = pd.read_csv(paths[3], sep=r"\s+")
    hashes = pd.read_csv(paths[4], sep="\t", header=None)
    meta, keep = cell_metadata(pdata, cells, hashes)
    ctg = pd.read_csv(paths[5]); verify_ctg(ctg)
    groups = meta[["drug", "dose_source_label", "dose_uM"]].drop_duplicates().sort_values(["drug", "dose_source_label"]).reset_index(drop=True)
    key_to_group = {(r.drug, r.dose_source_label):i for i,r in groups.iterrows()}
    group_ids = np.full(len(cells), -1, dtype=int)
    group_ids[keep] = [key_to_group[(r.drug,r.dose_source_label)] for r in meta.itertuples()]
    counts, totals, entries = aggregate_triplets(paths[0], len(genes), len(cells), group_ids, len(groups))
    if not np.array_equal(totals, pdata["n.umi"].to_numpy()):
        raise ValueError("Matrix counts do not equal deposited per-cell n.umi")
    groups["retained_cells"] = np.bincount(group_ids[keep], minlength=len(groups))
    scores, protocols = evaluate_dose(counts, groups)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"condition_counts.npz", counts=counts, gene_ids=genes.iloc[:,0].to_numpy(dtype=str))
    with np.load(args.output/"condition_counts.npz") as saved:
        np.testing.assert_array_equal(saved["counts"], counts)
    groups.to_csv(args.output/"conditions.csv", index=False)
    meta.to_csv(args.output/"retained_cells.csv.gz", index=False)
    ctg.to_csv(args.output/"CTG_source_values.csv", index=False)
    ctg_summary = ctg.groupby(["treatment", "dose", "real_dose"], as_index=False).agg(
        CTG_mean=("CTG", "mean"), CTG_source_rows=("CTG", "size"))
    links = groups.merge(ctg_summary, left_on=["drug", "dose_source_label", "dose_uM"],
                         right_on=["treatment", "dose", "real_dose"], how="left", validate="one_to_one")
    if links.CTG_mean.isna().any():
        raise ValueError("A response condition lacks its source CTG reference")
    links["link_level"] = "source_compound_dose_reference_not_replicate_pair"
    links["CTG_exposure_h_verified"] = None
    links["endpoint"] = "CellTiter_Glo_relative_ATP_viability_not_Annexin"
    links["functional_training_qualified"] = False
    links.to_csv(args.output/"CTG_condition_references.csv", index=False)
    scores.to_csv(args.output/"dose_diagnostics.csv", index=False)
    (args.output/"protocol.json").write_text(json.dumps(protocols, indent=2))
    report = dict(cells_source=len(cells), cells_hash_qc=len(meta), source_genes=len(genes), triplet_entries=entries,
                  source_UMI_totals_exact=True, condition_count=len(groups), malformed_well_cells=int(meta.plate_prefix.isna().sum()),
                  CTG_rows=len(ctg), CTG_endpoint="relative_ATP_viability_not_Annexin", CTG_to_RNA_replicate_pairing=False,
                  diagnostic_fitting=True, formal_training=False, state_increment_tested=False, true_sequence_tested=False,
                  full_research_feasibility_proven=False, dependence="Both plate prefixes aggregated per condition; no biological repeat count inferred from hashing",
                  sources=[dict(path=str(x.resolve()),sha256=digest(x)) for x in paths + [p/"Notebook_1_small_screen_analysis.R", p/"figure2.jpg"]], script_sha256=digest(__file__))
    report["outputs"] = {x.name:digest(x) for x in args.output.iterdir()}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {"sources","outputs"}},indent=2))
    print(scores.groupby(["kind","method"]).mse.mean().to_string())


if __name__ == "__main__":
    main()
