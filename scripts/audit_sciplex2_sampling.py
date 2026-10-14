"""Common-panel and conditional cell-resampling sensitivity of dose diagnostics.

Resampling observed cells measures sensitivity within the recovered material.
It is NOT independent biological replication, an apoptosis label, or a remedy
for selection of surviving/captured nuclei.
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
from scripts.audit_sciplex2_dose import evaluate_dose, PREFIX
from scripts.recover_public_expression import digest


def extract_panel(matrix, cell_mapping, gene_indices, n_genes):
    if len(set(gene_indices)) != len(gene_indices) or np.any(np.asarray(gene_indices)<0) or np.any(np.asarray(gene_indices)>=n_genes):
        raise ValueError("Invalid evaluation gene panel")
    gene_mapping = np.full(n_genes, -1, dtype=int)
    gene_mapping[gene_indices] = np.arange(len(gene_indices))
    panel = np.zeros((int(cell_mapping.max())+1, len(gene_indices)), dtype=np.int64)
    for chunk in pd.read_csv(matrix, sep="\t", names=["gene","cell","count"], header=None, dtype=np.int64, chunksize=500000):
        g,c,v = chunk.gene.to_numpy()-1, chunk.cell.to_numpy()-1, chunk["count"].to_numpy()
        if (g<0).any() or (g>=n_genes).any() or (c<0).any() or (c>=len(cell_mapping)).any() or (v<0).any():
            raise ValueError("Invalid raw count coordinate")
        rows,cols = cell_mapping[c],gene_mapping[g]
        use = (rows>=0)&(cols>=0)
        np.add.at(panel,(rows[use],cols[use]),v[use])
    return panel


def resample_noise(panel, total_umis, sample_n, draws, rng):
    panel, total_umis = np.asarray(panel), np.asarray(total_umis)
    if len(panel)!=len(total_umis) or len(panel)==0 or sample_n<1 or draws<1:
        raise ValueError("Invalid resampling dimensions")
    if (panel<0).any() or (total_umis<=0).any() or (panel.sum(axis=1)>total_umis).any():
        raise ValueError("Panel and full-library totals are inconsistent")
    truth = np.log1p(10000 * panel.sum(axis=0) / total_umis.sum())
    weights = rng.multinomial(sample_n, np.full(len(panel),1/len(panel)),size=draws)
    # Denominator is the FULL cell library, not the selected 500 genes.
    boot = np.log1p(10000 * (weights @ panel) / (weights @ total_umis)[:,None])
    errors = np.mean((boot-truth)**2,axis=1)
    return dict(resampling_mse_mean=float(errors.mean()),resampling_mse_p05=float(np.quantile(errors,.05)),
                resampling_mse_p95=float(np.quantile(errors,.95)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit",type=Path,required=True)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--draws",type=int,default=200)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError("New output directory required")
    audit=json.loads((args.audit/'manifest.json').read_text())
    for name,sha in audit['outputs'].items():
        if digest(args.audit/name)!=sha:raise ValueError("Upstream audit output changed")
    expected_sources = {Path(s['path']).name: s['sha256'] for s in audit['sources']}
    for suffix in ['UMI.count.matrix.gz', 'cell.annotations.txt.gz']:
        name = PREFIX + suffix
        if digest(args.source/name) != expected_sources[name]:
            raise ValueError('Selected source does not match upstream audit')
    groups=pd.read_csv(args.audit/'conditions.csv')
    with np.load(args.audit/'condition_counts.npz') as z:counts=z['counts']; genes=z['gene_ids']
    scores,protocol=evaluate_dose(counts,groups,common_panel=True)
    chosen=np.array(protocol[0]['gene_indices'])
    assert protocol[1]['gene_indices']==chosen.tolist()
    meta=pd.read_csv(args.audit/'retained_cells.csv.gz')
    cells=pd.read_csv(args.source/(PREFIX+'cell.annotations.txt.gz'),sep='\t',header=None)
    mapping=pd.Index(meta.Cell).get_indexer(cells.iloc[:,0])
    if (mapping>=0).sum()!=len(meta):raise ValueError("Retained cell mapping is incomplete")
    matrix=args.source/(PREFIX+'UMI.count.matrix.gz')
    panel=extract_panel(matrix,mapping,chosen,len(genes))
    totals=meta['n.umi'].to_numpy(dtype=np.int64)
    rows=[]; rng=np.random.default_rng(20261014)
    for i,group in groups.iterrows():
        use=meta.drug.eq(group.drug)&meta.dose_source_label.eq(group.dose_source_label)
        np.testing.assert_array_equal(panel[use].sum(axis=0),counts[i,chosen])
        if group.dose_source_label not in {5,100}:continue
        for regime,n in [('observed_n',int(use.sum())),('equal_9_cells',9)]:
            stat=resample_noise(panel[use],totals[use],n,args.draws,rng)
            rows.append(dict(drug=group.drug,held_source_label=float(group.dose_source_label),dose_uM=float(group.dose_uM),
                             observed_cells=int(use.sum()),resampled_cells=n,regime=regime,draws=args.draws,**stat))
    args.output.mkdir(parents=True)
    scores.to_csv(args.output/'common_panel_diagnostics.csv',index=False)
    pd.DataFrame(rows).to_csv(args.output/'cell_sampling_sensitivity.csv',index=False)
    np.savez_compressed(args.output/'selected_cell_counts.npz',counts=panel,full_cell_UMIs=totals,
                        gene_ids=genes[chosen],cell_ids=meta.Cell.to_numpy(dtype=str))
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    report=dict(upstream_manifest_sha256=digest(args.audit/'manifest.json'),source_matrix_sha256=digest(matrix),
                script_sha256=digest(__file__),dose_module_sha256=digest(ROOT/'scripts/audit_sciplex2_dose.py'),
                selected_gene_count=len(chosen),cell_count=len(meta),panel_aggregates_exact=True,
                biological_uncertainty_estimated=False,cell_fate_identified=False,formal_training=False,
                scope='Conditional resampling of observed cells only; not a confidence interval over biological replicates or unobserved dead cells')
    report['outputs']={f.name:digest(f) for f in args.output.iterdir()}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(scores.groupby(['kind','method']).mse.mean().to_string())


if __name__=='__main__':main()
