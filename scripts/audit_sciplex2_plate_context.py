"""Test plate-reference usefulness without claiming individual initial-state gain.

The held-plate vehicle expression is an available parallel reference input.
It is not a pre-treatment measurement of the same cells, nor an independent
biological-repeat certificate for the two source plate prefixes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_sciplex2_dose import aggregate_triplets,PREFIX
from scripts.recover_public_expression import digest


def evaluate(counts,groups,n_features=500):
    if len(groups)!=len(counts) or groups.duplicated(['plate_prefix','drug','dose_source_label']).any():
        raise ValueError('Group identity mismatch')
    if set(groups.plate_prefix)!={'A','B'} or (counts.sum(axis=1)<=0).any():
        raise ValueError('Two nonempty source plates required')
    y=np.log1p(10000*counts/counts.sum(axis=1,keepdims=True))
    rows=[];protocols=[]
    lookup={(r.plate_prefix,r.drug,r.dose_source_label):i for i,r in groups.iterrows()}
    for held,train in [('A','B'),('B','A')]:
        donor=groups.plate_prefix.eq(train).to_numpy()
        chosen=np.argsort(y[donor].var(axis=0),kind='stable')[-min(n_features,y.shape[1]):]
        protocols.append(dict(held_plate=held,donor_plate=train,gene_indices=chosen.tolist(),
                              feature_selection='donor_plate_only',held_reference_role='parallel_vehicle_not_t0'))
        for i,g in groups[groups.plate_prefix.eq(held)&groups.dose_source_label.gt(0)].iterrows():
            keys=[(train,g.drug,g.dose_source_label),(train,g.drug,0),(held,g.drug,0)]
            if any(k not in lookup for k in keys):
                raise ValueError('Missing donor condition or matching vehicle')
            source,source_ref,target_ref=[lookup[k] for k in keys]
            predictions={'donor_same_condition':y[source,chosen],
                         'held_plate_vehicle':y[target_ref,chosen],
                         'additive_plate_reference':y[source,chosen]-y[source_ref,chosen]+y[target_ref,chosen]}
            for method,pred in predictions.items():
                rows.append(dict(held_plate=held,drug=g.drug,dose_source_label=float(g.dose_source_label),
                                 dose_uM=float(g.dose_uM),test_cells=int(g.cell_count),
                                 donor_cells=int(groups.loc[source,'cell_count']),method=method,
                                 mse=float(np.mean((pred-y[i,chosen])**2))))
    return pd.DataFrame(rows),protocols


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('New output directory required')
    upstream=json.loads((args.audit/'manifest.json').read_text())
    for fn,sha in upstream['outputs'].items():
        if digest(args.audit/fn)!=sha:raise ValueError('Upstream output changed')
    src={Path(s['path']).name:s['sha256'] for s in upstream['sources']}
    for suffix in ['UMI.count.matrix.gz','cell.annotations.txt.gz','gene.annotations.txt.gz']:
        name=PREFIX+suffix
        if digest(args.source/name)!=src[name]:raise ValueError('Source changed')
    meta=pd.read_csv(args.audit/'retained_cells.csv.gz')
    excluded=int(meta.plate_prefix.isna().sum());meta=meta[meta.plate_prefix.notna()].copy()
    keys=['plate_prefix','drug','dose_source_label','dose_uM']
    groups=meta.groupby(keys).size().rename('cell_count').reset_index()
    key_to_i={tuple(r[k] for k in keys):i for i,r in groups.iterrows()}
    meta['group']=meta.apply(lambda r:key_to_i[tuple(r[k] for k in keys)],axis=1)
    cells=pd.read_csv(args.source/(PREFIX+'cell.annotations.txt.gz'),sep='\t',header=None)
    genes=pd.read_csv(args.source/(PREFIX+'gene.annotations.txt.gz'),sep='\t',header=None)
    cell_group=cells.iloc[:,0].map(meta.set_index('Cell')['group']).fillna(-1).to_numpy(dtype=int)
    counts,totals,entries=aggregate_triplets(args.source/(PREFIX+'UMI.count.matrix.gz'),len(genes),len(cells),cell_group,len(groups))
    pd_totals=cells.iloc[:,0].map(meta.set_index('Cell')['n.umi']).to_numpy()
    np.testing.assert_array_equal(totals[cell_group>=0],pd_totals[cell_group>=0])
    scores,protocols=evaluate(counts,groups)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'plate_condition_counts.npz',counts=counts,gene_ids=genes.iloc[:,0].to_numpy(dtype=str))
    groups.to_csv(args.output/'groups.csv',index=False);scores.to_csv(args.output/'scores.csv',index=False)
    (args.output/'protocols.json').write_text(json.dumps(protocols,indent=2))
    summaries={}
    # Declared descriptive sensitivity strata, not a selected success threshold.
    for minimum in [0,20,50]:
        sub=scores[(scores.test_cells>=minimum)&(scores.donor_cells>=minimum)]
        summaries[str(minimum)]=sub.groupby(['held_plate','method']).mse.mean().to_dict()
        print('minimum observed cells per donor/target condition',minimum)
        print(sub.groupby(['held_plate','method']).mse.agg(['mean','count']).to_string())
    report=dict(script_sha256=digest(__file__),upstream_manifest_sha256=digest(args.audit/'manifest.json'),
                source_matrix_sha256=src[PREFIX+'UMI.count.matrix.gz'],triplet_entries=entries,
                plate_condition_groups=len(groups),excluded_unresolved_plate_cells=excluded,
                source_UMI_totals_exact=True,individual_state_gain_identified=False,
                plate_effect_separated_from_biology=False,biological_independence_verified=False,
                fixed_additive_hypothesis=True,formal_training=False,full_research_qualified=False)
    report['outputs']={f.name:digest(f) for f in args.output.iterdir()}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
