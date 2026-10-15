"""Recover one public breast clone as a measured-state transfer challenge for A549 C.

Does not extend the therapeutic target beyond A549 or create drug/future-fate labels.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recover_public_expression import digest
from scripts.audit_tidyi_within_plate_increment import fit_probability,training_pca


def recover_counts(counts,meta,axis):
    if not counts.index.is_unique or not counts.columns.is_unique or not meta.index.is_unique:raise ValueError('Duplicate cell/gene identities')
    if set(counts.index)!=set(meta.index):raise ValueError('Source cell identities differ')
    if not set(axis.ensembl_id)<=set(counts.columns):raise ValueError('Common genes missing')
    meta=meta.loc[counts.index].copy();keep=meta.Cell_line.eq('MDA_MB_231_A')
    x=counts.loc[keep,axis.ensembl_id].to_numpy(float)
    if not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any():raise ValueError('Invalid actual counts')
    return x,meta.loc[keep].reset_index(names='cell_id'),meta.loc[~keep].reset_index(names='cell_id')


def score(p,y,labels):
    guess=p.argmax(axis=1)
    return dict(cells=len(y),accuracy=float(np.mean(guess==y)),balanced_accuracy=float(np.mean([np.mean(guess[y==k]==k) for k in range(3)])),
                mean_log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
                per_state_recall={s:float(np.mean(guess[y==k]==k)) for k,s in enumerate(labels)})


def evaluate(x,m,axis):
    if m.cell_id.duplicated().any():raise ValueError('Cross-source cell overlap')
    if not m.Well.str.fullmatch(r'[A-P](?:[1-9]|1[0-9]|2[0-4])').all():raise ValueError('Invalid well')
    labels=['PI+APC+','PI-APC+','PI-APC-'];y=pd.Categorical(m.cell_state,categories=labels).codes
    if (y<0).any():raise ValueError('Unknown source measurement label')
    total=x.sum(axis=1)
    if (total<=0).any():raise ValueError('Zero common-axis counts')
    mt=axis.gene_symbol.fillna('').str.startswith('MT-').to_numpy()
    q=np.c_[np.log1p(total),np.log1p((x>0).sum(axis=1)),x[:,mt].sum(axis=1)/total*100]
    rna=np.log1p(x/total[:,None]*1e4);position=np.eye(24)[m.Well.str[1:].astype(int).to_numpy()-1]
    plans=[('A549_P8_to_MDA_P6','A549',8,'MDA_MB_231_A',6),('MDA_P6_to_MDA_P14','MDA_MB_231_A',6,'MDA_MB_231_A',14),('MDA_P6_to_A549_P8','MDA_MB_231_A',6,'A549',8)]
    results=[];predictions=[]
    for name,bg1,p1,bg2,p2 in plans:
        train=(m.Cell_line.eq(bg1)&m.Plate.eq(p1)).to_numpy();test=(m.Cell_line.eq(bg2)&m.Plate.eq(p2)).to_numpy()
        if not train.any() or not test.any() or (train&test).any():raise ValueError('Invalid fixed transfer fold')
        if len(set(y[train]))!=3 or len(set(y[test]))!=3:raise ValueError('Absent class in transfer fold')
        mean=q[train].mean(axis=0);sd=q[train].std(axis=0);sd[sd==0]=1;qc=(q-mean)/sd
        pc,pt,select,load,center=training_pca(rna[train],rna[test],200,10)
        designs={'position':(position[train],position[test]),'QC':(qc[train],qc[test]),'RNA':(pc,pt),
                 'position_QC':(np.c_[position[train],qc[train]],np.c_[position[test],qc[test]]),
                 'position_QC_RNA':(np.c_[position[train],qc[train],pc],np.c_[position[test],qc[test],pt])}
        metrics={};out=m.loc[test,['cell_id','Cell_line','Plate','Well','cell_state']].copy();out['fold']=name
        for method,(a,b) in designs.items():
            p=fit_probability(np.c_[np.ones(len(a)),a],y[train],np.c_[np.ones(len(b)),b]);metrics[method]=score(p,y[test],labels)
            out[method+'_prediction']=[labels[k] for k in p.argmax(axis=1)]
            out[method+'_log_loss']=-np.log(np.maximum(p[np.arange(test.sum()),y[test]],1e-15))
        predictions.append(out)
        results.append(dict(name=name,training_cells=m.loc[train,'cell_id'].tolist(),test_cells=m.loc[test,'cell_id'].tolist(),
                            selected_gene_ids=axis.iloc[select].ensembl_id.tolist(),metrics=metrics))
    return dict(folds=results,QC='recomputed on common measured axis, not copied source QC',training_HVG=200,training_PCA=10,
                calibration='none',hyperparameter_search=False,independent_culture_proven=False,
                future_fate_validated=False,drug_endpoint_validated=False,natural_apoptosis_rate_validated=False,
                therapeutic_target_changed=False,limitation='same source sorting design; cell-line and plate transfer does not eliminate repeated layout confounding'),pd.concat(predictions,ignore_index=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['public-source','a549','output']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output required')
    receipt=json.loads((a.public_source/'mda_counts_receipt.json').read_text());cp=a.public_source/'SS3_Human_MDA_A_counts.csv'
    if digest(cp)!=receipt['sha256']:raise ValueError('Public counts changed')
    source_manifest=json.loads((a.a549/'manifest.json').read_text())
    for n in ['gene_axis.csv','tidyi_one_plate_counts.npz','tidyi_source_metadata.csv']:
        if digest(a.a549/n)!=source_manifest['outputs'][n]:raise ValueError('A549 recovery source changed')
    axis=pd.read_csv(a.a549/'gene_axis.csv');counts=pd.read_csv(cp,index_col=0);meta=pd.read_csv(a.public_source/'SS3_Human_MDA_A_metadata.csv',index_col=0)
    x,m,excluded=recover_counts(counts,meta,axis)
    ax=np.load(a.a549/'tidyi_one_plate_counts.npz');am=pd.read_csv(a.a549/'tidyi_source_metadata.csv').set_index('cell_id',drop=False)
    if not np.array_equal(ax['gene_ids'],axis.ensembl_id.to_numpy()):raise ValueError('A549 gene order mismatch')
    am=am.loc[ax['cell_ids']].reset_index(drop=True)
    all_x=np.vstack([ax['expression'],x]);all_meta=pd.concat([am,m],ignore_index=True)
    report,pred=evaluate(all_x,all_meta,axis)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'MDA_A_common_counts.npz',expression=x.astype(np.int32),cell_ids=m.cell_id.to_numpy(str),gene_ids=axis.ensembl_id.to_numpy(str))
    m.to_csv(a.output/'MDA_A_metadata.csv',index=False);excluded.to_csv(a.output/'excluded_unknown_background.csv',index=False);pred.to_csv(a.output/'transfer_predictions.csv',index=False)
    report['source_count_shape']=list(counts.shape);report['actual_source_values']=int(counts.size)
    report['strict_common_axis_values']=int(x.size);report['public_counts_receipt']=receipt
    report['metadata_source_url']='https://www.ebi.ac.uk/biostudies/files/E-MTAB-16354/SS3_Human_MDA_A_metadata.csv'
    report['actual_MDA_cells']=len(m);report['excluded_source_rows']=len(excluded);report['common_gene_count']=len(axis)
    report['source_hashes']={p.name:digest(p) for p in [cp,a.public_source/'SS3_Human_MDA_A_metadata.csv',a.a549/'gene_axis.csv',a.a549/'tidyi_one_plate_counts.npz']}
    report['script_sha256']=digest(__file__);report['outputs']={p.name:digest(p) for p in a.output.iterdir()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({**report,'folds':[{k:v for k,v in f.items() if k in ['name','metrics']} for f in report['folds']]},indent=2))


if __name__=='__main__':main()
