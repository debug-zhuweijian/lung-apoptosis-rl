"""Fixed, small within-plate diagnostic; not future-fate or independent-culture validation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recover_public_expression import digest


def fit_probability(train,y,test,classes=3):
    # Fixed C=1 (sum loss + 0.5 L2), never tuned on held-out rows.
    one=np.eye(classes)[y]
    def objective(w):
        w=w.reshape(train.shape[1],classes);z=train@w
        loss=np.sum(logsumexp(z,axis=1)-np.sum(z*one,axis=1))+.5*np.sum(w*w)
        gradient=train.T@(softmax(z,axis=1)-one)+w
        return loss,gradient.ravel()
    r=minimize(objective,np.zeros(train.shape[1]*classes),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-6})
    if not r.success:raise RuntimeError('Diagnostic optimization did not converge: '+r.message)
    return softmax(test@r.x.reshape(train.shape[1],classes),axis=1)


def training_pca(train,test,genes=200,components=10):
    # Every selection and projection is fit using the training rows only.
    select=np.argsort(-train.var(axis=0),kind='stable')[:min(genes,train.shape[1])]
    center=train[:,select].mean(axis=0);a=train[:,select]-center;b=test[:,select]-center
    _,s,v=np.linalg.svd(a,full_matrices=False);k=min(components,int((s>1e-8).sum()))
    if not k:raise ValueError('No training RNA variation')
    load=v[:k].T;tr=a@load;te=b@load;scale=tr.std(axis=0);scale[scale==0]=1
    return tr/scale,te/scale,select,load,center


def diagnostic(counts,meta,genes=200,components=10):
    if len(counts)!=len(meta) or not np.isfinite(counts).all() or (counts<0).any():raise ValueError('Invalid count binding')
    if meta.Plate.nunique()!=1 or not meta.Cell_line.eq('A549').all():raise ValueError('Requires one strictly identified A549 plate')
    if meta.Well.duplicated().any() or not meta.Well.str.fullmatch(r'[A-P](?:[1-9]|1[0-9]|2[0-4])').all():raise ValueError('Invalid well identifiers')
    labels=sorted(meta.cell_state.unique());y=pd.Categorical(meta.cell_state,categories=labels).codes
    if len(labels)!=3:raise ValueError('Requires three observed measurement-time states')
    row=meta.Well.str[0].to_numpy();col=meta.Well.str[1:].astype(int).to_numpy()-1
    position=np.eye(24)[col];qc=meta[['log1p_total_counts','pct_counts_mt','log1p_n_genes_by_counts']].to_numpy(float)
    if not np.isfinite(qc).all():raise ValueError('Missing QC measurement')
    lib=counts.sum(axis=1)
    if (lib<=0).any():raise ValueError('Zero common-axis library')
    rna=np.log1p(counts/lib[:,None]*1e4)
    output={k:np.full((len(y),3),np.nan) for k in ['position','position_QC','position_QC_RNA']};folds=[]
    for held in sorted(set(row)):
        test=row==held;train=~test
        mean=qc[train].mean(axis=0);sd=qc[train].std(axis=0);sd[sd==0]=1
        q=(qc-mean)/sd
        pc,pt,select,load,center=training_pca(rna[train],rna[test],genes,components)
        bases={'position':(position[train],position[test]),'position_QC':(np.c_[position[train],q[train]],np.c_[position[test],q[test]])}
        bases['position_QC_RNA']=(np.c_[bases['position_QC'][0],pc],np.c_[bases['position_QC'][1],pt])
        for name,(a,b) in bases.items():output[name][test]=fit_probability(a,y[train],b)
        folds.append(dict(held_well_row=held,training_cell_ids=meta.loc[train,'cell_id'].tolist(),test_cell_ids=meta.loc[test,'cell_id'].tolist(),selected_gene_indices=select.tolist(),actual_components=load.shape[1]))
    metrics={};pred=meta[['cell_id','Well','cell_state']].copy()
    for name,p in output.items():
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1):raise ValueError('Incomplete predictions')
        guess=p.argmax(axis=1);loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-15));recall=[np.mean(guess[y==k]==k) for k in range(3)]
        metrics[name]=dict(accuracy=float(np.mean(guess==y)),balanced_accuracy=float(np.mean(recall)),mean_log_loss=float(loss.mean()),per_state_recall=dict(zip(labels,recall)))
        pred[name+'_prediction']=[labels[k] for k in guess];pred[name+'_log_loss']=loss
    state=np.eye(3)[y][:,1:];rank=int(np.linalg.matrix_rank(np.c_[position,state]))
    return dict(cells=len(y),metrics=metrics,folds=folds,labels=labels,column_plus_state_design_rank=rank,
                column_plus_state_design_columns=26,features=dict(HVG=genes,PCA=components,QC=list(meta[['log1p_total_counts','pct_counts_mt','log1p_n_genes_by_counts']].columns)),
                independent_culture_validation=False,future_fate_prediction=False,causal_state_effect_identified=False,
                formal_project_model_training=False,held_out_unit='well_row',hyperparameter_selection='fixed before this run; no tuning',
                limitation='single sorted plate; QC can be biological and technical; conditional prediction is not transport or future fate'),pred


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    manifest=json.loads((a.source/'manifest.json').read_text())
    for n in ['tidyi_one_plate_counts.npz','tidyi_source_metadata.csv']:
        if digest(a.source/n)!=manifest['outputs'][n]:raise ValueError('Recovered source changed')
    data=np.load(a.source/'tidyi_one_plate_counts.npz');meta=pd.read_csv(a.source/'tidyi_source_metadata.csv').set_index('cell_id',drop=False)
    if not meta.index.is_unique:raise ValueError('Duplicate metadata cell')
    meta=meta.loc[data['cell_ids']].reset_index(drop=True)
    report,pred=diagnostic(data['expression'].astype(float),meta)
    report['source_manifest_sha256']=digest(a.source/'manifest.json');report['script_sha256']=digest(__file__)
    a.output.mkdir(parents=True);pred.to_csv(a.output/'held_row_predictions.csv',index=False)
    (a.output/'diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='folds'},indent=2))


if __name__=='__main__':main()
