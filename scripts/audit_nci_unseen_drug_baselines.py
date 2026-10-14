"""Retrospective drug-held-out expression baselines, not apoptosis/RL validation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recover_public_expression import digest


def kernel(fp1,dose1,time1,fp2,dose2,time2):
    a=np.asarray(fp1,float);b=np.asarray(fp2,float)
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():raise ValueError('Binary fingerprint required')
    arrays=[np.asarray(v,float) for v in [dose1,time1,dose2,time2]]
    if any((v<=0).any() or not np.isfinite(v).all() for v in arrays):raise ValueError('Positive measured dose/time required')
    dot=a@b.T;union=a.sum(axis=1)[:,None]+b.sum(axis=1)[None,:]-dot
    if (union<=0).any():raise ValueError('Empty fingerprint')
    d1,t1,d2,t2=arrays
    dist=(np.log10(d1)[:,None]-np.log10(d2)[None,:])**2+(np.log10(t1)[:,None]-np.log10(t2)[None,:])**2
    return dot/union*np.exp(-.5*dist)


def run(expression,samples,links,catalog,fp,genes=500):
    if samples.sample_id.duplicated().any() or links.sample_id.duplicated().any():raise ValueError('Duplicate sample')
    if not np.isfinite(expression).all():raise ValueError('Observed gene matrix required')
    links=links.set_index('sample_id').loc[samples.sample_id].reset_index()
    eligible=links.action_structure_feature_use_eligible.to_numpy(bool)&~samples.is_vehicle.to_numpy(bool)
    selected=np.flatnonzero(eligible);selected_samples=samples.iloc[selected].reset_index(drop=True)
    if selected_samples.empty:raise ValueError('No eligible measured treatment')
    row={x:i for i,x in enumerate(samples.sample_id)}
    control=np.array([row[v] for v in selected_samples.control_sample_id])
    if not samples.iloc[control].is_vehicle.all():raise ValueError('Control binding is not a vehicle')
    by_id=catalog.set_index('drug_id').fingerprint_row.to_dict()
    f=np.array([fp[int(by_id[v])] for v in links.iloc[selected].action_drug_id])
    delta=expression[selected]-expression[control]
    dose=selected_samples.dose_uM.to_numpy();time=selected_samples.exposure_h.to_numpy()
    gram=kernel(f,dose,time,f,dose,time)
    rows=[];folds=[]
    for held in sorted(selected_samples.drug_name.unique()):
        test=selected_samples.drug_name.eq(held).to_numpy();train=~test
        training_source_rows=np.unique(np.r_[selected[train],control[train]])
        chosen=np.argsort(-expression[training_source_rows].var(axis=0),kind='stable')[:min(genes,expression.shape[1])]
        y=delta[train][:,chosen];target=delta[test][:,chosen];mean=y.mean(axis=0)
        # Fixed alpha=1, unit log10 dose/time bandwidth. No tuning on held drug.
        weights=np.linalg.solve(gram[np.ix_(train,train)]+np.eye(train.sum()),y-mean)
        k=gram[np.ix_(test,train)]
        pred={'zero_effect':np.zeros_like(target),'training_mean_effect':np.broadcast_to(mean,target.shape),
              'nearest_structure_protocol':y[k.argmax(axis=1)],'fixed_kernel_ridge':mean+k@weights}
        folds.append(dict(held_drug=held,training_sample_ids=selected_samples.loc[train,'sample_id'].tolist(),
                          test_sample_ids=selected_samples.loc[test,'sample_id'].tolist(),selected_gene_indices=chosen.tolist()))
        for method,value in pred.items():
            for sample_id,error in zip(selected_samples.loc[test,'sample_id'],np.mean((value-target)**2,axis=1)):
                rows.append(dict(held_drug=held,sample_id=sample_id,method=method,delta_log2_RMA_MSE=float(error)))
    table=pd.DataFrame(rows);perdrug=table.groupby(['held_drug','method']).delta_log2_RMA_MSE.mean().unstack()
    summary={k:dict(mean_drug_MSE=float(perdrug[k].mean()),drugs_better_than_zero=int((perdrug[k]<perdrug.zero_effect).sum()),
                    drugs_better_than_training_mean=int((perdrug[k]<perdrug.training_mean_effect).sum())) for k in perdrug}
    return dict(eligible_treatment_samples=len(selected),held_drugs=selected_samples.drug_name.nunique(),results=summary,folds=folds,
                excluded_treated_sample_ids=samples.loc[~eligible&~samples.is_vehicle,'sample_id'].tolist(),
                kernel='binary Tanimoto times Gaussian on log10 uM and log10 h, bandwidth1 each',ridge_alpha=1,training_HVG=genes,
                hyperparameter_search=False,formal_project_model_training=False,independent_initial_state_value=False,
                strict_blind_preprocessing=False,apoptosis_prediction=False,selection_or_RL_benefit=False,
                limitation='published RMA uses all arrays within drug; retrospective development; dose/time reference controls define targets only'),table,perdrug


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['expression','chemistry','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output required')
    for path in [a.expression,a.chemistry]:
        m=json.loads((path/'manifest.json').read_text())
        for n,sha in m['outputs'].items():
            if digest(path/n)!=sha:raise ValueError('Recovered source changed: '+n)
    data=np.load(a.expression/'common_axis_expression.npz');samples=pd.read_csv(a.expression/'samples.csv')
    if not np.array_equal(data['sample_ids'],samples.sample_id):raise ValueError('Sample order mismatch')
    observed=data['observed_mask'].all(axis=0);x=data['expression'][:,observed]
    r,t,p=run(x,samples,pd.read_csv(a.chemistry/'sample_structure_links.csv'),pd.read_csv(a.chemistry/'chemical_catalog.csv'),np.load(a.chemistry/'fingerprints.npy'))
    a.output.mkdir(parents=True);t.to_csv(a.output/'condition_errors.csv',index=False);p.to_csv(a.output/'drug_mean_errors.csv')
    r['observed_common_gene_indices']=np.flatnonzero(observed).tolist();r['source_manifests']={str(path):digest(path/'manifest.json') for path in [a.expression,a.chemistry]};r['script_sha256']=digest(__file__)
    (a.output/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['folds','observed_common_gene_indices']},indent=2))


if __name__=='__main__':main()
