"""Fixed crossed drug/experiment holdout of observed RNA -> external STACK LF.

This is a retrospective cross-culture feature diagnostic, not initial-state
increment, predicted-response validation, apoptosis calibration or RL training.
"""
import argparse,ast,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.recover_public_expression import digest
from scripts.audit_nci_unseen_drug_baselines import kernel as structure_kernel


def response_features(train,test,genes=200):
    train=np.asarray(train,float);test=np.asarray(test,float)
    eligible=np.isfinite(train).all(axis=0)
    variance=np.zeros(train.shape[1]);variance[eligible]=np.var(train[:,eligible],axis=0)
    selected=np.flatnonzero(eligible & (variance>1e-12))
    selected=selected[np.argsort(-variance[selected],kind='stable')[:genes]]
    if len(selected)==0:raise ValueError('No variable training-observed RNA features')
    mean=train[:,selected].mean(axis=0);scale=train[:,selected].std(axis=0)
    z=(train[:,selected]-mean)/scale;held=test[:,selected]
    missing=~np.isfinite(held)
    # Explicit prediction-time imputation by TRAINING mean; source stays NaN.
    zt=(np.where(missing,mean,held)-mean)/scale
    return z,zt,selected,int(missing.sum())


def predict_kernel(k,kt,y,alpha=1.):
    mean=float(np.mean(y));return mean+kt@np.linalg.solve(k+alpha*np.eye(len(y)),y-mean)


def evaluate(x,fp,y,names,genes=200):
    x=np.asarray(x,float);fp=np.asarray(fp);y=np.asarray(y,float)
    n=len(names)
    if len(set(names))!=n or x.shape[0]!=n or fp.shape[0]!=n or y.shape!=(n,3) or n<4:raise ValueError('Crossed units differ')
    if not np.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('Valid measured fractions required')
    if not np.isin(fp,[0,1]).all():raise ValueError('Binary structural features required')
    ks=structure_kernel(fp,np.full(n,5),np.full(n,24),fp,np.full(n,5),np.full(n,24))
    rows=[];folds=[]
    for i,name in enumerate(names):
        tr=np.flatnonzero(np.arange(n)!=i);z,zt,selected,missing=response_features(x[tr],x[i:i+1],genes)
        kr=z@z.T/z.shape[1];krt=zt@z.T/z.shape[1];k=ks[np.ix_(tr,tr)];kt=ks[np.ix_([i],tr)]
        for r in range(3):
            reps=[j for j in range(3) if j!=r];train_y=y[np.ix_(tr,reps)].mean(axis=1)
            pred={'training_function_mean':float(train_y.mean()),
                'structure_only':float(predict_kernel(k,kt,train_y)[0]),
                'observed_RNA_only':float(predict_kernel(kr,krt,train_y)[0]),
                'structure_plus_observed_RNA':float(predict_kernel((k+kr)/2,(kt+krt)/2,train_y)[0])}
            for method,value in pred.items():rows.append(dict(held_drug=name,held_experiment=r+1,method=method,
                measured_fraction=float(y[i,r]),prediction=value,absolute_error=abs(value-y[i,r]),squared_error=(value-y[i,r])**2,
                outside_unit_interval=not 0<=value<=1,clipped=False))
            folds.append(dict(held_drug=name,held_experiment=r+1,training_drugs=[names[j] for j in tr],
                training_experiments=[j+1 for j in reps],selected_gene_indices=selected.tolist(),
                test_missing_features_imputed_to_training_mean=missing))
    predictions=pd.DataFrame(rows);metrics=[]
    for method,g in predictions.groupby('method',sort=False):
        metrics.append(dict(method=method,MAE=float(g.absolute_error.mean()),MSE=float(g.squared_error.mean()),
                            outside_unit_interval=int(g.outside_unit_interval.sum()),crossed_predictions=len(g)))
    report=dict(drugs=n,independent_source_experiments=3,crossed_holdouts=n*3,metrics=metrics,folds=folds,
        fixed_gene_limit=genes,fixed_kernel_ridge_alpha=1.,combined_kernel_weights=[.5,.5],hyperparameter_search=False,
        RNA_feature='log1p(author treated CP10k mean)-log1p(author pooled-control CP10k mean)',
        endpoint='SYTOX_STACK_lethal_fraction_not_Annexin',initial_state_increment=False,
        predicted_RNA_used=False,paired_RNA_functional_units=False,matched_culture=False,
        formal_G_or_RL_training=False,apoptosis_calibration=False,selection_benefit_validated=False,
        source_missing_values_zero_filled=False,cell_level_independence_assumed=False,
        statistical_pvalues_or_confidence_intervals=False,full_scientific_qualification=False)
    return report,predictions


def validate_cohort(expression_targets, catalog, eligibility_column):
    """Require the declared cohort; never silently fit a downloaded subset."""
    selected = catalog.loc[catalog[eligibility_column].eq(True), 'tahoe_drug']
    if selected.empty or selected.duplicated().any():
        raise ValueError('Declared chemical cohort is empty or duplicated')
    observed = []
    for target in expression_targets:
        parsed = ast.literal_eval(str(target))
        if not isinstance(parsed, list) or len(parsed) != 1 or len(parsed[0]) != 3:
            raise ValueError('Single chemical condition required')
        drug, dose, unit = parsed[0]
        if isinstance(dose, bool) or dose != 5 or unit != 'uM':
            raise ValueError('Declared 5uM cohort differs')
        observed.append(drug)
    if len(observed) != len(set(observed)) or set(observed) != set(selected):
        raise ValueError('Incomplete or duplicate declared expression cohort')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['expression','chemistry','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--eligibility-column',choices=['strict_structure_link_eligible','nominal_reference_link_eligible','nominal_functional_transfer_eligible'],default='strict_structure_link_eligible')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    for directory in [a.expression,a.chemistry]:
        m=json.loads((directory/'manifest.json').read_text())
        for name,sha in m['outputs'].items():
            if digest(directory/name)!=sha:raise ValueError('Input package bytes changed')
    z=np.load(a.expression/'aligned_group_means.npz',allow_pickle=False);c=pd.read_csv(a.chemistry/'nominal_structures.csv')
    f=pd.read_csv(a.chemistry/'functional_observations.csv');chem=pd.read_csv(a.chemistry/'chemical_catalog.csv');fp=np.load(a.chemistry/'fingerprints.npy',allow_pickle=False)
    validate_cohort(z['targets'], c, a.eligibility_column)
    names=[];fps=[];targets=[]
    for t in z['targets']:
        drug,dose,unit=ast.literal_eval(str(t))[0]
        row=c[c.tahoe_drug.eq(drug)&c[a.eligibility_column].eq(True)]
        if len(row)!=1 or dose!=5 or unit!='uM':raise ValueError('RNA condition identity differs')
        r=row.iloc[0];obs=f[f.source_compound_name.eq(r.source_compound_name)].sort_values('replicate')
        if obs.replicate.tolist()!=[1,2,3] or not obs.valid_measured_fraction.all() or obs.author_autofluorescence_exclusion.any():raise ValueError('Assay coverage or QC differs')
        v=chem[chem.drug_id.eq(r.drug_id)]
        if len(v)!=1:raise ValueError('Fingerprint identity differs')
        fps.append(fp[int(v.fingerprint_row.iloc[0])]);names.append(r.source_compound_name);targets.append(obs.lethal_fraction.to_numpy())
    x=np.log1p(z['target_mean'])-np.log1p(z['reference_mean'])
    report,pred=evaluate(x,np.array(fps),np.array(targets),names)
    a.output.mkdir(parents=True);pred.to_csv(a.output/'predictions.csv',index=False)
    pd.DataFrame(dict(drug=names,expression_target=z['targets'])).to_csv(a.output/'identity_order.csv',index=False)
    report.update(eligibility_column=a.eligibility_column,source_manifests={str(d/'manifest.json'):digest(d/'manifest.json') for d in [a.expression,a.chemistry]},
        script_sha256=digest(__file__),outputs={q.name:digest(q) for q in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__':main()
