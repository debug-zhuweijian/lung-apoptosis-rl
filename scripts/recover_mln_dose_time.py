"""Recover the actual A549 factorial and evaluate distinct held axes."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import read_soft,digest
from lung.data.external_gene_axis import align_gene_axis


def bind(columns,records):
    rows=[]
    for acc,m,_ in records:
        c=dict(x.split(': ',1) for x in m['characteristics_ch1'])
        if c['cell line']!='A549':raise ValueError('Wrong background')
        match=re.fullmatch(r'MLN4924 ([0-9.]+)µM',c['treatment'])
        if not match:raise ValueError('Unreviewed dose units')
        dose=float(match[1]);hour=int(c['time'].removesuffix('h'))
        rep=int(re.search(r'Replicate ([12])$',m['title'][0])[1])
        col='S'+m['description'][0]
        if col not in columns:raise ValueError('Source sample identifier missing')
        rows.append(dict(sample_id=acc,source_column=col,dose_uM=dose,exposure_h=hour,source_replicate=rep,
                         background='A549',drug='DMSO_vehicle' if dose==0 else 'MLN4924',
                         vehicle_percent=.1,initial_state_measured=False,Annexin_pair=False))
    s=pd.DataFrame(rows)
    expected={(d,h,r) for d in [0,.25,.5] for h in [8,24] for r in [1,2]}
    if len(s)!=12 or set(zip(s.dose_uM,s.exposure_h,s.source_replicate))!=expected or set(s.source_column)!=set(columns):
        raise ValueError('Incomplete dose-time-control factorial')
    return s


def diagnostics(values,samples,feature_count=500):
    values=np.asarray(values,float)
    if not np.isfinite(values).all() or (values<0).any() or (values.sum(axis=1)<=0).any():raise ValueError('Invalid count matrix')
    z=np.log1p(values/values.sum(axis=1)[:,None]*1e6)
    dose=samples.dose_uM.to_numpy();hour=samples.exposure_h.to_numpy()
    folds={'dose':dose!=.5,'time':hour==8,'protocol':~((dose==.5)&(hour==24))}
    results=[];spec={}
    for name,train in folds.items():
        idx=np.argsort(-z[train].var(axis=0),kind='stable')[:min(feature_count,z.shape[1])]
        def mean(d,h):
            pick=(dose==d)&(hour==h)
            if pick.sum()!=2:raise ValueError('Missing condition or replicate')
            return z[pick][:,idx].mean(axis=0)
        def delta(d,h):return mean(d,h)-mean(0,h)
        targets={'dose':[(.5,8),(.5,24)],'time':[(.25,24),(.5,24)],'protocol':[(.5,24)]}[name]
        spec[name]=dict(training_sample_ids=samples.loc[train,'sample_id'].tolist(),
                        held_sample_ids=samples.loc[~train,'sample_id'].tolist(),feature_indices=idx.tolist(),
                        test_vehicle_role='evaluation_only' if name=='time' else 'available_training_control')
        for d,h in targets:
            if name=='dose':
                base=delta(.25,h);pred={'no_drug_effect':np.zeros_like(base),'copy_lower_dose':base,'linear_dose':2*base}
            elif name=='time':
                base=delta(d,8);pred={'no_drug_effect':np.zeros_like(base),'copy_early_effect':base,'linear_time':3*base}
            else:
                base=delta(.5,8);low=delta(.25,24)
                pred={'no_drug_effect':np.zeros_like(base),'copy_same_dose_early':base,'copy_same_time_lower_dose':low,
                      'additive_rectangle':base+low-delta(.25,8)}
            target=delta(d,h)
            for method,p in pred.items():
                results.append(dict(fold=name,dose_uM=d,exposure_h=h,baseline=method,
                                    delta_log1pCPM_MSE=float(np.mean((p-target)**2)),genes=len(idx)))
    return dict(results=results,folds=spec,formal_model_training=False,
                evaluation='retrospective_source_protocol_diagnostic_not_blind_validation',
                baseline_assumptions='fixed proportionality/additivity on log1pCPM effects; not biological laws')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','gene-axis','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New directory required')
    f=pd.read_csv(a.source/'counts.txt.gz',sep='\t');_,records=read_soft(a.source/'GSE134190.soft.gz')
    if f.columns[:3].tolist()!=['gene_ID','Symbol','description'] or not f.gene_ID.is_unique:raise ValueError('Source gene layout changed')
    s=bind(f.columns[3:],records);columns=s.source_column.tolist();v=f[columns].to_numpy(float)
    if (v<0).any() or not np.isfinite(v).all() or (v!=np.floor(v)).any():raise ValueError('Source count values invalid')
    axis=pd.read_csv(a.gene_axis);frame=f.rename(columns={'gene_ID':'ensembl_id'})
    aligned,mask,ledger=align_gene_axis(frame,axis,identifier='ensembl_id',samples=columns,scale='bulk_estimated_counts')
    for r in ledger[ledger.mapping_status.eq('matched')].itertuples():
        if not np.array_equal(aligned[:,int(r.common_gene_index)],v[r.source_row]):raise ValueError('Axis changed source value')
    result=diagnostics(v.T,s)
    a.output.mkdir(parents=True);s.to_csv(a.output/'samples.csv',index=False);ledger.to_csv(a.output/'gene_mapping.csv',index=False)
    f.iloc[:,:3].to_csv(a.output/'source_gene_annotation.csv',index=False)
    np.savez_compressed(a.output/'source_expression.npz',expression=v.T,sample_ids=s.sample_id.to_numpy(str),gene_ids=f.gene_ID.to_numpy(str))
    np.savez_compressed(a.output/'common_axis_expression.npz',expression=aligned,observed_mask=mask,sample_ids=s.sample_id.to_numpy(str))
    (a.output/'axis_holdout_diagnostics.json').write_text(json.dumps(result,indent=2))
    report=dict(accession='GSE134190',doi='10.1073/pnas.1905973116',genes=len(f),samples=len(s),
                numeric_values=v.size,common_axis_observed=int(mask[0].sum()),common_axis_missing=int((~mask[0]).sum()),
                background='A549',doses_uM=[0,.25,.5],exposure_hours=[8,24],source_repeats_per_condition=2,
                contemporaneous_vehicle_present=True,vehicle_percent=.1,scale='bulk_estimated_counts',
                processing='source integer counts from StringTie assembly; not assumed direct molecule counting',
                independent_drug_axis=False,true_sequential_actions=False,Annexin_supervision=False,
                source_initial_state_variation_identified=False,full_research_qualified=False,
                source_urls={'counts':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE134nnn/GSE134190/suppl/GSE134190_gene_sample_count_with_symbol.txt.gz',
                             'metadata':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE134nnn/GSE134190/soft/GSE134190_family.soft.gz',
                             'paper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6789910/'},
                source_hashes={n:digest(a.source/n) for n in ['counts.txt.gz','GSE134190.soft.gz','paper.html']},
                gene_axis_sha256=digest(a.gene_axis),script_sha256=digest(__file__),outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
