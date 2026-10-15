"""Recover actual A549 time-course RNA; keep absent time-matched controls explicit."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import read_soft,digest
from lung.data.external_gene_axis import align_gene_axis


def gene_identity(labels):
    ids=labels.astype(str).str.split('|',n=1).str[0]
    if not ids.str.fullmatch(r'ENSG\d+(?:\.\d+)?').all():raise ValueError('Invalid compound gene identifier')
    if ids.str.replace(r'\.\d+$','',regex=True).duplicated().any():raise ValueError('Duplicate Ensembl after source label parsing')
    return ids


def bind_samples(columns,records):
    rows=[]
    for acc,m,_ in records:
        chars=dict(x.split(': ',1) for x in m['characteristics_ch1'])
        if chars['cell line']!='A549':continue
        stem=Path(m['supplementary_file_1'][0]).name.removeprefix(acc+'_').removesuffix('.txt.gz')
        if stem not in columns:raise ValueError('GEO sample absent from matrix')
        day=int(chars['time'].removeprefix('Day '));t=chars['treatment']
        if day not in [0,2,5,10] or (t!='-' if day==0 else t!='trametinib (10nM)'):
            raise ValueError('Unexpected time/dose protocol')
        rep=int(re.search(r'biol rep(\d+)$',m['title'][0]).group(1))
        rows.append(dict(sample_id=acc,source_column=stem,background='A549',time_days=day,
                         elapsed_h=day*24,drug='baseline_no_treatment' if day==0 else 'Trametinib',
                         dose_uM=0 if day==0 else .01,nominal_replicate=rep,
                         replicate_pairing_across_time_verified=False,contemporaneous_vehicle_available=False,
                         source_title=m['title'][0],source_description=m['description'][0]))
    out=pd.DataFrame(rows)
    if len(out)!=12 or set(columns[columns.str.startswith('A549_')])!=set(out.source_column):raise ValueError('A549 coverage mismatch')
    if out.groupby('time_days').nominal_replicate.apply(set).to_dict()!={d:{1,2,3} for d in [0,2,5,10]}:raise ValueError('Time/replicate grid mismatch')
    return out


def time_diagnostic(values, days, feature_count=500):
    """Predeclared interpolation/extrapolation baselines, no test-selected features."""
    values=np.asarray(values,float);days=np.asarray(days)
    if values.ndim!=2 or len(values)!=len(days) or (values<0).any() or not np.isfinite(values).all():
        raise ValueError('Invalid time diagnostic matrix')
    totals=values.sum(axis=1)
    if (totals<=0).any():raise ValueError('Empty library')
    z=np.log1p(values/totals[:,None]*1e6)
    results=[];features={}
    for held,train_days in [(5,[2,10]),(10,[2,5])]:
        train=np.isin(days,train_days);test=days==held
        if test.sum()!=3 or any((days==d).sum()!=3 for d in train_days):raise ValueError('Three source replicates per time required')
        idx=np.argsort(-z[train].var(axis=0),kind='stable')[:min(feature_count,z.shape[1])]
        features[str(held)]=idx.tolist()
        left,right=[z[days==d][:,idx].mean(axis=0) for d in train_days]
        target=z[test][:,idx].mean(axis=0)
        pred={'nearest_time':left if abs(held-train_days[0])<=abs(held-train_days[1]) else right,
              'linear_time':left+(right-left)*(held-train_days[0])/(train_days[1]-train_days[0]),
              'linear_log_time':left+(right-left)*np.log(held/train_days[0])/np.log(train_days[1]/train_days[0])}
        for name,p in pred.items():
            results.append(dict(held_day=held,training_days=train_days,baseline=name,features=len(idx),
                                mean_expression_MSE=float(np.mean((p-target)**2)),
                                individual_library_MSE=float(np.mean((p-z[test][:,idx])**2))))
    return results,features


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','gene-axis','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output required')
    frame=pd.read_csv(a.source/'GSE247512_gene_count.csv.gz')
    _,records=read_soft(a.source/'GSE247512.soft.gz');samples=bind_samples(frame.columns,records)
    frame['ensembl_id']=gene_identity(frame.gene_id)
    vals=frame[samples.source_column].to_numpy(float)
    if not np.isfinite(vals).all() or (vals<0).any() or (vals!=np.floor(vals)).any():raise ValueError('Published count matrix invalid')
    axis=pd.read_csv(a.gene_axis)
    aligned,mask,ledger=align_gene_axis(frame,axis,identifier='ensembl_id',samples=samples.source_column.tolist(),scale='bulk_estimated_counts')
    # Integer-valued source called raw counts but StringTie extraction is reported.
    # Retain conservative estimated-count category instead of asserting molecule counting.
    for r in ledger[ledger.mapping_status.eq('matched')].itertuples():
        if not np.array_equal(aligned[:,int(r.common_gene_index)],vals[r.source_row]):raise ValueError('Alignment changed value')
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'expression_source_counts.npz',expression=vals.T,gene_ids=frame.ensembl_id.to_numpy(str),sample_ids=samples.sample_id.to_numpy(str))
    np.savez_compressed(a.output/'expression_common_axis.npz',expression=aligned,observed_mask=mask,sample_ids=samples.sample_id.to_numpy(str))
    samples.to_csv(a.output/'samples.csv',index=False);ledger.to_csv(a.output/'gene_mapping.csv',index=False)
    diagnostics,features=time_diagnostic(vals.T,samples.time_days.to_numpy())
    (a.output/'time_diagnostic.json').write_text(json.dumps(dict(results=diagnostics,training_only_feature_indices=features,
        scope='factual treated-culture time prediction; no time-matched vehicle, causal drug-duration or sequential claim'),indent=2))
    report=dict(accession='GSE247512',doi='10.1038/s41392-025-02382-w',source_genes=len(frame),
                source_matrix_samples=len(frame.columns)-2,A549_samples=len(samples),numeric_A549_values=vals.size,
                scale='bulk_estimated_counts',source_scale_label='raw counts; extraction and normalization with StringTie',
                common_axis_observed_genes=int(mask[0].sum()),common_axis_missing_genes=int((~mask[0]).sum()),
                days=[0,2,5,10],dose_uM=.01,contemporaneous_control_samples=0,
                pure_drug_time_effect_identified=False,true_sequential_actions=False,individual_cell_pairing=False,
                combination_RNA_claim_in_overall_design_but_not_in_matrix=True,
                combination_replicates_conflict='GEO overall design 3 per treatment; paper Methods 1 per treatment; actual 48-column matrix only time-course',
                matched_Annexin_protocol=False,full_research_qualified=False,
                source_hashes={n:digest(a.source/n) for n in ['GSE247512_gene_count.csv.gz','GSE247512.soft.gz','paper.xml']},
                gene_axis_sha256=digest(a.gene_axis),script_sha256=digest(__file__),outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
