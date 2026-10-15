"""Recover only actual A549 NCI-TPW arrays, with explicit annotation and RMA scale."""
from __future__ import annotations
import argparse,gzip,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import read_soft,table_block,digest
from lung.data.external_gene_axis import align_gene_axis


def probe_gene_map(annotation,gene_info):
    if not annotation.ID.is_unique or not gene_info.GeneID.is_unique:raise ValueError('Duplicate annotation identifiers')
    reference={}
    for row in gene_info.itertuples():
        ids=re.findall(r'(?:^|\|)Ensembl:(ENSG\d+)(?=\||$)',str(row.dbXrefs))
        reference[str(row.GeneID)]=ids[0] if len(set(ids))==1 else None
    rows=[]
    for r in annotation.to_dict('records'):
        ids=[v.strip() for v in str(r['Gene ID']).split('///') if v.strip()]
        if len(ids)!=1 or not ids[0].isdigit():status='ambiguous_or_missing_source_entrez';ens=None
        else:
            ens=reference.get(ids[0]);status='unique_entrez_to_ensembl' if ens else 'missing_or_ambiguous_current_ensembl'
        rows.append(dict(probe_id=r['ID'],source_gene_symbol=r['Gene symbol'],source_entrez=r['Gene ID'],
                         ensembl_id=ens,mapping_status=status))
    return pd.DataFrame(rows)


def sample_protocol(title,metadata):
    m=re.fullmatch(r'A549_(.+)_([0-9.]+)nM_(2|6|24)h',title)
    if not m or metadata['platform_id']!=['GPL571'] or metadata['source_name_ch1']!=['human tumor cell line A549']:
        raise ValueError('Unexpected array background/platform/title')
    source_drug=m[1];nm=float(m[2]);hour=int(m[3])
    drug={'gemcitibine':'gemcitabine'}.get(source_drug,source_drug)
    return dict(source_drug_label=source_drug,drug_name=drug,dose_nM=nm,dose_uM=nm/1000,
                exposure_h=hour,is_vehicle=nm==0,control_group=source_drug+':'+str(hour),
                nominal_background='A549',vehicle_DMSO_percent=.25,
                chemical_structure_verified=False,independent_biological_repeat_verified=False,
                treatment_design='single_continuous_exposure',same_cell_pair=False)


def axis_diagnostics(expression,samples,feature_count=500):
    # Native log2 RMA. No CPM/log transform is applied to an already logged assay.
    x=np.asarray(expression,float)
    if not np.isfinite(x).all():raise ValueError('Diagnostic requires observed genes only')
    result=[];spec={};skipped=[]
    for drug,group in samples.groupby('drug_name'):
        doses=sorted(group.dose_nM.unique())
        if len(doses)!=3 or doses[0]!=0:raise ValueError('Unexpected dose grid')
        low,high=doses[1:];own=samples.drug_name.eq(drug).to_numpy()
        if len(group)!=9:
            skipped.append(dict(drug=drug,reason='incomplete_3_by_3_protocol_grid',available_samples=len(group)))
            continue
        dose=samples.dose_nM.to_numpy();hour=samples.exposure_h.to_numpy()
        folds={'dose':own&(dose!=high),'time':own&(hour!=24),'protocol':own&~((dose==high)&(hour==24))}
        def sample(d,h):
            ix=np.flatnonzero(own&(dose==d)&(hour==h))
            if len(ix)!=1:raise ValueError('Missing or duplicate condition')
            return ix[0]
        for fold,train in folds.items():
            idx=np.argsort(-x[train].var(axis=0),kind='stable')[:min(feature_count,x.shape[1])]
            def delta(d,h):return x[sample(d,h),idx]-x[sample(0,h),idx]
            key=drug+':'+fold
            spec[key]=dict(training_sample_ids=samples.loc[train,'sample_id'].tolist(),
                           held_sample_ids=samples.loc[own&~train,'sample_id'].tolist(),
                           observed_gene_feature_indices=idx.tolist(),target_vehicle_for_truth_only=fold=='time')
            targets={'dose':[(high,h) for h in [2,6,24]],'time':[(low,24),(high,24)],'protocol':[(high,24)]}[fold]
            for d,h in targets:
                if fold=='dose':
                    base=delta(low,h);pred={'zero_effect':np.zeros_like(base),'copy_lower_dose':base,'proportional_dose':base*d/low}
                elif fold=='time':
                    base=delta(d,6);pred={'zero_effect':np.zeros_like(base),'copy_6h':base,
                                        'linear_time_extrapolation':delta(d,2)+(base-delta(d,2))*5.5}
                else:
                    base=delta(high,6);other=delta(low,24)
                    pred={'zero_effect':np.zeros_like(base),'copy_same_dose_6h':base,'copy_lower_dose_24h':other,
                          'additive_rectangle':base+other-delta(low,6)}
                target=delta(d,h)
                for method,y in pred.items():
                    result.append(dict(drug=drug,fold=fold,dose_nM=d,time_h=h,baseline=method,
                                       delta_log2_RMA_MSE=float(np.mean((y-target)**2))))
    return dict(results=result,folds=spec,excluded_incomplete_drugs=skipped,
                limitation='single array per condition; published RMA used all arrays within drug; retrospective diagnostic, not strict blind validation',
                formal_model_training=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','gene-axis','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    selection=json.loads((a.source/'selected_samples.json').read_text())
    receipts=json.loads((a.source/'sample_download_receipts.json').read_text())
    if len(receipts)!=134 or {r['accession'] for r in receipts}!={r['accession'] for r in selection}:raise ValueError('A549 source coverage mismatch')
    summaries=json.loads((a.source/'a549_gds_summaries.json').read_text())['result']['200116436']
    full=[r for r in summaries['samples'] if r['title'].startswith('A549_')]
    if {r['accession'] for r in full}!={r['accession'] for r in selection}:raise ValueError('Series metadata coverage differs')
    values=[];samples=[];probes=None;raw_metadata={}
    for record in receipts:
        p=a.source/record['file']
        if digest(p)!=record['sha256']:raise ValueError('Source sample bytes changed')
        text,blocks=read_soft(p)
        if len(blocks)!=1:raise ValueError('Expected one sample per source response')
        acc,m,block=blocks[0]
        if acc!=record['accession'] or m['title']!=[record['title']]:raise ValueError('Sample identity drift')
        t=table_block(block,'sample')
        if not t.ID_REF.is_unique:raise ValueError('Duplicate probe')
        if probes is None:probes=t.ID_REF.tolist()
        if probes!=t.ID_REF.tolist():raise ValueError('Probe order changed')
        v=pd.to_numeric(t.VALUE,errors='raise').to_numpy(float)
        if not np.isfinite(v).all():raise ValueError('Nonfinite measured RMA')
        values.append(v);samples.append(dict(sample_id=acc,source_title=m['title'][0],**sample_protocol(m['title'][0],m)))
        raw_metadata[acc]=m
    values=np.asarray(values);samples=pd.DataFrame(samples)
    if samples.duplicated(['drug_name','dose_nM','exposure_h']).any():raise ValueError('Duplicate protocol requires separate replicate review')
    controls=samples[samples.is_vehicle]
    if controls.control_group.duplicated().any() or not set(samples.control_group)<=set(controls.control_group):raise ValueError('Missing/ambiguous time-and-drug-specific control')
    samples['control_sample_id']=samples.control_group.map(controls.set_index('control_group').sample_id)
    with gzip.open(a.source/'GPL571.annot.gz','rt') as f:annotation=table_block(f.read(),'platform')
    if set(probes)!=set(annotation.ID):raise ValueError('Platform probe coverage mismatch')
    annotation=annotation.set_index('ID',drop=False).loc[probes].reset_index(drop=True)
    gi=pd.read_csv(a.source/'Homo_sapiens.gene_info.gz',sep='\t',dtype=str,keep_default_na=False)
    annotation_taxa=gi['#tax_id'].value_counts().to_dict()
    gi=gi[gi['#tax_id'].eq('9606')].copy()
    if gi.empty:raise ValueError('Human annotation absent')
    mapping=probe_gene_map(annotation,gi)
    expression=pd.DataFrame(values.T,columns=samples.sample_id.tolist())
    expression['ensembl_id']=mapping.ensembl_id
    # An explicit log-intensity summary, not summing transcript counts.
    gene=expression.dropna(subset=['ensembl_id']).groupby('ensembl_id',sort=True).mean().reset_index()
    axis=pd.read_csv(a.gene_axis)
    aligned,mask,ledger=align_gene_axis(gene,axis,identifier='ensembl_id',samples=samples.sample_id.tolist(),scale='microarray_log2_RMA')
    for r in ledger[ledger.mapping_status.eq('matched')].itertuples():
        if not np.array_equal(aligned[:,int(r.common_gene_index)],gene.iloc[r.source_row,1:].to_numpy(float)):raise ValueError('Axis changed summarized values')
    missing=[]
    for drug,part in samples.groupby('drug_name'):
        for dose in sorted(part.dose_nM.unique()):
            for h in [2,6,24]:
                if not ((part.dose_nM==dose)&(part.exposure_h==h)).any():missing.append(dict(drug_name=drug,dose_nM=dose,exposure_h=h))
    diagnostic=axis_diagnostics(aligned[:,mask.all(axis=0)],samples)
    a.output.mkdir(parents=True)
    (a.output/'axis_diagnostics.json').write_text(json.dumps(diagnostic,indent=2))
    a.output.mkdir(parents=True,exist_ok=True);samples.to_csv(a.output/'samples.csv',index=False);mapping.to_csv(a.output/'probe_gene_mapping.csv',index=False);ledger.to_csv(a.output/'common_gene_mapping.csv',index=False)
    np.savez_compressed(a.output/'source_probe_expression.npz',expression=values,probe_ids=np.array(probes),sample_ids=samples.sample_id.to_numpy(str))
    np.savez_compressed(a.output/'common_axis_expression.npz',expression=aligned,observed_mask=mask,sample_ids=samples.sample_id.to_numpy(str))
    (a.output/'source_metadata.json').write_text(json.dumps(raw_metadata,indent=2))
    report=dict(accession='GSE116436',doi='10.1158/0008-5472.CAN-18-0989',source_series_samples=summaries['n_samples'],
                A549_actual_samples=len(samples),probe_count=len(probes),numeric_source_values=values.size,
                drug_labels=samples.drug_name.nunique(),time_h=[2,6,24],actual_controls=len(controls),missing_protocols=missing,
                common_axis_observed=int(mask[0].sum()),common_axis_missing=int((~mask[0]).sum()),
                annotation_taxa=annotation_taxa,probe_mapping_status=mapping.mapping_status.value_counts().to_dict(),source_scale='microarray_log2_RMA',
                aggregation='arithmetic mean log2 intensity across probes with unique source Entrez and unique current Ensembl',
                RMA_fit_scope='published within-drug all-cell-line arrays; not refitted on training split',
                strict_train_only_preprocessing=False,whole_NCI_RAW_downloaded=False,Annexin_labels=False,
                formal_training=False,full_research_qualified=False,
                annotation_revision='GPL571 annotation 2016-08-09; current NCBI human gene_info pinned by bytes',
                source_hashes={n:digest(a.source/n) for n in ['GPL571.annot.gz','Homo_sapiens.gene_info.gz','selected_samples.json','sample_download_receipts.json','a549_gds_summaries.json','paper.html']},
                source_samples=receipts,gene_info_source=json.loads((a.source/'gene_info_source.json').read_text()),
                gene_axis_sha256=digest(a.gene_axis),script_sha256=digest(__file__),outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['source_samples']},indent=2))


if __name__=='__main__':main()
