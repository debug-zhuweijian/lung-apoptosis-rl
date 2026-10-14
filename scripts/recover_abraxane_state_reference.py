"""Recover A549/Abr's crossed state/Abraxane reference without inventing apoptosis.

The common-axis output is a single-RefSeq feature proxy, not a reconstructed gene
sum. Ambiguous source suffixes and many-source-to-one-gene mappings stay excluded.
"""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import etree
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.recover_public_expression import read_soft,digest
from scripts.recover_nci_a549_factorial import probe_gene_map
from lung.data.external_gene_axis import align_gene_axis

SAMPLES=[('GSM1367785','A549-P-1','A549-P-1','A549',0),
 ('GSM1367786','A549-P-2','A549-P-2','A549',0),
 ('GSM1367787','A549-P-100','A549-P-100','A549',100),
 ('GSM1367788','A549/Abr','A549-R-0','A549/Abr',0),
 ('GSM1367789','A549/Abr-100','A549-R-100','A549/Abr',100)]


def bind_samples(columns,records):
    by_id={a:m for a,m,_ in records};rows=[]
    if set(by_id)!={r[0] for r in SAMPLES}:raise ValueError('Unexpected GEO sample coverage')
    if list(columns)!=[r[2] for r in SAMPLES]:raise ValueError('Source matrix columns/order changed')
    for acc,title,col,state,dose in SAMPLES:
        m=by_id[acc];chars=dict(v.split(': ',1) for v in m['characteristics_ch1'])
        if m['title']!=[title] or chars['strain']!=state or m['organism_ch1']!=['Homo sapiens']:
            raise ValueError('Source sample identity changed')
        rows.append(dict(sample_id=acc,source_title=title,source_column=col,state=state,
            dose_nM_paclitaxel_equivalent=dose,exposure_hours=4,
            drug_material='albumin_bound_paclitaxel_Abraxane' if dose else 'drug_free_parallel_culture',
            source_column_binding='curated title/condition correspondence; R denotes author resistant A549/Abr',
            biological_replicate_independence_verified=False,measured_predose_same_cell=False,
            history='parental' if state=='A549' else '6_month_stepwise_selection_10_to_400nM;100nM_maintenance;at_least_1week_drug_free_before_experiment'))
    return pd.DataFrame(rows)


def map_refseq(labels,link,gene_info):
    if len(link.columns)!=19:raise ValueError('UCSC ncbiRefSeqLink schema changed')
    a=link[[0,6]].copy();a.columns=['refseq','gene_id'];a['base']=a.refseq.str.replace(r'\.\d+$','',regex=True)
    valid=a.gene_id.str.fullmatch(r'[1-9]\d*');a=a[valid]
    candidates=a.groupby('base').gene_id.agg(lambda x:sorted(set(x)))
    unique={k:v[0] for k,v in candidates.items() if len(v)==1}
    rows=[]
    for label in labels:
        # Never strip _2 etc: these are author features, not RefSeq versions.
        canonical=bool(re.fullmatch(r'[NX][MR]_\d+(?:\.\d+)?',label))
        base=re.sub(r'\.\d+$','',label) if canonical else None
        rows.append(dict(ID=label,**{'Gene ID':unique.get(base,''),'Gene symbol':''},source_label_canonical=canonical))
    f=pd.DataFrame(rows);m=probe_gene_map(f,gene_info);m['source_label_canonical']=f.source_label_canonical
    m.loc[~f.source_label_canonical,'mapping_status']='author_suffix_unresolved'
    return m


def functional_table(xml):
    root=etree.parse(str(xml));rows=[]
    tables=root.xpath('//table-wrap[@id="pone.0131429.t001"]//tbody/tr')
    if len(tables)!=7:raise ValueError('Primary functional table changed')
    for t in tables:
        c=[''.join(x.itertext()).strip() for x in t.findall('td')]
        q=re.fullmatch(r'(.+)\((.+)\)',c[0])
        if not q:raise ValueError('Drug/table unit absent')
        for state,value in zip(['A549','A549/Abr'],c[1:3]):
            match=re.fullmatch(r'([0-9.]+)±([0-9.]+)',value)
            if not match:raise ValueError('Unknown primary summary format')
            rows.append(dict(state=state,source_drug=q[1],IC50_mean=float(match[1]),IC50_SD=float(match[2]),
                source_unit=q[2],endpoint='48h_MTT_relative_metabolic_viability_IC50_not_Annexin',
                exposure_hours=48,source_statistic='mean_SD_at_least_three_separate_experiments',
                individual_replicates_available=False,author_resistance_index=float(c[3]),author_P=c[4],
                material='albumin_bound_paclitaxel' if q[1]=='Abraxane' else 'source_reported_drug',
                unit_context='table_nM_methods_ng_per_mL' if q[1]=='DTX' else ('table_uM_methods_ug_per_mL' if q[1]=='5-FU' else 'as_reported'),
                exact_RNA_endpoint_dose_time_pair=False))
    return pd.DataFrame(rows)


def descriptive_interaction(values):
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[0]!=5 or not np.isfinite(x).all() or (x<0).any():raise ValueError('Five finite source libraries required')
    z=np.log2(1+x);parent=z[2]-z[:2].mean(axis=0);resistant=z[4]-z[3]
    return dict(source_feature_count=x.shape[1],scale='log2(1+source_FPKM)',
        parental_response_mean_square=float(np.mean(parent**2)),resistant_response_mean_square=float(np.mean(resistant**2)),
        descriptive_state_by_treatment_interaction_mean_square=float(np.mean((resistant-parent)**2)),
        parental_reference_disagreement_mean_square=float(np.mean((z[0]-z[1])**2)),
        interaction_confounded_with_unreplicated_treated_libraries=True,statistical_significance_test=False,
        RNA_increment_beyond_state_ID_identified=False,formal_training=False)


def recover(source,gene_info,gene_axis,output):
    if output.exists():raise FileExistsError('New output required')
    f=pd.read_csv(source/'GSE56742_5samples.fpkm.txt.gz',sep='\t');_,records=read_soft(source/'GSE56742_family.soft.gz')
    if f.columns[0]!='Transcript' or f.Transcript.isna().any() or f.Transcript.duplicated().any():raise ValueError('Invalid source feature identity')
    samples=bind_samples(f.columns[1:],records);values=f.iloc[:,1:].to_numpy(float).T
    diagnostic=descriptive_interaction(values)
    link=pd.read_csv(source/'ncbiRefSeqLink.txt.gz',sep='\t',header=None,dtype=str,keep_default_na=False)
    gi=pd.read_csv(gene_info,sep='\t',dtype=str,keep_default_na=False);gi=gi[gi['#tax_id'].eq('9606')]
    mapping=map_refseq(f.Transcript,link,gi);g=pd.DataFrame(values.T,columns=samples.sample_id);g['ensembl_id']=mapping.ensembl_id
    axis=pd.read_csv(gene_axis);aligned,mask,ledger=align_gene_axis(g,axis,identifier='ensembl_id',samples=samples.sample_id.tolist(),scale='bulk_FPKM')
    output.mkdir(parents=True);samples.to_csv(output/'samples.csv',index=False);mapping.to_csv(output/'refseq_gene_mapping.csv.gz',index=False);ledger.to_csv(output/'axis_mapping.csv.gz',index=False)
    np.savez_compressed(output/'source_FPKM.npz',expression=values,feature_ids=f.Transcript.to_numpy(str),sample_ids=samples.sample_id.to_numpy(str))
    np.savez_compressed(output/'single_refseq_proxy_axis.npz',expression=aligned,observed_mask=mask,gene_ids=axis.sort_values('common_gene_index').ensembl_id.to_numpy(str),sample_ids=samples.sample_id.to_numpy(str))
    with np.load(output/'source_FPKM.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['expression'],values)
    with np.load(output/'single_refseq_proxy_axis.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['expression'],aligned);np.testing.assert_array_equal(z['observed_mask'],mask)
        assert np.isnan(z['expression'][~mask]).all()
    functional_table(source/'paper.xml').to_csv(output/'functional_IC50_reference.csv',index=False)
    (output/'descriptive_interaction.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    (output/'source_metadata.json').write_text(json.dumps({a:m for a,m,_ in records},indent=2)+'\n')
    report=dict(accession='GSE56742',doi='10.1371/journal.pone.0131429',source_feature_count=len(f),actual_libraries=5,
        actual_source_values=values.size,source_feature_unit='author_RefSeq_feature_FPKM',
        GEO_processing_calls_RPKM_four_samples_but_actual_table_has_five=True,
        common_axis_observed_proxy_features=int(mask[0].sum()),common_axis_missing=int((~mask[0]).sum()),
        proxy_not_gene_total_FPKM=True,transcript_summing_performed=False,
        author_suffix_rows=int((~mapping.source_label_canonical).sum()),
        mapped_many_to_one_rows_rejected=int(ledger.mapping_status.eq('duplicate_source_identifier').sum()),
        full_production_model_input_qualified=False,qualified_Annexin_observations=0,
        functional_group_IC50_summaries=14,matched_RNA_endpoint_dose_time_pairs=0,
        complete_two_state_by_Abraxane_treated_control_reference_grid=True,independent_state_replicates_or_RNA_increment_verified=False,
        retrospective_descriptive_only=True,formal_training=False,full_research_feasibility=False,
        source_hashes={n:digest(source/n) for n in ['GSE56742_5samples.fpkm.txt.gz','GSE56742_family.soft.gz','paper.xml','ncbiRefSeqLink.txt.gz','ncbiRefSeqLink.sql']},
        current_gene_info_sha256=digest(gene_info),gene_axis_sha256=digest(gene_axis),script_sha256=digest(__file__),
        outputs={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','gene-info','gene-axis','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.source,a.gene_info,a.gene_axis,a.output),indent=2))
