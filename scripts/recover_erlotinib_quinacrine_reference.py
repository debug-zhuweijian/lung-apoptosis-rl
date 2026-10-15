"""Recover GSE57422 source arrays and printed Annexin quadrants without pseudo-pairs."""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.recover_public_expression import read_soft,table_block,digest
from scripts.recover_nci_a549_factorial import probe_gene_map
from lung.data.external_gene_axis import align_gene_axis

# Original Fig2A, personally inspected: TL PI+/Annexin-, TR double+, BL double-, BR Annexin+/PI-.
QUADRANTS={'left_1uM':[(4.02,1.54,90.9,3.51),(4.42,3.85,84.2,7.58),
                      (18.3,28.9,40.2,12.6),(26.5,27.5,37.4,8.56)],
           'right_5uM':[(3.24,4.00,88.8,3.91),(5.75,2.60,88.6,3.08),
                       (34.0,44.3,19.3,2.43),(42.0,43.4,12.1,2.42)]}
FIGURE_SHA='1b407e3ae6740ed37f063448352add09663bae30679fe40b040e2fb75c85db06'


def sample_protocol(accession,meta):
    title=meta['title'][0]
    m=re.fullmatch(r'(A549|H1975) cells, (.+)',title)
    if not m or meta['platform_id']!=['GPL17692']:raise ValueError('Unexpected source background/platform')
    cell,condition=m.groups();baseline=condition=='untreated'
    erlotinib=quinacrine=0.;hour=0
    if not baseline:
        match=re.fullmatch(r'(.+), (6|12|24|48) h',condition)
        if not match:raise ValueError('Unparsed source condition')
        hour=int(match[2]);drugs=match[1]
        components=drugs.split(' + ')
        parsed=[]
        for component in components:
            drug=re.fullmatch(r'(erlotinib|quinacrine) ([0-9.]+) μM',component)
            if not drug:raise ValueError('Unparsed drug/dose')
            parsed.append(drug[1])
            if drug[1]=='erlotinib':erlotinib=float(drug[2])
            else:quinacrine=float(drug[2])
        if len(set(parsed))!=len(parsed):raise ValueError('Duplicate drug component')
    chars='; '.join(meta.get('characteristics_ch1',[]))
    return dict(sample_id=accession,source_title=title,nominal_cell_line=cell,
        erlotinib_uM=erlotinib,quinacrine_uM=quinacrine,exposure_hours=hour,
        is_baseline=baseline,control_time_hours=0,parallel_time_matched_control=False,
        design='baseline' if baseline else ('simultaneous_combination' if erlotinib and quinacrine else 'single_continuous'),
        source_characteristics=chars,source_genotype_conflict='G61H' in chars,
        paper_A549_genotype_conflicts_with_canonical_background=cell=='A549',
        independent_replicates_verified=False,paired_functional_replicate=False)


def parse_assignments(annotation):
    rows=[]
    for row in annotation.to_dict('records'):
        ids=set();symbols=set()
        for entry in row['gene_assignment'].split(' /// '):
            parts=[v.strip() for v in entry.split('//')]
            if len(parts)>=5:
                if parts[4].isdigit():ids.add(parts[4])
                if parts[1]!='---':symbols.add(parts[1])
        ambiguous_symbol_with_missing_id = len(symbols)>1 and len(ids)==1
        rows.append({'ID':row['ID'],'Gene ID':('ambiguous_source_assignments' if ambiguous_symbol_with_missing_id else ' /// '.join(sorted(ids))),
                     'Gene symbol':' /// '.join(sorted(symbols)),
                     'source_gene_assignment':row['gene_assignment'],
                     'source_crosshyb_type':row['crosshyb_type'],'source_category':row['category']})
    return pd.DataFrame(rows)


def functional_reference():
    rows=[]
    for panel,quads in QUADRANTS.items():
        dose=1 if panel=='left_1uM' else 5
        for label,q in zip(['control','erlotinib','quinacrine','combination'],quads):
            if any(v<0 or v>100 for v in q) or abs(sum(q)-100)>.15:raise ValueError('Invalid printed quadrants')
            rows.append(dict(panel=panel,condition=label,cell_line='A549',exposure_hours=48,
               erlotinib_uM=dose if label in ['erlotinib','combination'] else 0,
               quinacrine_uM=5 if label in ['quinacrine','combination'] else 0,
               PI_positive_Annexin_negative=q[0],double_positive=q[1],double_negative=q[2],Annexin_positive_PI_negative=q[3],
               Annexin_positive_percent=q[1]+q[3],non_double_negative_percent=100-q[2],
               value_level='printed_single_flow_plot_not_reconstructed_group_mean',raw_replicates_available=False,
               matched_RNA_sample_id={'erlotinib':'GSM1382375','combination':'GSM1382382'}.get(label) if dose==1 else None,
               RNA_link_type='nominal_condition_only_not_replicate_pair',control_RNA_time_matched=False))
    return pd.DataFrame(rows)


def recover(source,gene_info,gene_axis,output):
    if output.exists():raise FileExistsError('New output version required')
    if digest(source/'figure2.jpg')!=FIGURE_SHA:raise ValueError('Reviewed figure source changed')
    text,records=read_soft(source/'GSE57422_family.soft.gz');annotation=table_block(text.split('^SAMPLE = ',1)[0],'platform')
    values=[];samples=[];probes=None;metadata={}
    for acc,m,block in records:
        spec=sample_protocol(acc,m)
        if spec['nominal_cell_line']!='A549':continue
        t=table_block(block,'sample')
        if not t.ID_REF.is_unique:raise ValueError('Duplicate probe')
        if probes is None:probes=t.ID_REF.tolist()
        if probes!=t.ID_REF.tolist():raise ValueError('Sample probe order differs')
        v=pd.to_numeric(t.VALUE,errors='raise').to_numpy(float)
        if not np.isfinite(v).all():raise ValueError('Nonfinite source RMA')
        values.append(v);samples.append(spec);metadata[acc]=m
    if len(samples)!=12:raise ValueError('Expected 12 nominal A549 arrays')
    values=np.asarray(values);samples=pd.DataFrame(samples)
    if samples.duplicated(['erlotinib_uM','quinacrine_uM','exposure_hours']).any():raise ValueError('Unexpected repeats')
    if set(annotation.ID)!=set(probes):raise ValueError('Platform coverage mismatch')
    annotation=annotation.set_index('ID',drop=False).loc[probes].reset_index(drop=True)
    parsed=parse_assignments(annotation)
    gi=pd.read_csv(gene_info,sep='\t',dtype=str,keep_default_na=False);gi=gi[gi['#tax_id'].eq('9606')]
    mapping=probe_gene_map(parsed,gi)
    mapping['source_crosshyb_type']=parsed.source_crosshyb_type
    mapping['source_gene_assignment']=parsed.source_gene_assignment
    expression=pd.DataFrame(values.T,columns=samples.sample_id.tolist());expression['ensembl_id']=mapping.ensembl_id
    gene=expression.dropna(subset=['ensembl_id']).groupby('ensembl_id',sort=True).mean().reset_index()
    aligned,mask,ledger=align_gene_axis(gene,pd.read_csv(gene_axis),identifier='ensembl_id',samples=samples.sample_id.tolist(),scale='microarray_log2_RMA')
    for row in ledger[ledger.mapping_status.eq('matched')].itertuples():
        np.testing.assert_equal(aligned[:,int(row.common_gene_index)],gene.iloc[row.source_row,1:].to_numpy(float))
    output.mkdir(parents=True);samples.to_csv(output/'samples.csv',index=False);mapping.to_csv(output/'probe_gene_mapping.csv.gz',index=False)
    ledger.to_csv(output/'common_gene_mapping.csv',index=False);functional_reference().to_csv(output/'functional_reference.csv',index=False)
    np.savez_compressed(output/'source_expression.npz',expression=values,probe_ids=np.array(probes),sample_ids=samples.sample_id.to_numpy(str))
    np.testing.assert_equal(np.load(output/'source_expression.npz')['expression'],values)
    np.savez_compressed(output/'common_axis_expression.npz',expression=aligned,observed_mask=mask,sample_ids=samples.sample_id.to_numpy(str))
    (output/'source_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    manifest=dict(accession='GSE57422',source_arrays=len(records),nominal_A549_arrays=12,source_probe_count=len(probes),
        actual_numeric_source_values=values.size,common_axis_observed=int(mask[0].sum()),common_axis_missing=int((~mask[0]).sum()),
        source_scale='author_RMA_log2_intensity_not_counts',aggregation='mean_log2_intensity_only_for_unique_source_Entrez_to_unique_current_Ensembl',
        genotype_flagged_samples=samples.loc[samples.source_genotype_conflict,'sample_id'].tolist(),
        entire_A549_background_has_paper_genotype_conflict=True,strict_train_only_preprocessing=False,
        observed_quinacrine_48h_RNA=False,matched_RNA_function_condition_references=2,
        actual_paired_RNA_function_replicates=0,initial_state_increment_validated=False,full_research_qualified=False,
        source_hashes={n:digest(source/n) for n in ['GSE57422_family.soft.gz','bioc.xml','figure2.jpg']},
        annotation_sha256=digest(gene_info),axis_sha256=digest(gene_axis),script_sha256=digest(__file__),
        outputs={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','gene-info','gene-axis','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.source,a.gene_info,a.gene_axis,a.output),indent=2))
