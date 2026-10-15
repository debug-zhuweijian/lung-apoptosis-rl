"""Restore public expression packets from source files, without local experiments.

Example:
  python scripts/recover_public_expression_bundle.py --source-root path/to/cloud-readback \
      --output experiments/public-expression-recovery-v1

The output is a new, source-bound namespace. It is not a replacement for frozen
preprocessing, model parameters, experimental roles, or independent evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from lung.data.public_source_recovery import (bind_geo_samples, file_sha256,
    load_expression_packet, read_geo_samples, verify_source_lock, write_expression_packet)


def recover_axis(source: Path) -> pd.DataFrame:
    genes = pd.read_parquet(source / 'tahoe/gene_metadata.parquet')
    with (source / 'tidyi/SS3_Human_A549_counts.csv').open(encoding='utf-8', newline='') as handle:
        raw = next(csv.reader(handle))[1:]
    if len(raw) != len(set(raw)) or not genes.ensembl_id.is_unique or not genes.token_id.is_unique:
        raise ValueError('Source gene/token identities are not unique')
    axis = genes.loc[genes.ensembl_id.isin(raw)].sort_values('token_id').reset_index(drop=True)
    if axis.ensembl_id.tolist() != sorted(axis.ensembl_id):
        raise ValueError('Public token order and project identity order disagree')
    axis.insert(0, 'common_gene_index', np.arange(len(axis)))
    return axis


def binding(sample_id, column, title, line, group, rep, time_h, protocol, controls, **extras):
    return dict(sample_id=sample_id, source_column=column, source_title=title, cell_line=line,
                condition_id=group, replicate_index=rep, measurement_time_h=time_h,
                source_background_id=group.split(':')[0] + ':' + line,
                ordered_protocol_json=json.dumps(protocol, ensure_ascii=False, sort_keys=True),
                control_sample_ids_json=json.dumps(controls), observed_pretreatment_cell=False,
                alias_basis='explicit treatment label and replicate suffix correspondence',
                **extras)


def recover_a375(source: Path, output: Path, axis: pd.DataFrame) -> dict:
    acc = 'GSE306207'
    frame = pd.read_excel(source / acc / 'GSE306207_A375_Seq_All_Samples_TPM.xlsx',
                          sheet_name='A375_Seq_All_Samples_TPM')
    records = read_geo_samples(source / acc / f'{acc}_family.soft.gz')
    controls = ['GSM9194773', 'GSM9194774']
    bindings = []
    for group, label, suffix, start, time_h, protocol in [
        ('control', 'DMSO', 'NC', 9194773, None, [{'kind': 'vehicle', 'name': 'DMSO', 'duration_h': None, 'concentration': None}]),
        ('ibet72', 'IBET726', 'IBET726', 9194775, 72, [{'kind': 'drug_phase', 'drug': 'I-BET726', 'dose': 2, 'unit': 'uM', 'phase_duration_h': 72}]),
        ('withdrawal144', 'IBET726_w3d', 'IBET726W', 9194777, 144, [
            {'kind': 'drug_phase', 'drug': 'I-BET726', 'dose': 2, 'unit': 'uM', 'phase_duration_h': 72},
            {'kind': 'reported_drug_free_phase', 'phase_duration_h': 72, 'washout_procedure': None}]),
        ('sequence144', 'IBET726_Oxa', '726_Oxa', 9194779, 144, [
            {'kind': 'drug_phase', 'drug': 'I-BET726', 'dose': 2, 'unit': 'uM', 'phase_duration_h': 72},
            {'kind': 'drug_phase', 'drug': 'Oxaliplatin', 'dose': 10, 'unit': 'uM', 'phase_duration_h': 72,
             'prior_drug_removal_procedure': None, 'prior_drug_retention': None}])]:
        for rep in [1, 2]:
            bindings.append(binding(f'GSM{start+rep-1}', f'S{rep}_A375_{suffix}', f'A375_{label}_{rep}',
                'A375', acc + ':' + group, rep, time_h, protocol, controls,
                replicate_type='reported_suffix; biological independence not certified by GEO',
                reference_time_match='control sampling clock not explicit in GEO'))
    samples = bind_geo_samples(frame, records, bindings, identifier='gene')
    return write_expression_packet(output / acc, frame.rename(columns={'gene':'ensembl_id'}), axis, samples,
        identifier='ensembl_id', scale='bulk_TPM', metadata={
            'source_id': acc, 'assembly':'GRCh38', 'quantification':'RSEM TPM',
            'source_of_protocol':'GEO Series overall_design and Sample treatment_protocol_ch1',
            'a549_transfer_validated':False, 'paired_apoptosis_endpoint_available':False,
            'sampling_clock_control_h':None})


def recover_hm(source: Path, output: Path, axis: pd.DataFrame) -> dict:
    acc = 'GSE333828'
    raw = pd.read_csv(source / acc / f'{acc}_raw_counts.tsv.gz', sep='\t')
    records = read_geo_samples(source / acc / f'{acc}_family.soft.gz')
    bindings = []
    controls = ['GSM9774923', 'GSM9774924', 'GSM9774925']
    for group, label, suffix, start in [('control','Notreatment','NT',9774923), ('HM97662','HM97662','HM',9774926)]:
        protocol = [{'kind':'untreated' if group == 'control' else 'drug_phase', 'phase_duration_h':144,
                     'drug':None if group == 'control' else 'HM97662', 'dose':None if group=='control' else 1,
                     'unit':None if group=='control' else 'uM', 'medium_change_interval_h':72,
                     'dose_replenishment_details':None}]
        for rep in [1,2,3]:
            bindings.append(binding(f'GSM{start+rep-1}',f'A549_{suffix}-{rep}',f'A549_{label}_rep{rep}',
                'A549',acc+':A549:'+group,rep,144,protocol,controls,
                replicate_type='biological; GEO overall_design and article RNA methods',
                genotype='SMARCA4 deficient/SMARCA2 proficient', reference_time_match='144h concurrent group',
                drug_structure_status='unresolved; no surrogate fingerprint'))
    frame = raw[['gene_id']+[b['source_column'] for b in bindings]]
    samples = bind_geo_samples(frame, records, bindings, identifier='gene_id')
    return write_expression_packet(output / acc, frame.rename(columns={'gene_id':'ensembl_id'}), axis, samples,
        identifier='ensembl_id', scale='bulk_estimated_counts', metadata={
            'source_id':acc, 'raw_file_has_other_backgrounds':True, 'selected_background':'A549',
            'quantification':'Salmon gene-level estimated counts, despite raw_counts filename',
            'rounding_performed':False, 'medium_change_source':'PMC13476540 sec0008',
            'rna_function_same_replicate_pairing':False})


def recover_macaya(source: Path, output: Path, axis: pd.DataFrame) -> dict:
    acc = 'GSE161218'
    records, bindings, frame, gene_ids = {}, [], None, None
    controls = ['GSM4892080','GSM4892081','GSM4892082']
    for index in range(9):
        sample = f'GSM{4892080+index}'
        group = ['CTL','Lest','Tram'][index//3]; rep=index%3+1
        title=f'{group}_{rep}'
        records.update(read_geo_samples(source / 'macaya' / (sample+'.soft')))
        filename=sample+'_'+title+'.counts.txt.gz'
        table=pd.read_csv(source/'macaya'/filename, sep='\t', comment='#')
        if list(table.columns[:6]) != ['Geneid','Chr','Start','End','Strand','Length'] or len(table.columns)!=7:
            raise ValueError('Unexpected featureCounts source schema')
        if gene_ids is None:
            gene_ids=table.Geneid.tolist(); frame=pd.DataFrame({'ensembl_id':gene_ids})
        elif table.Geneid.tolist()!=gene_ids:
            raise ValueError('featureCounts source gene order changed')
        frame[sample]=pd.to_numeric(table.iloc[:,6],errors='raise')
        drug,dose={'CTL':(None,None),'Lest':('Lestaurtinib',0.625),'Tram':('Trametinib',0.5)}[group]
        protocol=[{'kind':'control' if group=='CTL' else 'drug_phase','drug':drug,'dose':dose,
                   'unit':None if dose is None else 'uM','phase_duration_h':16,'vehicle_composition':None}]
        bindings.append(binding(sample,sample,title,'H1792',acc+':'+group,rep,16,protocol,controls,
            replicate_type='reported RNA replicate; not paired to functional experiment number',
            reference_time_match='16h RNA design; functional readout is separately measured at24h',
            treatment_identity_basis='sample title, not the common and/or treatment paragraph'))
    samples=bind_geo_samples(frame,records,bindings,identifier='ensembl_id')
    return write_expression_packet(output/acc,frame,axis,samples,identifier='ensembl_id',scale='bulk_raw_counts',
        metadata={'source_id':acc,'cell_line':'H1792, NOT A549','assembly':'hg38','annotation':'Gencode v28',
                  'quantification':'featureCounts v1.6.0, exon gene counts; not voom values',
                  'function_time_h':24,'rna_time_h':16,'paired_rna_function_samples':False})


def recover_tidyi(source: Path, output: Path, axis: pd.DataFrame) -> dict:
    base=source/'tidyi'
    meta=pd.read_csv(base/'SS3_Human_A549_metadata.csv',index_col=0)
    counts=pd.read_csv(base/'SS3_Human_A549_counts.csv',index_col=0)
    if not meta.index.is_unique or not counts.index.is_unique or set(meta.index)!=set(counts.index):
        raise ValueError('TIDYI cell identities are not one-to-one')
    counts=counts.loc[meta.index]
    values=counts.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values<0) or np.any(values!=np.floor(values)):
        raise ValueError('TIDYI deposited count values are invalid')
    eligible=(meta.Cell_line.eq('A549') & meta.cell_state.isin(['PI-APC-','PI-APC+','PI+APC+']) & (values.sum(1)>0))
    exclusions=pd.DataFrame({'cell_id':meta.index, 'line':meta.Cell_line,'label':meta.cell_state,
        'eligible':eligible,'reason':np.where(~meta.Cell_line.eq('A549'),'missing_or_non_A549_identity',
            np.where(~meta.cell_state.isin(['PI-APC-','PI-APC+','PI+APC+']),'unknown_label',
                np.where(values.sum(1)<=0,'empty_expression','')))}).loc[~eligible]
    exclusions.to_csv(output/'tidyi_exclusions.csv',index=False)
    selected=meta.loc[eligible]
    frame=counts.loc[eligible].T.reset_index().rename(columns={'index':'ensembl_id'})
    samples=pd.DataFrame({'sample_id':selected.index,'source_column':selected.index,'cell_line':'A549',
        'role':'unassigned','source_background_id':'E-MTAB-16354:A549', 'plate':selected.Plate.to_numpy(),
        'well':selected.Well.to_numpy(),'original_sorting_label':selected.cell_state.to_numpy(),
        'statistical_unit':'sorted_cell; not independent biological replicate',
        'paired_to_function_replicate':False, 'measurement_time_h':None,
        'drug_dose_unit':None,'ordered_protocol_json':None,
        'source_fields_json':[json.dumps(json.loads(row.to_json()),sort_keys=True) for _,row in selected.iterrows()]})
    result=write_expression_packet(output/'TIDYI_official',frame,axis,samples,identifier='ensembl_id',
        scale='single_cell_raw_counts',metadata={'source_id':'E-MTAB-16354 official CSV',
            'all_source_rows':len(meta),'source_role':'same-cell RNA and sorting label',
            'natural_population_apoptosis_fraction':False,'second_author_plate_recovered':False})
    discrepancy=pd.DataFrame({'cell_id':meta.index,'source_total_counts':meta.total_counts,
        'export_sum':values.sum(1),'source_detected_genes':meta.n_genes_by_counts,
        'export_detected_genes':(values>0).sum(1)})
    discrepancy.to_csv(output/'tidyi_qc_discrepancies.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--lock',type=Path,default=ROOT/'configs/public_alignment_sources_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); source=args.source_root.resolve(); output=args.output.resolve()
    lock=json.loads(args.lock.read_text(encoding='utf-8'))
    verified=verify_source_lock(source,lock['sources'])
    if output.exists(): raise FileExistsError('Use a new output namespace')
    axis=recover_axis(source)
    output.mkdir(parents=True); axis.to_parquet(output/'gene_axis.parquet',index=False)
    packets={}
    for name,fn in [('GSE306207',recover_a375),('GSE333828',recover_hm),('GSE161218',recover_macaya),('TIDYI_official',recover_tidyi)]:
        packets[name]=fn(source,output,axis)
        print(name,packets[name]['sample_count'],packets[name]['observed_axis_genes'],flush=True)
    checks={}
    for name,bg in [('GSE306207','A375'),('GSE333828','A549'),('GSE161218','H1792'),('TIDYI_official','A549')]:
        pkt=load_expression_packet(output/name,expected_scale=packets[name]['measurement_domain'],
            expected_gene_ids=axis.ensembl_id.tolist(),expected_background=bg)
        checks[name]={'background':bg,'real_consumer_readback':True,'sample_count':len(pkt['samples'])}
    result={'schema':'public-source-recovery-run-v1','source_lock_sha256':file_sha256(args.lock),
        'implementation_sha256':file_sha256(Path(__file__)), 'source_files_verified':verified,
        'gene_axis_status':'source-reconstructed identity order, NOT byte-certified frozen preprocessing',
        'gene_axis_sha256':file_sha256(output/'gene_axis.parquet'),'packets':packets,'consumer_checks':checks,
        'frozen_roles_changed':False,'model_training_or_effect_verification':False,'whole_project_qualified':False}
    (output/'recovery_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
