"""Recover real STACK assay counts, audit LF arithmetic, and bind a nominal RNA condition."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import digest,read_soft
from lung.data.external_gene_axis import align_gene_axis
PINS={'single_drug.xlsx':'2f51796806586e48501753b6aba6d152a42a49e78a0028d3c9b0ed1575a36c71','pim_combination.xlsx':'ad216298a2d8c2267482f9402af85a4c609c40ca40a1be360fa1f75f9d54e925','paper.xml':'7eb943bd037375f8213dac9c53d278caa4f9f496ae6fe31acd4270d4aacc8c36'}
EXCLUDED={'Sunitinib Malate','Mitoxantrone HCl','PHA-665752'}


def lf_from_counts(green,red,overlap):
    green=np.asarray(green,float);red=np.asarray(red,float);overlap=np.asarray(overlap,float)
    if green.shape!=red.shape or green.shape!=overlap.shape:raise ValueError('Count dimensions differ')
    peak=np.fmax.accumulate(green,axis=0);live=red-overlap
    with np.errstate(divide='ignore',invalid='ignore'):calc=peak/(peak+live)
    unknown_history=np.maximum.accumulate(~np.isfinite(green),axis=0)
    valid=np.isfinite(calc)&(green>=0)&(red>=0)&(overlap>=0)&(live>=0)&(peak>=0)&(calc>=0)&(calc<=1)&~unknown_history
    return calc,valid,peak,live


def read_book(path,arm):
    book=openpyxl.load_workbook(path,read_only=True,data_only=True);records=[];summaries=[]
    for rep in [1,2,3]:
        values={};raw={};names=None;times=None
        for kind in ['Lethal Fraction','SG+ Counts','mKate2+ Counts','Overlap Counts']:
            rows=list(book[f'{kind} rep {rep}'].values);header=list(rows[0][1:]);clock=np.array([r[0] for r in rows[1:]],float)
            if names is not None and (header!=names or not np.array_equal(clock,times)):raise ValueError('Assay field identity/order mismatch')
            names=header;times=clock;raw[kind]=pd.DataFrame([r[1:] for r in rows[1:]])
            values[kind]=raw[kind].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        if len(names)!=384 or not np.array_equal(times,np.arange(0,145,4)):raise ValueError('Unexpected plate/time coverage')
        drugs=[n for n in names if n and n!='Empty']
        if len(drugs)!=261 or len(set(drugs))!=261:raise ValueError('Unexpected source compound coverage')
        calc,valid,peak,live=lf_from_counts(values['SG+ Counts'],values['mKate2+ Counts'],values['Overlap Counts'])
        lf=values['Lethal Fraction'];comparable=np.isfinite(calc)&np.isfinite(lf)
        if not np.allclose(calc[comparable],lf[comparable],atol=1e-10,rtol=0):raise ValueError('Source LF arithmetic mismatch')
        for i,t in enumerate(times):
            for j,n in enumerate(names):
                role='vehicle_reference' if n=='Empty' else ('unannotated_well' if not n else 'library_compound')
                source_ok=np.isfinite(lf[i,j]) and 0<=lf[i,j]<=1
                records.append(dict(arm=arm,replicate=rep,source_column_1based=j+2,source_row_1based=i+2,
                                    source_compound_name=n,role=role,time_h=t,library_dose_uM=5 if role=='library_compound' else None,
                                    base_pimasertib_dose_uM=5 if arm=='pim_plus_library' else 0,
                                    known_pimasertib_component_uM=(5 if arm=='pim_plus_library' else 0)+(5 if n=='Pimasertib (AS-703026)' else 0),
                                    source_LF_raw=raw['Lethal Fraction'].iloc[i,j],lethal_fraction=lf[i,j],recomputed_LF=calc[i,j],
                                    SG_raw=raw['SG+ Counts'].iloc[i,j],mKate2_raw=raw['mKate2+ Counts'].iloc[i,j],overlap_raw=raw['Overlap Counts'].iloc[i,j],
                                    running_max_green=peak[i,j],adjusted_live=live[i,j],valid_measured_fraction=bool(valid[i,j] and source_ok),
                                    within_paper_120h_window=bool(t<=120),author_autofluorescence_exclusion=n in EXCLUDED,
                                    self_combination=arm=='pim_plus_library' and n=='Pimasertib (AS-703026)',
                                    endpoint='SYTOX_STACK_lethal_fraction_not_Annexin',treatment_design='simultaneous_coadministration' if arm=='pim_plus_library' else 'single_agent',
                                    RNA_pair=False,formal_training_role=False))
        summaries.append(dict(arm=arm,replicate=rep,source_numeric_LF=int(np.isfinite(lf).sum()),source_nonfinite_LF=int((~np.isfinite(lf)).sum()),
                              source_LF_above_one=int((lf>1).sum()),negative_adjusted_live=int((live<0).sum()),
                              max_recomputation_difference=float(np.max(np.abs(calc[comparable]-lf[comparable])))))
    return pd.DataFrame(records),summaries


def ranking_repeatability(frame):
    subset=frame[(frame.arm=='single_agent')&frame.time_h.eq(24)&frame.role.eq('library_compound')&~frame.author_autofluorescence_exclusion]
    if len(subset)!=774 or not subset.valid_measured_fraction.all():raise ValueError('24h eligible source coverage/quality failed')
    table=subset.pivot(index='source_compound_name',columns='replicate',values='lethal_fraction')
    folds=[]
    for held in [1,2,3]:
        training=table.drop(columns=held).mean(axis=1);chosen=training.sort_values(ascending=False,kind='stable').index[0];test=table[held]
        folds.append(dict(held_experiment=held,training_experiments=[i for i in [1,2,3] if i!=held],selected_compound=chosen,
                          held_selected_LF=float(test[chosen]),held_best_observed_LF=float(test.max()),held_regret=float(test.max()-test[chosen]),
                          held_uniform_random_expected_LF=float(test.mean())))
    return dict(folds=folds,source_experiments=3,eligible_compounds=258,
                meaning='repeatability of ranking from measured outcomes; not unseen-drug prediction, RNA/G/RL performance or clinical recommendation')


def recover_rna(source,axis,output):
    f=pd.read_csv(source/'fpkm_A549.csv.gz').rename(columns={'geneID':'ensembl_id'})
    if not f.ensembl_id.is_unique:raise ValueError('Duplicate source RNA gene')
    _,blocks=read_soft(source/'GSE140172.soft.gz');samples=[]
    for acc,m,_ in blocks:
        c=dict(s.split(': ',1) for s in m['characteristics_ch1']);name=m['title'][0]
        if c.get('cell line')!='A549':continue
        if name not in f or c['pimasertib dose'] not in ['N/A','500 nM','5 microM']:raise ValueError('RNA identity/dose not bound')
        dose={'N/A':0.,'500 nM':.5,'5 microM':5.}[c['pimasertib dose']]
        samples.append(dict(sample_id=acc,source_column=name,dose_uM=dose,collection_h=96,drug_name='pimasertib' if dose else 'DMSO',
                            reporter_A549_N_supported_by_paper=True,vehicle_split_1_to_8_at_24h=dose==0,
                            collection='media removal and HBSS wash before adherent cell harvest',survivor_selection_concern=True,
                            same_cell_before_after_pair=False,paired_functional_replicate=False))
    sm=pd.DataFrame(samples)
    if len(sm)!=6 or sm.source_column.duplicated().any():raise ValueError('RNA source coverage mismatch')
    cols=sm.source_column.tolist();x,mask,ledger=align_gene_axis(f,axis,identifier='ensembl_id',samples=cols,scale='bulk_FPKM')
    values=f[cols].to_numpy(float)
    for r in ledger[ledger.mapping_status.eq('matched')].itertuples():
        if not np.array_equal(x[:,int(r.common_gene_index)],values[r.source_row]):raise ValueError('FPKM mapping changed values')
    sm.to_csv(output/'RNA_samples.csv',index=False);ledger.to_csv(output/'RNA_gene_mapping.csv',index=False)
    np.savez_compressed(output/'RNA_common_axis_FPKM.npz',expression=x,observed_mask=mask,gene_ids=axis.ensembl_id.to_numpy(str),sample_ids=sm.sample_id.to_numpy(str))
    np.savez_compressed(output/'RNA_source_FPKM.npz',expression=values.T,gene_ids=f.ensembl_id.to_numpy(str),sample_ids=sm.sample_id.to_numpy(str))
    return dict(source_genes=len(f),actual_samples=6,source_values=int(values.size),common_observed_genes=int(mask[0].sum()),common_unresolved_genes=int((~mask[0]).sum()),
                scale='bulk_FPKM',cross_source_scale_calibrated=False,high_dose_sample_ids=sm.loc[sm.dose_uM.eq(5),'sample_id'].tolist())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['source','gene-axis','output']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output required')
    for n,h in PINS.items():
        if digest(a.source/n)!=h:raise ValueError('Reviewed assay/paper version changed')
    frames=[];audit=[]
    for filename,arm in [('single_drug.xlsx','single_agent'),('pim_combination.xlsx','pim_plus_library')]:
        f,r=read_book(a.source/filename,arm);frames.append(f);audit.extend(r)
    frame=pd.concat(frames,ignore_index=True);ranking=ranking_repeatability(frame)
    a.output.mkdir(parents=True);frame.to_csv(a.output/'all_assay_observations.csv.gz',index=False)
    subset=frame[(frame.arm=='single_agent')&frame.time_h.eq(24)&frame.role.eq('library_compound')];subset.to_csv(a.output/'single_agent_24h.csv',index=False)
    rna=recover_rna(a.source,pd.read_csv(a.gene_axis),a.output)
    bridge=frame[(frame.arm=='single_agent')&frame.time_h.eq(96)&frame.source_compound_name.eq('Pimasertib (AS-703026)')]
    if len(bridge)!=3 or not bridge.valid_measured_fraction.all():raise ValueError('Nominal RNA-linked functional condition invalid')
    bridge.to_csv(a.output/'pimasertib_5uM_96h_function.csv',index=False)
    report=dict(source_dataset='10.17632/9bccjb75wp.1',RNA_dataset='GSE140172',observations=len(frame),audit=audit,ranking_repeatability=ranking,RNA=rna,
                nominal_condition_bridge=dict(drug='pimasertib',dose_uM=5,time_h=96,real_function_values=bridge.lethal_fraction.tolist(),
                                              function_experiments=3,RNA_libraries=2,replicate_pairs_verified=False,
                                              culture_geometry_and_control_passaging_matched=False,survivor_RNA_bias=True,
                                              statistical_calibration_identified=False),
                endpoint_mechanism='Q-VD-OPh rescue supports apoptosis for tested MEKi conditions in paper; not blanket library specificity or Annexin calibration',
                source_time_extent_h=144,paper_screen_duration_h=120,all_source_errors_preserved=True,clipping_performed=False,
                complete_apoptosis_chain_qualified=False,actual_sequence_data=False,formal_project_training=False,
                source_hashes={n:digest(a.source/n) for n in [*PINS,'fpkm_A549.csv.gz','GSE140172.soft.gz','file_list.json']},gene_axis_sha256=digest(a.gene_axis),
                script_sha256=digest(__file__),outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
