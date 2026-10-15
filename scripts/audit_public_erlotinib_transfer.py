"""Nominal matched-condition cross-source check, not a fitted transport model.

NCI has time-matched vehicle arrays. GSE57422 has only a 0h reference. Their
relative effects are deliberately flagged as different estimands; no p-values
across genes or artificial biological replicates are produced.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.recover_public_expression import digest


def compare_vectors(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.ndim!=1 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite observed vectors required')
    if len(a)<2:raise ValueError('At least two measured genes required')
    rho=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
    return dict(genes=len(a),spearman=rho,sign_agreement=float(np.mean(np.sign(a)==np.sign(b))),
                A_zero_effect_MSE=float(np.mean(a*a)),B_zero_effect_MSE=float(np.mean(b*b)),
                cross_source_copy_MSE=float(np.mean((a-b)**2)),independent_sample_count=None,
                gene_level_pvalue_computed=False)


def load_source(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for name in ['samples.csv','common_axis_expression.npz']:
        if digest(folder/name)!=manifest['outputs'][name]:raise ValueError('Input bytes changed')
    samples=pd.read_csv(folder/'samples.csv');arrays=np.load(folder/'common_axis_expression.npz')
    if not np.array_equal(arrays['sample_ids'],samples.sample_id.to_numpy(str)):raise ValueError('Sample order mismatch')
    x=arrays['expression'];mask=arrays['observed_mask']
    if x.shape!=mask.shape or len(x)!=len(samples) or mask.dtype!=bool:raise ValueError('Observation mask mismatch')
    if not np.isnan(x[~mask]).all() or not np.isfinite(x[mask]).all():raise ValueError('Missing values or measured values invalid')
    return manifest,samples,x,mask

def one_index(mask):
    indices=np.flatnonzero(mask)
    if len(indices)!=1:raise ValueError('Exactly one recorded array per condition required')
    return int(indices[0])


def run(nci,other,output):
    if output.exists():raise FileExistsError('New diagnostic output required')
    ma,sa,xa,oa=load_source(nci);mb,sb,xb,ob=load_source(other)
    if ma['gene_axis_sha256']!=mb['axis_sha256']:raise ValueError('Ordered common gene axes differ')
    if ma['source_scale']!='microarray_log2_RMA' or mb['source_scale']!='author_RMA_log2_intensity_not_counts':
        raise ValueError('This check is defined only for the two source-native log2 RMA scales')
    observed=oa.all(axis=0)&ob.all(axis=0);a=xa[:,observed];b=xb[:,observed];pairs=[];arrays={}
    for hour in [6,24]:
        i=one_index((sa.drug_name=='erlotinib')&(sa.dose_uM==1)&(sa.exposure_h==hour))
        c=one_index(sa.sample_id==sa.iloc[i].control_sample_id)
        j=one_index((sb.erlotinib_uM==1)&(sb.quinacrine_uM==0)&(sb.exposure_hours==hour))
        d=one_index(sb.is_baseline)
        if sa.iloc[c].exposure_h!=hour or sb.iloc[d].exposure_hours!=0:raise ValueError('Source reference clocks changed')
        da,db=a[i]-a[c],b[j]-b[d]
        pairs.append(dict(time_hours=hour,A_treatment=sa.iloc[i].sample_id,A_reference=sa.iloc[c].sample_id,
                          B_treatment=sb.iloc[j].sample_id,B_reference=sb.iloc[d].sample_id,
                          A_reference_hours=hour,B_reference_hours=0,
                          expression_level_spearman=compare_vectors(a[i],b[j])['spearman'],
                          effect_comparison=compare_vectors(da,db),reference_estimands_equal=False))
        arrays[hour]=(a[i],b[j],a[c],b[d])
    temporal_a=arrays[24][0]-arrays[6][0];temporal_b=arrays[24][1]-arrays[6][1]
    # Both are changes in treated cultures. Neither is an isolated causal drug effect.
    temporal=compare_vectors(temporal_a,temporal_b)
    temporal['definition']='24h_minus_6h_treated_RNA; includes untreated growth/time drift in both sources'
    temporal['unobserved_control_imputed']=False
    output.mkdir(parents=True)
    np.savez_compressed(output/'observed_comparison_vectors.npz',common_gene_indices=np.flatnonzero(observed),
                        A_effect_6h=arrays[6][0]-arrays[6][2],B_effect_6h=arrays[6][1]-arrays[6][3],
                        A_effect_24h=arrays[24][0]-arrays[24][2],B_effect_24h=arrays[24][1]-arrays[24][3],
                        A_treated_time_increment=temporal_a,B_treated_time_increment=temporal_b)
    report=dict(A_source='GSE116436 NCI-TPW',B_source='GSE57422',nominal_drug='erlotinib',dose_uM=1,
                shared_observed_genes=int(observed.sum()),pairs=pairs,treated_temporal_increment=temporal,
                protocol='fixed all-common-observed gene panel; no variable-gene selection, alignment fitting or tuning',
                source_background_conflict=mb['entire_A549_background_has_paper_genotype_conflict'],
                experimental_material_and_culture_equivalence_verified=False,
                gene_level_independent_replication=False,unseen_drug_generalization_test=False,
                calibrated_endpoint_test=False,complete_cross_source_qualification=False,
                raw_array_normalization_refitted=False,formal_training=False,
                source_manifest_hashes={'NCI':digest(nci/'manifest.json'),'Dermawan':digest(other/'manifest.json')},
                ordered_gene_axis_hash=ma['gene_axis_sha256'],script_sha256=digest(__file__),
                outputs={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['nci','other','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.nci,a.other,a.output),indent=2,allow_nan=False))
