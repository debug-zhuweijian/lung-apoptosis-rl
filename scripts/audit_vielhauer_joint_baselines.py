"""Execute fixed simple joint-choice baselines on recovered published means.

Independent new retrospective artifact. No old split or full-P qualification is
bypassed. Source authentication rereads historical outcomes; not a blind trial.
"""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.build_vielhauer_duration_challenge import load_challenge
from scripts.recover_public_expression import digest
from lung.data.joint_selection_evaluation import metric_result,maximize


def role(row):
    dose=row['dose_nM']
    if dose in [.05,1000.]:return 'fit'
    if dose==10:return 'selection'
    if dose in [.5,40.]:return 'final_test'
    raise ValueError('Unexpected source dose')


def freeze_predictions(metadata,fit_labels):
    ids=[r['condition_id'] for r in metadata]
    if len(ids)!=len(set(ids)) or len(ids)!=20:raise ValueError('Complete unique 20-candidate source domain required')
    expected={(d,q,t) for d in ['bocnho','cbi_indole2'] for q in [.05,.5,10.,40.,1000.] for t in [6,24]}
    if {(r['drug_id'],r['dose_nM'],r['exposure_duration_h']) for r in metadata}!=expected:
        raise ValueError('Complete unique physical candidate grid required')
    fit=[r for r in metadata if role(r)=='fit']
    if len(fit)!=8 or set(fit_labels)!={r['condition_id'] for r in fit}:raise ValueError('Exactly fit-role labels required')
    if not all(math.isfinite(v) and 0<=v<=1 for v in fit_labels.values()):raise ValueError('Invalid observed fractions')
    means={};anchors={}
    for drug in ['bocnho','cbi_indole2']:
        for duration in [6,24]:
            pair=sorted([r for r in fit if r['drug_id']==drug and r['exposure_duration_h']==duration],key=lambda r:r['dose_nM'])
            if len(pair)!=2 or [r['dose_nM'] for r in pair]!=[.05,1000.]:raise ValueError('Source fit grid incomplete')
            y=[fit_labels[r['condition_id']] for r in pair];means[(drug,duration)]=sum(y)/2;anchors[(drug,duration)]=y
    constant=sum(fit_labels.values())/8;pred={name:{} for name in ['fit_mean','drug_protocol_fit_mean','drug_protocol_logdose_interpolation']}
    for r in metadata:
        cid=r['condition_id'];key=(r['drug_id'],r['exposure_duration_h']);lo,hi=anchors[key]
        fraction=math.log(r['dose_nM']/.05)/math.log(1000/.05)
        pred['fit_mean'][cid]=constant;pred['drug_protocol_fit_mean'][cid]=means[key]
        pred['drug_protocol_logdose_interpolation'][cid]=lo+fraction*(hi-lo)
    return pred


def evaluate_role(metadata,labels,predictions,wanted):
    rows=[r for r in metadata if role(r)==wanted];ids=[r['condition_id'] for r in rows]
    if len(ids)!={'selection':4,'final_test':8}[wanted]:raise ValueError('Complete role domain required')
    first_drug=sorted({r['drug_id'] for r in rows})[0];low_dose=min(r['dose_nM'] for r in rows)
    constraints={'joint_all':lambda r:True,'fixed_drug':lambda r:r['drug_id']==first_drug,
      'fixed_dose':lambda r:r['dose_nM']==low_dose,'fixed_exposure':lambda r:r['exposure_duration_h']==24,
      'fixed_all_three':lambda r:r['drug_id']==first_drug and r['dose_nM']==low_dose and r['exposure_duration_h']==24}
    result={};truth={cid:labels[cid] for cid in ids}
    for name,all_scores in predictions.items():
        scores={cid:all_scores[cid] for cid in ids}
        result[name]={policy:metric_result(scores,truth,maximize(scores,[r['condition_id'] for r in rows if keep(r)])) for policy,keep in constraints.items()}
    constant=predictions['fit_mean'];result['uniform_random_expected']=metric_result({cid:constant[cid] for cid in ids},truth,{cid:1/len(ids) for cid in ids})
    return result


def run(challenge,output):
    if output.exists():raise FileExistsError('New output directory required')
    data=load_challenge(challenge);rows=data['active_conditions'];keys=['condition_id','drug_id','dose_nM','exposure_duration_h']
    metadata=[{k:r[k] for k in keys} for r in rows]
    fit_labels={r['condition_id']:r['measured_endpoint']['mean_fraction'] for r in rows if role(r)=='fit'}
    predictions=freeze_predictions(metadata,fit_labels)
    output.mkdir(parents=True);packet=dict(protocol='fixed_three_simple_baselines_no_selection_or_final_tuning',
        metadata=metadata,fit_label_ids=sorted(fit_labels),predictions=predictions,
        source_manifest_sha256=digest(challenge/'manifest.json'),script_sha256=digest(__file__))
    frozen=output/'frozen_predictions.json';frozen.write_text(json.dumps(packet,indent=2)+'\n');frozen_hash=digest(frozen)
    # No held-out outcome is provided to the prediction function above.
    labels={r['condition_id']:dict(label_fraction=r['measured_endpoint']['mean_fraction'],
        reading_lower_fraction=r['measured_endpoint']['pixel_lower_fraction'],reading_upper_fraction=r['measured_endpoint']['pixel_upper_fraction']) for r in rows}
    evaluation={r:evaluate_role(metadata,labels,predictions,r) for r in ['selection','final_test']}
    if digest(frozen)!=frozen_hash:raise ValueError('Predictions changed after evaluation')
    (output/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    report=dict(source='Vielhauer2013 cloud raster v2',new_experiment_not_overwriting_frozen_roles=True,
        fit_conditions=8,selection_conditions=4,final_conditions=8,policy_count_per_predictor=5,
        prediction_baselines=3,source_label_level='published_group_mean',source_RNA_used=False,
        predictions_use_fit_labels_only=True,selection_or_final_used_for_tuning=False,
        originally_prospectively_blinded=False,independent_biological_replication=False,
        historical_source_authentication_reads_all_labels=True,output_scores_frozen_before_metric_execution=True,
        reading_intervals_are_biological_CI=False,formal_P_G_Q_or_RL_evaluated=False,
        initial_state_increment_or_unseen_drug_evaluated=False,full_research_qualification=False,
        frozen_predictions_sha256=frozen_hash,source_manifest_sha256=digest(challenge/'manifest.json'),
        actual_metric_code_sha256=digest(ROOT/'src/lung/data/joint_selection_evaluation.py'),
        script_sha256=digest(__file__),outputs={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return {name:{m:dict(selected=v['joint_all']['expected_measured_selected_fraction'],regret=v['joint_all']['measured_candidate_regret'],MSE=v['joint_all']['mean_squared_error']) for m,v in values.items() if m!='uniform_random_expected'} for name,values in evaluation.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--challenge',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.challenge,a.output),indent=2))
