"""Reproduce manually reviewed S2F/S2M published bars; no biological inference."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/20261011-data-alignment-round16/state_function_new_sources'
OUTPUT = ROOT / '08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/function'
BACKGROUNDS = ['HCC827', 'HCC2935', 'NCI-H1975', 'PC9']
# Native image coordinates, reviewed against full image and enlarged crop.
# Tops are outer published bar boundaries, not SD caps or replicate points.
PANELS = {
    'S2F': {'image': 'Martin2022_image3.jpg', 'ticks': [(1310, .1), (1256, 1), (1202, 10), (1148, 100)],
            'xs': [[143,150,157,164,171,177],[191,198,205,212,219,225],[335,342,349,356,363,370],[383,390,397,404,411,418]],
            'tops': [[1256,1203,1254,1199,1236,1188],[1256,1220,1251,1221,1226,1205],[1256,1241,1257,1242,1227,1213],[1256,1163,1263,1165,1232,1154]],
            'candidates': [('DMSO',[]),('Osi',[('osimertinib',160)]),('Venetoclax',[('venetoclax',300)]),('Osi/Venetoclax',[('osimertinib',160),('venetoclax',300)]),('AZD4320',[('AZD4320',300)]),('Osi/AZD4320',[('osimertinib',160),('AZD4320',300)])]},
    'S2M': {'image': 'Martin2022_image4.jpg', 'ticks': [(955,.1),(900,1),(845,10),(791,100)],
            'xs': [[169,179,188,198],[217,227,237,246],[362,372,382,391],[411,420,430,440]],
            'tops': [[901,835,884,811],[901,867,894,856],[901,879,901,867],[901,817,887,815]],
            'candidates': [('DMSO',[]),('Osi',[('osimertinib',160)]),('K-975',[('K-975',100)]),('Osi/K-975',[('osimertinib',160),('K-975',100)])]},
}

def calibration(ticks):
    """OLS log10(value)=intercept+slope*y using all visible axis ticks."""
    ys = [float(t[0]) for t in ticks]
    logs = [math.log10(t[1]) for t in ticks]
    ybar, lbar = sum(ys)/len(ys), sum(logs)/len(logs)
    slope = sum((y-ybar)*(l-lbar) for y,l in zip(ys,logs))/sum((y-ybar)**2 for y in ys)
    return lbar-slope*ybar, slope

def value_at(y, ticks):
    intercept, slope = calibration(ticks)
    return 10**(intercept+slope*y)

def build_rows():
    rows=[]
    for panel,spec in PANELS.items():
        source_hash=hashlib.sha256((SOURCE/spec['image']).read_bytes()).hexdigest()
        for background,xs,tops in zip(BACKGROUNDS,spec['xs'],spec['tops']):
            block=f'Martin2022_{panel}_{background}_parental_72h'
            for x,y,(label,components) in zip(xs,tops,spec['candidates']):
                # 2 px top reading + 1 px common axis location bound, conservatively added.
                row={'observation_id':f'{block}_{label.replace("/","+")}', 'measurement_domain_id':f'Martin2022_{panel}',
                     'decision_block_id':block,'cell_background':background,'state':'parental',
                     'candidate_label':label,'candidate_type':'vehicle' if not components else 'single' if len(components)==1 else 'combination',
                     'components_json':json.dumps([{'drug':d,'concentration':c,'unit':'nM'} for d,c in components]),
                     'exposure_h':72,'observation_window_h':72,'panel':panel,'source_image':spec['image'],'source_image_sha256':source_hash,
                     'bar_center_x_px':x,'bar_top_y_px':y,'bar_top_reading_bound_px':2,'axis_location_bound_px':1,
                     'published_mean_digitized':value_at(y,spec['ticks']), 'pixel_reading_lower':value_at(y+3,spec['ticks']),
                     'pixel_reading_upper':value_at(y-3,spec['ticks']), 'uncertainty_type':'finite_pixel_reading_bound_not_CI_not_SD',
                     'endpoint':'relative_caspase_3_7_signal_AUC_72h', 'unit':'published_relative_activity_AUC_scale_not_fully_specified',
                     'well_replicates_declared':3,'biological_replicates':'unknown','replicate_observations_available':False,
                     'rna_bridge_status':'same_source_parallel_background_bridge_explicit_assumption_unvalidated',
                     'annexin_fraction_eligible':False,'a549_eligible':False}
                rows.append(row)
    return rows

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows=build_rows()
    with (OUTPUT/'published_means.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    domains=[]
    for panel,spec in PANELS.items():
        a,b=calibration(spec['ticks'])
        domains.append({'measurement_domain_id':f'Martin2022_{panel}','axis':'log10','ticks_y_pixel_value':spec['ticks'],
                        'formula':'value = 10 ** (intercept + slope * native_image_y)', 'intercept':a,'slope':b,
                        'plot_label':'Relative caspase-3/7 signal' if panel=='S2F' else 'Relative caspase-3/7 activity',
                        'axis_tick_residual_log10_max':max(abs(a+b*y-math.log10(v)) for y,v in spec['ticks'])})
    blocks=[{'decision_block_id':block,'measurement_domain_id':members[0]['measurement_domain_id'],
             'cell_background':members[0]['cell_background'],'complete_common_candidate_set':True,
             'candidate_observation_ids':[r['observation_id'] for r in members],
             'core_single_observation_ids':[r['observation_id'] for r in members if r['candidate_type']=='single'],
             'vehicle_observation_ids':[r['observation_id'] for r in members if r['candidate_type']=='vehicle'],
             'combination_auxiliary_observation_ids':[r['observation_id'] for r in members if r['candidate_type']=='combination']}
            for block in dict.fromkeys(r['decision_block_id'] for r in rows)
            for members in [[r for r in rows if r['decision_block_id']==block]]]
    manifest={'version':'Martin2022-function-v1','source':'PMC10035388; supplemental Figure S2F/S2M',
              'source_evidence':['PMC10035388.xml sec2-4 Incucyte Growth Assay','sec2-5 Apoptosis Assay','sec2-9 RNA Sequencing','Martin2022_supplement.md Figure S2 legend'],
              'source_hashes':{n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ['PMC10035388.xml','Martin2022_supplement.md','Martin2022_image3.jpg','Martin2022_image4.jpg']},
              'counts':{'published_bar_observation_identities':len(rows),'raw_independent_biological_function_replicates':0,
                        'measurement_domains':2,'decision_blocks':len(blocks),'single_active_bars':20,'vehicle_bars':8,'combination_bars':12},
              'domains':domains,'blocks':blocks,
              'normalization':'distinct green spots / confluence; normalized to average DMSO value at 72 hours (Methods sec2-5)',
              'sampling_interval_h':4,'auc_algorithm':'unknown','auc_unit':'not fully specified','auc_dmso_baseline_one_assumed':False,
              'bridge':'same source parallel RNA: parental acute RNA uses 500 nM osimertinib for 24h; function uses 160 nM for 72h. No sample matched RNA/function pair established; no Y duplication across RNA replicates.',
              's2n_quarantine':{'state':'established DTP','backgrounds':['PC9','HCC4006','NCI-H1975'],'function_followup_h':72,
                               'osimertinib_nM':500,'K975_nM':100,'dtp_induction_duration':'experiment dependent 10–21d; S2N exact duration unverified','numeric_labels_extracted':False},
              'limitations':['No Annexin fraction, absolute apoptosis fraction, individual-cell fate, A549 supervision, or validated RNA/function bridge.',
                             'Same label in different panels retained separately; panel provenance is not evidence of independent biological experiments.',
                             'SD bars not digitized into repeats; reading bounds omit biological/normalization/integration uncertainties.']}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest['counts']))

if __name__=='__main__':
    main()
