"""Recover published A549 ordered-treatment Annexin summaries (Huang 2018).

Fig. 4 annotations were visually transcribed, not digitized from bar heights.
This is an external protocol reference, without RNA or a unique OGA structure.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recover_public_expression import digest

FIGURE_SHA = '1d725aa3fc6d00eb7ee61f0416db53e832525f4e2e370a40cdc38a778a26ffb6'
# early mean/SD, late mean/SD; percent, published rounded summaries.
ANNOTATIONS = {
    'control': (3.7, .1, 5.4, .0),
    'OGA24': (10.8, .2, 5.5, .2),
    'DDP24': (12.8, .2, 18.2, .1),
    'DDP_OGA24': (36.8, .2, 25.9, .4),
    'DDP12_then_OGA12': (19.4, .2, 33.0, .4),
    'OGA12_then_DDP12': (26.9, .6, 14.2, .3),
}


def references():
    rows=[]
    for key,(early,early_sd,late,late_sd) in ANNOTATIONS.items():
        for endpoint,value,sd in [('Annexin_positive_PI_negative',early,early_sd),
                                  ('Annexin_positive_PI_positive',late,late_sd),
                                  ('Annexin_positive_total',early+late,None)]:
            rows.append(dict(protocol_id=key,endpoint=endpoint,value=round(value,1),sd=sd,unit='percent',
                             value_origin='sum_of_published_quadrant_means' if sd is None else 'published_figure_annotation_mean',
                             source_reported_independent_replications=3,raw_replicates_available=False,
                             cells_counted_per_sample=10000,RNA_available=False,whole_research_qualified=False))
    protocols={}
    for key in ANNOTATIONS:
        if '_then_' in key:
            drugs=['cisplatin','OGA_mixture'] if key.startswith('DDP') else ['OGA_mixture','cisplatin']
            intervals=[(0,12),(12,24)]
        elif key=='control': drugs=[];intervals=[]
        elif key=='DDP_OGA24':drugs=['cisplatin','OGA_mixture'];intervals=[(0,24),(0,24)]
        else:drugs=['cisplatin' if key=='DDP24' else 'OGA_mixture'];intervals=[(0,24)]
        protocols[key]=dict(cell_background='A549_BCRC60074',medium='F12K_10percent_FBS',horizon_h=24,
                            control='untreated_24h',stage_medium_replacement=('_then_' in key),
                            stage_extra_wash_count=None,
                            events=[dict(drug=d,start_h=a,end_h=b,dose=2 if d=='cisplatin' else 100,
                                         dose_unit='ug/ml',unique_structure_verified=False if d=='OGA_mixture' else None)
                                    for d,(a,b) in zip(drugs,intervals)])
    return pd.DataFrame(rows),protocols


def contrasts(table):
    wide=table.pivot(index='protocol_id',columns='endpoint',values='value')
    order=(wide.loc['DDP12_then_OGA12']-wide.loc['OGA12_then_DDP12']).to_dict()
    return dict(same_exposure_order_difference_percentage_points=order,
                total_Annexin_best_observed_protocol=wide.Annexin_positive_total.idxmax(),
                statistical_significance_of_derived_total=None,
                scope='Same-dose reverse orders can be compared; simultaneous 24h doubles each drug exposure relative to 12h+12h',
                model_prediction_evaluated=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper',type=Path,required=True);ap.add_argument('--figure',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('New output directory required')
    if digest(args.figure)!=FIGURE_SHA:raise ValueError('Figure bytes changed: visual re-review required')
    paper=' '.join(ET.parse(args.paper).getroot().itertext())
    for marker in ['10.3390/ijms19061769','cultivated solution was removed','three independent replications']:
        if marker not in paper:raise ValueError('Source protocol text changed: re-review required')
    table,protocols=references();result=contrasts(table)
    args.output.mkdir(parents=True)
    table.to_csv(args.output/'endpoint_summaries.csv',index=False)
    (args.output/'protocols.json').write_text(json.dumps(protocols,indent=2))
    (args.output/'contrasts.json').write_text(json.dumps(result,indent=2))
    manifest=dict(paper_doi='10.3390/ijms19061769',paper_sha256=digest(args.paper),figure_sha256=digest(args.figure),
                  script_sha256=digest(__file__),transcription='Root visual reading of Figure 4 numerical annotations',
                  number_of_treatment_protocols=6,measured_quadrant_summaries=12,derived_total_means=6,
                  OGA='Source-defined enzymatic pectin mixture <1 kDa, no unique molecule or molar concentration assigned',
                  RNA_available=False,source_replications_reconstructed=False,
                  total_sd_missing_reason='Early/late covariance not published; SDs cannot be added',
                  external_sequence_reference=True,model_sequence_validated=False,full_chain_qualified=False)
    manifest['outputs']={f.name:digest(f) for f in args.output.iterdir()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
