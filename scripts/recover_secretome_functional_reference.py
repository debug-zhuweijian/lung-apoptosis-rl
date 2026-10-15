"""Recover source A549 Annexin/confluence observations, without RNA mispairing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import openpyxl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recover_public_expression import digest
LABELS=['Control CM','Control CM + Galunisertib 50 µM','Control CDDP CM',
        'CDDP CM + Galunisertibe 50 µM','Control CM + Staurosporine 0.5 µM','CDDP CM + Staurosporine 0.5 µM']


def extract_blocks(rows):
    if len(rows)!=17:raise ValueError('Unexpected A549 source layout')
    result=[]
    for block,label in enumerate(LABELS):
        header=[v for v in rows[block*3] if v is not None]
        if len(header)!=1 or ' '.join(str(header[0]).split())!=label:raise ValueError('Source condition label mismatch')
        values=rows[block*3+1]
        if len(values)!=3 or any(not isinstance(x,(int,float)) or not np.isfinite(x) or x<0 for x in values):
            raise ValueError('Three finite nonnegative source observations required')
        if block<5 and any(x is not None for x in rows[block*3+2]):raise ValueError('Unexpected intervening source row')
        result.append((label,values))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','ena','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    book=openpyxl.load_workbook(a.source,read_only=True,data_only=True)
    blocks=extract_blocks(list(book['Fig 3b_A549 n=3'].values))
    # Source companion confluence data repeat entire control triplet under CDDP CM.
    growth=extract_blocks(list(book['Fig 3a _A549 n=3'].values))
    duplicate_growth=np.array_equal(growth[0][1],growth[2][1])
    ena=pd.read_csv(a.ena,sep='\t');human=ena[ena.scientific_name.eq('Homo sapiens')]
    if len(human)!=12 or not human.sample_title.str.startswith('A549_T_').all():raise ValueError('RNA source units changed: re-review required')
    rows=[];protocols=[]
    for i,(label,values) in enumerate(blocks):
        context='cisplatin_donor_CM' if 'CDDP' in label else 'control_donor_CM'
        drug='galunisertib' if 'Galunisertib' in label else 'staurosporine' if 'Staurosporine' in label else 'vehicle'
        dose=50 if drug=='galunisertib' else .5 if drug=='staurosporine' else 0
        pid=f'A549_CM_{i}'
        protocols.append(dict(protocol_id=pid,cell_line='A549',context=context,drug=drug,dose_uM=dose,
                              recipient_preincubation_h=24,drug_exposure_h_methods=48,figure_caption_h=72,
                              exact_clock_alignment_verified=False,donor_senescence_induction_days=7,
                              donor_medium_conditioning_h=48,recipient_received_cisplatin_directly=False,
                              residual_carryover_measured=False,matched_recipient_A549_RNA=False))
        for j,value in enumerate(values):
            rows.append(dict(protocol_id=pid,source_label=label,source_row=i*3+2,source_column=j+1,
                             reported_repeat_index=j+1,value=float(value),endpoint='Annexin_V_count_per_cell_confluence',
                             unit='source_count_per_confluence_unit',is_fraction=False,
                             paired_to_RNA_repeat=False,paired_repeat_identity_across_conditions_verified=False))
    a.output.mkdir(parents=True);pd.DataFrame(rows).to_csv(a.output/'endpoint_observations.csv',index=False)
    pd.DataFrame(protocols).to_csv(a.output/'protocols.csv',index=False)
    pd.DataFrame(rows).groupby('protocol_id').value.agg(['mean','std','count']).to_csv(a.output/'descriptive_summaries.csv')
    report=dict(source_doi='10.1038/s43587-025-01054-2',source_workbook_sha256=digest(a.source),ena_metadata_sha256=digest(a.ena),
                observations=18,conditions=6,source_reported_A549_independent_repeats=3,
                observation_unit='Annexin V-positive image counts divided by cell confluence; not percent cells',
                companion_growth_control_triplet_duplicated=bool(duplicate_growth),
                companion_growth_figure_table_agreement_verified=False,companion_growth_used_for_joint_fit=False,
                human_RNA_samples_are_donor_not_recipient=True,human_RNA_runs=len(human),RNA_expression_recovered=False,
                matched_RNA_function=False,full_chain_qualified=False,script_sha256=digest(__file__))
    report['outputs']={f.name:digest(f) for f in a.output.iterdir()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
