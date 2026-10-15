"""Recover GSE222187 individual TPM columns, excluding redundant group averages.

The resistant subline and nutrient schedule are retained. Assay protocols do
not automatically fill unreported RNA timing or maintenance-drug washout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.recover_public_expression import read_soft,digest
GROUPS=['CON','DDP','TRF','TRF_DDP']
SAMPLES=[g+str(i) for g in GROUPS for i in [1,2,3]]


def validate(frame):
    if frame.gene_id.isna().any() or not frame.gene_id.is_unique:raise ValueError('Invalid gene identity')
    if frame.columns[:13].tolist()!=['gene_id']+SAMPLES:raise ValueError('Unexpected individual-sample columns')
    values=frame[SAMPLES].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid TPM')
    for g in GROUPS:
        if not np.allclose(frame[[g+str(i) for i in [1,2,3]]].mean(axis=1),frame[g],atol=0.000051,rtol=0):
            raise ValueError('Published rounded group mean disagrees')
    extra=[c for c in frame if c not in ['gene_id']+SAMPLES+GROUPS]
    if any(frame[c].notna().any() for c in extra):raise ValueError('Unrecognized nonempty extra column')
    return values


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['matrix','soft','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    frame=pd.read_csv(a.matrix,sep='\t');values=validate(frame)
    _,records=read_soft(a.soft)
    meta={m['title'][0]:(acc,m) for acc,m,_ in records}
    if set(meta)!=set(SAMPLES):raise ValueError('GEO title binding incomplete')
    rows=[]
    for title in SAMPLES:
        acc,m=meta[title]
        if m['source_name_ch1']!=['A549/DDP']:raise ValueError('Resistant background mismatch')
        group=title[:-1]
        rows.append(dict(sample_id=acc,source_column=title,source_group=group,background='A549/DDP',
                         nominal_replicate=int(title[-1]),RNA_exposure_h=None,RNA_dose_ug_ml=None,
                         maintenance_cisplatin_ug_ml_reported=1,maintenance_washout_verified=False,
                         nutrient_cycle='6h_normal_18h_serum_and_glucose_free' if group.startswith('TRF') else 'normal',
                         medium_change_PBS_washes_reported=2,paired_to_Annexin_repeat=False))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'expression_TPM.npz',expression=values.T,gene_ids=frame.gene_id.to_numpy(dtype=str),
                        sample_ids=np.array([r['sample_id'] for r in rows]))
    pd.DataFrame(rows).to_csv(a.output/'samples.csv',index=False)
    frame[['gene_id']+GROUPS].to_csv(a.output/'source_group_means_not_samples.csv',index=False)
    report=dict(source_accession='GSE222187',source_doi='10.1007/s12032-022-01923-5',
                scale='TPM',genes=len(frame),individual_sample_columns=12,group_means_excluded_from_sample_count=4,
                numeric_expression_values=values.size,source_group_mean_rounding_verified=True,
                TPM_library_sums=dict(zip(SAMPLES,values.sum(axis=0).tolist())),
                frozen_axis_aligned=False,exact_RNA_function_protocol_verified=False,full_chain_qualified=False,
                source_hashes={'matrix':digest(a.matrix),'soft':digest(a.soft)},script_sha256=digest(__file__))
    report['outputs']={f.name:digest(f) for f in a.output.iterdir()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
