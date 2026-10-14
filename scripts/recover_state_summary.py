"""Recover source-level expression summaries without inventing replicate matrices.

The 2024 splicing tables use de-novo transcript IDs. An Ensembl annotation is
not permission to sum isoform FPKM or to turn three group columns into nine samples.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.recover_public_expression import digest


def join_summaries(a,b):
    keys=['gene_id','gene id','geneName','length']
    for frame in [a,b]:
        if frame.gene_id.isna().any() or not frame.gene_id.is_unique:
            raise ValueError('Source transcript identity is missing or duplicated')
    if set(a.gene_id)!=set(b.gene_id):raise ValueError('Source transcript coverage differs')
    b=b.set_index('gene_id').loc[a.gene_id].reset_index()
    for key in keys:
        if not a[key].equals(b[key]):raise ValueError('Source transcript annotations disagree')
    if not np.array_equal(a.FPKM_Dormancy.to_numpy(),b.FPKM_Dormancy.to_numpy()):
        raise ValueError('Shared dormancy summary disagrees')
    values=np.column_stack([a.FPKM_NC,a.FPKM_Dormancy,b.FPKM_Relapse]).astype(np.float64)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid source FPKM')
    annotation=a[keys].copy().rename(columns={'gene_id':'source_transcript_id','gene id':'source_ensembl_annotation'})
    annotation['gene_mapping_status']=np.where(annotation.source_ensembl_annotation.eq('-'),'missing_source_annotation',
        np.where(annotation.source_ensembl_annotation.duplicated(keep=False),'multiple_source_transcripts_per_gene','one_source_transcript_annotation'))
    return values,annotation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--supplement',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('New output directory required')
    with zipfile.ZipFile(args.supplement) as z:
        if z.testzip() is not None:raise ValueError('Corrupt source supplement')
        a=pd.read_csv(io.BytesIO(z.read('mmc1.xls')),sep='\t')
        b=pd.read_csv(io.BytesIO(z.read('mmc2.xls')),sep='\t')
    values,annotation=join_summaries(a,b)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'source_transcript_group_FPKM.npz',expression=values.T,
                        transcript_ids=annotation.source_transcript_id.to_numpy(dtype=str),
                        group_labels=np.array(['NC','Dormancy','Relapse']))
    annotation.to_csv(args.output/'transcript_annotations.csv',index=False)
    report=dict(source_doi='10.1016/j.csbj.2023.12.039',
                shared_short_read_project='PRJNA730205',long_read_assembly_project='PRJNA793575',
                independent_of_EphB1_2022_RNA=False,supplement_sha256=digest(args.supplement),
                script_sha256=digest(__file__),transcripts=len(annotation),group_summaries=3,
                numeric_source_values=values.size,group_unit='published_condition_summary_not_individual_sample',
                scale='de_novo_transcript_FPKM',shared_dormancy_exact=True,
                biological_replicate_matrix_recovered=False,Annexin_RNA_pairing=False,
                gene_axis_aligned=False,full_chain_qualified=False,
                mapping_status_counts=annotation.gene_mapping_status.value_counts().to_dict())
    report['outputs']={f.name:digest(f) for f in args.output.iterdir()}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
