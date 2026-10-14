"""Recover a small verified ARCHS4 reprocessing of public GSE270062.

Preserve symbol collisions and source row order. Integer-valued Kallisto exports
are not original raw counts; no simultaneous RNA/Annexin pairing is asserted.
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
from scripts.recover_public_expression import digest,read_soft

SAMPLES=['GSM8333896','GSM8333897','GSM8333898','GSM8333899']


def verify_export(table,genes,metadata):
    if table.columns.tolist()!=SAMPLES or set(metadata)!=set(SAMPLES):
        raise ValueError('Sample identity or order mismatch')
    if table.index.astype(str).tolist()!=genes:
        raise ValueError('ARCHS4 exported gene row order disagrees with gene metadata')
    for sample in SAMPLES:
        if metadata[sample]['series']!='GSE270062' or metadata[sample]['source']!='A549':
            raise ValueError('ARCHS4 sample background or series mismatch')
    values=table.apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid expression values')
    annotation=pd.DataFrame({'source_row':np.arange(len(table)),'source_gene_symbol':table.index.astype(str)})
    annotation['identity_status']=np.where(annotation.source_gene_symbol.eq(''),'missing_symbol',
        np.where(annotation.source_gene_symbol.duplicated(keep=False),'duplicate_symbol_requires_gene_ID','unique_source_symbol'))
    return values,annotation


def attach_provider_axis(annotation,provider):
    symbols=annotation.source_gene_symbol.tolist()
    ids=provider['ensembl_gene']
    if provider['symbol']!=symbols or len(ids)!=len(symbols) or len(set(ids))!=len(ids):
        raise ValueError('Provider axis is inconsistent')
    out=annotation.copy()
    out['provider_ensembl_at_same_row']=ids
    # Release binding is incomplete. Unique symbols identify rows unambiguously;
    # equal-symbol row permutations cannot be excluded for symbol collisions.
    unique=out.identity_status.eq('unique_source_symbol')
    out['resolved_ensembl_id']=out.provider_ensembl_at_same_row.where(unique)
    out['ensembl_link_status']=np.where(unique,'unique_symbol_in_verified_provider_axis','row_candidate_only_release_binding_incomplete')
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['matrix','genes','metadata','soft','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--provider-axis',type=Path)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('New output directory required')
    with zipfile.ZipFile(args.matrix) as z:
        if z.testzip() is not None:raise ValueError('Corrupt matrix archive')
        table=pd.read_csv(io.BytesIO(z.read('matrix.tsv')),sep='\t',index_col=0,keep_default_na=False)
    gene_meta=json.loads(args.genes.read_text());metadata=json.loads(args.metadata.read_text())
    if gene_meta['species']!='human':raise ValueError('Wrong species gene metadata')
    values,annotation=verify_export(table,gene_meta['genes'],metadata)
    if args.provider_axis is not None:
        annotation=attach_provider_axis(annotation,json.loads(args.provider_axis.read_text()))
    _,records=read_soft(args.soft)
    if [r[0] for r in records]!=SAMPLES:raise ValueError('GEO sample records changed')
    rows=[]
    for i,(acc,meta,_) in enumerate(records):
        chars=dict(x.split(': ',1) for x in meta['characteristics_ch1'])
        if chars['cell line']!='A549':raise ValueError('GEO background changed')
        protocol=' '.join(meta['treatment_protocol_ch1'])
        if '6 μM EM-2 for 24 h' not in protocol:raise ValueError('GEO exposure changed')
        expected='DMSO' if i<2 else '6 uM EM-2'
        if chars['treatment']!=expected:raise ValueError('GEO treatment identity changed')
        rows.append(dict(sample=acc,title=meta['title'][0],cell_line='A549',series='GSE270062',
                         drug='vehicle' if i<2 else 'EM-2',dose_uM=0 if i<2 else 6,
                         exposure_h=24,nominal_replicate=i%2+1,paired_to_function_replicate=False,
                         independent_function_time_h=48))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'expression.npz',expression=values.T,sample_ids=np.array(SAMPLES),
                        source_gene_symbols=annotation.source_gene_symbol.to_numpy(dtype=str))
    annotation.to_csv(args.output/'gene_rows.csv',index=False);pd.DataFrame(rows).to_csv(args.output/'samples.csv',index=False)
    report=dict(series='GSE270062',provider='ARCHS4 uniformly reprocessed source reads',
                provider_endpoint='https://maayanlab.cloud/sigpy/data/samples',
                provider_documentation='https://archs4.org/help',
                source_paper_doi='10.3389/fphar.2025.1580217',
                rows=len(values),samples=len(SAMPLES),values=values.size,
                scale='bulk_estimated_counts_ARCHS4_integer_export',
                documentation_annotation='GRCh38_Ensembl107_Kallisto',
                exact_provider_release_bound_to_export=False,
                source_row_order_verified=True,source_sample_metadata_verified=True,
                symbol_collision_rows=int(annotation.identity_status.eq('duplicate_symbol_requires_gene_ID').sum()),
                frozen_gene_axis_aligned=False,contemporaneous_RNA_Annexin=False,
                same_cell_fate=False,full_chain_qualified=False,
                script_sha256=digest(__file__),
                source_hashes={k:digest(getattr(args,k)) for k in ['matrix','genes','metadata','soft']})
    if args.provider_axis is not None:
        report['provider_axis_sha256']=digest(args.provider_axis)
        report['unique_symbol_Ensembl_links']=int(annotation.resolved_ensembl_id.notna().sum())
        report['symbol_collision_IDs_are_candidates_only']=True
    report['outputs']={f.name:digest(f) for f in args.output.iterdir()}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
