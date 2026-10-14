"""Reconstruct the documented public gene-axis definition and align real sources.

This is a new cloud recovery, not an assertion that an unavailable frozen
manifest was read. Native scales, missingness and source sample units persist.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import digest
from lung.data.external_gene_axis import align_gene_axis

SOURCE_HASHES={
 'SS3_Human_A549_counts.csv':'b33aa1b03d71972f8e8c33864be0f0e2747ffa8b56289d00f27cd8c276cc3360',
 'SS3_Human_A549_metadata.csv':'6ee300a8ba36642950c94d3be7455beb51255e42d8f311273d00aa9e0868a38b',
 'gene_metadata.parquet':'d6104d1ca570d94832be0d27ffe8b7e4e54cfd366c9080239219f9c406eb751c'}


def build_axis(genes,counts,meta):
    if not counts.index.is_unique or not counts.columns.is_unique or not meta.index.is_unique:
        raise ValueError('Duplicate source cell/gene identity')
    if set(counts.index)!=set(meta.index):raise ValueError('Cell metadata identity mismatch')
    if genes.ensembl_id.isna().any() or not genes.ensembl_id.is_unique:
        raise ValueError('Invalid Tahoe gene identity')
    values=counts.to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():
        raise ValueError('Invalid source integer counts')
    meta=meta.loc[counts.index].copy()
    axis=genes.loc[genes.ensembl_id.isin(counts.columns)].sort_values('ensembl_id').reset_index(drop=True)
    axis.insert(0,'common_gene_index',np.arange(len(axis)))
    axis['tidyi_source_column']=counts.columns.get_indexer(axis.ensembl_id)
    return axis,meta


def verified_bundle(path):
    m=json.loads((path/'manifest.json').read_text())
    for name,sha in m['outputs'].items():
        if digest(path/name)!=sha:raise ValueError('Recovered source output hash mismatch: '+name)
    return m


def save_aligned(out,name,frame,axis,samples,scale,metadata,provenance):
    values,mask,ledger=align_gene_axis(frame,axis,identifier='ensembl_id',samples=samples,scale=scale)
    if metadata.sample_id.tolist()!=samples:raise ValueError('Sample metadata row binding mismatch')
    dest=out/name;dest.mkdir()
    np.savez_compressed(dest/'aligned_expression.npz',expression=values,observed_mask=mask,
                        gene_ids=axis.ensembl_id.to_numpy(dtype=str),sample_ids=np.array(samples),value_scale=np.array(scale))
    ledger.to_csv(dest/'gene_mapping.csv',index=False);metadata.to_csv(dest/'samples.csv',index=False)
    result=dict(samples=len(samples),target_genes=len(axis),observed_genes=int(mask[0].sum()),
                missing_genes=int((~mask[0]).sum()),measured_zeros=int(((values==0)&mask).sum()),
                native_scale=scale,normalization_performed=False,
                missing_values_are_nan=True,source_sample_identity_preserved=True,
                mapping_status_counts=ledger.mapping_status.value_counts().to_dict(),
                source_manifest_sha256=provenance,cross_source_scale_comparable=False,function_link_qualified=False)
    result['outputs']={f.name:digest(f) for f in dest.iterdir()}
    (dest/'manifest.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['axis-source','public-recovery','em2','trf','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    for name,sha in SOURCE_HASHES.items():
        if digest(a.axis_source/name)!=sha:raise ValueError('Public axis source differs from documented snapshot: '+name)
    genes=pd.read_parquet(a.axis_source/'gene_metadata.parquet')
    counts=pd.read_csv(a.axis_source/'SS3_Human_A549_counts.csv',index_col=0)
    meta=pd.read_csv(a.axis_source/'SS3_Human_A549_metadata.csv',index_col=0)
    axis,meta=build_axis(genes,counts,meta)
    if len(axis)!=32461:raise ValueError('Documented public axis size changed')
    a.output.mkdir(parents=True);axis.to_csv(a.output/'gene_axis.csv',index=False)
    eligible=meta.Cell_line.eq('A549')
    np.savez_compressed(a.output/'tidyi_one_plate_counts.npz',expression=counts.loc[eligible,axis.ensembl_id].to_numpy(dtype=np.int32),
                        gene_ids=axis.ensembl_id.to_numpy(dtype=str),cell_ids=counts.index[eligible].to_numpy(dtype=str))
    meta.assign(strict_A549_eligible=eligible).to_csv(a.output/'tidyi_source_metadata.csv',index_label='cell_id')
    outcomes={}
    for name in ['hm97662','gse161800']:
        p=a.public_recovery/name;verified_bundle(p)
        with np.load(p/'source_expression.npz',allow_pickle=False) as z:
            sample=z['sample_ids'].tolist();frame=pd.DataFrame(z['expression'].T,columns=sample)
            if not z['observed_mask'].all():raise ValueError('Source has unhandled missing observations')
            frame.insert(0,'ensembl_id',z['feature_ids']);scale=str(z['value_scale'])
        metadata=pd.read_csv(p/'samples.csv')
        metadata.insert(0,'sample_id',metadata.geo_accession)
        if not metadata.sample_id.is_unique:raise ValueError('Duplicate GEO identity')
        metadata=metadata.set_index('sample_id',drop=False).loc[sample].reset_index(drop=True)
        outcomes[name]=save_aligned(a.output,name,frame,axis,sample,scale,metadata,digest(p/'manifest.json'))
    p=a.em2;verified_bundle(p)
    ann=pd.read_csv(p/'gene_rows.csv');metadata=pd.read_csv(p/'samples.csv').rename(columns={'sample':'sample_id'})
    with np.load(p/'expression.npz',allow_pickle=False) as z:
        sample=z['sample_ids'].tolist();frame=pd.DataFrame(z['expression'].T,columns=sample)
        if not np.array_equal(z['source_gene_symbols'],ann.source_gene_symbol):raise ValueError('EM2 annotation row mismatch')
    frame.insert(0,'ensembl_id',ann.resolved_ensembl_id)
    outcomes['em2']=save_aligned(a.output,'em2',frame,axis,sample,'bulk_estimated_counts',metadata,digest(p/'manifest.json'))
    p=a.trf;verified_bundle(p);metadata=pd.read_csv(p/'samples.csv')
    with np.load(p/'expression_TPM.npz',allow_pickle=False) as z:
        sample=z['sample_ids'].tolist();frame=pd.DataFrame(z['expression'].T,columns=sample);frame.insert(0,'ensembl_id',z['gene_ids'])
    outcomes['trf']=save_aligned(a.output,'trf',frame,axis,sample,'bulk_TPM',metadata,digest(p/'manifest.json'))
    report=dict(definition='Documented sorted Ensembl intersection of fixed Tahoe vocabulary and original TIDYI CSV',
                tahoe_revision='2dc57900b7981cfcf5e211527169a0b006546a95',source_hashes=SOURCE_HASHES,
                common_genes=len(axis),tidyi_source_rows=len(counts),strict_A549_rows=int(eligible.sum()),
                excluded_unconfirmed_background=int((~eligible).sum()),second_TIDYI_plate_recovered=False,
                frozen_manifest_read=False,original_frozen_file_byte_equality_verified=False,
                source_QC_total_count_difference_cells=int((counts.sum(axis=1)!=meta.total_counts).sum()),
                sources=outcomes,formal_training=False,full_research_qualified=False,script_sha256=digest(__file__))
    report['outputs']={str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*') if f.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
