"""Read complete target records from existing verified ranges; no network path.

A partial source recovery does not shrink the declared scientific cohort. The
full-cohort evaluator must still reject this output until missing targets exist.
"""
import argparse,io,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.recover_pie_a549_functional_reference import (SIZE,URL,FOOTER_HASH,CachedRanges,selected_ranges,align_means)
from scripts.recover_public_expression import digest


def partition_coverage(required,available):
    if not required or any(not groups for groups in required.values()):raise ValueError('Every declared target needs source groups')
    rows=[]
    for target,groups in required.items():
        missing=sorted(set(groups)-set(available))
        rows.append(dict(target=target,complete_source_groups=not missing,required_groups=sorted(groups),missing_groups=missing))
    return rows


def declared_targets(catalog):
    required_columns={'tahoe_drug','nominal_functional_transfer_eligible'}
    if not required_columns.issubset(catalog.columns):
        raise ValueError('Use nominal_structures.csv with explicit cohort eligibility')
    c=catalog.loc[catalog.nominal_functional_transfer_eligible.eq(True)]
    if c.empty or c.tahoe_drug.isna().any() or c.tahoe_drug.duplicated().any():
        raise ValueError('Distinct declared drug cohort required')
    return [str([(n,5.,'uM')]) for n in c.tahoe_drug]


def recover(source,catalog,axis_path,metadata_path,output):
    if output.exists():raise FileExistsError('New output directory required')
    footer_path=source/'PIE_A549_footer.bin';receipt_path=source/'PIE_ranges/receipt.json'
    if digest(footer_path)!=FOOTER_HASH:raise ValueError('Pinned footer changed')
    footer=footer_path.read_bytes();meta=pq.ParquetFile(io.BytesIO(b'PAR1'+footer)).metadata
    before_receipt_hash=digest(receipt_path);receipt=json.loads(receipt_path.read_text())
    if receipt['url']!=URL:raise ValueError('Cache source differs')
    prior={r['group']:r for r in receipt['groups']}
    if len(prior)!=len(receipt['groups']):raise ValueError('Duplicate group receipts')
    targets=declared_targets(pd.read_csv(catalog))
    required={t:{r['group'] for r in selected_ranges(meta,[t],160_000_000)} for t in targets}
    descriptions={r['group']:r for t in targets for r in selected_ranges(meta,[t],160_000_000)}
    available={};parts=[(SIZE-len(footer),footer)]
    for group,g in descriptions.items():
        path=source/'PIE_ranges'/f'group_{group:03d}.bin'
        if not path.exists():continue
        record=prior.get(group)
        if record is None or record['path']!=path.name or any(record[k]!=g[k] for k in ['start','end_exclusive','bytes']):
            raise ValueError('Existing range lacks its exact metadata receipt')
        if path.stat().st_size!=g['bytes'] or digest(path)!=record['sha256']:raise ValueError('Range integrity failed')
        available[group]=record
    coverage=partition_coverage(required,available)
    complete=[r['target'] for r in coverage if r['complete_source_groups']]
    if not complete:raise ValueError('No complete target; do not fabricate partial expression')
    used=sorted(set().union(*(required[t] for t in complete)))
    if sum(available[g]['bytes'] for g in used)>160_000_000:raise ValueError('Offline memory bound exceeded')
    for g in used:parts.append((available[g]['start'],(source/'PIE_ranges'/available[g]['path']).read_bytes()))
    reader=pq.ParquetFile(CachedRanges(SIZE,parts));frames=[]
    for group in used:
        f=reader.read_row_group(group).to_pandas();frames.append(f[f.target.isin(complete)])
    raw=pd.concat(frames,ignore_index=True)
    if set(raw.target)!=set(complete) or raw[['target','feature']].duplicated().any():raise ValueError('Target coverage or feature identity failed')
    axis=pd.read_csv(axis_path);gene_meta=pd.read_parquet(metadata_path)
    values,refs,mask,ledger,conditions=align_means(raw,axis,gene_meta)
    output.mkdir(parents=True);raw.to_parquet(output/'source_group_means.parquet',index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(output/'source_group_means.parquet'),raw)
    conditions.to_csv(output/'conditions.csv',index=False);ledger.to_csv(output/'gene_mapping.csv.gz',index=False)
    np.savez_compressed(output/'aligned_group_means.npz',target_mean=values,reference_mean=refs,observed_mask=mask,
       gene_ids=axis.sort_values('common_gene_index').ensembl_id.to_numpy(str),targets=conditions.target.to_numpy(str))
    with np.load(output/'aligned_group_means.npz',allow_pickle=False) as z:
        for key,expected in [('target_mean',values),('reference_mean',refs),('observed_mask',mask)]:
            np.testing.assert_array_equal(z[key],expected)
        if not np.isnan(z['target_mean'][~mask]).all() or not np.isnan(z['reference_mean'][~mask]).all():
            raise ValueError('Missing genes became measured abundance')
    (output/'declared_cohort_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    if digest(receipt_path)!=before_receipt_hash:raise ValueError('Readonly cache receipt changed')
    manifest=dict(source_url=URL,source_catalog_sha256=digest(catalog),eligibility_column='nominal_functional_transfer_eligible',
       declared_conditions=len(targets),recovered_complete_conditions=len(complete),pending_conditions=len(targets)-len(complete),
       declared_cohort_complete=len(complete)==len(targets),full_cohort_evaluation_allowed=len(complete)==len(targets),
       actual_mean_records=len(raw),observed_axis_values=int(mask.sum())*2,raw_cell_records_read=0,
       source_scale='population_mean_CP10k',plate_matched_reference=False,full_protocol_match=False,
       used_ranges=[available[g] for g in used],source_bytes_already_cached=sum(available[g]['bytes'] for g in used),
       new_network_bytes=0,cache_modified=False,cache_receipt_sha256=before_receipt_hash,footer_sha256=FOOTER_HASH,
       missing_targets_not_silently_excluded_from_declared_goal=True,model_training=False,scientific_feasibility_complete=False,
       gene_metadata_sha256=digest(metadata_path),gene_axis_sha256=digest(axis_path),script_sha256=digest(__file__),
       outputs={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','catalog','gene-axis','gene-metadata','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.source,a.catalog,a.gene_axis,a.gene_metadata,a.output),indent=2))
