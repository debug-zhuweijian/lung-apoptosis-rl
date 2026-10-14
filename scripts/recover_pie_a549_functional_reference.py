"""Recover only selected public A549 group-mean RNA records using bounded ranges.

PIE author-processed CP10k means are not raw counts or individual cell trajectories.
No missing gene is filled with zero. Assay labels remain in their separate table.
"""
import argparse,hashlib,io,json,struct,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from lung.data.external_gene_axis import align_gene_axis
from scripts.recover_public_expression import digest
URL='https://huggingface.co/datasets/arcinstitute/PIE_tahoe100m/resolve/aa919261564cfd63155f734760cf38d79e089b6c/de/CVCL_0023.parquet'
SIZE=482895984
FOOTER_HASH='c912f642790adac6668a57929191cb0054601c1510929fd1791ee296de85645c'


class CachedRanges(io.RawIOBase):
    def __init__(self,size,parts):self.size=size;self.parts=parts;self.pos=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        if whence not in [0,1,2]:raise ValueError('Unknown seek mode')
        pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if not 0<=pos<=self.size:raise ValueError('Invalid seek')
        self.pos=pos;return pos
    def read(self,n=-1):
        if n<0:raise ValueError('Whole file read forbidden')
        n=min(n,self.size-self.pos)
        if not n:return b''
        for start,b in self.parts:
            if start<=self.pos and self.pos+n<=start+len(b):
                off=self.pos-start;self.pos+=n;return b[off:off+n]
        raise ValueError('Unfetched source range; cannot fill missing bytes')


def download_range(start,end):
    import requests
    if not 0<=start<end<=SIZE or end-start>8_000_000:raise ValueError('Range outside bounded source')
    r=requests.get(URL,headers={'Range':f'bytes={start}-{end-1}'},timeout=(20,90),stream=True)
    expected=f'bytes {start}-{end-1}/{SIZE}'
    if r.status_code!=206 or r.headers.get('content-range')!=expected:
        r.close();raise ValueError('Server did not return exact requested range')
    chunks=[];n=0
    for b in r.iter_content(65536):
        n+=len(b)
        if n>end-start:r.close();raise ValueError('Response exceeds declared range')
        chunks.append(b)
    r.close()
    if n!=end-start:raise ValueError('Truncated source range')
    return b''.join(chunks)


def selected_ranges(metadata,targets):
    groups=[]
    for i in range(metadata.num_row_groups):
        g=metadata.row_group(i);s=g.column(0).statistics
        if s is None:raise ValueError('Missing target range statistics; do not silently skip')
        if not any(s.min<=t<=s.max for t in targets):continue
        offsets=[min(x for x in [g.column(j).dictionary_page_offset,g.column(j).data_page_offset] if x is not None) for j in range(g.num_columns)]
        lo=min(offsets);hi=max(offsets[j]+g.column(j).total_compressed_size for j in range(g.num_columns))
        groups.append(dict(group=i,start=lo,end_exclusive=hi,bytes=hi-lo))
    if not groups or sum(g['bytes'] for g in groups)>80_000_000:raise ValueError('Requested subset exceeds 80MB bound')
    return groups


def recover_rows(source,targets,fetch=False):
    source.mkdir(parents=True,exist_ok=True);cache=source/'PIE_ranges';cache.mkdir(exist_ok=True)
    path=source/'PIE_A549_footer.bin'
    if not path.exists():
        if not fetch:raise ValueError('Source footer missing; use --fetch for bounded public download')
        b=download_range(SIZE-131072,SIZE)
        if b[-4:]!=b'PAR1':raise ValueError('Not a Parquet footer')
        n=struct.unpack('<I',b[-8:-4])[0]
        if n+8>len(b):raise ValueError('Footer larger than bounded request')
        path.write_bytes(b[-n-8:])
    footer=path.read_bytes()
    if digest(path)!=FOOTER_HASH:raise ValueError('Pinned footer changed')
    metadata=pq.ParquetFile(io.BytesIO(b'PAR1'+footer)).metadata
    groups=selected_ranges(metadata,targets);prior={}
    if (cache/'receipt.json').exists():
        receipt=json.loads((cache/'receipt.json').read_text())
        if receipt['url']!=URL:raise ValueError('Source URL changed')
        prior={r['group']:r for r in receipt['groups']}
    records=[];parts=[(SIZE-len(footer),footer)]
    for g in groups:
        f=cache/f'group_{g["group"]:03d}.bin'
        if f.exists():
            r=prior.get(g['group'])
            if r is None or any(r[k]!=g[k] for k in ['start','end_exclusive','bytes']) or digest(f)!=r['sha256']:
                raise ValueError('Cached source range lacks matching integrity receipt')
            b=f.read_bytes()
        else:
            if not fetch:raise ValueError('Source range missing; no invented bytes')
            b=download_range(g['start'],g['end_exclusive']);f.write_bytes(b)
        if len(b)!=g['bytes']:raise ValueError('Range length mismatch')
        record=dict(g,path=f.name,sha256=hashlib.sha256(b).hexdigest());records.append(record);parts.append((g['start'],b))
        prior[g['group']]=record
        (cache/'receipt.json').write_text(json.dumps(dict(url=URL,source_revision='aa919261564cfd63155f734760cf38d79e089b6c',
            full_file_sha256_verified=False,selected_range_bytes=sum(x['bytes'] for x in groups),
            groups=sorted(prior.values(),key=lambda x:x['group'])),indent=2))
    f=pq.ParquetFile(CachedRanges(SIZE,parts));frames=[]
    for g in groups:
        d=f.read_row_group(g['group']).to_pandas();frames.append(d[d.target.isin(targets)])
    d=pd.concat(frames,ignore_index=True)
    if set(d.target)!=set(targets) or d[['target','feature']].duplicated().any():raise ValueError('Incomplete or duplicate target-gene records')
    return d,records


def align_means(frame,axis,metadata):
    if metadata.ensembl_id.duplicated().any():raise ValueError('Duplicate reference identity')
    unique=~metadata.gene_symbol.duplicated(keep=False)
    mapping=dict(zip(metadata.loc[unique,'gene_symbol'],metadata.loc[unique,'ensembl_id']))
    targets=[];refs=[];masks=[];ledgers=[];conditions=[]
    for target,g in frame.groupby('target',sort=True):
        if g.feature.duplicated().any():raise ValueError('Duplicate tested gene')
        if not g.target_ncells.gt(0).all() or not g.ref_ncells.gt(0).all() or g.target_ncells.nunique()!=1 or g.ref_ncells.nunique()!=1:raise ValueError('Population counts differ within condition')
        x=g[['target_mean','ref_mean']].copy();x['ensembl_id']=g.feature.map(mapping)
        a,mask,ledger=align_gene_axis(x,axis,identifier='ensembl_id',samples=['target_mean','ref_mean'],scale='population_mean_CP10k')
        targets.append(a[0]);refs.append(a[1]);masks.append(mask[0]);ledger['target']=target;ledger['source_symbol']=g.feature.to_numpy();ledgers.append(ledger)
        conditions.append(dict(target=target,source_tested_genes=len(g),observed_common_genes=int(mask[0].sum()),
            author_reported_target_cells=int(g.target_ncells.iloc[0]),author_reported_reference_cells=int(g.ref_ncells.iloc[0]),
            raw_individual_cells_downloaded=False,control_definition='all A549 DMSO_TF cells pooled by author pipeline',
            plate_matched_reference=False,scale='population_mean_CP10k',missing_gene_semantics='not_tested_or_not_uniquely_mapped_not_measured_zero'))
    return np.array(targets),np.array(refs),np.array(masks),pd.concat(ledgers,ignore_index=True),pd.DataFrame(conditions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','catalog','gene-axis','gene-metadata','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--fetch',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    c=pd.read_csv(a.catalog);c=c[c.strict_structure_link_eligible.eq(True)]
    if c.empty or c.tahoe_drug.duplicated().any():raise ValueError('Unique qualified drug names required')
    names=[str([(n,5.0,'uM')]) for n in c.tahoe_drug]
    raw,ranges=recover_rows(a.source,names,a.fetch);axis=pd.read_csv(a.gene_axis);meta=pd.read_parquet(a.gene_metadata)
    values,refs,mask,ledger,conditions=align_means(raw,axis,meta)
    a.output.mkdir(parents=True);raw.to_parquet(a.output/'source_group_means.parquet',index=False)
    np.savez_compressed(a.output/'aligned_group_means.npz',target_mean=values,reference_mean=refs,observed_mask=mask,
        gene_ids=axis.sort_values('common_gene_index').ensembl_id.to_numpy(dtype=str),targets=conditions.target.to_numpy(dtype=str))
    ledger.to_csv(a.output/'gene_mapping.csv.gz',index=False);conditions.to_csv(a.output/'conditions.csv',index=False)
    report=dict(source_url=URL,source_revision='aa919261564cfd63155f734760cf38d79e089b6c',
        source_file_bytes=SIZE,full_source_file_downloaded=False,full_file_sha256_verified=False,
        downloaded_row_groups=len(ranges),selected_range_bytes=sum(x['bytes'] for x in ranges),ranges=ranges,
        actual_mean_records=len(raw),conditions=len(conditions),reported_target_cells=int(conditions.author_reported_target_cells.sum()),
        raw_cell_records_read=0,observed_axis_values=int(mask.sum())*2,source_scale='population_mean_CP10k',
        source_gene_filter='author gpudge 0.7.0 >=5CPM tested genes; absent entries remain masked',
        phenotype='RNA response group means, not functional death labels',same_cell_pair=False,full_protocol_match=False,
        model_training=False,scientific_feasibility_complete=False,
        source_catalog_sha256=digest(a.catalog),gene_metadata_sha256=digest(a.gene_metadata),gene_axis_sha256=digest(a.gene_axis),
        footer_sha256=FOOTER_HASH,script_sha256=digest(__file__),outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='ranges'},indent=2))

if __name__=='__main__':main()
