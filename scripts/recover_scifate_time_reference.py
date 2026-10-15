"""Stream actual sci-fate A549 total/new RNA, without inventing past-cell pairs."""
from __future__ import annotations
import argparse,gzip,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import digest
from lung.data.external_gene_axis import align_gene_axis


def coordinate_rows(path,ng,nc):
    with gzip.open(path,'rt') as f:
        if next(f).strip()!='%%MatrixMarket matrix coordinate integer general':raise ValueError('Unexpected matrix header')
        g,c,nnz=map(int,next(f).split())
        if (g,c)!=(ng,nc):raise ValueError('Matrix/annotation dimensions disagree')
        previous=-1;count=0
        for line in f:
            g,c,v=map(int,line.split());g-=1;c-=1;key=c*ng+g
            if not(0<=g<ng and 0<=c<nc) or v<=0 or key<=previous:raise ValueError('Unsorted, duplicate or invalid coordinate')
            previous=key;count+=1;yield key,g,c,v
        if count!=nnz:raise ValueError('MatrixMarket nnz mismatch')


def joint_sums(total_rows,new_rows,ng,nc,time_index):
    total=np.zeros((6,ng),dtype=np.int64);new=np.zeros_like(total)
    percell=np.zeros((nc,2),dtype=np.int64);nnew=0;ntotal=0
    it=iter(new_rows);item=next(it,None)
    for key,g,c,v in total_rows:
        total[time_index[c],g]+=v;percell[c,0]+=v;ntotal+=1
        if item is not None and item[0]<key:raise ValueError('New RNA coordinate missing from total RNA')
        if item is not None and item[0]==key:
            if item[3]>v:raise ValueError('New RNA exceeds total RNA')
            new[time_index[c],g]+=item[3];percell[c,1]+=item[3];nnew+=1;item=next(it,None)
    if item is not None:raise ValueError('New RNA extends beyond total matrix')
    return total,new,percell,ntotal,nnew


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','gene-axis','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    for receipt in ['scifate_annotation_sources.json','scifate_count_sources.json']:
        for r in json.loads((a.source/receipt).read_text()):
            if digest(a.source/r['url'].rsplit('/',1)[1])!=r['sha256']:raise ValueError('Source cache changed')
    def path(typ,suffix=''):return a.source/f'GSM3770930_A549_{typ}{suffix}.txt.gz'
    cells=pd.read_csv(path('cell_annotate'));ncells=pd.read_csv(path('cell_annotate','_newly_synthesised'))
    genes=pd.read_csv(path('gene_annotate'));ngenes=pd.read_csv(path('gene_annotate','_newly_synthesised'))
    if not genes.equals(ngenes):raise ValueError('Total/new gene annotation mismatch')
    if not cells[['sample','treatment_time']].equals(ncells[['sample','treatment_time']]):raise ValueError('Total/new cell binding mismatch')
    if not cells['sample'].is_unique or not genes.gene_id.is_unique:raise ValueError('Duplicate source identities')
    days=cells.treatment_time.str.removesuffix('h').astype(int)
    if set(days)!={0,2,4,6,8,10}:raise ValueError('Unexpected time conditions')
    ng,nc=len(genes),len(cells)
    total,new,percell,nt,nn=joint_sums(coordinate_rows(path('gene_count'),ng,nc),coordinate_rows(path('gene_count','_newly_synthesised'),ng,nc),ng,nc,(days//2).to_numpy())
    cells['matrix_total_UMI']=percell[:,0];cells['matrix_new_UMI']=percell[:,1]
    cells['matrix_new_fraction']=np.divide(percell[:,1],percell[:,0],out=np.full(nc,np.nan),where=percell[:,0]>0)
    cells['source_exon_intron_difference']=cells.all_exon+cells.all_intron-percell[:,0]
    cells['source_read_total_difference']=cells.all_reads-percell[:,0]
    axis=pd.read_csv(a.gene_axis);frame=pd.DataFrame(total.T,columns=[str(i) for i in range(6)])
    frame['ensembl_id']=genes.gene_id
    at,mask,ledger=align_gene_axis(frame,axis,identifier='ensembl_id',samples=[str(i) for i in range(6)],scale='bulk_raw_counts')
    # Interface scale validates nonnegative integral sums; biological unit stays aggregated single-cell UMI.
    frame[[str(i) for i in range(6)]]=new.T
    an,nmask,nledger=align_gene_axis(frame,axis,identifier='ensembl_id',samples=[str(i) for i in range(6)],scale='bulk_raw_counts')
    if not np.array_equal(mask,nmask):raise ValueError('Total/new gene masks differ')
    for r in ledger[ledger.mapping_status.eq('matched')].itertuples():
        j=int(r.common_gene_index)
        if not np.array_equal(at[:,j],total[:,r.source_row]) or not np.array_equal(an[:,j],new[:,r.source_row]):
            raise ValueError('Common-axis values changed')
    a.output.mkdir(parents=True)
    cells.to_csv(a.output/'cell_measurement_audit.csv',index=False);genes.to_csv(a.output/'source_genes.csv',index=False)
    ledger.to_csv(a.output/'gene_mapping.csv',index=False)
    np.savez_compressed(a.output/'time_sums.npz',total_UMI=total,new_UMI=new,time_h=np.arange(0,12,2),gene_ids=genes.gene_id.to_numpy(str))
    np.savez_compressed(a.output/'common_axis_time_sums.npz',total_UMI=at,new_UMI=an,observed_mask=mask,time_h=np.arange(0,12,2))
    report=dict(accession='GSE131351',sample='GSM3770930',doi='10.1038/s41587-020-0480-9',
                cells=nc,genes=ng,source_total_nnz=nt,source_new_nnz=nn,time_cell_counts=days.value_counts().sort_index().to_dict(),
                common_axis_observed=int(mask[0].sum()),common_axis_missing=int((~mask[0]).sum()),
                new_UMI_subset_of_total_exact=True,independent_biological_replicates_verified=False,
                source_paper_final_filtered_cells=6680,paper_final_filter_reproduced=False,
                annotation_exon_intron_mismatch_cells=int((cells.source_exon_intron_difference!=0).sum()),
                annotation_read_total_mismatch_cells=int((cells.source_read_total_difference!=0).sum()),
                value_unit='gene_assigned_single_cell_UMI_summed_by_time_not_biological_replicate',
                actual_same_cell_total_new_measurement=True,actual_same_cell_before_after_pair=False,
                inferred_past_expression_exported=False,Annexin_endpoint=False,true_sequential_drug_actions=False,
                DEX_uM=.1,label_window_h=2,full_research_qualified=False,
                source_files=[r for receipt in ['scifate_annotation_sources.json','scifate_count_sources.json']
                              for r in json.loads((a.source/receipt).read_text())],
                source_protocol_hashes={name:digest(a.source/name) for name in ['GSE131351.soft.gz','scifate.pdf']},
                gene_axis_sha256=digest(a.gene_axis),script_sha256=digest(__file__),
                outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
