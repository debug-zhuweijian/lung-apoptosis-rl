"""Recover public A549 subculture raw counts, not class-averaged/future outcomes.

Three contrast sheets contain the same parental libraries in different orders.
Join by Ensembl identity, verify repeated counts, and retain source symbol damage.
"""
import argparse,json,sys,datetime
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.recover_public_expression import digest
from lung.data.external_gene_axis import align_gene_axis

SHEETS={'parental_vs_holo':['Sample_P1','Sample_P2','Sample_P3','Sample_1_1','Sample_1_2','Sample_1_5'],
        'parental_vs_mero':['Sample_P1','Sample_P2','Sample_P3','Sample_2_2','Sample_2_3','Sample_2_4'],
        'parental_vs_para':['Sample_P1','Sample_P2','Sample_P3','Sample_3_2','Sample_3_4','Sample_3_7']}
FUNCTIONAL_CLONES={'holo':{'1.1','1.5','1.10'},'mero':{'2.2','2.3','2.4'},'para':{'3.2','3.4','3.11'}}


def combine(frames):
    merged=None;symbols=[]
    for sheet,columns in SHEETS.items():
        f=frames[sheet]
        if f.ensembl_gene_id.duplicated().any() or not f.ensembl_gene_id.str.fullmatch(r'ENSG\d+').all():
            raise ValueError('Invalid or duplicate source Ensembl identity')
        if any(c not in f for c in columns):raise ValueError('Source RNA columns differ')
        values=f[columns].to_numpy(float)
        if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():
            raise ValueError('Raw integer counts required, not normalized values')
        indexed=f.set_index('ensembl_gene_id')[columns].sort_index()
        if merged is None:merged=indexed.copy()
        else:
            if not indexed.index.equals(merged.index):raise ValueError('Gene coverage differs across contrast sheets')
            for col in columns:
                if col in merged:
                    if not np.array_equal(merged[col].to_numpy(),indexed[col].to_numpy()):
                        raise ValueError('Repeated source library counts differ')
                else:merged[col]=indexed[col]
        for row in f[['ensembl_gene_id','hgnc_symbol']].itertuples(index=False):
            if isinstance(row.hgnc_symbol,(datetime.datetime,datetime.date)):
                symbols.append(dict(sheet=sheet,ensembl_id=row.ensembl_gene_id,
                    source_symbol_display=str(row.hgnc_symbol),source_symbol_type=type(row.hgnc_symbol).__name__,
                    mapping_key='ensembl_gene_id',guessed_symbol_repair=False))
    return merged.reset_index().rename(columns={'ensembl_gene_id':'ensembl_id'}),pd.DataFrame(symbols)


def validate_library_totals(raw, summary):
    if summary['sample'].duplicated().any():raise ValueError('Duplicate summary library')
    rows=[]
    columns=[c for c in raw if c.startswith('Sample_')]
    if set(summary['sample'])!={c.removeprefix('Sample_') for c in columns}:
        raise ValueError('Library summary coverage differs')
    for col in columns:
        name=col.removeprefix('Sample_');expected=summary.loc[summary['sample'].eq(name),'# reads mapping to a gene'].iloc[0]
        actual=int(raw[col].sum())
        if not np.isfinite(expected) or actual!=expected:raise ValueError('Raw counts do not reproduce reported mapped reads')
        rows.append(dict(source_column=col,reported_mapped_reads=int(expected),recovered_count_sum=actual,difference=0))
    return pd.DataFrame(rows)


def sample_metadata(columns):
    rows=[]
    for col in columns:
        suffix=col.removeprefix('Sample_')
        if suffix in ['P1','P2','P3']:context='parental';clone=None;overlap=False
        elif suffix in ['1_1','1_2','1_5','2_2','2_3','2_4','3_2','3_4','3_7']:
            context={'1':'holo','2':'mero','3':'para'}[suffix[0]];clone=suffix.replace('_','.');overlap=clone in FUNCTIONAL_CLONES[context]
        else:raise ValueError('Unknown source subculture label')
        rows.append(dict(sample_id='Tieche2018:'+suffix,source_column=col,background='A549',context=context,
            source_clone_id=clone,statistical_unit='parental_culture' if clone is None else 'named_clonal_subculture',
            split_group='Tieche2018:'+suffix,baseline_expression=True,drug_action_at_RNA_collection=None,
            explicit_RNA_collection_h=None,source_default_analysis_day=6,
            source_RNA_collection_clock_independently_verified=False,
            clone_ID_also_listed_in_functional_panel=overlap,clone_specific_functional_values_recovered=False,
            source_functional_panel_clone_IDs=json.dumps(sorted(FUNCTIONAL_CLONES.get(context,set()))),
            paired_RNA_functional_replicate=False,paired_single_cell_before_after=False))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','gene-axis','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    import hashlib
    workbook=a.source/'mmc3.xlsx'
    if hashlib.md5(workbook.read_bytes()).hexdigest()!='b121b42425192cdbed6ed220353ed416':raise ValueError('Pinned workbook differs')
    frames={s:pd.read_excel(workbook,sheet_name=s,usecols=range(10)) for s in SHEETS}
    raw,damaged=combine(frames);totals=validate_library_totals(raw,pd.read_excel(workbook,sheet_name='count_summary'));columns=[c for c in raw if c.startswith('Sample_')];samples=sample_metadata(columns)
    axis=pd.read_csv(a.gene_axis);aligned,mask,ledger=align_gene_axis(raw,axis,identifier='ensembl_id',samples=columns,scale='bulk_raw_counts')
    values=raw[columns].to_numpy(float)
    for r in ledger[ledger.mapping_status.eq('matched')].itertuples():
        if not np.array_equal(aligned[:,int(r.common_gene_index)],values[r.source_row]):raise ValueError('Mapped count changed')
    a.output.mkdir(parents=True);totals.to_csv(a.output/'library_count_totals.csv',index=False);samples.to_csv(a.output/'samples.csv',index=False);damaged.to_csv(a.output/'source_symbol_date_audit.csv',index=False);ledger.to_csv(a.output/'gene_mapping.csv.gz',index=False)
    np.savez_compressed(a.output/'source_raw_counts.npz',expression=values.T.astype(np.int64),gene_ids=raw.ensembl_id.to_numpy(str),sample_ids=samples.sample_id.to_numpy(str))
    np.savez_compressed(a.output/'common_axis_counts.npz',expression=aligned,observed_mask=mask,gene_ids=axis.sort_values('common_gene_index').ensembl_id.to_numpy(str),sample_ids=samples.sample_id.to_numpy(str))
    report=dict(source_doi='10.1016/j.neo.2018.09.008',source_genes=len(raw),source_libraries=len(samples),actual_count_values=int(values.size),
        named_subcultures=9,parental_cultures=3,common_observed_genes=int(mask[0].sum()),common_missing_genes=int((~mask[0]).sum()),
        source_symbol_date_rows=len(damaged),unique_genes_with_source_symbol_date=int(damaged.ensembl_id.nunique()),
        source_gene_annotation='GRCh38.87',source_scale='HTSeq-count raw counts',author_normalized_columns_used=False,
        DE_rank_row_order_used_for_alignment=False,repeated_parental_counts_equal_after_gene_ID_join=True,all_12_count_sums_equal_author_mapped_read_totals=True,
        clone_IDs_also_in_functional_panel=int(samples.clone_ID_also_listed_in_functional_panel.sum()),
        clone_specific_functional_values_recovered=False,source_functional_plot_statistic='median_and_error_of_three_different_subcultures_per_class',
        functional_endpoint='trypan_blue_excluded_live_cell_count_after_exposure_and_recovery_not_Annexin',
        independent_initial_RNA_increment_validated=False,complete_scientific_qualification=False,
        source_hashes={q.name:digest(q) for q in [workbook,a.source/'mmc2.docx',a.source/'paper.xml',a.gene_axis]},
        script_sha256=digest(__file__),outputs={q.name:digest(q) for q in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
