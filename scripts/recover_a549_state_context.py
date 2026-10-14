"""Recover measured A549 pretreatment-context RNA with explicit protocol conflicts."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import read_soft,digest
from lung.data.external_gene_axis import align_gene_axis


def protocol(condition):
    common=dict(collection_h=168,initial_single_cell_expression_measured=False,chemical_structure_verified=False)
    if condition=='ctr':return dict(**common,drug_name=None,dose_uM=None,exposure_h=0,recovery_h=0,protocol_conflict=False,protocol_role='parallel_proliferating_reference',vehicle_composition_verified=False)
    if condition=='doxo':return dict(**common,drug_name='doxorubicin',dose_uM=.1,exposure_h=72,recovery_h=96,protocol_conflict=False,protocol_role='drug_pulse_then_withdrawal')
    if condition=='ink':return dict(**common,drug_name='INK128',dose_uM=.2,exposure_h=168,recovery_h=0,protocol_conflict=False,protocol_role='continuous_exposure_refreshed_every_72h')
    if condition=='pb':return dict(**common,drug_name='palbociclib',dose_uM=5,exposure_h=None,recovery_h=None,protocol_conflict=True,protocol_role='unresolved_exposure_history',paper_specific_exposure_h=168,GEO_generic_exposure_h=72,GEO_generic_recovery_h=96)
    raise ValueError('Unknown source treatment')


def bind_samples(blocks,columns):
    rows=[]
    for accession,m,_ in blocks:
        title=m.get('title',[''])[0]
        if not title.startswith('A549 '):continue
        match=re.fullmatch(r'A549 (ctr|pb|doxo|ink) (1|2|2-2|3)',title)
        chars=dict(x.split(': ',1) for x in m['characteristics_ch1'])
        if not match or chars.get('cell line')!='A549':raise ValueError('Inconsistent background/title')
        condition,label=match.groups()
        if chars.get('treatment')!=condition or chars.get('replicate')!=label.split('-')[0]:raise ValueError('Replicate/treatment annotation conflict')
        source=m['description']
        if len(source)!=1 or source[0] not in columns:raise ValueError('Missing explicit source column binding')
        rows.append(dict(sample_id=accession,source_column=source[0],source_title=title,source_replicate_label=chars['replicate'],
                         library_label=label,conservative_split_group='A549_replicate_'+chars['replicate'],
                         four_independent_biological_replicates_verified=False,condition=condition,background='A549',
                         **protocol(condition)))
    result=pd.DataFrame(rows)
    if len(result)!=16 or result.sample_id.duplicated().any() or result.source_column.duplicated().any():raise ValueError('A549 library coverage mismatch')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','gene-axis','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output required')
    _,blocks=read_soft(a.source/'GSE246690.soft.gz');frame=pd.read_excel(a.source/'counts.xlsx')
    frame=frame.rename(columns={frame.columns[0]:'ensembl_id'})
    if not frame.ensembl_id.is_unique or not frame.ensembl_id.str.fullmatch(r'ENSG\d+(?:\.\d+)?').all():raise ValueError('Invalid unique human gene IDs')
    samples=bind_samples(blocks,frame.columns);cols=samples.source_column.tolist();values=frame[cols].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():raise ValueError('Invalid published integer counts')
    axis=pd.read_csv(a.gene_axis);aligned,mask,ledger=align_gene_axis(frame,axis,identifier='ensembl_id',samples=cols,scale='bulk_raw_counts')
    for r in ledger[ledger.mapping_status.eq('matched')].itertuples():
        if not np.array_equal(aligned[:,int(r.common_gene_index)],values[r.source_row]):raise ValueError('Mapped count changed')
    a.output.mkdir(parents=True);samples.to_csv(a.output/'samples.csv',index=False);ledger.to_csv(a.output/'gene_mapping.csv',index=False)
    np.savez_compressed(a.output/'source_A549_counts.npz',expression=values.T.astype(np.int32),gene_ids=frame.ensembl_id.to_numpy(str),sample_ids=samples.sample_id.to_numpy(str))
    np.savez_compressed(a.output/'common_axis_counts.npz',expression=aligned,observed_mask=mask,gene_ids=axis.ensembl_id.to_numpy(str),sample_ids=samples.sample_id.to_numpy(str))
    report=dict(accession='GSE246690',doi='10.1158/0008-5472.CAN-24-0529',source_genes=len(frame),source_libraries=len(frame.columns)-1,
                actual_A549_libraries=len(samples),actual_A549_source_values=int(values.size),common_observed_genes=int(mask[0].sum()),common_missing_genes=int((~mask[0]).sum()),
                source_replicate_labels=samples.source_replicate_label.value_counts().to_dict(),conservative_split_groups=samples.conservative_split_group.nunique(),
                processing_conflicts=['GEO says mouse reference GRCh38 although human ENSG IDs and paper human hg38',
                                      'GEO featureCounts/STAR2.7.10a versus paper STAR GeneCounts2.7.10b; original pipeline not re-executed'],
                mapping_status_counts=ledger.mapping_status.value_counts().to_dict(),
                identifier_suffix_limit='numeric suffixes retained in source; collisions after existing suffix normalization are excluded, not summed',
                source_scale='published_integer_gene_counts',palbociclib_history_conflict=True,
                functional_challenge=dict(endpoint='crystal_violet_relative_viability_not_Annexin',initial_contexts=4,
                                          normalization='each context own untreated reference',duration_candidates_h=[48,72],
                                          duration_conflict='Results48h versus Methods/Fig2B72h',raw_numeric_observations_recovered=False,
                                          RNA_measured_before_second_intervention=True,post_navitoclax_RNA_available_in_this_matrix=False,
                                          individual_RNA_function_pairs_verified=False),
                independent_state_increment_validated=False,complete_sequence_model_validated=False,full_research_qualified=False,
                source_urls=dict(counts='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE246nnn/GSE246690/suppl/GSE246690_vramponi_202207_rnaseq_raw_counts.xlsx',
                                 GEO='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246690',paper='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7617193/fullTextXML'),
                source_hashes={p.name:digest(p) for p in [a.source/'counts.xlsx',a.source/'GSE246690.soft.gz',a.source/'paper.xml',a.gene_axis]},
                script_sha256=digest(__file__),outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
