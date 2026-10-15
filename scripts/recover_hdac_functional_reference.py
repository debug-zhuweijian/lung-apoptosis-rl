"""Recover actual A549 viability and author cross-modal statistics, not Annexin labels."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

SHA='45c8aaebf80a89619a6320eb9415d7f22afd929ea4aec5e078f716c369f767fd'
DRUGS={'Abexinostat','Belinostat','CUDC101','Ralimetinib','TSA','Panobinostat','Vorinostat'}
DOSES={0,.001,.01,.1,1,2.5,10}


def extract_viability(frame):
    required=['Line','Drug','Concentration','Ratio 1','Ratio 2','Ratio 3','Average','SD','SE']
    if list(frame.columns)!=required:raise ValueError('Source columns changed')
    a=frame.loc[frame.Line.eq('A549')].copy()
    if len(a)!=49 or set(a.Drug)!=DRUGS or a.duplicated(['Drug','Concentration']).any():
        raise ValueError('Complete unique A549 source grid required')
    rows=[]
    for index,r in a.iterrows():
        if r.Concentration not in DOSES:raise ValueError('Unreviewed source concentration')
        vals=r[['Ratio 1','Ratio 2','Ratio 3']].to_numpy(float)
        if not np.isfinite(vals).all() or (vals<0).any():raise ValueError('Invalid source ratios')
        for j,v in enumerate(vals):
            rows.append(dict(source_sheet='Figure 1B',source_excel_row=int(index)+2,source_column=['D','E','F'][j],
                cell_background='A549',drug_source_label=r.Drug,dose_uM=float(r.Concentration),
                source_numeric_index=j+1,value=float(v),unit='CellTiterGlo_treated_over_DMSO',
                control_normalization_anchor=bool(r.Concentration==0),independent_control_identity=None,
                cross_condition_repeat_identity=None,annexin_label=False,apoptosis_fraction=None,
                nominal_exposure_h=24,clock_qualification='caption 24h; culture Methods repeats 24h incubation wording',
                source_reported_average=r.Average,source_reported_SD=r.SD,source_reported_SE=r.SE))
    if not a.groupby('Drug').Concentration.apply(lambda x:set(x)==DOSES).all():
        raise ValueError('Missing drug dose')
    return pd.DataFrame(rows)


def extract_crossmodal(frame):
    if list(frame.columns)!=['Gene','Drug','Uniprot','Line','Protein log2FC','RNA normalized_effect']:
        raise ValueError('Author statistic schema changed')
    if not frame.Line.eq('A549').all() or frame.duplicated(['Drug','Uniprot']).any():
        raise ValueError('Background or protein identity ambiguity')
    if set(frame.Drug)!={'Abexinostat','Belinostat','Panobinostat','TSA'} or not frame.groupby('Drug').size().eq(5560).all():
        raise ValueError('Author matched feature domain changed')
    if not np.isfinite(frame[['Protein log2FC','RNA normalized_effect']].to_numpy(float)).all():
        raise ValueError('Missing or nonfinite author statistic')
    return frame.copy(),{str(k):float(g['Protein log2FC'].corr(g['RNA normalized_effect'])) for k,g in frame.groupby('Drug')}


def recover(source,output):
    if output.exists():raise FileExistsError('New output directory required')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SHA:raise ValueError('Reviewed source workbook changed')
    obs=extract_viability(pd.read_excel(source,sheet_name='Figure 1B'))
    effects,corr=extract_crossmodal(pd.read_excel(source,sheet_name='Figure 5A'))
    output.mkdir(parents=True)
    obs.to_csv(output/'A549_viability_source.csv',index=False)
    effects.to_parquet(output/'author_RNA_protein_effects.parquet',index=False)
    np.testing.assert_equal(pd.read_parquet(output/'author_RNA_protein_effects.parquet').to_numpy(),effects.to_numpy())
    np.testing.assert_equal(pd.read_csv(output/'A549_viability_source.csv').value.to_numpy(),obs.value.to_numpy())
    stats=dict(source_doi='10.1038/s42003-025-09455-0',source_workbook_sha256=SHA,
        source_conditions=49,active_conditions=42,active_ratio_values=126,normalization_anchor_values=21,
        normalization_anchors_are_independent_controls=False,source_values_above_one=int((obs.value>1).sum()),
        source_summary_columns_recomputed=False,source_SE_definition_verified=False,
        annexin_labels=False,clock_fully_qualified=False,
        RNA_protein_statistic_rows=len(effects),RNA_protein_correlations=corr,
        RNA_quantity='published dose-regression normalized coefficient from sci-Plex; not expression count or dose-specific log fold change',
        RNA_source_independent_from_sciPlex=False,RNA_fate_pairing=False,
        protein_quantity='bulk protein log2 fold change against DMSO; UniProt isoform IDs retained',
        replicate_identity_between_modalities=False,initial_state_increment_verified=False,
        full_research_feasibility_proven=False,
        files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(stats,indent=2)+'\n');return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workbook',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.workbook,a.output),indent=2))
