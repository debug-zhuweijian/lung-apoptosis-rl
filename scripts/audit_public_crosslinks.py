"""Audit nominal chemistry and cross-source condition coverage on recovered data.

The output includes positive AND rejected joins, without fitting models or
promoting metadata-only wells to observed expression/function samples.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from lung.data.chemistry import build_chemical_catalog
from lung.data.public_source_recovery import (compare_full_structures,file_sha256,verify_source_lock,
                                             validate_function_observations, load_expression_packet, sciplex_nominal_dose_sidecar)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--followup-root',type=Path,required=True)
    parser.add_argument('--expression-root',type=Path,required=True)
    parser.add_argument('--function-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve()
    for path,lock in [(a.source_root,'public_alignment_sources_v1.json'),(a.followup_root,'public_alignment_followup_v1.json')]:
        verify_source_lock(path,json.loads((ROOT/'configs'/lock).read_text())['sources'])
    if out.exists():raise FileExistsError('Choose a new cross-source audit namespace')
    out.mkdir(parents=True)
    chemistry=[]
    for path in sorted((a.followup_root/'chemistry').glob('*.json')):
        values=json.loads(path.read_text())['PropertyTable']['Properties']
        if len(values)!=1:raise ValueError('Nominal name lookup did not resolve uniquely')
        p=values[0];name=path.stem.replace('_',' ')
        if not p.get('SMILES'):raise ValueError('Full/isomeric SMILES missing; no connectivity fallback')
        chemistry.append(dict(drug_id='pubchem:'+str(p['CID']),drug_name=name,canonical_smiles=p['SMILES'],
            pubchem_cid=p['CID'],pubchem_inchikey=p['InChIKey'],molecular_weight_g_mol=p['MolecularWeight'],
            source_sha256=file_sha256(path),source_name_lookup=name,
            material_lot_verified=False,coordination_geometry_certified=False if '[Pt' in p['SMILES'] else None))
    original=pd.DataFrame(chemistry)
    cat,fp,settings=build_chemical_catalog(original[['drug_id','drug_name','canonical_smiles']])
    cat=cat.merge(original.drop(columns='canonical_smiles'),on=['drug_id','drug_name'],validate='one_to_one')
    cat.to_csv(out/'nominal_chemical_catalog.csv',index=False)
    np.savez_compressed(out/'nominal_fingerprints.npz',fingerprints=fp,drug_ids=cat.drug_id.to_numpy(dtype=str))
    with np.load(out/'nominal_fingerprints.npz',allow_pickle=False) as x:np.testing.assert_array_equal(x['fingerprints'],fp)
    # Recompute independently from the saved molecular graph via the same fixed
    # representation contract; this certifies feature bytes, not reagent identity.
    rebuilt,fp2,_=build_chemical_catalog(pd.DataFrame({'drug_id':cat.drug_id,'drug_name':cat.drug_name,
                                                      'canonical_smiles':cat.canonical_full_smiles}))
    np.testing.assert_array_equal(fp,fp2)
    byname=original.set_index('drug_name').to_dict('index')
    tahoe=pd.read_parquet(a.source_root/'tahoe/drug_metadata.parquet')
    sc=pd.read_csv(a.followup_root/'sciplex/A549_MCF7_K562_hashTable_metadata.txt.gz',sep='\t')
    a549=sc.loc[sc.cell_type.eq('A549')].copy()
    if a549.duplicated(['plate_oligo','well_oligo']).any():raise ValueError('Duplicate physical sciPlex well')
    a549['condition_id']='GSM4150378:'+a549.plate_oligo+':'+a549.well_oligo
    a549['metadata_only']=True; a549['RNA_readback_verified']=False
    # Primary-paper interpretation is separate from the absent source unit field.
    a549=sciplex_nominal_dose_sidecar(a549)
    a549['role']='unassigned'
    a549.to_csv(out/'sciplex_A549_original_wells.csv',index=False)
    comparisons=[]
    def compare(source,pos,name,smiles,reference,extra=None):
        r={'source':source,'source_position':pos,'source_drug_name':name,'nominal_reference_name':reference,
           'source_smiles':smiles,'reference_cid':byname[reference]['pubchem_cid']}
        try:r.update(compare_full_structures(smiles,byname[reference]['canonical_smiles']))
        except ValueError as e:r.update(full_graph_equal=False,connectivity_equal=False,error=str(e))
        r['eligible_as_exact_structure_join']=r['full_graph_equal']
        r['material_identity_verified']=False
        r.update(extra or {});comparisons.append(r)
    # Explicit aliases, not fuzzy joins. Salt/solvate labels remain separate.
    aliases={'Trametinib':'Trametinib','Trametinib (GSK1120212)':'Trametinib',
             'Oxaliplatin':'Oxaliplatin','Resveratrol':'Resveratrol'}
    for i,r in tahoe.iterrows():
        n=r['drug']
        if n in aliases:compare('Tahoe',f'drug_metadata.parquet:row{i}',n,r['canonical_smiles'],aliases[n])
    for i,r in a549[['name','SMILES']].drop_duplicates().iterrows():
        if r['name'] in aliases:compare('sciPlex',f'hashTable_metadata:row{i}',r['name'],r['SMILES'],aliases[r['name']])
    # Check cited supplementary metadata before considering it as an identity bridge.
    table=pd.read_excel(a.followup_root/'papers/42256_2026_1192_MOESM4_ESM.xlsx',sheet_name='Supplementary Table 7',header=1)
    seqaliases={'Trametinib (GSK1120212)':'Trametinib','midostaurin':'Midostaurin',
                'cisplatin':'Cisplatin','Oxaliplatin':'Oxaliplatin'}
    for i,r in table.iterrows():
        name=str(r['Name']).strip()
        if name in seqaliases:compare('SequenTx_Table7',f'Supplementary Table 7!row{i+3}',name,r['SMILES'],seqaliases[name],
                {'source_table_not_certified_as_executable_model_input':True})
    pd.DataFrame(comparisons).to_csv(out/'structure_join_audit.csv',index=False)
    # Link each functional observation to its own complete list of active components.
    # No Cartesian expansion of an observation into multiple Y training rows.
    component_links=[]
    for name in ['Macaya2023','Sarin2017_2018']:
        folder=a.function_root/name
        manifest=json.loads((folder/'functional_receipt.json').read_text())
        for filename in ['functional_observations.csv','condition_level_links.csv']:
            if file_sha256(folder/filename)!=manifest['files'][filename]:
                raise ValueError('Functional packet checksum mismatch')
        f=folder/'functional_observations.csv';obs=pd.read_csv(f);validate_function_observations(obs)
        for r in obs.to_dict('records'):
            drugs=json.loads(r['agents_json']) if name=='Macaya2023' else [{'name':r['drug'],'dose':r['dose_uM'],'unit':'uM'}]
            resolved=[dict(d,nominal_drug_id='pubchem:'+str(byname[d['name']]['pubchem_cid']),
                           material_identity_verified=False) for d in drugs]
            component_links.append(dict(observation_id=r['observation_id'],source_id=name,
                cell_line=r['cell_line'],endpoint_domain=r['endpoint_domain'],role='unassigned',
                components_json=json.dumps(resolved,sort_keys=True),
                measured_endpoint_status=r['value_status'],control_material_verified=False if not drugs else None))
    pd.DataFrame(component_links).to_csv(out/'functional_nominal_component_links.csv',index=False)
    # The ordered phases remain in the expression packet; join only drug labels,
    # retaining unobserved HM97662 structure as unresolved (no zero fingerprint).
    protocol_links=[]
    for name in ['GSE306207','GSE333828','GSE161218']:
        domains={'GSE306207':'bulk_TPM','GSE333828':'bulk_estimated_counts','GSE161218':'bulk_raw_counts'}
        backgrounds={'GSE306207':'A375','GSE333828':'A549','GSE161218':'H1792'}
        samples=load_expression_packet(a.expression_root/name,expected_scale=domains[name],
            expected_background=backgrounds[name])['samples']
        for key,group in samples.groupby('condition_id',sort=False):
            if group.ordered_protocol_json.nunique()!=1:raise ValueError('Replicate protocol mismatch')
            protocol=json.loads(group.ordered_protocol_json.iloc[0])
            for event in protocol:
                drug=event.get('drug')
                if drug:
                    event['nominal_drug_id']='pubchem:'+str(byname[drug]['pubchem_cid']) if drug in byname else None
                    event['structure_status']='nominal_lookup_not_material_verified' if drug in byname else 'unresolved'
            protocol_links.append(dict(condition_id=key,source_id=name,cell_line=group.cell_line.iloc[0],
                sample_ids_json=json.dumps(group.sample_id.tolist()),ordered_protocol_json=json.dumps(protocol,sort_keys=True),
                role='unassigned',RNA_values_verified=True,paired_function_replicates=False))
    pd.DataFrame(protocol_links).to_csv(out/'RNA_protocol_nominal_component_links.csv',index=False)
    bothe=[]
    for f in sorted((a.followup_root/'bothe').glob('DEseq2*.txt')):
        frame=pd.read_csv(f,sep='\t')
        bothe.append({'file':f.name,'sha256':file_sha256(f),'rows':len(frame),'columns':frame.columns.tolist(),
            'data_type':'published_contrast_statistics_not_individual_counts','sample_count_recovered':0,
            'can_reconstruct_12_sample_matrix':False})
    receipt={'nominal_chemicals':len(cat),'fingerprint_shape':list(fp.shape),'fingerprint_settings':settings,
        'fingerprint_bytes_recomputed_equal':True,'sciplex_A549_planned_wells':len(a549),
        'sciplex_times_as_recorded':sorted(a549.time_point.unique().tolist()),
        'sciplex_dose_unit_in_file':None,'sciplex_nominal_unit_from_primary_paper':'nM',
        'sciplex_dose_um_grid':sorted(a549.dose_um.unique().tolist()),'sciplex_expression_read':False,
        'nominal_structure_comparisons':comparisons,'bothe_public_contrasts':bothe,
        'functional_observation_component_links':len(component_links),
        'RNA_condition_protocol_links':len(protocol_links),'whole_project_qualified':False,
        'files':{f.name:file_sha256(f) for f in out.iterdir() if f.is_file()}}
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['nominal_structure_comparisons','files']},indent=2))
if __name__=='__main__':main()
