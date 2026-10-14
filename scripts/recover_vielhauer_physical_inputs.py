"""Bind re-read nominal structures to actual 20-condition physical protocols.

Uses the existing physical protocol tensor/conditioner. No frozen source, weights,
reward or split is replaced. The optional conditioner execution is untrained QA.
"""
import argparse,json,sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.build_vielhauer_duration_challenge import load_challenge
from scripts.recover_public_expression import digest
from lung.data.chemistry import build_chemical_catalog
from lung.data.physical_units import canonical_molar_protocol,UNIT_TRANSFORM_ID
from lung.models.full_protocol import (FullPhysicalProtocolBatch,FullProtocolConditioner,EVENT_TYPES,DOSE_UNITS,SCALAR_FIELDS)
EXPECTED={'bocnho':(24755111,'XTLVCBPHMQCFAN-JOCHJYFZSA-N','C36H32ClN5O5'),
          'cbi_indole2':(11016947,'OMTPLEBPJORDFA-LJQANCHMSA-N','C31H23ClN4O3')}
FIGURE_SHA='ebdaaba83c066b28f8b038984a16c1705b782c553832863ca53335d878b4112f'
PRIMARY_SHA='b8f28d05844d5f297b19e94bc2c58ab7b37e86d5d6d449bd0198a7e80ce7d492'


def validate_structure(drug,property_record):
    cid,key,formula=EXPECTED[drug];p=property_record
    if p['CID']!=cid or p['InChIKey']!=key or p['MolecularFormula']!=formula:raise ValueError('Reviewed nominal identity changed')
    m=Chem.MolFromSmiles(p['SMILES'])
    if m is None or Chem.MolToInchiKey(m)!=key or rdMolDescriptors.CalcMolFormula(m)!=formula:raise ValueError('Full structure differs from reviewed identity')
    centers=Chem.FindMolChiralCenters(m,includeUnassigned=True)
    if len(centers)!=1 or centers[0][1]!='S' or len(Chem.GetMolFrags(m))!=1 or Chem.GetFormalCharge(m)!=0:
        raise ValueError('Reviewed stereo/material form differs')
    return Chem.MolToSmiles(m,isomericSmiles=True)


def physical_batch(conditions,catalog,fingerprints):
    if len(conditions)!=20 or len({r['condition_id'] for r in conditions})!=20:raise ValueError('Full source candidate domain required')
    lookup=catalog.set_index('drug_id').fingerprint_row.to_dict();n=20;s=4
    fp=torch.zeros(n,s,2048);fk=torch.zeros(n,s,dtype=torch.bool);vehicle=torch.zeros_like(fk)
    kind=torch.zeros(n,s,dtype=torch.long);unit=torch.zeros_like(kind)
    values=torch.zeros(n,s,len(SCALAR_FIELDS));known=torch.zeros_like(values,dtype=torch.bool);lengths=[];grid=set();records=[]
    for i,r in enumerate(conditions):
        drug=r['drug_id'];dose=r['dose_nM'];duration=r['exposure_duration_h'];endpoint=r['terminal_time_h']
        if drug not in EXPECTED or dose not in [.05,.5,10.,40.,1000.] or duration not in [6.,24.] or endpoint!=24:
            raise ValueError('Unsupported source drug, dose or clock')
        key=(drug,dose,duration)
        if key in grid:raise ValueError('Duplicate physical candidate')
        grid.add(key)
        if r['dose_uM']!=dose/1000:raise ValueError('Dose unit correspondence changed')
        events=[dict(event='drug_interval',unit='nM',dose=dose,start_hours=0,end_hours=duration,exposure_hours=duration)]
        if duration==6:
            if r['washout_at_h']!=6 or r['drug_free_recovery_h']!=18 or r['protocol_arm']!='pulse_6h_washout_recovery18h':
                raise ValueError('Pulse source history changed')
            events.extend([dict(event='wash',start_hours=6,end_hours=6),
                dict(event='media_change',start_hours=6,end_hours=6,media_change=1),
                dict(event='recovery',start_hours=6,end_hours=24,recovery_hours=18)])
        elif r['washout_at_h'] is not None or r['drug_free_recovery_h']!=0 or r['protocol_arm']!='continuous_24h':
            raise ValueError('Continuous source history changed')
        lengths.append(len(events));fp[i,0]=torch.tensor(fingerprints[lookup[drug]],dtype=torch.float32);fk[i,0]=True
        for j,e in enumerate(events):
            kind[i,j]=EVENT_TYPES.index(e['event']);unit[i,j]=DOSE_UNITS.index(e.get('unit','unknown'))
            for k,f in enumerate(SCALAR_FIELDS):
                if f in e:values[i,j,k]=e[f];known[i,j,k]=True
        records.append(dict(condition_id=r['condition_id'],source_material_id=r['source_material_identity'],drug_id=drug,
            fingerprint_row=int(lookup[drug]),endpoint_hours=24,events=events,
            wash_count=None,prior_drug_retention=None,initial_RNA_reference=None,
            floating_cells_retained_at_washout=None,nominal_structure_only=True))
    b=FullPhysicalProtocolBatch(fp,fk,vehicle,kind,unit,values,known,torch.tensor(lengths),torch.full((n,),24.),torch.ones(n,dtype=torch.bool));b.validate()
    return b,records


def recover(chemistry,challenge,output):
    if output.exists():raise FileExistsError('New output directory required')
    if digest(chemistry/'fig1.jpg')!=FIGURE_SHA or digest(chemistry/'PMC2974002.1.xml')!=PRIMARY_SHA:
        raise ValueError('Personally reviewed primary identity sources changed')
    props={drug:json.loads((chemistry/f'PubChem_{spec[0]}.json').read_text())['PropertyTable']['Properties'][0] for drug,spec in EXPECTED.items()}
    entries=[dict(drug_id=drug,drug_name=drug,canonical_smiles=validate_structure(drug,p)) for drug,p in props.items()]
    catalog,fp,chemistry_report=build_chemical_catalog(pd.DataFrame(entries))
    prepared=load_challenge(challenge);conditions=prepared['active_conditions'];source_batch,records=physical_batch(conditions,catalog,fp)
    batch=canonical_molar_protocol(source_batch)
    output.mkdir(parents=True);catalog.to_csv(output/'chemical_catalog.csv',index=False);np.save(output/'fingerprints.npy',fp)
    source_arrays={f.name:getattr(source_batch,f.name).numpy() for f in fields(source_batch)}
    np.savez_compressed(output/'source_unit_physical_batch.npz',**source_arrays)
    arrays={f.name:getattr(batch,f.name).numpy() for f in fields(batch)};np.savez_compressed(output/'physical_protocol_batch.npz',**arrays)
    with np.load(output/'physical_protocol_batch.npz',allow_pickle=False) as z:
        for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
        restored=FullPhysicalProtocolBatch(**{key:torch.from_numpy(z[key]) for key in arrays});restored.validate()
    (output/'input_records.json').write_text(json.dumps(records,indent=2)+'\n')
    # Actual production conditioner, small untrained dimensions: structural QA only.
    with torch.random.fork_rng():
        torch.manual_seed(14);model=FullProtocolConditioner(SimpleNamespace(d_model=8,context_dim=4));model.eval()
        with torch.no_grad():out=model(torch.zeros(20,8),restored)
    if not torch.isfinite(out).all():raise ValueError('Conditioner failed on actual physical records')
    torch.testing.assert_close(batch.values,restored.values)
    conditions=pd.DataFrame(conditions);differences=[]
    for (drug,dose),group in conditions.groupby(['drug_id','dose_nM']):
        indices=group.index.tolist();differences.append(bool(not torch.equal(out[indices[0]],out[indices[1]])))
    if not all(differences):raise ValueError('Pulse and continuous histories collapsed in QA')
    report=dict(source='Vielhauer2013',actual_conditions=20,actual_nominal_structures=2,
        dose_unit_transform=UNIT_TRANSFORM_ID,model_dose_unit='uM',source_dose_unit='nM',
        source_unit_tensor_retained=True,existing_weights_compatible_without_requalification=False,
        fingerprint_on_bits=fp.sum(axis=1).astype(int).tolist(),fingerprint_drug_order=catalog.drug_id.tolist(),
        fingerprint_shape=list(fp.shape),tensor_shape=list(batch.fingerprint.shape),actual_events=int(batch.lengths.sum()),
        all_observation_clocks_24h=True,unknown_wash_count_preserved=bool(not batch.known[:,:,4].any()),
        unknown_retention_preserved=bool(not batch.known[:,:,5].any()),
        conditioner_actual_class='lung.models.full_protocol.FullProtocolConditioner',
        conditioner_untrained_QA_only=True,conditioner_output_shape=list(out.shape),pulse_continuous_distinct_pairs=sum(differences),
        matched_initial_RNA=False,matched_response_weights=False,formal_training=False,
        final_endpoint_effect_evaluated=False,full_research_qualification=False,
        historical_batch_composition_or_ee_verified=False,original_frozen_inputs_modified=False,
        chemical_catalog=chemistry_report,primary_2013_figure_sha256=FIGURE_SHA,primary_2010_identity_sha256=PRIMARY_SHA,
        pubchem_properties_sha256={drug:digest(chemistry/f'PubChem_{spec[0]}.json') for drug,spec in EXPECTED.items()},
        challenge_manifest_sha256=digest(challenge/'manifest.json'),script_sha256=digest(__file__),
        actual_consumer_sha256=digest(ROOT/'src/lung/models/full_protocol.py'),
        dose_transform_code_sha256=digest(ROOT/'src/lung/data/physical_units.py'),outputs={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['chemistry','challenge','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.chemistry,a.challenge,a.output),indent=2))
