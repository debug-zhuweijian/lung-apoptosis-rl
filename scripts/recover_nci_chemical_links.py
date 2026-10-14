"""Bind nominal NCI-TPW drug structures without assigning drug molecules to vehicles."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.link_endpoint_chemical_identities import normalize_name
from scripts.recover_public_expression import digest
from scripts.recover_macaya_chemical_links import compare_tahoe
from lung.data.drug_identity import describe_structure
from lung.data.chemistry import build_chemical_catalog
from rdkit import Chem


def validated_catalog(resolved,properties,raw):
    props={r['CID']:r for r in properties['PropertyTable']['Properties']}
    if len(props)!=len(properties['PropertyTable']['Properties']):raise ValueError('Duplicate property CID')
    rows=[]
    for r in resolved:
        if r['name_resolution_status']!='unique_pubchem_name_and_synonyms_verified':raise ValueError('Unverified nominal identity')
        cid=r['selected_cid'];syn=[x for x in raw if x.get('InformationList',{}).get('Information',[{}])[0].get('CID')==cid]
        if len(syn)!=1 or normalize_name(r['source_name']) not in {normalize_name(n) for n in syn[0]['InformationList']['Information'][0]['Synonym']}:
            raise ValueError('Source name lacks cached synonym binding')
        p=props[cid];d=describe_structure(p.get('SMILES',p.get('IsomericSMILES')))
        if d['rdkit_inchikey']!=p['InChIKey']:raise ValueError('Structure/key disagreement requires review')
        mol=Chem.MolFromSmiles(d['canonical_full_smiles'])
        platinum=any(a.GetAtomicNum()==78 for a in mol.GetAtoms())
        # Registry consistency does not verify coordination geometry.
        geometry_review=platinum and (d['fragment_count']>1 or d['defined_stereo_count']==0)
        rows.append(dict(coordination_geometry_review_required=geometry_review,
                         structure_feature_use_eligible=not geometry_review,
                         source_compound_id='NCI_TPW:'+r['source_name'],drug_id='pubchem:'+str(cid),drug_name=r['source_name'],
                         canonical_smiles=d['canonical_full_smiles'],component_role='drug',nominal_structure_verified=True,
                         experimental_reagent_form_verified=False,**d))
    c=pd.DataFrame(rows)
    if c.drug_name.duplicated().any():raise ValueError('Duplicate name')
    return c


def link_samples(samples,catalog):
    if set(samples.drug_name)!=set(catalog.drug_name):raise ValueError('Sample compound coverage mismatch')
    if not samples.is_vehicle.isin([True,False]).all():raise ValueError('Invalid vehicle role')
    if not samples.is_vehicle.eq(samples.dose_nM.eq(0)).all():raise ValueError('Vehicle/dose inconsistency')
    links=samples.merge(catalog[['drug_name','drug_id','structure_feature_use_eligible']] if 'structure_feature_use_eligible' in catalog else catalog[['drug_name','drug_id']],on='drug_name',how='left',validate='many_to_one')
    links=links.rename(columns={'drug_id':'experimental_block_drug_id'})
    links['action_drug_id']=links.experimental_block_drug_id.where(~links.is_vehicle,None)
    links['nominal_action_structure_verified']=~links.is_vehicle
    links['action_structure_feature_use_eligible']=(~links.is_vehicle)&links.get('structure_feature_use_eligible',False)
    links['experimental_reagent_form_verified']=False
    links['paired_RNA_endpoint']=False
    return links


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['source','expression','tahoe','output']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    sources=json.loads((a.source/'sources.json').read_text());raw=[]
    for r in sources.values():
        p=a.source/r['path']
        if digest(p)!=r['sha256']:raise ValueError('Public response bytes changed')
        raw.append(json.loads(p.read_text()))
    resolved=json.loads((a.source/'resolved.json').read_text());props=json.loads((a.source/'properties.json').read_text())
    if props not in raw:raise ValueError('Unbound property response')
    for r in resolved:
        if any(e not in sources.values() for e in r['lookup_evidence']):raise ValueError('Unbound identity evidence')
    catalog=validated_catalog(resolved,props,raw)
    samples=pd.read_csv(a.expression/'samples.csv');links=link_samples(samples,catalog)
    chemical,fp,meta=build_chemical_catalog(catalog)
    tahoe=pd.read_parquet(a.tahoe)
    matches=compare_tahoe(catalog,tahoe)
    matches['tahoe_unspecified_stereo_count']=[describe_structure(tahoe.loc[tahoe.drug.eq(n),'canonical_smiles'].iloc[0])['unspecified_stereo_count'] for n in matches.tahoe_drug]
    matches['source_unspecified_stereo_count']=[catalog.loc[catalog.source_compound_id.eq(n),'unspecified_stereo_count'].iloc[0] for n in matches.source_compound_id]
    matches['stereo_limit']=np.where(matches.tahoe_unspecified_stereo_count.gt(0),'Tahoe stereo omitted; not evidence of opposite isomer','see structure relation')
    a.output.mkdir(parents=True)
    catalog.to_parquet(a.output/'nominal_structures.parquet',index=False)
    links.to_csv(a.output/'sample_structure_links.csv',index=False)
    chemical.to_csv(a.output/'chemical_catalog.csv',index=False);np.save(a.output/'fingerprints.npy',fp)
    matches.to_csv(a.output/'tahoe_matches.csv',index=False)
    report=dict(nominal_drugs=len(catalog),sample_rows=len(links),coordination_review_drugs=catalog.loc[catalog.coordination_geometry_review_required,'drug_name'].tolist(),
                treated_rows=int((~links.is_vehicle).sum()),
                controls_without_action_drug=int(links.action_drug_id.isna().sum()),fingerprint=meta,
                tahoe_eligible_drugs=matches.loc[matches.nominal_full_form_eligible,'source_compound_id'].unique().tolist(),
                independent_endpoint_link=False,experimental_lots_verified=False,full_research_qualified=False,
                source_hashes={p.name:digest(p) for p in a.source.glob('*.json')},
                expression_manifest_sha256=digest(a.expression/'manifest.json'),tahoe_sha256=digest(a.tahoe),
                script_sha256=digest(__file__),outputs={p.name:digest(p) for p in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
