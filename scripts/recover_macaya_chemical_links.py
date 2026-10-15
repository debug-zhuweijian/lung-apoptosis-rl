"""Bind recovered Macaya endpoints to nominal structures and audited Tahoe matches.

Offline inputs are public-source cache outputs, not evidence of experimental lots.
Do not infer RNA samples, dose/time pairing or drug coverage from name matches.
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
from lung.data.chemistry import build_chemical_catalog
from lung.data.drug_identity import describe_structure,compare_structures,material_form_review_reason
from lung.data.endpoint_alignment import load_alignment
from scripts.recover_public_expression import digest


def resolve_verified_rows(resolved,properties):
    props={r['CID']:r for r in properties['PropertyTable']['Properties']}
    rows=[]
    if len(props)!=len(properties['PropertyTable']['Properties']):raise ValueError('Duplicate property CID')
    for r in resolved:
        if r['name_resolution_status']!='unique_pubchem_name_and_synonyms_verified':raise ValueError('Unverified nominal name')
        p=props[r['selected_cid']];smiles=p.get('SMILES',p.get('IsomericSMILES'))
        d=describe_structure(smiles)
        if d['rdkit_inchikey']!=p['InChIKey']:raise ValueError('Structure/key mismatch')
        rows.append(dict(source_compound_id='Macaya2023:'+r['source_name'],drug_id='pubchem:'+str(p['CID']),
                         drug_name=r['source_name'],canonical_smiles=d['canonical_full_smiles'],
                         component_role='drug',nominal_structure_verified=True,
                         experimental_reagent_form_verified=False,**d))
    if {r['drug_name'] for r in rows}!={'Trametinib','Lestaurtinib','Midostaurin'} or len(rows)!=3:
        raise ValueError('Unexpected core compound coverage')
    return pd.DataFrame(rows)


def compare_tahoe(catalog,tahoe):
    links=[]
    for r in catalog.to_dict('records'):
        for t in tahoe.to_dict('records'):
            relation=compare_structures(r['canonical_full_smiles'],t['canonical_smiles'])
            if relation in ['different_structures','unresolved']:continue
            reason=material_form_review_reason(t['drug'],describe_structure(t['canonical_smiles']))
            links.append(dict(source_compound_id=r['source_compound_id'],tahoe_drug=t['drug'],
                              structure_relation=relation,material_form_review_reason=reason,
                              nominal_full_form_eligible=relation=='same_full_form' and reason is None,
                              dose_time_background_matched=False,paired_RNA_endpoint=False))
    return pd.DataFrame(links)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','endpoint','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    # Verify original fetched responses, then bind derived summaries back to them.
    sources=json.loads((a.source/'sources.json').read_text())
    raw=[]
    for record in sources.values():
        p=a.source/record['path']
        if digest(p)!=record['sha256']:raise ValueError('Public response cache changed')
        raw.append(json.loads(p.read_text()))
    resolved=json.loads((a.source/'resolved.json').read_text())
    for r in resolved:
        for e in r['lookup_evidence']:
            if e not in list(sources.values()):raise ValueError('Missing name evidence')
        syn=[v for v in raw if v.get('InformationList',{}).get('Information',[{}])[0].get('CID')==r['selected_cid']]
        if len(syn)!=1 or r['source_name'].casefold() not in {x.casefold() for x in syn[0]['InformationList']['Information'][0]['Synonym']}:
            raise ValueError('Exact source name not in cached synonyms')
    properties=json.loads((a.source/'properties.json').read_text())
    if properties not in raw:raise ValueError('Derived properties not bound to source response')
    ts=json.loads((a.source/'tahoe_source.json').read_text())
    if digest(a.source/'tahoe_drug_metadata.parquet')!=ts['sha256']:raise ValueError('Tahoe source changed')
    catalog=resolve_verified_rows(resolved,properties)
    endpoint=load_alignment(a.endpoint,ROOT)
    links=endpoint['interventions'].merge(catalog,left_on='compound_id',right_on='source_compound_id',how='left',validate='many_to_one')
    if links.drug_id.isna().any() or len(links)!=len(endpoint['interventions']):raise ValueError('Unresolved intervention')
    chemical,fp,meta=build_chemical_catalog(catalog)
    matches=compare_tahoe(catalog,pd.read_parquet(a.source/'tahoe_drug_metadata.parquet'))
    a.output.mkdir(parents=True)
    catalog.to_parquet(a.output/'nominal_structures.parquet',index=False)
    links.to_parquet(a.output/'intervention_structure_links.parquet',index=False)
    chemical.to_csv(a.output/'chemical_catalog.csv',index=False);np.save(a.output/'fingerprints.npy',fp)
    matches.to_csv(a.output/'tahoe_matches.csv',index=False)
    receipt=dict(nominal_compounds=len(catalog),intervention_rows=len(links),fingerprint=meta,
                 tahoe_nominal_eligible_names=matches.loc[matches.nominal_full_form_eligible,'tahoe_drug'].tolist(),
                 matched_RNA_endpoint=False,experimental_lots_verified=False,full_research_qualified=False,
                 endpoint_manifest_sha256=digest(a.endpoint/'manifest.json'),tahoe_source=ts,
                 source_hashes={p.name:digest(p) for p in a.source.glob('*.json')},
                 outputs={p.name:digest(p) for p in a.output.iterdir()},script_sha256=digest(__file__))
    (a.output/'manifest.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
