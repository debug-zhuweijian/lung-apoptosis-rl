"""Build qualified public structure/dose links, retaining separate assay units.

No name-only match becomes a biological pair; no endpoint replicate is expanded
into artificial RNA replicate pairs. Inference and formal model fitting are absent.
"""
import argparse,ast,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.link_endpoint_chemical_identities import normalize_name
from scripts.recover_public_expression import digest
from lung.data.drug_identity import describe_structure,compare_structures,material_form_review_reason,review_stereo_omission
from lung.data.chemistry import build_chemical_catalog


def primary_name(name):
    # Only known alternative identifiers may be removed; material suffixes stay.
    parts=re.findall(r'\(([^)]*)\)',name)
    if any(re.search(r'hydrat|chloride|bromide|sulfate|acetate|citrate|maleate|mesylate|salt|solvat|sodium|potassium|calcium|%',p,re.I) for p in parts) or any(re.fullmatch(r'[RSrs+−-]+|HCl|HBr|cis|trans|rac|racemate',p.strip(),re.I) for p in parts):
        return name.strip()
    return re.sub(r'\([^)]*\)','',name).strip()


def validate_name_resolution(candidates,resolutions):
    by_name={r['source_name']:r for r in resolutions}
    if len(by_name)!=len(resolutions) or set(by_name)!={r['source_name'] for r in candidates}:
        raise ValueError('Independent name-query coverage differs')
    for c in candidates:
        r=by_name[c['source_name']]
        if r['queried_primary_name']!=primary_name(c['source_name']) or r['status']!=200 or r['cids']!=[c['cid']]:
            raise ValueError('Independent source-name CID is unresolved or conflicting')


def resolve(candidates,synonyms,properties,tahoe,complete_omitted_stereo=False,independent_name_resolution=None):
    if complete_omitted_stereo:
        if independent_name_resolution is None:raise ValueError('Independent name evidence required for nominal completion')
        validate_name_resolution(candidates,independent_name_resolution)
    syn={r['CID']:r['Synonym'] for r in synonyms['InformationList']['Information']}
    props={r['CID']:r for r in properties['PropertyTable']['Properties']}
    if len(props)!=len(properties['PropertyTable']['Properties']) or tahoe.drug.duplicated().any():raise ValueError('Duplicate identities')
    result=[]
    for row in candidates:
        cid=row['cid'];name=row['source_name'];key=normalize_name(primary_name(name))
        if key not in {normalize_name(x) for x in syn[cid]}:raise ValueError('Source material/name not verified by synonyms')
        t=tahoe.loc[tahoe.drug.eq(row['tahoe_name'])]
        if len(t)!=1 or int(t.iloc[0].pubchem_cid)!=cid:raise ValueError('Tahoe CID changed')
        alias_parts=[a.strip() for block in re.findall(r'\(([^)]*)\)',name) for a in block.split(',')]
        unverified_aliases=[a for a in alias_parts if normalize_name(a) not in {normalize_name(x) for x in syn[cid]}]
        prop=props[cid];sm=prop.get('SMILES',prop.get('IsomericSMILES'));a=describe_structure(sm);b=describe_structure(t.iloc[0].canonical_smiles)
        if a['rdkit_inchikey']!=prop['InChIKey']:raise ValueError('Registry structure/key mismatch')
        relation=compare_structures(sm,t.iloc[0].canonical_smiles)
        metal=any(x.GetAtomicNum()==78 for x in Chem.MolFromSmiles(sm).GetAtoms())
        reason=material_form_review_reason(name,a) or material_form_review_reason(row['tahoe_name'],b)
        review=review_stereo_omission(t.iloc[0].canonical_smiles,sm)
        repaired=bool(complete_omitted_stereo and review['eligible'] and reason is None and not metal and not unverified_aliases)
        original_eligible=relation=='same_full_form' and reason is None and not metal and not unverified_aliases
        result.append(dict(source_compound_name=name,tahoe_drug=row['tahoe_name'],drug_id=f'pubchem:{cid}',
            source_compound_id='Inde2020:'+name,drug_name=name,component_role='drug',canonical_smiles=a['canonical_full_smiles'],
            nominal_primary_name_verified=True,full_source_label_verified=not unverified_aliases,
            unverified_parenthetical_aliases=json.dumps(unverified_aliases),source_primary_name=primary_name(name),structure_relation=relation,
            source_unspecified_stereo=a['unspecified_stereo_count'],tahoe_unspecified_stereo=b['unspecified_stereo_count'],
            material_review_reason=reason,coordination_review_required=metal,
            strict_structure_link_eligible=original_eligible,
            nominal_reference_link_eligible=bool(original_eligible or repaired),
            tahoe_original_smiles=t.iloc[0].canonical_smiles,
            tahoe_derived_reference_smiles=a['canonical_full_smiles'] if repaired else t.iloc[0].canonical_smiles,
            omitted_stereo_reference_completion_used=repaired,stereo_completion_review=review['reason'],
            remaining_reference_unspecified_stereo=a['unspecified_stereo_count'],
            original_Tahoe_source_record_modified=False,experimental_stereochemistry_verified=False,
            historical_reagent_form_verified=False,**a))
    out=pd.DataFrame(result)
    if out.source_compound_name.duplicated().any() or out.drug_id.duplicated().any():raise ValueError('Ambiguous source candidate')
    return out


def parse_single_dose(text,expected_drug):
    value=ast.literal_eval(text)
    if not isinstance(value,list) or len(value)!=1:raise ValueError('Single compound condition required')
    name,dose,unit=value[0]
    if normalize_name(name)!=normalize_name(expected_drug):raise ValueError('Condition drug mismatch')
    factors={'uM':1.,'nM':.001,'mM':1000.}
    if unit not in factors or isinstance(dose,bool):raise ValueError('Explicit molar concentration required')
    dose=float(dose)*factors[unit]
    if not np.isfinite(dose) or dose<=0:raise ValueError('Positive dose required')
    return dose


def bind(catalog,samples,observations):
    if samples['sample'].duplicated().any():raise ValueError('Duplicate Tahoe sample')
    if observations[['replicate','source_column_1based']].duplicated().any():raise ValueError('Duplicate functional unit')
    f=observations.loc[observations.source_compound_name.isin(catalog.source_compound_name)].copy()
    if not f.time_h.eq(24).all() or not f.library_dose_uM.eq(5).all() or not f.arm.eq('single_agent').all():raise ValueError('Functional clock/dose/design changed')
    f['functional_group_id']='Inde2020:24h:5uM:'+f.source_compound_name
    links=[]
    for c in catalog.to_dict('records'):
        v=f.loc[f.source_compound_name.eq(c['source_compound_name'])]
        if sorted(v.replicate.tolist())!=[1,2,3]:raise ValueError('Independent experiment coverage changed')
        for t in samples.loc[samples.drug.eq(c['tahoe_drug'])].to_dict('records'):
            dose=parse_single_dose(t['drugname_drugconc'],t['drug'])
            if not np.isclose(dose,5,rtol=0,atol=1e-12):continue
            links.append(dict(source_compound_name=c['source_compound_name'],drug_id=c['drug_id'],tahoe_drug=c['tahoe_drug'],
                tahoe_sample=t['sample'],tahoe_plate=t['plate'],dose_uM=dose,
                functional_group_id=v.functional_group_id.iloc[0],functional_observation_count=len(v),
                strict_structure_link_eligible=c['strict_structure_link_eligible'],
                nominal_reference_link_eligible=c.get('nominal_reference_link_eligible',c['strict_structure_link_eligible']),
                a549_cells_in_sample_verified=False,paired_RNA_functional_replicates=False,
                matched_culture=False,functional_background='A549 nuclear-mKate2 reporter',
                RNA_background='Tahoe pooled 3D cell villages; A549 membership requires row verification',
                observed_endpoint='SYTOX_STACK_lethal_fraction_not_Annexin',functional_time_h=24,
                tahoe_time_h=24,tahoe_time_basis='published source Methods; not encoded in sample metadata',
                identity_level='nominal drug/dose/time candidate; not full protocol equality'))
    return pd.DataFrame(links),f


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','tahoe','functional','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--complete-omitted-stereo',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    for rec in json.loads((a.source/'sources_with_names.json').read_text()):
        if digest(a.source/rec['path'])!=rec['sha256']:raise ValueError('Public chemistry cache changed')
    for rec in json.loads((a.tahoe/'sources.json').read_text()):
        if digest(a.tahoe/rec['path'])!=rec['sha256']:raise ValueError('Public metadata cache changed')
    fm=json.loads((a.functional/'manifest.json').read_text())
    if digest(a.functional/'single_agent_24h.csv')!=fm['outputs']['single_agent_24h.csv']:raise ValueError('Functional source binding changed')
    candidates=json.loads((a.source/'name_candidates.json').read_text()); resolutions=json.loads((a.source/'name_resolution.json').read_text())
    for r in resolutions:
        actual=json.loads((a.source/r['cache_path']).read_text()).get('IdentifierList',{}).get('CID',[])
        if r['cids']!=actual:raise ValueError('Name-query result differs from cached response')
    validate_name_resolution(candidates,resolutions)
    t=pd.read_parquet(a.tahoe/'drug_metadata.parquet');s=pd.read_parquet(a.tahoe/'sample_metadata.parquet')
    c=resolve(candidates,json.loads((a.source/'synonyms.json').read_text()),json.loads((a.source/'properties.json').read_text()),t,a.complete_omitted_stereo,resolutions)
    links,f=bind(c,s,pd.read_csv(a.functional/'single_agent_24h.csv'))
    chem,fp,meta=build_chemical_catalog(c);a.output.mkdir(parents=True)
    c.to_csv(a.output/'nominal_structures.csv',index=False);links.to_csv(a.output/'condition_links.csv',index=False)
    f.to_csv(a.output/'functional_observations.csv',index=False);chem.to_csv(a.output/'chemical_catalog.csv',index=False);np.save(a.output/'fingerprints.npy',fp)
    report=dict(nominal_reference_drugs=int(c.nominal_reference_link_eligible.sum()),
        derived_stereo_completions=int(c.omitted_stereo_reference_completion_used.sum()),
        original_Tahoe_records_modified=0,nominal_drugs=len(c),strict_structure_drugs=int(c.strict_structure_link_eligible.sum()),candidate_sample_links=len(links),
        strict_structure_sample_links=int(links.strict_structure_link_eligible.sum()),functional_values=len(f),
        preserved_independent_experiments=3,no_replicate_cross_product=True,full_protocol_matches=0,
        actual_RNA_rows_recovered_by_this_script=0,complete_scientific_qualification=False,
        unmatched_library_names_are_not_proven_absent=True,all_library_drugs_resolved=False,fingerprint=meta,
        independent_source_name_queries_verified=True,source_cache_manifest_sha256=digest(a.source/'sources_with_names.json'),
        source_hashes={str(q):digest(q) for q in [a.source/'name_resolution.json',a.source/'name_candidates.json',a.source/'synonyms.json',a.source/'properties.json',a.tahoe/'sample_metadata.parquet',a.tahoe/'drug_metadata.parquet',a.functional/'single_agent_24h.csv']},
        script_sha256=digest(__file__),outputs={q.name:digest(q) for q in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
