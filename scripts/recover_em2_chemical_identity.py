"""Bind EM-2 source drawings to a nominal full-stereo reference, not an alias guess.

A molecular formula alone cannot identify this compound. Root visually reviewed
2025 Fig.1 and 2024 synthesis Fig.1; byte checks preserve those exact sources.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.recover_public_expression import digest
from lung.data.chemistry import build_chemical_catalog
from lung.data.drug_identity import describe_structure

KEY='CSIBMGLPBAXXSG-MKOUKINSSA-N'
FIGURE_SHA='a223aeb7a954051fdaae269ed70c1796731fc26742558acfbdc70044c5d0e837'
SYNTHESIS_SHA='aff890eb48994ccbdb8492183eab47e9603427aaf2de33b465d5a4729545c2ee'
# Independent connectivity transcription from the source drawing (no stereo inferred here).
CONNECTIVITY='COC12C=C(C)C(O1)C1OC(=O)C(=C)C1C(OC(=O)C(=C)C)CC(C)=C2'


def validate_nominal_structure(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None:raise ValueError('Invalid reference SMILES')
    if rdMolDescriptors.CalcMolFormula(m)!='C20H24O6':raise ValueError('Source formula mismatch')
    if Chem.MolToSmiles(m,isomericSmiles=False)!=Chem.MolToSmiles(Chem.MolFromSmiles(CONNECTIVITY),isomericSmiles=False):
        raise ValueError('Source connectivity mismatch despite possible formula agreement')
    if Chem.MolToInchiKey(m)!=KEY:raise ValueError('Nominal stereochemistry mismatch or unspecified')
    return describe_structure(smiles)


def pubchem_smiles(data,expected_cid):
    compounds=data['PC_Compounds']
    if len(compounds)!=1 or compounds[0]['id']['id']['cid']!=expected_cid:raise ValueError('PubChem reference identity mismatch')
    return next(p['value']['sval'] for p in compounds[0]['props'] if p['urn']['label']=='SMILES' and p['urn'].get('name')=='Absolute')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['source','expression','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    if digest(a.source/'EM2_2025_fig1.webp')!=FIGURE_SHA or digest(a.source/'synthesis2024.pdf')!=SYNTHESIS_SHA:
        raise ValueError('Source drawings changed: visual re-review required')
    cj=json.loads((a.source/'chembl2368404.json').read_text())
    if cj['molecule_chembl_id']!='CHEMBL2368404':raise ValueError('ChEMBL identity mismatch')
    smiles=cj['molecule_structures']['canonical_smiles'];identity=validate_nominal_structure(smiles)
    ps=pubchem_smiles(json.loads((a.source/'CSIBMGLPBAXXSG-MKOUKINSSA-N.json').read_text()),73354392)
    if validate_nominal_structure(ps)['rdkit_inchikey']!=identity['rdkit_inchikey']:raise ValueError('Provider disagreement')
    parent=pubchem_smiles(json.loads((a.source/'phantomolin.json').read_text()),5281493)
    if not parent.startswith('CCO') or Chem.MolToInchiKey(Chem.MolFromSmiles(parent))!='YNGLXZUKGYZCFL-YUZMFRNZSA-N':
        raise ValueError('Phantomolin derivation reference changed')
    validate_nominal_structure(parent[1:])  # 2024 Fig.1: ethoxy -> methoxy, same stereochemical core.
    src=json.loads((a.expression/'manifest.json').read_text())
    if src['series']!='GSE270062':raise ValueError('Wrong RNA source')
    for name,sha in src['outputs'].items():
        if digest(a.expression/name)!=sha:raise ValueError('RNA recovery output changed')
    samples=pd.read_csv(a.expression/'samples.csv')
    if samples.drug.tolist()!=['vehicle','vehicle','EM-2','EM-2']:raise ValueError('RNA compound binding changed')
    catalog=pd.DataFrame([dict(drug_id='EM2_GSE270062_nominal',drug_name='2-deethoxy-2beta-methoxyphantomolin (EM-2)',
                               canonical_smiles=smiles,component_role='drug')])
    chemical,fp,fpmeta=build_chemical_catalog(catalog)
    samples['nominal_structure_id']=samples.drug.map({'EM-2':identity['identity_id']})
    samples['nominal_compound_reference']=samples.drug.map({'EM-2':'PubChem:73354392;ChEMBL:CHEMBL2368404'})
    a.output.mkdir(parents=True);chemical.to_csv(a.output/'chemical.csv',index=False);np.save(a.output/'fingerprints.npy',fp)
    samples.to_csv(a.output/'RNA_sample_links.csv',index=False)
    result=dict(source_paper_doi='10.3389/fphar.2025.1580217',synthesis_doi='10.1002/anie.202402050',
                nominal_structure_verified=True,verification='Root source-drawing visual comparison plus formula, independent graph transcription, phantomolin relation and two registry graph equality checks',
                pubchem_cid=73354392,chembl_id='CHEMBL2368404',identity=identity,fingerprint=fpmeta,
                experimental_lot_verified=False,matched_RNA_Annexin_protocol=False,full_chain_qualified=False,
                rejected_same_formula_candidate=163043788,
                expression_manifest_sha256=digest(a.expression/'manifest.json'),script_sha256=digest(__file__),
                source_hashes={n:digest(a.source/n) for n in ['EM2_2025_fig1.webp','synthesis2024.pdf','chembl2368404.json','CSIBMGLPBAXXSG-MKOUKINSSA-N.json','phantomolin.json']})
    result['outputs']={f.name:digest(f) for f in a.output.iterdir()}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
