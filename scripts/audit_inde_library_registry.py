"""Audit all 261 independent primary-name queries against Tahoe registry IDs.

CID agreement proposes a record for the existing structure/material/alias review;
it never itself proves experimental form, protocol or RNA/functional pairing.
"""
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.link_inde_tahoe_conditions import primary_name
from scripts.link_endpoint_chemical_identities import normalize_name
from scripts.recover_public_expression import digest


def audit(names,responses,tahoe):
    if len(set(names))!=len(names) or len({r['source_name'] for r in responses})!=len(responses):
        raise ValueError('Duplicate source identity/query')
    if set(names)!={r['source_name'] for r in responses}:raise ValueError('Incomplete library query coverage')
    if tahoe.drug.duplicated().any():raise ValueError('Duplicate Tahoe record labels')
    rows=[]
    for r in responses:
        if r['queried_primary_name']!=primary_name(r['source_name']):raise ValueError('Query differs from preserved primary material name')
        ids=r['cids'];hits=tahoe[tahoe.pubchem_cid.eq(ids[0])] if r['status']==200 and len(ids)==1 else tahoe.iloc[:0]
        selected=None;reason='no_unique_primary_CID'
        if r['status']==200 and len(ids)==1:
            if len(hits)==0:reason='no_exact_CID_record_not_proven_absent'
            elif len(hits)==1:selected=hits.iloc[0].drug;reason='unique_CID_candidate_pending_structure_alias_review'
            else:
                exact=hits[hits.drug.map(normalize_name).eq(normalize_name(r['queried_primary_name']))]
                if len(exact)==1:selected=exact.iloc[0].drug;reason='same_CID_multiple_forms_primary_name_selects_one_pending_review'
                else:reason='multiple_same_CID_material_records_require_review'
        rows.append(dict(source_name=r['source_name'],queried_primary_name=r['queried_primary_name'],
                         cid=ids[0] if r['status']==200 and len(ids)==1 else None,
                         query_status=r['status'],all_query_CIDs=json.dumps(ids),
                         same_CID_Tahoe_names=json.dumps(hits.drug.tolist()),tahoe_name=selected,
                         candidate_reason=reason,experimental_form_verified=False,
                         condition_pairing_verified=False))
    ledger=pd.DataFrame(rows)
    ledger['multiple_source_names_same_CID']=ledger.cid.notna() & ledger.cid.duplicated(keep=False)
    ledger['unique_candidate_for_existing_review']=ledger.tahoe_name.notna() & ~ledger.multiple_source_names_same_CID
    candidates=[dict(source_name=r.source_name,tahoe_name=r.tahoe_name,cid=int(r.cid))
                for r in ledger[ledger.unique_candidate_for_existing_review].itertuples()]
    return ledger,candidates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','functional','tahoe','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output directory required')
    receipts=json.loads((a.source/'sources.json').read_text());by_path={r['path']:r for r in receipts}
    if len(by_path)!=len(receipts):raise ValueError('Duplicate response receipt')
    responses=json.loads((a.source/'name_resolution.json').read_text())
    for r in responses:
        rec=by_path[r['cache_path']];path=a.source/r['cache_path']
        if digest(path)!=rec['sha256']:raise ValueError('Public query bytes changed')
        raw=json.loads(path.read_text());actual=raw.get('IdentifierList',{}).get('CID',[]) if r['status']==200 else []
        if actual!=r['cids']:raise ValueError('Query summary differs from raw response')
    fm=json.loads((a.functional/'manifest.json').read_text());f=a.functional/'single_agent_24h.csv'
    if digest(f)!=fm['outputs'][f.name]:raise ValueError('Functional source bytes changed')
    names=pd.read_csv(f).source_compound_name.drop_duplicates().tolist()
    ledger,candidates=audit(names,responses,pd.read_parquet(a.tahoe));a.output.mkdir(parents=True)
    ledger.to_csv(a.output/'library_CID_audit.csv',index=False)
    (a.output/'name_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    report=dict(library_names=len(names),query_responses=len(responses),unique_primary_CIDs=int(ledger.cid.notna().sum()),
       exact_CID_Tahoe_candidates=int(ledger.tahoe_name.notna().sum()),unique_candidates_for_existing_review=len(candidates),
       duplicate_source_CID_names=ledger.loc[ledger.multiple_source_names_same_CID,'source_name'].tolist(),
       unresolved_names=ledger.loc[ledger.cid.isna(),'source_name'].tolist(),
       structure_and_alias_review_complete=False,experimental_form_verified=False,RNA_recovery_performed=False,
       complete_scientific_qualification=False,
       source_hashes={'name_resolution':digest(a.source/'name_resolution.json'),'receipts':digest(a.source/'sources.json'),
                      'functional':digest(f),'tahoe':digest(a.tahoe)},
       script_sha256=digest(__file__),outputs={q.name:digest(q) for q in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
