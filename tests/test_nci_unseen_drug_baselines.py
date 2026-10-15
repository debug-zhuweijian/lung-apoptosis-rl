import numpy as np
import pandas as pd
import pytest
from scripts.audit_nci_unseen_drug_baselines import kernel,run


def fixture():
    samples=[];links=[];catalog=[]
    for k in range(3):
        drug=f'd{k}';catalog.append(dict(drug_id=drug,fingerprint_row=k))
        for j in range(3):
            sid=f'{k}_{j}';samples.append(dict(sample_id=sid,drug_name=drug,is_vehicle=j==0,dose_uM=float(j),exposure_h=6,control_sample_id=f'{k}_0'))
            links.append(dict(sample_id=sid,action_structure_feature_use_eligible=j>0,action_drug_id=drug if j else None))
    x=np.random.default_rng(9).normal(10,1,(9,7))
    return x,pd.DataFrame(samples),pd.DataFrame(links),pd.DataFrame(catalog),np.array([[1,1,0],[1,0,1],[0,1,1]])


def test_kernel_identity_symmetry_and_bounded_similarity():
    f=np.array([[1,0,1],[1,1,0]]);k=kernel(f,[1,2],[6,24],f,[1,2],[6,24])
    np.testing.assert_allclose(k,k.T);np.testing.assert_allclose(np.diag(k),1)
    assert (k>=0).all() and (k<=1).all() and np.linalg.eigvalsh(k).min()>0


def test_vehicle_dose_is_not_treated_as_log_drug_input():
    with pytest.raises(ValueError,match='Positive'):
        kernel([[1]],[0],[24],[[1]],[1],[24])


def test_nonbinary_descriptor_rejected():
    with pytest.raises(ValueError,match='Binary'):
        kernel([[.5]],[1],[6],[[1]],[1],[6])


def test_whole_drug_is_held_out_and_controls_not_treated_as_actions():
    r,t,p=run(*fixture(),genes=4)
    assert r['eligible_treatment_samples']==6 and r['held_drugs']==3 and len(t)==24
    for f in r['folds']:
        assert not set(f['training_sample_ids'])&set(f['test_sample_ids'])
        prefix=f['test_sample_ids'][0].split('_')[0]
        assert all(not v.startswith(prefix+'_') for v in f['training_sample_ids'])
    assert not r['apoptosis_prediction'] and not r['selection_or_RL_benefit'] and not r['strict_blind_preprocessing']


def test_held_drug_expression_cannot_select_training_genes():
    args=fixture();a=run(*args,genes=4)[0]
    args[0][:3,0]=1000000;b=run(*args,genes=4)[0]
    assert a['folds'][0]['selected_gene_indices']==b['folds'][0]['selected_gene_indices']


def test_unresolved_structure_treatment_is_excluded():
    x,s,l,c,fp=fixture();l.loc[l.sample_id.eq('0_1'),'action_structure_feature_use_eligible']=False
    r,_,_=run(x,s,l,c,fp,genes=4)
    assert r['eligible_treatment_samples']==5 and r['excluded_treated_sample_ids']==['0_1']
