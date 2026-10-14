"""Synthetic failure cases for source identity and held-dose isolation."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("sci2", Path(__file__).resolve().parents[1]/"scripts/audit_sciplex2_dose.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_dose_labels_are_not_micromolar_and_bad_well_is_retained():
    p = pd.DataFrame(dict(Cell=["c1","c2","c3"], top_to_second_best_ratio=[11,20,10], qval=[0,0,0],
                          hash_umis=[31,40,40],top_oligo=["Dex_100_AA01","SAHA_100_E09","BMS_1_AB01"]),index=["c1","c2","c3"])
    cells=pd.DataFrame(["c1","c2","c3"]); hashes=pd.DataFrame(p.top_oligo.tolist())
    meta,keep=MOD.cell_metadata(p,cells,hashes)
    assert keep.tolist()==[True,True,False]
    assert meta.dose_uM.tolist()==[50,1000]
    assert pd.isna(meta.loc['c2','plate_prefix'])
    assert meta.loc['c2','source_well_label']=='E09'
    assert meta.solvent.tolist()==['ethanol','DMSO']
    with pytest.raises(ValueError,match='order'):
        MOD.cell_metadata(p,cells.iloc[::-1],hashes)


def test_ctg_not_silently_reinterpreted_as_dose_label():
    c=pd.DataFrame(dict(CTG=[.5,1.05],replicate=[1,1],cell_type=['A549']*2,
                        treatment=['Nutlin','Dex'],dose=[100,100],real_dose=[250,50]))
    MOD.verify_ctg(c)  # CTG above 1 is retained, not clipped to an apoptosis probability.
    with pytest.raises(ValueError,match='mapping'):
        MOD.verify_ctg(c.assign(real_dose=[100,100]))
    with pytest.raises(ValueError,match='Duplicate'):
        MOD.verify_ctg(pd.concat([c,c]))


def test_matrix_aggregation_keeps_excluded_cells_for_total_check(tmp_path):
    p=tmp_path/'m.tsv'
    p.write_text('1\t1\t2\n2\t1\t3\n1\t2\t9\n2\t3\t4\n')
    counts,totals,n=MOD.aggregate_triplets(p,2,3,np.array([0,-1,1]),2)
    np.testing.assert_array_equal(counts,[[2,3],[0,4]])
    np.testing.assert_array_equal(totals,[5,9,4]); assert n==4
    p.write_text('3\t1\t2\n')
    with pytest.raises(ValueError,match='coordinate'):
        MOD.aggregate_triplets(p,2,3,np.array([0,-1,1]),2)


def test_feature_selection_cannot_see_held_dose():
    groups=pd.DataFrame([dict(drug=d,dose_source_label=z,dose_uM=z*f) for d,f in MOD.FACTORS.items() for z in [0,.1,.5,1,5,10,50,100]])
    rng=np.random.default_rng(1); counts=rng.integers(1,100,size=(32,12))
    first,protocol=MOD.evaluate_dose(counts,groups,n_features=5)
    changed=counts.copy(); changed[groups.dose_source_label.eq(5),0]+=100000
    _,other=MOD.evaluate_dose(changed,groups,n_features=5)
    assert protocol[0]['gene_indices']==other[0]['gene_indices']
    assert len(first)==40
    for p in protocol:
        assert all(x['dose_source_label']!=p['held_source_label'] for x in p['train_conditions'])
        assert len(p['test_conditions'])==4
    assert set(first.method)=={'control_only','drug_mean','nearest_dose','logdose_ridge','logdose_interpolation_clamped'}
