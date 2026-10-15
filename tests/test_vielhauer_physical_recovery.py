from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from scripts.recover_vielhauer_physical_inputs import physical_batch,validate_structure


def fixture():
    rows=[]
    for drug in ['bocnho','cbi_indole2']:
        for dose in [.05,.5,10.,40.,1000.]:
            for duration in [6.,24.]:
                rows.append(dict(condition_id=f'{drug}:{dose}:{duration}',drug_id=drug,dose_nM=dose,dose_uM=dose/1000,
                   exposure_duration_h=duration,terminal_time_h=24,washout_at_h=6 if duration==6 else None,
                   drug_free_recovery_h=18 if duration==6 else 0,
                   protocol_arm='pulse_6h_washout_recovery18h' if duration==6 else 'continuous_24h',source_material_identity=drug))
    return rows,pd.DataFrame({'drug_id':['bocnho','cbi_indole2'],'fingerprint_row':[0,1]}),np.eye(2,2048,dtype=np.uint8)


def test_actual_exposure_is_not_observation_clock_or_invented_washes():
    b,r=physical_batch(*fixture());assert b.lengths.sum()==50
    assert (b.endpoint_hours==24).all() and not b.known[:,:,4].any() and not b.known[:,:,5].any()
    assert b.fingerprint_known.sum()==20 and len(r)==20


def test_labels_do_not_enter_physical_input_values():
    rows,c,f=fixture();a,_=physical_batch(rows,c,f)
    for r in rows:r['measured_endpoint']={'mean_fraction':.99}
    b,_=physical_batch(rows,c,f)
    assert np.array_equal(a.values.numpy(),b.values.numpy())


@pytest.mark.parametrize('field,value',[('terminal_time_h',6),('dose_uM',1000),('washout_at_h',None)])
def test_invalid_clock_unit_or_missing_wash_rejected(field,value):
    rows,c,f=fixture();rows[0][field]=value
    with pytest.raises(ValueError):physical_batch(rows,c,f)


def test_candidate_duplicate_and_subset_are_not_complete_domains():
    rows,c,f=fixture()
    with pytest.raises(ValueError):physical_batch(rows[:-1],c,f)
    rows[-1]=deepcopy(rows[0]);rows[-1]['condition_id']='other-id'
    with pytest.raises(ValueError):physical_batch(rows,c,f)


def test_wrong_stereoisomer_cannot_match_expected_key():
    record=dict(CID=11016947,InChIKey='OMTPLEBPJORDFA-LJQANCHMSA-N',MolecularFormula='C31H23ClN4O3',
      SMILES='O=C(Nc1ccc2[nH]c(C(=O)N3C[C@H](CCl)c4c3cc(O)c3ccccc43)cc2c1)c1cc2ccccc2[nH]1')
    with pytest.raises(ValueError,match='Full structure'):validate_structure('cbi_indole2',record)
