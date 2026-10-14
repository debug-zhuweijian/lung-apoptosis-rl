from dataclasses import replace
import torch
from lung.data.physical_units import canonical_molar_protocol
from lung.models.full_protocol import DOSE_UNITS
from test_vielhauer_physical_recovery import fixture,physical_batch


def test_same_molar_intervention_is_identical_after_normalization():
    b,_=physical_batch(*fixture());v=b.values.clone();u=b.dose_unit.clone()
    v[:,0,0]/=1000;u[:,0]=DOSE_UNITS.index('uM');other=replace(b,values=v,dose_unit=u)
    a=canonical_molar_protocol(b);c=canonical_molar_protocol(other)
    torch.testing.assert_close(a.values,c.values,rtol=1e-6,atol=1e-10)
    assert torch.equal(a.dose_unit,c.dose_unit)
    assert b.dose_unit[:,0].eq(DOSE_UNITS.index('nM')).all()
    assert torch.equal(a.values[:,:,1:],b.values[:,:,1:]) and torch.equal(a.known,b.known)


def test_millimolar_and_missing_values_without_inference():
    b,_=physical_batch(*fixture());v=b.values.clone();u=b.dose_unit.clone();k=b.known.clone()
    u[0,0]=DOSE_UNITS.index('mM');v[0,0,0]=.001
    k[1,0,0]=False;v[1,0,0]=float('nan')
    c=canonical_molar_protocol(replace(b,values=v,dose_unit=u,known=k))
    assert c.values[0,0,0]==1 and torch.isnan(c.values[1,0,0]) and not c.known[1,0,0]
    assert c.dose_unit[1,0]==u[1,0]


def test_percentage_doses_are_never_guessed_as_molar():
    b,_=physical_batch(*fixture());u=b.dose_unit.clone();u[0,0]=DOSE_UNITS.index('percent_v_v')
    changed=replace(b,dose_unit=u);c=canonical_molar_protocol(changed)
    assert c.dose_unit[0,0]==u[0,0] and c.values[0,0,0]==b.values[0,0,0]


def test_canonicalization_is_idempotent():
    b,_=physical_batch(*fixture());a=canonical_molar_protocol(b);c=canonical_molar_protocol(a)
    assert torch.equal(a.values,c.values) and torch.equal(a.dose_unit,c.dose_unit)
