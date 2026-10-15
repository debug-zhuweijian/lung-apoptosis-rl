"""Explicit molar-unit canonicalization for newly registered protocol artifacts.

Never use this transform with weights qualified for raw source-unit inputs without
requalification. Original values/units belong in the source-bound input records.
"""
from dataclasses import replace
import torch
from ..models.full_protocol import DOSE_UNITS,SCALAR_FIELDS

UNIT_TRANSFORM_ID='physical_molar_uM_v1'


def canonical_molar_protocol(batch):
    """Return a copy: known active interval doses nM/mM/uM -> uM only.

    No mass concentration, percentage, missing dose, retention or time is inferred.
    This changes the feature contract, not the physical intervention or source.
    """
    batch.validate()
    dose=SCALAR_FIELDS.index('dose');values=batch.values.clone();units=batch.dose_unit.clone()
    active=torch.arange(batch.values.shape[1],device=batch.lengths.device)[None]<batch.lengths[:,None]
    intervals=(batch.event_type<=1)&active&batch.known[:,:,dose]
    for name,factor in [('nM',.001),('mM',1000.),('uM',1.)]:
        selected=intervals&(batch.dose_unit==DOSE_UNITS.index(name))
        values[:,:,dose][selected]=batch.values[:,:,dose][selected]*factor
        units[selected]=DOSE_UNITS.index('uM')
    result=replace(batch,values=values,dose_unit=units);result.validate();return result
