"""Fixed tiny interface checks only; no checkpoint or real data access."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lung.models.full_protocol import (FullPhysicalProtocolBatch, FullProtocolConditioner,
                                      FullProtocolLatentAdapter)
from lung.models.latent_staged import LatentMember


def physical(pulse=False):
    values = torch.zeros(1, 2, 8)
    values[0, 0] = torch.tensor([1., 0., 6. if pulse else 24., 6. if pulse else 24., 0., 1., 0., 0.])
    values[0, 1] = torch.tensor([0., 6., 24., 0., 2., 0., 1., 18.])
    return FullPhysicalProtocolBatch(torch.ones(1, 2, 2048), torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool), torch.tensor([[0, 3]]), torch.tensor([[1, 0]]),
        values, torch.ones(1, 2, 8, dtype=torch.bool), torch.tensor([2 if pulse else 1]),
        torch.tensor([24.]), torch.tensor([True]))


class TinyMember(nn.Module):
    cfm_loss = LatentMember.cfm_loss
    sample_latent = LatentMember.sample_latent

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_dim=2, sigma_min=.001)
        self.linear = nn.Linear(2 + 4 + 4, 2)
        self.calls = []

    def condition_encoder(self, **kwargs):
        return kwargs['base_condition']

    def velocity(self, state, tau, reference_h, condition):
        self.calls.append((reference_h.detach().clone(), condition.detach().clone(), tau.clone()))
        return self.linear(torch.cat((state, reference_h, condition), dim=-1))


def adapter():
    torch.manual_seed(43)
    return FullProtocolLatentAdapter(TinyMember(), FullProtocolConditioner(
        SimpleNamespace(d_model=4, context_dim=2)))


def test_pulse_and_continuous_enter_actual_velocity_differently():
    model = adapter()
    args = dict(base_condition=torch.zeros(1, 4), noise=torch.ones(1, 2), tau=torch.tensor([.5]))
    ref = torch.ones(1, 4)
    model(torch.zeros(1, 2), ref, physical(), **args)
    model(torch.zeros(1, 2), ref, physical(True), **args)
    assert not torch.equal(model.member.calls[0][1], model.member.calls[1][1])
    assert torch.equal(model.member.calls[0][0], ref)
    assert physical().endpoint_hours.item() == physical(True).endpoint_hours.item()


def test_unknown_is_not_measured_zero_and_action_identity_survives():
    model = adapter()
    p = physical()
    known = p.known.clone()
    known[..., 4] = False
    known[..., 6] = False
    unknown = replace(p, known=known)
    base = torch.zeros(1, 4)
    assert not torch.equal(model.full_protocol(base, p), model.full_protocol(base, unknown))
    other = replace(p, fingerprint=torch.zeros_like(p.fingerprint))
    assert not torch.equal(model.full_protocol(base, p), model.full_protocol(base, other))


def test_ordered_repeated_4_24_4_history_changes_condition():
    model = adapter()
    p = physical(True)
    values = torch.zeros(1, 3, 8)
    values[0, :, 1] = torch.tensor([0., 4., 28.])
    values[0, :, 2] = torch.tensor([4., 28., 32.])
    values[0, :, 3] = torch.tensor([4., 24., 4.])
    values[0, 1, 3] = 0.
    values[0, 1, 7] = 24.
    fp = torch.zeros(1, 3, 2048)
    fp[0, 0, 0] = 1
    fp[0, 2, 1] = 1
    p = replace(p, fingerprint=fp, fingerprint_known=torch.ones(1,3,dtype=torch.bool),
        vehicle=torch.zeros(1,3,dtype=torch.bool), event_type=torch.tensor([[0,3,0]]),
        dose_unit=torch.tensor([[2,0,2]]), values=values, known=torch.ones_like(values,dtype=torch.bool),
        lengths=torch.tensor([3]), endpoint_hours=torch.tensor([32.]))
    reverse_drugs = replace(p, fingerprint=fp.flip(1))
    base = torch.zeros(1,4)
    assert not torch.equal(model.full_protocol(base,p), model.full_protocol(base,reverse_drugs))


def test_cfm_autograd_reaches_new_parameters_without_fit():
    model = adapter()
    loss = model(torch.ones(1,2), torch.ones(1,4), physical(True),
        base_condition=torch.zeros(1,4), noise=torch.ones(1,2), tau=torch.tensor([.4]))
    loss.backward()
    for module in (model.full_protocol.drug, model.full_protocol.history, model.full_protocol.fusion):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_sampling_keeps_reference_at_every_vf_and_generation_tau_unitless():
    model = adapter()
    ref = torch.ones(1,4)
    result = model.sample_latent(ref, physical(True), noise=torch.ones(1,2), steps=2,
                                base_condition=torch.zeros(1,4))
    assert result.shape == (1,2)
    assert all(torch.equal(call[0], ref) for call in model.member.calls)
    assert all((call[2] >= 0).all() and (call[2] <= 1).all() for call in model.member.calls)


def test_old_checkpoint_rejected_even_non_strict_direct_load():
    model = adapter()
    with pytest.raises(ValueError, match='checkpoint'):
        model.load_registered_state({}, {'stage':'latent_cfm'})
    with pytest.raises(ValueError):
        model.load_state_dict(model.member.state_dict())
    with pytest.raises(ValueError):
        model.load_state_dict(model.member.state_dict(), strict=False)


def test_bad_physical_interval_rejected_and_no_outcome_y_field():
    p = physical(True)
    with pytest.raises(ValueError, match='observation clock'):
        replace(p, endpoint_hours=torch.tensor([5.])).validate()
    assert 'Y' not in p.__dataclass_fields__


@pytest.mark.parametrize('field', ['values', 'fingerprint', 'endpoint_hours'])
@pytest.mark.parametrize('dtype', [torch.bool, torch.int64, torch.complex64])
def test_non_real_floating_physical_values_rejected(field, dtype):
    p = physical(True)
    with pytest.raises(ValueError, match='real floating'):
        replace(p, **{field: getattr(p, field).to(dtype)}).validate()


@pytest.mark.parametrize('index,value', [(4,.5), (5,2.), (6,2.), (3,5.)])
def test_invalid_known_count_flag_or_exposure_rejected(index, value):
    p = physical(True)
    values = p.values.clone()
    values[0,0,index] = value
    with pytest.raises(ValueError):
        replace(p, values=values).validate()


def test_known_start_beyond_endpoint_rejected_even_with_unknown_end():
    p = physical(True)
    values, known = p.values.clone(), p.known.clone()
    values[0,1,1] = 30
    known[0,1,2] = False
    with pytest.raises(ValueError, match='start exceeds'):
        replace(p, values=values, known=known).validate()


def test_nonbinary_fingerprint_and_inconsistent_vehicle_rejected():
    p = physical()
    with pytest.raises(ValueError, match='binary bits'):
        replace(p, fingerprint=p.fingerprint * 2).validate()
    with pytest.raises(ValueError, match='vehicle flag'):
        replace(p, vehicle=torch.ones_like(p.vehicle)).validate()


def test_unknown_invalid_placeholder_sanitized_without_becoming_known_zero():
    p = physical(True)
    values, known = p.values.clone(), p.known.clone()
    values[...,4] = float('nan')
    known[...,4] = False
    unknown = replace(p, values=values, known=known)
    p.validate()
    unknown.validate()
    model = adapter()
    base = torch.zeros(1,4)
    assert torch.isfinite(model.full_protocol(base,unknown)).all()
    assert not torch.equal(model.full_protocol(base,p),model.full_protocol(base,unknown))


def test_known_recovery_duration_matches_interval():
    p = physical(True)
    values = p.values.clone()
    values[0,1,7] = 8.
    with pytest.raises(ValueError, match='Recovery interval duration'):
        replace(p, values=values).validate()
    known = p.known.clone()
    known[0,1,7] = False
    replace(p, values=values, known=known).validate()


@pytest.mark.parametrize('index', [3,7])
def test_known_physical_duration_cannot_exceed_endpoint_with_unknown_start_end(index):
    p = physical(True)
    values, known = p.values.clone(), p.known.clone()
    values[0,0,index] = 100.
    known[0,0,1:3] = False
    with pytest.raises(ValueError, match='duration exceeds'):
        replace(p, values=values, known=known).validate()


def test_start_plus_known_exposure_cannot_exceed_endpoint_with_unknown_end():
    p = physical()
    values, known = p.values.clone(), p.known.clone()
    values[0,0,1] = 23.
    values[0,0,3] = 2.
    known[0,0,2] = False
    with pytest.raises(ValueError, match='start plus physical duration'):
        replace(p, values=values, known=known).validate()
