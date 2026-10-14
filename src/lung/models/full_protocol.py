"""New endpoint-P version: ordered physical protocol enters every latent VF.

No RNA trajectory, outcome Y, source decoder or training runner is created here.
Generation tau retains LatentMember's [0,1] meaning, not treatment hours.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .encoders import DrugEncoder, HistoryEncoder, mlp

FEATURE_SCHEMA = "full_physical_protocol_v1"
PARAMETER_IDENTITY = "full_physical_protocol_latent_v1"
SCALAR_FIELDS = ("dose", "start_hours", "end_hours", "exposure_hours",
                 "wash_count", "prior_retention", "media_change", "recovery_hours")
EVENT_TYPES = ("drug_interval", "vehicle_interval", "wash", "recovery", "media_change")
DOSE_UNITS = ("unknown", "uM", "nM", "mM", "percent_v_v")


@dataclass(frozen=True)
class FullPhysicalProtocolBatch:
    fingerprint: torch.Tensor  # [B,S,2048], unknown entries sanitized with mask
    fingerprint_known: torch.Tensor  # [B,S] bool
    vehicle: torch.Tensor  # [B,S] bool
    event_type: torch.Tensor  # [B,S] long, EVENT_TYPES
    dose_unit: torch.Tensor  # [B,S] long, DOSE_UNITS; no implicit unit conversion
    values: torch.Tensor  # [B,S,8], SCALAR_FIELDS, original declared units
    known: torch.Tensor  # [B,S,8] bool; unknown never interpreted as measured zero
    lengths: torch.Tensor  # [B] long; valid prefix preserves authored order
    endpoint_hours: torch.Tensor  # [B], common observation clock
    endpoint_known: torch.Tensor  # [B] bool

    def validate(self):
        for name in ("fingerprint", "values", "endpoint_hours"):
            if not getattr(self, name).is_floating_point():
                raise ValueError(f"{name} must be real floating point, not bool/integer/complex.")
        if self.fingerprint.ndim != 3:
            raise ValueError("Protocol fingerprints require [B,S,2048].")
        b, s, f = self.fingerprint.shape
        if f != 2048 or s < 1:
            raise ValueError("Protocol fingerprints require [B,S>=1,2048].")
        for name in ("fingerprint_known", "vehicle", "event_type", "dose_unit"):
            if getattr(self, name).shape != (b, s):
                raise ValueError(f"Invalid {name} shape.")
        if self.values.shape != (b, s, 8) or self.known.shape != self.values.shape:
            raise ValueError("Protocol values/known require [B,S,8].")
        for name in ("fingerprint_known", "vehicle", "known", "endpoint_known"):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must be bool.")
        for name, count in (("event_type", len(EVENT_TYPES)), ("dose_unit", len(DOSE_UNITS))):
            v = getattr(self, name)
            if v.dtype != torch.long or (v < 0).any() or (v >= count).any():
                raise ValueError(f"Invalid {name} registry id.")
        if self.lengths.shape != (b,) or self.lengths.dtype != torch.long:
            raise ValueError("lengths must be integer [B].")
        if (self.lengths < 1).any() or (self.lengths > s).any():
            raise ValueError("Invalid protocol lengths.")
        if self.endpoint_hours.shape != (b,) or self.endpoint_known.shape != (b,):
            raise ValueError("Endpoint fields require [B].")
        active = torch.arange(s, device=self.lengths.device)[None] < self.lengths[:, None]
        selected = self.values[self.known & active[..., None]]
        if not torch.isfinite(selected).all() or (selected < 0).any():
            raise ValueError("Known physical fields must be finite and nonnegative.")
        if not torch.isfinite(self.fingerprint[self.fingerprint_known & active]).all():
            raise ValueError("Known fingerprint must be finite.")
        bits = self.fingerprint[self.fingerprint_known & active]
        if ((bits != 0) & (bits != 1)).any():
            raise ValueError("Known Morgan fingerprint must contain binary bits.")
        for index in (5, 6):
            flags = self.values[..., index][self.known[..., index] & active]
            if ((flags != 0) & (flags != 1)).any():
                raise ValueError("Known retention/media flags must be binary.")
        washes = self.values[..., 4][self.known[..., 4] & active]
        if (washes != washes.round()).any():
            raise ValueError("Known wash count must be an integer.")
        if (((self.event_type == 1) & ~self.vehicle & active)
                | ((self.event_type == 0) & self.vehicle & active)).any():
            raise ValueError("Drug/vehicle interval event contradicts vehicle flag.")
        drug_dose = (self.event_type == 0) & self.known[..., 0] & active
        if ((self.dose_unit == 0) & drug_dose).any():
            raise ValueError("Known drug dose requires an explicit unit.")
        end = self.endpoint_hours[self.endpoint_known]
        if not torch.isfinite(end).all() or (end <= 0).any():
            raise ValueError("Known observation clock must be positive.")
        starts, ends = self.values[..., 1], self.values[..., 2]
        paired = self.known[..., 1] & self.known[..., 2] & active
        if ((ends < starts) & paired).any():
            raise ValueError("Interval ends precede starts.")
        if ((ends > self.endpoint_hours[:, None]) & self.known[..., 2]
                & self.endpoint_known[:, None] & active).any():
            raise ValueError("Interval exceeds observation clock.")
        if ((starts > self.endpoint_hours[:, None]) & self.known[..., 1]
                & self.endpoint_known[:, None] & active).any():
            raise ValueError("Interval start exceeds observation clock.")
        exposure_known = paired & self.known[..., 3] & (self.event_type <= 1)
        if (~torch.isclose(ends - starts, self.values[..., 3]) & exposure_known).any():
            raise ValueError("Drug/vehicle interval exposure must equal end minus start.")
        recovery_known = paired & self.known[..., 7] & (self.event_type == 3)
        if (~torch.isclose(ends - starts, self.values[..., 7]) & recovery_known).any():
            raise ValueError("Recovery interval duration must equal end minus start.")
        for index in (3, 7):
            duration = self.values[..., index]
            clock_known = self.known[..., index] & self.endpoint_known[:, None] & active
            if ((duration > self.endpoint_hours[:, None]) & clock_known).any():
                raise ValueError("Known physical duration exceeds observation clock.")
            applicable = (self.event_type <= 1) if index == 3 else (self.event_type == 3)
            if ((starts + duration > self.endpoint_hours[:, None]) & clock_known
                    & self.known[..., 1] & applicable).any():
                raise ValueError("Known start plus physical duration exceeds observation clock.")
        if ((starts[:, 1:] < starts[:, :-1]) & self.known[:, 1:, 1]
                & self.known[:, :-1, 1] & active[:, 1:]).any():
            raise ValueError("Protocol must retain chronological order.")


class FullProtocolConditioner(nn.Module):
    """Trainable history and endpoint fusion; old ConditionEncoder stays intact."""
    def __init__(self, config):
        super().__init__()
        d, c = config.d_model, config.context_dim
        self.drug = DrugEncoder(d)
        self.event = nn.Embedding(len(EVENT_TYPES), c)
        self.unit = nn.Embedding(len(DOSE_UNITS), c)
        self.history = HistoryEncoder(SimpleNamespace(d_model=d,
                                                      history_input_dim=d + 2*c + 18))
        self.fusion = mlp([2*d + 2, 2*d, d])

    def forward(self, base_condition, physical):
        physical.validate()
        b, s, _ = physical.fingerprint.shape
        active = torch.arange(s, device=physical.lengths.device)[None] < physical.lengths[:, None]
        fp = torch.where((physical.fingerprint_known & active)[..., None], physical.fingerprint,
                         torch.zeros_like(physical.fingerprint))
        drug = self.drug(fp.reshape(-1, 2048), physical.vehicle.reshape(-1)).reshape(b, s, -1)
        values = torch.where(physical.known & active[..., None], physical.values,
                             torch.zeros_like(physical.values))
        features = torch.cat((drug, self.event(physical.event_type), self.unit(physical.dose_unit),
                              values.log1p(), physical.known.to(values.dtype),
                              physical.fingerprint_known[..., None].to(values.dtype),
                              physical.vehicle[..., None].to(values.dtype)), dim=-1)
        ordered = self.history(features, physical.lengths)
        endpoint = torch.where(physical.endpoint_known, physical.endpoint_hours,
                               torch.zeros_like(physical.endpoint_hours))
        return self.fusion(torch.cat((base_condition, ordered, endpoint.log1p()[:, None],
                                      physical.endpoint_known[:, None].to(values.dtype)), dim=-1))


class FullProtocolLatentAdapter(nn.Module):
    """New parameter namespace around a latent member, never an old checkpoint alias.

    Registration/qualification by producer is still required before real sampling.
    Target is detached cached endpoint RNA latent; not an apoptosis Y label.
    """
    feature_schema = FEATURE_SCHEMA
    parameter_identity = PARAMETER_IDENTITY

    def __init__(self, member, conditioner):
        super().__init__()
        self.member = member
        self.full_protocol = conditioner

    def encode_condition(self, physical, **condition_kwargs):
        base = self.member.condition_encoder(**condition_kwargs)
        return self.full_protocol(base, physical)

    def forward(self, latent_target, reference_h, physical, *, noise=None, tau=None,
                **condition_kwargs):
        condition = self.encode_condition(physical, **condition_kwargs)
        return self.member.cfm_loss(latent_target, reference_h, condition, noise=noise, tau=tau)

    def sample_latent(self, reference_h, physical, *, noise=None, steps=32, generator=None,
                      **condition_kwargs):
        condition = self.encode_condition(physical, **condition_kwargs)
        return self.member.sample_latent(reference_h, condition, noise=noise, steps=steps,
                                         generator=generator)

    def load_registered_state(self, state, metadata):
        if metadata.get("feature_schema") != FEATURE_SCHEMA or metadata.get(
                "parameter_identity") != PARAMETER_IDENTITY:
            raise ValueError("Old/unregistered checkpoint cannot load full protocol version.")
        return self.load_state_dict(state, strict=True)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not any(key.startswith("full_protocol.") for key in state_dict):
            raise ValueError("Old checkpoint has no full protocol parameters.")
        if not strict:
            raise ValueError("Full protocol parameter loading requires strict identity.")
        return super().load_state_dict(state_dict, strict=True, assign=assign)
