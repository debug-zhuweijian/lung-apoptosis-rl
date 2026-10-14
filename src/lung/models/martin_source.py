"""New source-aware Martin E/D contract, not trained cross-source qualification.

Losses are this project's continuous-observation designs. Estimated counts are
not integer UMI supervision. This module does not implement response P or G/Q.
"""
from dataclasses import fields
from types import SimpleNamespace

import torch
from torch import nn

from ..data.martin_source_inputs import (BACKGROUNDS, DOMAINS, META_FIELDS, PROTOCOLS,
                                       SOURCE_ID, canonical, digest, require, sha)
from .encoders import mlp
from .full_protocol import FullPhysicalProtocolBatch, FullProtocolConditioner

PARAMETER_IDENTITY = 'martin_source_bulk_E_D_v1'


class MartinSourceModel(nn.Module):
    """Mask-aware E, source/domain D and ordered nominal history conditioning.

Reference reconstruction uses its own 24h DMSO clock. Full candidate protocol
context is consumed by target decoding, with no reference/target pairing claim.
"""
    parameter_identity = PARAMETER_IDENTITY

    def __init__(self, gene_ids, contract, *, width=32):
        super().__init__()
        require(type(width) is int and width >= 4, 'WIDTH_REQUIRED')
        self.gene_ids = tuple(gene_ids)
        require(bool(self.gene_ids) and len(set(self.gene_ids)) == len(self.gene_ids)
                and contract['gene_ids_sha256'] == digest(self.gene_ids)
                and contract['source_id'] == SOURCE_ID
                and contract['domains'] == list(DOMAINS)
                and contract['backgrounds'] == list(BACKGROUNDS)
                and contract['protocols'] == list(PROTOCOLS), 'MODEL_SOURCE_AXIS_CONTEXT')
        self._contract_json = canonical({k: v for k, v in contract.items() if k != 'gene_ids'})
        from pathlib import Path
        directory = Path(__file__).resolve().parent
        self._identity_json = canonical({'parameter_identity': PARAMETER_IDENTITY,
                                          'input_contract': self.contract, 'width': width,
                                          'model_dependency_sha256': {name: sha(directory / name) for name in
                                                                      ('martin_source.py', 'full_protocol.py', 'encoders.py')}})
        self.source = nn.Embedding(1, width)  # index 0 explicitly means GSE193258
        self.background = nn.Embedding(len(BACKGROUNDS), width)
        self.domain = nn.Embedding(len(DOMAINS), width)
        self.clock_origin = nn.Embedding(2, width)  # reference own / target nominal phase
        self.clock = mlp([1, width, width])
        self.expression = mlp([2 * len(self.gene_ids), width, width])
        self.protocol = nn.Embedding(len(PROTOCOLS), width)
        self.metadata = mlp([2 * len(META_FIELDS), width, width])
        self.full_protocol = FullProtocolConditioner(SimpleNamespace(d_model=width, context_dim=max(1, width // 4)))
        self.decoders = nn.ModuleDict({d: mlp([2 * width, width, len(self.gene_ids)]) for d in DOMAINS})

    @property
    def contract(self):
        import json
        return json.loads(self._contract_json)

    @property
    def identity(self):
        import json
        return json.loads(self._identity_json)

    def _validate(self, batch):
        batch.validate()
        require(batch.gene_ids == self.gene_ids and batch.contract_json == self._contract_json,
                'REGISTERED_BATCH_MODEL_IDENTITY')

    def _context(self, batch, *, target_observation=False):
        self._validate(batch)
        parameter = self.source.weight
        b = len(batch.sample_ids)
        def ids(value):
            return torch.full((b,), value, dtype=torch.long, device=parameter.device)
        clock_values = batch.physical.endpoint_hours if target_observation else batch.measurement_clock
        clock = clock_values.to(parameter).log1p()[:, None]
        return (self.source(ids(0)) + self.background(ids(BACKGROUNDS.index(batch.background)))
                + self.domain(ids(DOMAINS.index(batch.domain)))
                + self.clock_origin(ids(int(target_observation or batch.part == 'target'))) + self.clock(clock))

    def encode(self, batch):
        context = self._context(batch)
        mask = batch.observed_mask.to(device=context.device)
        values = batch.model_values.to(context)
        finite = torch.where(mask, values, torch.zeros_like(values))
        # These computational zeros cannot become observations: mask is an
        # independent encoder input and remains the criterion for every loss.
        return self.expression(torch.cat((finite, mask.to(values.dtype)), dim=-1)) + context

    def protocol_context(self, batch):
        context = self._context(batch, target_observation=True)
        values, known = batch.metadata_values.to(context), batch.metadata_known.to(device=context.device)
        meta = self.metadata(torch.cat((torch.where(known, values, torch.zeros_like(values)).log1p(),
                                        known.to(values.dtype)), dim=-1))
        protocol = torch.full((len(batch.sample_ids),), PROTOCOLS.index(batch.protocol),
                              dtype=torch.long, device=context.device)
        physical = FullPhysicalProtocolBatch(**{
            f.name: (getattr(batch.physical, f.name).to(context) if getattr(batch.physical, f.name).is_floating_point()
                     else getattr(batch.physical, f.name).to(device=context.device))
            for f in fields(batch.physical)})
        return self.full_protocol(context + self.protocol(protocol) + meta, physical)

    def decode(self, latent, batch, *, as_target=None):
        context = self._context(batch)
        require(as_target is None or type(as_target) is bool, 'STRICT_TARGET_BOOL')
        target = batch.part == 'target' if as_target is None else as_target
        if target:
            context = self.protocol_context(batch)
        require(latent.shape == context.shape and latent.is_floating_point()
                and bool(torch.isfinite(latent).all()), 'FINITE_SOURCE_LATENT')
        values = self.decoders[batch.domain](torch.cat((latent.to(context), context), dim=-1))
        if batch.domain == 'estimated_counts':
            values = torch.nn.functional.softplus(values)  # nonnegative log1p(count estimate)
        mask = batch.observed_mask.to(device=values.device)
        source = (values.to(torch.float64) * batch.scale.to(device=values.device)
                  + batch.offset.to(device=values.device) if batch.domain == 'log2TPM'
                  else values.to(torch.float64).expm1())
        require(bool(torch.isfinite(source[mask]).all()), 'NONFINITE_SOURCE_DECODE')
        return {'model_values': values, 'source_values': torch.where(mask, source, torch.nan),
                'observed_mask': mask, 'gene_ids': self.gene_ids, 'domain': batch.domain,
                'part': 'target' if target else 'reference', 'input_contract': self.contract,
                'request_id': batch.request_id, 'sample_ids': batch.sample_ids,
                'source_id': batch.source_id, 'background': batch.background, 'protocol': batch.protocol,
                'system': batch.system, 'fold_id': batch.fold_id, 'role': batch.role,
                'parameter_identity': PARAMETER_IDENTITY, 'scientific_qualified': False}

    def forward(self, batch):
        return self.decode(self.encode(batch), batch)

    def observation_loss(self, output, batch):
        """Per-replicate masked MSE; no count likelihood or functional Y claim."""
        self._validate(batch)
        require(output['domain'] == batch.domain and output['gene_ids'] == self.gene_ids
                and output['part'] == batch.part and output['input_contract'] == self.contract
                and output['parameter_identity'] == PARAMETER_IDENTITY, 'OUTPUT_SUPERVISION_IDENTITY')
        require(all(output.get(key) == getattr(batch, key) for key in
                    ('request_id', 'sample_ids', 'source_id', 'background', 'protocol', 'system', 'fold_id', 'role')),
                'OUTPUT_REQUEST_SAMPLE_CONTEXT_MISMATCH')
        prediction = output['model_values']
        mask = batch.observed_mask.to(device=prediction.device)
        require(prediction.shape == mask.shape and prediction.is_floating_point()
                and bool(torch.isfinite(prediction[mask]).all())
                and torch.equal(output['observed_mask'], mask), 'OBSERVED_LOSS_MASK')
        target = torch.where(mask, batch.model_values.to(prediction), torch.zeros_like(prediction))
        residual = torch.where(mask, prediction, torch.zeros_like(prediction)) - target
        return (residual.square().sum(-1) / mask.sum(-1)).mean()

    def validate_checkpoint_metadata(self, metadata):
        require(canonical(metadata.get('model_identity')) == self._identity_json,
                'CHECKPOINT_DOMAIN_FOLD_AXIS_OR_CONTEXT')
        require(metadata.get('training_role') == 'fit' and metadata.get('final_RNA_used') is False,
                'CHECKPOINT_ROLE_OR_FINAL')

    def load_registered_state(self, state, metadata):
        self.validate_checkpoint_metadata(metadata)
        return nn.Module.load_state_dict(self, state, strict=True)

    def load_state_dict(self, *args, **kwargs):
        raise ValueError('MARTIN_SOURCE_USE_REGISTERED_SOURCE_METADATA')
