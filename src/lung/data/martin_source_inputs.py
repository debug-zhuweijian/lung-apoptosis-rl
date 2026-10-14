"""Source-bound Martin bulk observations for E/D; preparing tensors is not ML.

The frozen RNA/protocol/coordinate readers authenticate the source. This layer
never fits coordinates, opens final RNA, invents pairing, or attaches functional Y.
"""
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .martin_native_rna import BASE, DOMAINS, sha
from .martin_observation_coordinates import MartinObservationCoordinates, OUTPUT
from .martin_observed_protocol import MartinObservedProtocol
from ..models.full_protocol import FullPhysicalProtocolBatch

SCHEMA = 'martin-source-E-D-inputs-v1'
SOURCE_ID = 'GSE193258'
BACKGROUNDS = ('H1975', 'HCC827', 'HCC2935', 'PC9')
PROTOCOLS = tuple('Martin2022_RNA_' + name for name in
                  ('DTP', 'acute_post_treatment', 'long_washout', 'short_washout'))
META_FIELDS = ('reference_preculture_days', 'reference_DMSO_hours',
               'wash_operation_duration', 'wash_volume',
               'intracellular_prior_drug_retention', 'medium_volume',
               'DMSO_concentration', 'event_list_exhaustive')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError('MARTIN_SOURCE_' + reason)


def physical_repeat(physical, count):
    """One reported condition repeated across RNA replicates; not new experiments."""
    physical.validate()
    require(physical.lengths.shape == (1,), 'SINGLE_REPORTED_GROUP_REQUIRED')
    return FullPhysicalProtocolBatch(**{
        f.name: getattr(physical, f.name).repeat_interleave(count, dim=0)
        for f in fields(physical)})


@dataclass(frozen=True)
class MartinSourceBatch:
    raw_values: torch.Tensor  # source log2(TPM+.01) or continuous estimated counts
    model_values: torch.Tensor  # float64 affine/log1p; missing remains NaN
    observed_mask: torch.Tensor
    offset: torch.Tensor  # inverse affine; NaN outside the source mask
    scale: torch.Tensor
    measurement_clock: torch.Tensor  # source clock, not a same-cell t0 clock
    metadata_values: torch.Tensor
    metadata_known: torch.Tensor
    physical: FullPhysicalProtocolBatch
    domain: str
    units: str
    gene_ids: tuple
    sample_ids: tuple
    source_id: str
    background: str
    protocol: str
    part: str
    request_id: str
    system: str
    fold_id: str
    role: str
    contract_json: str

    @property
    def contract(self):
        return json.loads(self.contract_json)

    def validate(self):
        values, mask = self.raw_values, self.observed_mask
        require(self.domain in DOMAINS and self.part in ('reference', 'target'), 'DOMAIN_OR_PART')
        require(self.role in ('fit', 'selection'), 'FINAL_RNA_NOT_ENABLED')
        require(self.source_id == SOURCE_ID and self.background in BACKGROUNDS
                and self.protocol in PROTOCOLS, 'SOURCE_CONTEXT_IDENTITY')
        require(values.dtype == torch.float64 and values.ndim == 2
                and mask.dtype == torch.bool and values.shape == mask.shape
                and self.model_values.dtype == torch.float64
                and self.model_values.shape == values.shape, 'FLOAT64_VALUES_BOOL_MASK')
        b, g = values.shape
        require(g == len(self.gene_ids) == len(set(self.gene_ids))
                and b == len(self.sample_ids) == len(set(self.sample_ids))
                and bool(mask.any(-1).all()), 'ORDERED_AXIS_SAMPLES_OR_EMPTY_OBSERVATION')
        for array in (values, self.model_values):
            require(bool(torch.isfinite(array[mask]).all())
                    and bool(torch.isnan(array[~mask]).all()), 'MISSING_IS_NAN_WITH_MASK')
        require(self.offset.dtype == self.scale.dtype == torch.float64
                and self.offset.shape == self.scale.shape == (g,)
                and bool((mask == mask[0]).all()), 'INVERSE_COORDINATES')
        observed = mask[0]
        require(bool(torch.isfinite(self.offset[observed]).all())
                and bool(torch.isfinite(self.scale[observed]).all())
                and bool((self.scale[observed] > 0).all())
                and bool(torch.isnan(self.offset[~observed]).all())
                and bool(torch.isnan(self.scale[~observed]).all()), 'INVERSE_MASK_OR_SCALE')
        expected = ((values - self.offset) / self.scale if self.domain == 'log2TPM'
                    else values.log1p())
        require(torch.allclose(self.model_values[mask], expected[mask], rtol=0, atol=0),
                'OBSERVATION_TRANSFORM_CHANGED')
        if self.domain == 'estimated_counts':
            require(bool((values[mask] >= 0).all()), 'ESTIMATED_COUNTS_NONNEGATIVE')
        require(self.measurement_clock.dtype == torch.float64
                and self.measurement_clock.shape == (b,)
                and bool(torch.isfinite(self.measurement_clock).all())
                and bool((self.measurement_clock > 0).all()), 'SOURCE_MEASUREMENT_CLOCK')
        require(self.metadata_values.shape == self.metadata_known.shape == (b, len(META_FIELDS))
                and self.metadata_values.dtype == torch.float64
                and self.metadata_known.dtype == torch.bool
                and bool(torch.isfinite(self.metadata_values[self.metadata_known]).all())
                and bool((self.metadata_values[self.metadata_known] >= 0).all())
                and bool(torch.isnan(self.metadata_values[~self.metadata_known]).all()),
                'PROTOCOL_METADATA_MASK')
        self.physical.validate()
        require(self.physical.lengths.shape == (b,), 'PHYSICAL_REPLICATE_COUNT')
        if self.part == 'reference':
            require(bool(self.metadata_known[:, 1].all())
                    and torch.equal(self.measurement_clock, self.metadata_values[:, 1]), 'REFERENCE_OWN_CLOCK')
        else:
            require(bool(self.physical.endpoint_known.all())
                    and torch.equal(self.measurement_clock, self.physical.endpoint_hours.to(torch.float64)),
                    'TARGET_OBSERVATION_CLOCK')
        contract = self.contract
        require(contract['schema'] == SCHEMA and contract['source_id'] == self.source_id
                and contract['gene_ids_sha256'] == digest(self.gene_ids)
                and contract['system'] == self.system and contract['fold_id'] == self.fold_id
                and contract['units'][self.domain] == self.units
                and contract['domains'] == list(DOMAINS)
                and contract['backgrounds'] == list(BACKGROUNDS)
                and contract['protocols'] == list(PROTOCOLS)
                and contract['same_cell_t0'] is False and contract['functional_Y'] is False
                and contract['scientific_qualified'] is False, 'MODEL_CONTRACT_MISMATCH')
        permitted = contract['fit_sample_ids' if self.role == 'fit' else 'selection_sample_ids']
        require(set(self.sample_ids) <= set(permitted), 'ROLE_GSM_MEMBERSHIP')


def prepare_part(part, condition, physical, coordinates, contract):
    """Pure tensor preparation; called only after the registered readers verify."""
    group, link = condition['group'], condition['role_link']
    request = part['request']
    name = 'target' if part['target_expression_used'] else 'reference'
    require(part['domain'] in DOMAINS and part['coordinate'] == 'joint'
            and part['functional_Y'] is None and part['same_cell_t0'] is False
            and part['unpaired_reference_target'] is True
            and part['function_72h_four_drug_target'] is False, 'OBSERVATION_SEMANTICS')
    require(link['request_id'] == request['request_id']
            and link['fold_id'] == request['fold_id'] and link['role'] == request['role']
            and part['role_system'] == contract['system'] and link['fold_id'] == contract['fold_id']
            and group['source_id'] == SOURCE_ID
            and group['source_background'] == request['background_id']
            and group['base_design']['protocol_name'] == request['protocol_id']
            and group[name + '_ids'] == part['sample_ids']
            and group[name + '_columns'] == part['source_column_indices'], 'RNA_PROTOCOL_LINK')
    require(part['gene_ids'] == tuple(contract['gene_ids'])
            and link['role'] in ('fit', 'selection'), 'AXIS_OR_FINAL_ROLE')
    values, mask = part['values'].copy(), part['observed_mask'].copy()
    if part['domain'] == 'log2TPM':
        offset, scale = coordinates['mean'].copy(), coordinates['scale'].copy()
        model_values = (values - offset) / scale
    else:
        offset = np.where(mask[0], 0., np.nan)
        scale = np.where(mask[0], 1., np.nan)
        model_values = torch.from_numpy(values).log1p().numpy()
    clock = (group['endpoint_hours'] if name == 'target'
             else group['reference_own_clock']['vehicle_DMSO_hours'])
    metadata = [group['reference_own_clock']['drug_free_preculture_days'],
                group['reference_own_clock']['vehicle_DMSO_hours'],
                group['unknown']['wash_operation_duration'], group['unknown']['wash_volume'],
                group['unknown']['intracellular_prior_drug_retention'],
                group['medium_evidence']['volume'], group['medium_evidence']['DMSO_concentration'],
                float(group['unknown']['event_list_exhaustive'])]
    meta_values = np.array([np.nan if v is None else v for v in metadata], dtype=np.float64)
    b = len(part['sample_ids'])
    identity = {k: v for k, v in contract.items() if k != 'gene_ids'}
    batch = MartinSourceBatch(
        torch.from_numpy(values), torch.from_numpy(model_values), torch.from_numpy(mask),
        torch.from_numpy(offset), torch.from_numpy(scale), torch.full((b,), clock, dtype=torch.float64),
        torch.from_numpy(np.broadcast_to(meta_values, (b, len(metadata))).copy()),
        torch.from_numpy(np.broadcast_to(np.isfinite(meta_values), (b, len(metadata))).copy()),
        physical_repeat(physical, b), part['domain'], part['units'], tuple(part['gene_ids']), tuple(part['sample_ids']),
        SOURCE_ID, group['source_background'], request['protocol_id'], name, request['request_id'],
        part['role_system'], link['fold_id'], link['role'], canonical(identity))
    batch.validate()
    return batch


class MartinSourceInputs:
    """Only fit/selection data; no final permission or model execution is exposed."""
    def __init__(self, root, *, allowed_roles=('fit',)):
        if not allowed_roles or set(allowed_roles) - {'fit', 'selection'}:
            raise PermissionError('MARTIN_SOURCE_FINAL_NOT_ENABLED')
        self.root = Path(root).resolve()
        self.protocol = MartinObservedProtocol(self.root, self.root / BASE / 'observed-protocol-inputs-v1',
                                               allowed_roles=allowed_roles)
        try:
            self.coordinates = MartinObservationCoordinates(allowed_roles=allowed_roles)
            require(self.coordinates.reader.root == self.root, 'REGISTERED_ROOT_CHANGED')
        except BaseException:
            self.protocol.close()
            raise

    def close(self):
        self.coordinates.close()
        self.protocol.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def contract(self, *, system, fold_id):
        require((system, fold_id) in self.coordinates._folds, 'UNKNOWN_FOLD')
        genes = self.coordinates.reader.axes['joint_ensembl_ids']
        # Authenticated role metadata only. No final expression or functional Y
        # is opened to enumerate which GSM identities belong to selection.
        selection_ids = sorted({sid for r in self.protocol._rna._requests[system].values()
                                if r['fold_id'] == fold_id and r['role'] == 'selection'
                                for sid in r['physical_reference_ids'] + r['target_sample_ids']})
        return {'schema': SCHEMA, 'source_id': SOURCE_ID, 'measurement': 'bulk_RNA_biological_replicates',
                'domains': list(DOMAINS), 'backgrounds': list(BACKGROUNDS), 'protocols': list(PROTOCOLS),
                'units': self.coordinates.reader.axes['units'],
                'gene_ids': genes, 'gene_ids_sha256': digest(genes),
                'RNA_manifest_sha256': sha(self.root / BASE / 'native-rna-inputs-v1/manifest.json'),
                'coordinates_manifest_sha256': sha(OUTPUT / 'manifest.json'),
                'protocol_manifest_sha256': sha(self.root / BASE / 'observed-protocol-inputs-v1/manifest.json'),
                'source_input_consumer_sha256': sha(__file__),
                'system': system, 'fold_id': fold_id,
                'fit_sample_ids': self.coordinates._folds[(system, fold_id)]['fit_sample_ids'],
                'selection_sample_ids': selection_ids,
                'preprocessing': {'log2TPM': 'registered_fit_unique_GSM_affine_float64_ddof0',
                                  'estimated_counts': 'continuous_estimate_log1p_no_rounding_no_library_normalization'},
                'same_cell_t0': False, 'functional_Y': False, 'scientific_qualified': False}

    def read_request(self, request_id, *, system, fold_id, role, domain='log2TPM', supervision=False):
        require(type(supervision) is bool and domain in DOMAINS, 'DOMAIN_OR_STRICT_BOOL')
        condition = self.protocol.condition(request_id, system=system, fold_id=fold_id, role=role)
        # Public coordinate read revalidates the frozen source/fit moments. It is
        # data verification of the registered transform, never model fitting.
        checked = self.coordinates.read_request(request_id, system=system, fold_id=fold_id,
                                                 role=role, supervision=supervision)
        if domain == 'log2TPM':
            parts = checked
        elif supervision:
            parts = self.protocol.supervision(request_id, system=system, fold_id=fold_id,
                                                role=role, domain=domain, coordinate='joint')
        else:
            parts = {'reference': self.protocol.reference(request_id, system=system, fold_id=fold_id,
                                                          role=role, domain=domain, coordinate='joint')}
        physical = self.protocol.physical_batch([request_id], system=system, fold_id=fold_id, role=role)
        contract = self.contract(system=system, fold_id=fold_id)
        return {name: prepare_part(part, condition, physical,
                                  self.coordinates._folds[(system, fold_id)], contract)
                for name, part in parts.items()}
