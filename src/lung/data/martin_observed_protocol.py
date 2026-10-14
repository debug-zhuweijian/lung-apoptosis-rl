"""Reported Martin RNA regimens in the existing full-physical input schema.

Observed RNA remains delegated to the closed fixed-source reader. CPU physical
tensors are data, not models, latent descriptors or production qualification.
"""
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from .martin_native_rna import BASE, ROLES, MartinNativeRNA, read_json, sha, write_json
from ..models.full_protocol import (DOSE_UNITS, EVENT_TYPES, FEATURE_SCHEMA,
                                    SCALAR_FIELDS, FullPhysicalProtocolBatch)

SCHEMA = 'martin-observed-RNA-fullphysical-inputs-v1'
BASE_CONTRACT = 'martin_nominal_RNA_phase_base_contract_v1'
RNA = BASE + '/native-rna-inputs-v1'
ARTICLE = 'experiments/20261011-data-alignment-round16/state_function_new_sources/PMC10035388.xml'
PINS = MappingProxyType({
    ARTICLE: '84b139de72a9beb74e28edfb81e889e13b4dde4752f72b3f88f000a4966f90d3',
    BASE + '/rna/geo_samples_source.json': '2f5f8c243f344e1a53777169c8f373073640ea7c75f8eb27dbfdfb2ae3550bba',
    BASE + '/rna/rna_samples.csv': 'd16f9f0456625174801991fee04ff2014e210362d6cfa505d57dd685842abab9',
    BASE + '/chemistry/manifest.json': '4aca710a439aa5665fc805f34d72b25ed92c03a618f587b64af4a44595b60827',
    BASE + '/chemistry/items.json': 'cb1da492a8486aaa6cec1e97f214c658e0992d97333310197b59c44db09bc95a',
    BASE + '/chemistry/morgan2048_chiral.npy': '41e630a67f7d77ce56073f101af2739b83647cb504a1fcc158dc06b7cc478eab',
    RNA + '/manifest.json': 'dd13e4870baf785fff59081578fb89df209f70066346ec72c367429aabb2d745',
    'src/lung/data/martin_native_rna.py': '4f85643234cd99f5b5e5b1f6a6127e49ca4ed2268b9050ec0e2fdc4a82a21352',
    'src/lung/data/full_protocol_inputs.py': '6d067664e80cd9f7035387fc904c0c6271d6792b4d934b58aec2fe4854de6a37',
    'src/lung/models/full_protocol.py': 'c6bdce150ed8a0cc53e3e26d3f438597805f918c6a4a4e1fa4f6304e1aa64e27',
})


def clone(value):
    return json.loads(json.dumps(value, allow_nan=False))


def same_json(left, right):
    """Exact JSON scalar types, list order and finite numeric metadata."""
    try:
        return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
            right, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('NONFINITE_OR_INVALID_PROTOCOL_JSON_METADATA') from None


def _event(kind, *, unit='unknown', **values):
    return {'event_type': kind, 'dose_unit': unit, 'vehicle': False,
            'values': {field: values.get(field) for field in SCALAR_FIELDS}}


def _group(request, chemical):
    p, background = request['actual_protocol'], request['background_id']
    state = p['long_short_history']
    long_hours = {'H1975': 72, 'HCC827': 96, 'PC9': 168, 'HCC2935': 240}
    if background not in long_hours or state not in ('acute_post_treatment', 'DTP', 'short_washout', 'long_washout'):
        raise ValueError('UNKNOWN_OBSERVED_RNA_GROUP')
    exposure = 24 if state == 'acute_post_treatment' else 504
    recovery = 24 if state == 'short_washout' else long_hours[background] if state == 'long_washout' else 0
    endpoint = exposure + recovery
    if (request['dose_uM'] != .5 or request['chemical_id'] != chemical['chemical_id']
            or p['exposure_duration_h'] != exposure or p['washout_h'] != recovery
            or p['observation_time_h'] != endpoint or request['function_72h_four_drug_target'] is not False):
        raise ValueError('OBSERVED_RNA_DOSE_CLOCK_OR_CHEMICAL_CHANGED')
    interval = _event('drug_interval', unit='nM', dose=500, start_hours=0, end_hours=exposure, exposure_hours=exposure)
    interval['fingerprint'] = {'chemical_id': chemical['chemical_id'], 'row': 0,
                               'row_sha256': chemical['fingerprint_sha256'], 'scope': 'nominal_parent_graph_only'}
    events = [interval]
    if recovery:
        events += [_event('wash', start_hours=504, end_hours=504, wash_count=2),
                   _event('media_change', start_hours=504, end_hours=504, media_change=1),
                   _event('recovery', start_hours=504, end_hours=endpoint, recovery_hours=recovery)]
    return {'group_id': 'GSE193258:' + background + ':' + state, 'source_id': 'GSE193258',
            'source_background': background, 'paper_background': 'NCI-H1975' if background == 'H1975' else background,
            'source_state': state, 'target_ids': request['target_sample_ids'],
            'target_columns': request['target_column_indices'], 'reference_ids': request['physical_reference_ids'],
            'reference_columns': request['reference_column_indices'], 'endpoint_hours': endpoint, 'events': events,
            'time_origin': 'first_osimertinib_start_nominal_phase_clock',
            'point_event_time_convention': 'wash_media_transition_clock_not_measured_operation_duration',
            'reference_own_clock': {'drug_free_preculture_days': 21, 'vehicle_DMSO_hours': 24, 'same_cell_t0': False},
            'medium_evidence': {'recovery_drug_free_media': True if recovery else None,
                                'wash_buffer': 'PBS' if recovery else None,
                                'formula': None, 'serum': None, 'volume': None, 'DMSO_concentration': None},
            'unknown': {'wash_operation_start_end_wallclock': None, 'wash_operation_duration': None,
                        'wash_volume': None, 'intracellular_prior_drug_retention': None,
                        'maintenance_media_events': None, 'event_list_exhaustive': False},
            'maintenance_rule': 'RNA_sec2_9_does_not_authorize_sec2_4_twice_weekly_growth_assay_schedule',
            'evidence': {'paper': ARTICLE, 'section': 'sec2-9 RNA Sequencing',
                         'GSM_file': BASE + '/rna/geo_samples_source.json', 'target_GSMs': request['target_sample_ids']},
            'base_design': {'schema': BASE_CONTRACT, 'dose_uM': .5, 'duration_h': exposure,
                            'horizon_h': endpoint, 'elapsed_h': 0., 'design_choice_not_paper_instruction': True,
                            'source_name': 'GSE193258', 'background_name': background,
                            'protocol_name': request['protocol_id'], 'source_index': None,
                            'background_index': None, 'protocol_index': None,
                            'numeric_model_kwargs_ready': False, 'encoder_decoder_weights_status': 'pending'},
            'unpaired_reference_target': True, 'same_cell_t0': False, 'function_72h_four_drug_target': False,
            'model_or_function_qualified': False}


def _reconstruct(root, allowed_roles, allow_final):
    root = Path(root).resolve()
    bindings = dict(PINS)
    for name, expected in PINS.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError('FIXED_PROTOCOL_SOURCE_CHANGED: ' + name)
    code = 'src/lung/data/martin_observed_protocol.py'
    if sha(root / code) != sha(__file__):
        raise ValueError('EXECUTING_PROTOCOL_LOADER_COPY_CHANGED')
    bindings[code] = sha(__file__)
    builder = 'scripts/prepare_martin_observed_protocol.py'
    trusted_builder = Path(__file__).resolve().parents[3] / builder
    if sha(root / builder) != sha(trusted_builder):
        raise ValueError('TRUSTED_PROTOCOL_BUILDER_COPY_CHANGED')
    bindings[builder] = sha(trusted_builder)
    reader = MartinNativeRNA(root, root / RNA, allowed_roles=allowed_roles, allow_final=allow_final)
    try:
        metadata = reader.manifest
        chem = next(c for c in read_json(root / BASE / 'chemistry/items.json') if c['source_drug_name'] == 'osimertinib')
        if (chem['chemical_id'] != 'pubchem:71496458' or chem['fingerprint_row'] != 0
                or chem['experimental_stock_form_certified'] is not False):
            raise ValueError('NOMINAL_OSIMERTINIB_IDENTITY_CHANGED')
        fp = np.load(root / BASE / 'chemistry/morgan2048_chiral.npy', allow_pickle=False)
        if fp.dtype != np.uint8 or fp.shape != (4, 2048) or not np.isin(fp, [0, 1]).all():
            raise ValueError('NOMINAL_FINGERPRINT_CHANGED')
        fp = fp[0].copy()
        fp.setflags(write=False)
        groups, links = {}, {'lobo': [], 'cycle': []}
        for system in links:
            for request in read_json(root / RNA / (system + '_requests.json')):
                group = _group(request, chem)
                previous = groups.setdefault(group['group_id'], group)
                if not same_json(previous, group):
                    raise ValueError('SAME_PHYSICAL_GROUP_CHANGED_ACROSS_ROLES')
                links[system].append({'request_id': request['request_id'], 'fold_id': request['fold_id'],
                                      'role': request['role'], 'group_id': group['group_id']})
        if len(groups) != 16 or tuple(len(links[s]) for s in links) != (64, 64):
            raise ValueError('OBSERVED_GROUP_OR_ROLE_VIEW_COUNTS_CHANGED')
        coordinates = {'RNA_manifest_sha256': PINS[RNA + '/manifest.json'], 'RNA_loader_revision': metadata['loader_revision'],
                       'coordinates_sha256': metadata['files']['coordinates.json'],
                       'gene_mask_sha256': metadata['files']['joint_observed_mask.npy'],
                       'source_domains': reader.axes['units'], 'source_gene_count': 19712,
                       'joint_gene_count': 32461, 'observed_joint_genes': 16167,
                       'normalization': 'none_source_values_preserved', 'RNA_model_domain_qualified': False}
        return reader, {'groups': list(groups.values()), 'role_links': links, 'RNA_coordinates': coordinates,
                        'source_bindings': bindings, 'fingerprint': fp}
    except BaseException:
        reader.close()
        raise


def _manifest(snapshot, file_hashes):
    return {'schema': SCHEMA, 'feature_schema': FEATURE_SCHEMA, 'base_contract': BASE_CONTRACT,
            'source_bindings': snapshot['source_bindings'], 'files': file_hashes,
            'RNA_coordinates': snapshot['RNA_coordinates'], 'groups': 16, 'events': 40,
            'role_requests': {'lobo': 64, 'cycle': 64}, 'target_samples': 48, 'reference_samples': 12,
            'point_event_time_convention': 'transition_clock_not_measured_operation_duration',
            'numeric_model_kwargs_ready': False, 'model_or_function_qualified': False,
            'functional_Y_read': False, 'ML_forward': False, 'PT_load': False, 'training_performed': False}


def prepare(root, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('NEW_OBSERVED_PROTOCOL_VERSION_REQUIRED')
    reader, snapshot = _reconstruct(root, ROLES, True)
    try:
        output.mkdir(parents=True)
        for name, value in (('groups.json', snapshot['groups']), ('role_links.json', snapshot['role_links'])):
            write_json(output / name, value)
        manifest = _manifest(snapshot, {name: sha(output / name) for name in ('groups.json', 'role_links.json')})
        write_json(output / 'manifest.json', manifest)
        return manifest
    finally:
        reader.close()


class MartinObservedProtocol:
    def __init__(self, root, directory, *, allowed_roles=('fit',), allow_final=False):
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self._rna, snapshot = _reconstruct(self.root, allowed_roles, allow_final)
        try:
            m = read_json(self.directory / 'manifest.json')
            expected = {name: sha(self.directory / name) for name in ('groups.json', 'role_links.json')}
            if (not same_json(m, _manifest(snapshot, expected))
                    or not same_json(read_json(self.directory / 'groups.json'), snapshot['groups'])
                    or not same_json(read_json(self.directory / 'role_links.json'), snapshot['role_links'])):
                raise ValueError('SOURCE_OBSERVED_PROTOCOL_RECONSTRUCTION_CHANGED')
            self._snapshot = snapshot
            self._groups = {g['group_id']: g for g in snapshot['groups']}
            self._links = {s: {r['request_id']: r for r in rows} for s, rows in snapshot['role_links'].items()}
            self._manifest = m
        except BaseException:
            self._rna.close()
            raise

    @property
    def manifest(self):
        return clone(self._manifest)

    def close(self):
        self._rna.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def requests(self, *, system, fold_id, role):
        return self._rna.requests(system=system, fold_id=fold_id, role=role)

    def condition(self, request_id, *, system, fold_id, role):
        permitted = {r['request_id'] for r in self.requests(system=system, fold_id=fold_id, role=role)}
        if request_id not in permitted:
            raise PermissionError('OBSERVED_PROTOCOL_ROLE_SYSTEM_OR_FOLD_MISMATCH')
        link = self._links[system][request_id]
        return {'role_link': clone(link), 'group': clone(self._groups[link['group_id']]),
                'RNA_coordinates': clone(self._snapshot['RNA_coordinates'])}

    def physical_batch(self, request_ids, *, system, fold_id, role):
        if not isinstance(request_ids, (list, tuple)) or not request_ids:
            raise ValueError('NONEMPTY_SOURCE_REQUEST_IDS_REQUIRED')
        records = [self.condition(rid, system=system, fold_id=fold_id, role=role)['group'] for rid in request_ids]
        b, s = len(records), max(len(r['events']) for r in records)
        fp = torch.zeros((b, s, 2048), dtype=torch.float32)
        fk, vehicle = torch.zeros((b, s), dtype=torch.bool), torch.zeros((b, s), dtype=torch.bool)
        kinds, units = torch.zeros((b, s), dtype=torch.long), torch.zeros((b, s), dtype=torch.long)
        values = torch.zeros((b, s, len(SCALAR_FIELDS)), dtype=torch.float32)
        known = torch.zeros_like(values, dtype=torch.bool)
        for i, record in enumerate(records):
            for j, e in enumerate(record['events']):
                kinds[i, j], units[i, j] = EVENT_TYPES.index(e['event_type']), DOSE_UNITS.index(e['dose_unit'])
                for k, field in enumerate(SCALAR_FIELDS):
                    if e['values'][field] is not None:
                        values[i, j, k], known[i, j, k] = e['values'][field], True
                if e.get('fingerprint') is not None:
                    fp[i, j], fk[i, j] = torch.from_numpy(self._snapshot['fingerprint'].astype(np.float32)), True
        batch = FullPhysicalProtocolBatch(fp, fk, vehicle, kinds, units, values, known,
            torch.tensor([len(r['events']) for r in records], dtype=torch.long),
            torch.tensor([r['endpoint_hours'] for r in records], dtype=torch.float32), torch.ones(b, dtype=torch.bool))
        batch.validate()
        return batch

    def reference(self, request_id, *, system, fold_id, role, domain='log2TPM', coordinate='source'):
        self.condition(request_id, system=system, fold_id=fold_id, role=role)
        return self._rna.reference(request_id, system=system, role=role, domain=domain, coordinate=coordinate)

    def supervision(self, request_id, *, system, fold_id, role, domain='log2TPM', coordinate='source'):
        self.condition(request_id, system=system, fold_id=fold_id, role=role)
        return self._rna.supervision(request_id, system=system, role=role, domain=domain, coordinate=coordinate)
