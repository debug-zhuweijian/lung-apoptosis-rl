"""Explicit final-only Martin source E/D evaluation, never model selection.

Metadata preflight uses frozen package/role/statistic metadata only. It does not
construct the old readers whose constructors reconstruct all RNA/statistics.
Actual evaluation loads completed frozen namespace weights and final rows only.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from lung.data.martin_native_rna import BASE, DOMAINS
from lung.data.martin_source_inputs import (BACKGROUNDS, META_FIELDS, PROTOCOLS, SCHEMA as INPUT_SCHEMA,
    SOURCE_ID, MartinSourceBatch, canonical, digest, physical_repeat, require, sha)
from lung.models.full_protocol import DOSE_UNITS, EVENT_TYPES, SCALAR_FIELDS, FullPhysicalProtocolBatch
from lung.models.martin_source import MartinSourceModel
from . import martin_source_learning as learning
from .checkpoint import atomic_json, load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'martin-source-final-evaluation-v1'
MANIFEST_PINS = {
    'native-rna-inputs-v1': 'dd13e4870baf785fff59081578fb89df209f70066346ec72c367429aabb2d745',
    'observed-protocol-inputs-v1': 'b4c40156b6601d9196356e8b87017617c0e3b5edce1eb32d82614497681203be',
    'observation-coordinates-v1': '19b36d542dac64dd1c285b9bfebf9e0d1debe0530569b6ef2c8eea1ef28f4ad1'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def role_counts(requests):
    reference = {sid for r in requests for sid in r['physical_reference_ids']}
    target = {sid for r in requests for sid in r['target_sample_ids']}
    require(not reference & target, 'FINAL_REFERENCE_TARGET_OVERLAP')
    return {'unique_GSM': len(reference | target), 'domain_views_per_GSM': 2,
        'unique_reference_GSM': len(reference), 'unique_target_GSM': len(target),
        'reference_domain_request_rows_before_dedup': 2 * sum(len(r['physical_reference_ids']) for r in requests),
        'target_domain_request_rows_before_dedup': 2 * sum(len(r['target_sample_ids']) for r in requests),
        'not_twice_as_many_biological_replicates': True}


class FinalSourceInputs:
    """Source-bound final factory. Metadata construction never reads RNA arrays."""
    def __init__(self, root, fold_id):
        self.root, self.fold_id = Path(root).resolve(), fold_id
        require(self.root == ROOT.resolve(), 'REGISTERED_FINAL_PROJECT_ROOT')
        base = self.root / BASE
        self.manifests = {}
        for name, expected in MANIFEST_PINS.items():
            path = base / name / 'manifest.json'
            require(sha(path) == expected, 'FROZEN_FINAL_PACKAGE_MANIFEST_CHANGED')
            self.manifests[name] = read(path)
        self.native = base / 'native-rna-inputs-v1'
        self.protocol = base / 'observed-protocol-inputs-v1'
        self.coordinates = base / 'observation-coordinates-v1'
        def metadata(directory, name):
            manifest = self.manifests[directory.name]
            require(sha(directory / name) == manifest['files'][name], 'FINAL_METADATA_BYTES_CHANGED')
            return read(directory / name)
        self.axes = metadata(self.native, 'coordinates.json')
        self.samples = metadata(self.native, 'samples.json')
        all_requests = metadata(self.native, 'cycle_requests.json')
        self.requests = [r for r in all_requests if r['fold_id'] == fold_id]
        require(len(self.requests) == 16 and len({r['request_id'] for r in self.requests}) == 16,
                'REGISTERED_CYCLE_FINAL_FOLD_REQUIRED')
        self.roles = {role: [r for r in self.requests if r['role'] == role] for role in ('fit', 'selection', 'final_test')}
        require([len(self.roles[r]) for r in self.roles] == [8, 4, 4], 'FINAL_FIXED_CYCLE_ROLE_COUNTS')
        self.groups = {g['group_id']: g for g in metadata(self.protocol, 'groups.json')}
        links = metadata(self.protocol, 'role_links.json')['cycle']
        self.links = {r['request_id']: r for r in links if r['fold_id'] == fold_id}
        require(set(self.links) == {r['request_id'] for r in self.requests}, 'FINAL_PROTOCOL_ROLE_LINKS')
        coordinate = self.manifests['observation-coordinates-v1']
        require(coordinate['source_manifest'] == self.manifests['native-rna-inputs-v1']
            and coordinate['ordered_gene_ids'] == self.axes['joint_ensembl_ids'], 'FINAL_FIXED_FIT_STATS_SOURCE_AXIS')
        selected = [r for r in coordinate['folds'] if r['system'] == 'cycle' and r['fold_id'] == fold_id]
        require(len(selected) == 1, 'FINAL_FIXED_FIT_COORDINATE_FOLD')
        self.fold = selected[0]
        self.counts = {r: role_counts(v) for r, v in self.roles.items()}
        self.ids = {r: sorted({sid for q in v for sid in q['physical_reference_ids'] + q['target_sample_ids']})
                    for r, v in self.roles.items()}
        require([len(self.ids[r]) for r in self.roles] == [30, 15, 15]
            and len(set(sum(self.ids.values(), []))) == 60 and self.ids['fit'] == self.fold['fit_sample_ids'],
            'FINAL_FIT_SELECTION_IDS_DISJOINT_AND_FROZEN')
        self.gene_ids = tuple(self.axes['joint_ensembl_ids'])
        require(len(self.gene_ids) == len(set(self.gene_ids)) == 32461, 'FINAL_EXACT_GENE_AXIS')
        self.contract = {'schema': INPUT_SCHEMA, 'source_id': SOURCE_ID,
            'measurement': 'bulk_RNA_biological_replicates', 'domains': list(DOMAINS),
            'backgrounds': list(BACKGROUNDS), 'protocols': list(PROTOCOLS), 'units': self.axes['units'],
            'gene_ids_sha256': digest(self.gene_ids),
            'RNA_manifest_sha256': MANIFEST_PINS['native-rna-inputs-v1'],
            'coordinates_manifest_sha256': MANIFEST_PINS['observation-coordinates-v1'],
            'protocol_manifest_sha256': MANIFEST_PINS['observed-protocol-inputs-v1'],
            'source_input_consumer_sha256': sha(Path(__file__).parents[1] / 'data/martin_source_inputs.py'),
            'system': 'cycle', 'fold_id': fold_id, 'fit_sample_ids': self.ids['fit'],
            'selection_sample_ids': self.ids['selection'],
            'preprocessing': {'log2TPM': 'registered_fit_unique_GSM_affine_float64_ddof0',
                'estimated_counts': 'continuous_estimate_log1p_no_rounding_no_library_normalization'},
            'same_cell_t0': False, 'functional_Y': False, 'scientific_qualified': False}
        self.scope = {'schema': SCHEMA, 'role': 'final_test', 'system': 'cycle', 'fold_id': fold_id,
            'input_contract_sha256': digest(self.contract), 'final_sample_ids': self.ids['final_test'],
            'request_ids': sorted(r['request_id'] for r in self.roles['final_test']),
            'package_manifests': MANIFEST_PINS, 'ordered_gene_ids_sha256': digest(self.gene_ids),
            'fixed_fit_coordinate_file': self.fold['file'],
            'fixed_fit_coordinate_sha256': coordinate['files_sha256'][self.fold['file']],
            'final_consumer_sha256': sha(__file__), 'statistic_refitting': False, 'model_selection': False}
        self._arrays, self._coords = {}, None

    def _load_arrays(self):
        """Called only by explicit final evaluation after completed-model checks."""
        if self._coords is None:
            path = self.coordinates / self.fold['file']
            require(sha(path) == self.scope['fixed_fit_coordinate_sha256'], 'FINAL_FIT_STATS_BYTES_CHANGED')
            with np.load(path, allow_pickle=False) as data:
                require(set(data.files) == {'mean', 'scale', 'observed_mask', 'constant_mask'}, 'FINAL_COORDINATE_SCHEMA')
                self._coords = {k: data[k].copy() for k in data.files}
            require(sha(path) == self.scope['fixed_fit_coordinate_sha256'], 'FINAL_STATS_CHANGED_DURING_READ')
            for name in ('joint_observed_mask.npy', *(f'joint_{d}.npy' for d in DOMAINS)):
                path = self.native / name
                require(sha(path) == self.manifests['native-rna-inputs-v1']['files'][name], 'FINAL_RNA_PACKAGE_BYTES_CHANGED')
                self._arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
            mask = self._arrays['joint_observed_mask.npy']
            require(mask.dtype == np.bool_ and mask.shape == (len(self.gene_ids),)
                and np.array_equal(mask, self._coords['observed_mask']), 'FINAL_OBSERVATION_MASK')
            require(self._coords['constant_mask'].dtype == np.bool_
                and self._coords['constant_mask'].shape == mask.shape
                and all(self._coords[k].dtype == np.float64 and self._coords[k].shape == mask.shape
                    and np.isfinite(self._coords[k][mask]).all() and np.isnan(self._coords[k][~mask]).all()
                    for k in ('mean', 'scale')) and (self._coords['scale'][mask] > 0).all(), 'FINAL_FIXED_FIT_STATISTIC_SCHEMA')

    def verify_source_bytes(self):
        """Bind evaluated values to the same sealed source after consumption."""
        for name, expected in MANIFEST_PINS.items():
            require(sha(self.root / BASE / name / 'manifest.json') == expected, 'FINAL_SOURCE_CHANGED_DURING_EVALUATION')
        for name in self._arrays:
            require(sha(self.native / name) == self.manifests['native-rna-inputs-v1']['files'][name],
                    'FINAL_SOURCE_CHANGED_DURING_EVALUATION')
        require(sha(self.coordinates / self.fold['file']) == self.scope['fixed_fit_coordinate_sha256'],
                'FINAL_SOURCE_CHANGED_DURING_EVALUATION')

    def close(self):
        for array in self._arrays.values():
            array._mmap.close()

    def batch(self, request, domain, part):
        require(domain in DOMAINS and part in ('reference', 'target') and request in self.roles['final_test'],
                'FINAL_ONLY_SOURCE_REQUEST')
        self._load_arrays()
        group = self.groups[self.links[request['request_id']]['group_id']]
        indices = request['reference_column_indices' if part == 'reference' else 'target_column_indices']
        ids = request['physical_reference_ids' if part == 'reference' else 'target_sample_ids']
        require(group[part + '_ids'] == ids and group[part + '_columns'] == indices
            and [self.samples[i]['gsm'] for i in indices] == ids and group['source_id'] == SOURCE_ID
            and group['source_background'] == request['background_id'], 'FINAL_SAMPLE_COLUMN_PROTOCOL_IDENTITY')
        array = self._arrays['joint_' + domain + '.npy']
        require(array.dtype == np.float64 and array.shape == (60, len(self.gene_ids)), 'FINAL_RNA_FLOAT64_SHAPE')
        values = np.array(array[indices], copy=True)
        mask = np.broadcast_to(self._arrays['joint_observed_mask.npy'], values.shape).copy()
        offset = self._coords['mean'].copy() if domain == 'log2TPM' else np.where(mask[0], 0., np.nan)
        scale = self._coords['scale'].copy() if domain == 'log2TPM' else np.where(mask[0], 1., np.nan)
        transformed = (values - offset) / scale if domain == 'log2TPM' else torch.from_numpy(values).log1p().numpy()
        metadata = [group['reference_own_clock']['drug_free_preculture_days'], group['reference_own_clock']['vehicle_DMSO_hours'],
            *(group['unknown'][k] for k in ('wash_operation_duration', 'wash_volume', 'intracellular_prior_drug_retention')),
            group['medium_evidence']['volume'], group['medium_evidence']['DMSO_concentration'], float(group['unknown']['event_list_exhaustive'])]
        metadata = np.asarray([np.nan if v is None else v for v in metadata], dtype=np.float64)
        physical = self.physical(group)
        count = len(ids)
        return FinalSourceBatch(torch.from_numpy(values), torch.from_numpy(transformed), torch.from_numpy(mask),
            torch.from_numpy(offset), torch.from_numpy(scale),
            torch.full((count,), group['endpoint_hours'] if part == 'target' else metadata[1], dtype=torch.float64),
            torch.from_numpy(np.broadcast_to(metadata, (count, len(META_FIELDS))).copy()),
            torch.from_numpy(np.broadcast_to(np.isfinite(metadata), (count, len(META_FIELDS))).copy()),
            physical_repeat(physical, count), domain, self.axes['units'][domain], self.gene_ids, tuple(ids),
            SOURCE_ID, request['background_id'], request['protocol_id'], part, request['request_id'], 'cycle',
            self.fold_id, 'final_test', canonical(self.contract), canonical(self.scope))

    def physical(self, group):
        fp_path = self.root / BASE / 'chemistry/morgan2048_chiral.npy'
        expected = self.manifests['observed-protocol-inputs-v1']['source_bindings'][(Path(BASE) / 'chemistry/morgan2048_chiral.npy').as_posix()]
        require(sha(fp_path) == expected, 'FINAL_CHEMICAL_FP_CHANGED')
        fp_source = np.load(fp_path, allow_pickle=False)[0].astype(np.float32)
        require(sha(fp_path) == expected and fp_source.shape == (2048,)
            and np.isin(fp_source, (0, 1)).all(), 'FINAL_CHEMICAL_FP_SCHEMA_OR_CHANGED')
        n = len(group['events'])
        fp, fk, vehicle = torch.zeros(1, n, 2048), torch.zeros(1, n, dtype=torch.bool), torch.zeros(1, n, dtype=torch.bool)
        kinds, units = torch.zeros(1, n, dtype=torch.long), torch.zeros(1, n, dtype=torch.long)
        values, known = torch.zeros(1, n, 8), torch.zeros(1, n, 8, dtype=torch.bool)
        for slot, event in enumerate(group['events']):
            kinds[0, slot], units[0, slot] = EVENT_TYPES.index(event['event_type']), DOSE_UNITS.index(event['dose_unit'])
            vehicle[0, slot] = event['vehicle']
            for index, field in enumerate(SCALAR_FIELDS):
                if event['values'][field] is not None:
                    values[0, slot, index], known[0, slot, index] = event['values'][field], True
            if event.get('fingerprint') is not None:
                fp[0, slot], fk[0, slot] = torch.from_numpy(fp_source), True
        return FullPhysicalProtocolBatch(fp, fk, vehicle, kinds, units, values, known, torch.tensor([n]),
            torch.tensor([group['endpoint_hours']], dtype=torch.float32), torch.tensor([True]))


@dataclass(frozen=True)
class FinalSourceBatch(MartinSourceBatch):
    final_scope_json: str

    def validate(self):
        """Independent final validation: no relabeling as fit or selection."""
        contract, scope, mask = self.contract, read_json(self.final_scope_json), self.observed_mask
        require(self.role == 'final_test' and scope['role'] == 'final_test' and scope['schema'] == SCHEMA
            and scope['input_contract_sha256'] == digest(contract) and self.system == scope['system'] == 'cycle'
            and self.fold_id == scope['fold_id'] and self.request_id in scope['request_ids']
            and set(self.sample_ids) <= set(scope['final_sample_ids'])
            and not set(self.sample_ids) & set(contract['fit_sample_ids'] + contract['selection_sample_ids']), 'FINAL_SCOPE_OR_SAMPLE_ROLE')
        require(self.source_id == SOURCE_ID and self.background in BACKGROUNDS and self.protocol in PROTOCOLS
            and self.domain in DOMAINS and self.part in ('reference', 'target'), 'FINAL_DOMAIN_SOURCE_CONTEXT')
        require(self.raw_values.dtype == self.model_values.dtype == torch.float64 and mask.dtype == torch.bool
            and self.raw_values.ndim == 2 and self.raw_values.shape == self.model_values.shape == mask.shape
            and len(self.sample_ids) == len(set(self.sample_ids)) == len(mask)
            and mask.shape[1] == len(self.gene_ids) == len(set(self.gene_ids))
            and bool(mask.any(-1).all()) and bool((mask == mask[0]).all()), 'FINAL_AXIS_REPLICATES_MASK')
        for value in (self.raw_values, self.model_values):
            require(bool(torch.isfinite(value[mask]).all()) and bool(torch.isnan(value[~mask]).all()), 'FINAL_NAN_MASK_VALUES')
        observed = mask[0]
        require(self.offset.dtype == self.scale.dtype == torch.float64
            and self.offset.shape == self.scale.shape == (len(self.gene_ids),)
            and bool(torch.isfinite(self.offset[observed]).all()) and bool(torch.isfinite(self.scale[observed]).all())
            and bool((self.scale[observed] > 0).all()) and bool(torch.isnan(self.offset[~observed]).all())
            and bool(torch.isnan(self.scale[~observed]).all()), 'FINAL_FIT_COORDINATE_MASK')
        expected = (self.raw_values - self.offset) / self.scale if self.domain == 'log2TPM' else self.raw_values.log1p()
        require(torch.allclose(self.model_values[mask], expected[mask], rtol=0, atol=0)
            and (self.domain != 'estimated_counts' or bool((self.raw_values[mask] >= 0).all())), 'FINAL_FIXED_TRANSFORM')
        self.physical.validate()
        b = len(mask)
        require(self.measurement_clock.dtype == torch.float64 and self.measurement_clock.shape == (b,)
            and self.metadata_values.dtype == torch.float64 and self.metadata_known.dtype == torch.bool
            and self.metadata_values.shape == self.metadata_known.shape == (b, len(META_FIELDS))
            and bool(torch.isfinite(self.metadata_values[self.metadata_known]).all())
            and bool((self.metadata_values[self.metadata_known] >= 0).all())
            and bool(torch.isnan(self.metadata_values[~self.metadata_known]).all())
            and self.physical.lengths.shape == (b,), 'FINAL_PROTOCOL_METADATA_MASK')
        clock = self.metadata_values[:, 1] if self.part == 'reference' else self.physical.endpoint_hours.to(torch.float64)
        require(bool(self.metadata_known[:, 1].all()) if self.part == 'reference' else bool(self.physical.endpoint_known.all()), 'FINAL_CLOCK_KNOWN')
        require(bool(torch.isfinite(clock).all()) and bool((clock > 0).all()) and torch.equal(self.measurement_clock, clock), 'FINAL_SOURCE_CLOCK')
        require(contract['schema'] == INPUT_SCHEMA and contract['source_id'] == SOURCE_ID
            and contract['gene_ids_sha256'] == digest(self.gene_ids) and contract['system'] == self.system
            and contract['fold_id'] == self.fold_id and contract['domains'] == list(DOMAINS)
            and contract['units'][self.domain] == self.units and contract['backgrounds'] == list(BACKGROUNDS)
            and contract['protocols'] == list(PROTOCOLS) and contract['same_cell_t0'] is False
            and contract['functional_Y'] is False and contract['scientific_qualified'] is False, 'FINAL_TRAINED_INPUT_CONTRACT')


def read_json(value):
    return json.loads(value)


def completed_run(directory, context):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(context.root / 'experiments'), 'FINAL_TRAINING_NAMESPACE_PATH')
    manifest, result = read(directory / 'run_manifest.json'), read(directory / 'result.json')
    require(manifest['schema'] == result['schema'] == learning.SCHEMA and manifest['objective'] == learning.OBJECTIVE
        and manifest['contract'] == context.contract and manifest['synthetic'] is False and result['synthetic'] is False
        and manifest['learning_source_sha256'] == sha(learning.__file__)
        and manifest['execution_dependencies'] == learning.execution_dependencies()
        and manifest['identity'] == result['identity'] == digest({k: v for k, v in manifest.items() if k != 'identity'})
        and result['run_manifest_sha256'] == sha(directory / 'run_manifest.json')
        and result['completion'] == {'status': 'complete', 'reason': result['reason']}
        and result['reason'] in ('selection_plateau', 'max_epochs') and result['final_RNA_used'] is False
        and result['scientific_qualified'] is False and type(result['optimizer_steps']) is int and result['optimizer_steps'] > 0
        and type(result['epochs_completed']) is int and 0 < result['epochs_completed'] <= manifest['settings']['epochs']
        and len(result['history']) == result['epochs_completed']
        and all(type(result[k]) is int and result[k] == 0 for k in ('P_outputs', 'G_outputs', 'Q_outputs'))
        and (result['reason'] != 'max_epochs' or result['epochs_completed'] == manifest['settings']['epochs']),
        'FINAL_COMPLETED_FROZEN_SOURCE_RUN_REQUIRED')
    for role in ('fit', 'selection'):
        require(result[role + '_counts'] == manifest[role + '_counts'] == context.counts[role], 'FINAL_SOURCE_TRAINING_COUNTS_CHANGED')
    for name, file in (('selected_checkpoint', 'best.pt'), ('last_checkpoint', 'last.pt')):
        require(result[name] == {'file': file, 'sha256': sha(directory / file)}, 'FINAL_SELECTED_OR_LAST_BYTES_CHANGED')
    return manifest, result


def final_preflight(root=ROOT, *, fold_id, trained_run=None):
    context = FinalSourceInputs(root, fold_id)
    status = 'completed_matching_source_model_pending'
    if trained_run is not None:
        completed_run(trained_run, context)
        status = 'frozen_model_metadata_ready_actual_final_evaluation_pending'
    return {'schema': SCHEMA, 'status': status, 'scope': context.scope, 'final_counts': context.counts['final_test'],
        'final_metadata_verified': True, 'RNA_values_revalidated': False, 'final_RNA_values_read': False,
        'fixed_fit_statistics_values_read': False, 'PT_loads': 0, 'model_forwards': 0, 'optimizer_steps': 0,
        'statistic_refitting': False, 'model_selection': False, 'functional_Y_read': False, 'scientific_qualified': False}


def load_frozen_model(directory, context, device):
    manifest, result = completed_run(directory, context)
    last = load_checkpoint(Path(directory) / 'last.pt', manifest['identity'])
    state = last['extra']['early_stop']
    require(all(type(state[k]) is int and state[k] >= 0 for k in ('bad', 'updates', 'steps'))
        and type(state['best']) is float and np.isfinite(state['best'])
        and state['updates'] == result['epochs_completed'] and type(last['extra']['cursor']) is int,
        'FINAL_ACTUAL_COMPLETED_STOP_STATE')
    stop = learning._EarlyStop({'early_stopping': manifest['settings']['early_stopping']})
    stop.load_state_dict(state)
    require(last['metadata']['stage'] == learning.SCHEMA and last['metadata']['synthetic'] is False
        and last['metadata']['objective'] == learning.OBJECTIVE and last['metadata']['training_role'] == 'fit'
        and last['metadata']['final_RNA_used'] is False and last['metadata']['scientific_qualified'] is False
        and type(last['epoch']) is int and last['epoch'] == result['epochs_completed']
        and type(last['counters']['optimizer_step']) is int and last['counters']['optimizer_step'] == result['optimizer_steps'] == stop.steps
        and last['extra']['cursor'] == 0 and last['extra']['order'] is None and last['extra']['history'] == result['history']
        and (stop.stopped if result['reason'] == 'selection_plateau' else last['epoch'] == manifest['settings']['epochs']),
        'FINAL_ACTUAL_COMPLETED_LAST_CHECKPOINT')
    saved = load_checkpoint(Path(directory) / 'best.pt', manifest['identity'])
    metadata = saved['metadata']
    require(metadata['stage'] == learning.SCHEMA and metadata['synthetic'] is False
        and metadata['objective'] == learning.OBJECTIVE and metadata['training_role'] == 'fit'
        and metadata['final_RNA_used'] is False and metadata['scientific_qualified'] is False
        and type(saved['counters']['optimizer_step']) is int and 0 < saved['counters']['optimizer_step'] <= result['optimizer_steps']
        and type(saved['epoch']) is int and 0 < saved['epoch'] <= result['epochs_completed'], 'FINAL_ACTUAL_SELECTED_CHECKPOINT')
    selected_stop = saved['extra']['early_stop']
    require(type(saved['extra']['cursor']) is int and saved['extra']['cursor'] == 0 and saved['extra']['order'] is None
        and saved['extra']['history'] == result['history'][:saved['epoch']]
        and saved['counters']['optimizer_step'] == result['history'][saved['epoch'] - 1]['optimizer_step']
        and type(selected_stop['steps']) is int and selected_stop['steps'] == saved['counters']['optimizer_step']
        and type(selected_stop['updates']) is int and selected_stop['updates'] == saved['epoch'],
        'FINAL_ACTUAL_SELECTED_HISTORY_AND_COUNTER')
    require(all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in saved['model'].values()), 'FINAL_FINITE_FROZEN_SOURCE_PARAMETERS')
    model = MartinSourceModel(context.gene_ids, context.contract, width=manifest['settings']['width']).to(device)
    model.validate_checkpoint_metadata(last['metadata'])
    model.load_registered_state(saved['model'], metadata)
    return model.eval().requires_grad_(False), result


def evaluate_final(root, *, fold_id, trained_run, output, device='cpu'):
    context = FinalSourceInputs(root, fold_id)
    output = Path(output).resolve()
    require(output.is_relative_to(context.root / 'experiments') and not output.exists(), 'NEW_FINAL_EVALUATION_OUTPUT_REQUIRED')
    model, result = load_frozen_model(trained_run, context, device)
    before = {name: sha(Path(trained_run) / name) for name in ('best.pt', 'last.pt', 'result.json', 'run_manifest.json')}
    records, seen = [], {d: {} for d in DOMAINS}
    try:
        with torch.no_grad():
            for request in sorted(context.roles['final_test'], key=lambda r: r['request_id']):
                for domain in DOMAINS:
                    for part in ('reference', 'target'):
                        batch = context.batch(request, domain, part)
                        batch.validate()
                        require(batch.final_scope_json == canonical(context.scope), 'FINAL_EXPECTED_EVALUATION_SCOPE')
                        for index, sid in enumerate(batch.sample_ids):
                            sample = learning.one_sample(batch, index)
                            identity = learning.observation_identity(sample)
                            previous = seen[domain].get(sid)
                            require(previous is None or previous == identity, 'FINAL_DUPLICATE_SAMPLE_CONFLICT')
                            if previous is not None:
                                continue
                            seen[domain][sid] = identity
                            predicted = model(sample)
                            mask = sample.observed_mask.to(predicted['source_values'].device)
                            raw = sample.raw_values.to(predicted['source_values'])
                            error = predicted['source_values'][mask] - raw[mask]
                            records.append({'sample_id': sid, 'domain': domain, 'units': sample.units, 'part': part,
                                'request_id': sample.request_id, 'background': sample.background, 'protocol': sample.protocol,
                                'role': 'final_test', 'observed_genes': int(mask.sum()),
                                'model_coordinate_MSE': float(model.observation_loss(predicted, sample)),
                                'raw_source_MSE': float(error.square().mean()), 'raw_source_MAE': float(error.abs().mean())})
        require(all(set(seen[d]) == set(context.ids['final_test']) for d in DOMAINS), 'COMPLETE_FINAL_TWO_DOMAIN_SOURCE_UNITS_REQUIRED')
        context.verify_source_bytes()
        require(all(sha(Path(trained_run) / name) == expected for name, expected in before.items()), 'FINAL_FROZEN_TRAINING_ASSET_CHANGED')
        value = {'schema': SCHEMA, 'scope': context.scope, 'training_identity': result['identity'], 'frozen_assets_sha256': before,
            'records': records, 'by_domain': {d: {k: float(np.mean([r[k] for r in records if r['domain'] == d]))
                for k in ('model_coordinate_MSE', 'raw_source_MSE', 'raw_source_MAE')} for d in DOMAINS},
            'final_counts': context.counts['final_test'], 'final_RNA_values_actually_read': True,
            'RNA_units_not_biological_replicates_times_two': True, 'fixed_fit_statistics_used': True,
            'model_selection': False, 'optimizer_steps': 0, 'functional_Y_read': False, 'scientific_qualified': False,
            'source_E_D_only_not_shared_latent_P_C_G_Q_validation': True}
        require(all(np.isfinite(r[k]) for r in records for k in ('model_coordinate_MSE', 'raw_source_MSE', 'raw_source_MAE')), 'FINITE_FINAL_ERRORS')
        output.mkdir(parents=True)
        atomic_json(output / 'final_metrics.json', value)
        return value
    finally:
        context.close()
