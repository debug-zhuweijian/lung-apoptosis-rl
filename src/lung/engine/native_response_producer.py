"""Per-request reference-conditioned expression production; metadata fails closed.

No checkpoint loader or torch import runs during metadata preflight. This module
does not read response targets, functional labels, or the standard-P cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 'native-response-producer-v1'
SCHEMA = 'native-response-descriptor-v1'
BASE = 'common_gene_library_10000_log1p'
UNIT = 'source_raw_counts_no_new_normalization'
SUPPORTED = {'tahoe_continuous_exposure_24h_v1', 'sciplex_author_reported_endpoint_v1'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(ok, reason):
    if not ok:
        raise ValueError('NATIVE_P_' + reason)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def protocol_status(request):
    if request['measurement_level'] != 'single_cell':
        return 'pending_bulk_decoder_domain_and_matching_weights'
    p = request['actual_protocol']
    if (request['protocol_id'] not in SUPPORTED or p.get('washout_h') not in (None, 0)
            or p.get('prior_drug') or any(p.get(k) for k in
            ('exposure_intervals', 'recovery_h', 'media_replacement', 'previous_drug_retention', 'interval_h'))):
        return 'pending_full_physical_protocol_conditioner_and_RNA_supervision'
    duration = p.get('source_time_h') if p.get('source_time_h') is not None else p.get('duration_h')
    if duration not in (24, 72) or (request['protocol_id'].startswith('tahoe') and duration != 24):
        return 'unsupported_native_endpoint'
    return 'native_single_exposure_endpoint_only'


def validate_metadata(metadata, bindings):
    """Check declared expression coordinates and matching AE/response weights."""
    require(isinstance(metadata, dict), 'MODEL_METADATA_REQUIRED')
    require(metadata.get('bindings') == bindings, 'MODEL_BINDINGS_MISMATCH')
    for name, stages in (('representation', {'latent_ae'}), ('response', {'latent_cfm', 'latent_members'})):
        stage = metadata.get(name, {})
        require(stage.get('stage') in stages, 'STAGE_MISMATCH')
        require(type(stage.get('optimizer_step')) is int and stage['optimizer_step'] > 0, 'TRAINED_COUNTER_REQUIRED')
        require(stage.get('data_identity') == bindings['data_identity'], 'MODEL_DATA_IDENTITY_MISMATCH')
        require(stage.get('gene_axis_sha256') == bindings['gene_axis_sha256'], 'MODEL_GENE_AXIS_MISMATCH')
        require(stage.get('normalization_sha256') == bindings['normalization_sha256'], 'MODEL_NORMALIZATION_MISMATCH')
        require(stage.get('context_vocabulary') == bindings['context_vocabulary'], 'MODEL_CONTEXT_VOCABULARY_MISMATCH')
        require(stage.get('base_transform') == BASE and stage.get('units') == UNIT, 'MODEL_BASE_UNITS_MISMATCH')
        require(stage.get('synthetic') is False, 'REAL_TRAINING_REQUIRED')
        require(isinstance(stage.get('checkpoint'), str) and bool(stage['checkpoint']), 'CHECKPOINT_PATH_REQUIRED')
        require(isinstance(stage.get('checkpoint_sha256'), str) and len(stage['checkpoint_sha256']) == 64, 'CHECKPOINT_HASH_REQUIRED')
    require(metadata.get('response', {}).get('representation_checkpoint_sha256') ==
            metadata['representation']['checkpoint_sha256'], 'AE_RESPONSE_BINDING_MISMATCH')


def preflight(requests, bindings, metadata=None):
    statuses = {}
    for request in requests:
        status = protocol_status(request)
        statuses[status] = statuses.get(status, 0) + 1
    reasons = []
    if metadata is None:
        reasons.append('matching_AE_response_metadata_missing')
    else:
        try:
            validate_metadata(metadata, bindings)
        except ValueError as error:
            reasons.append(str(error))
    return {'schema': VERSION, 'status_counts': statuses, 'bindings': bindings,
            'matching_metadata_valid': not reasons,
            'pending': reasons, 'weights_read': False, 'model_created': False,
            'forward_called': False, 'actual_model_inference': False,
            'actual_descriptors': 0, 'pool_C_required_for_expression_production': False}


@dataclass(frozen=True)
class NativeMetadata:
    """Actual ordered coordinates and native action table, loaded without models."""
    bindings: dict
    gene_ids: tuple
    mean: np.ndarray
    scale: np.ndarray
    conditions: object
    fingerprints: np.ndarray

    @classmethod
    def from_joint(cls, root, units):
        import pandas as pd
        from lung.data.native_response_units import JOINT
        root = Path(root)
        asset = root / JOINT
        manifest = read(asset / 'manifest.json')
        for name in ('genes.parquet', 'conditions.parquet', 'fingerprints.npy', 'normalization.npz', 'neighbors.npz'):
            require(sha(asset / name) == manifest['files'][name]['sha256'], 'TRAINING_ASSET_CHANGED')
        genes = tuple(pd.read_parquet(asset / 'genes.parquet').ensembl_id.astype(str))
        require(len(genes) == len(set(genes)), 'ORDERED_UNIQUE_GENES_REQUIRED')
        conditions = pd.read_parquet(asset / 'conditions.parquet').set_index('condition_id', drop=False)
        vocab = {name: sorted(conditions[name].dropna().unique().tolist())
                 for name in ('source_id', 'background_id', 'protocol_id')}
        with np.load(asset / 'normalization.npz', allow_pickle=False) as norm:
            mean, scale = norm['mean'].copy(), norm['scale'].copy()
        require(mean.shape == scale.shape == (len(genes),) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), 'NORMALIZATION_INVALID')
        fp = np.load(asset / 'fingerprints.npy', allow_pickle=False)
        bindings = {'data_identity': digest({'signature': manifest['signature'], 'files': manifest['files']}),
                    'gene_axis_sha256': sha(asset / 'genes.parquet'), 'ordered_gene_ids_sha256': digest(genes),
                    'normalization_sha256': sha(asset / 'normalization.npz'),
                    'neighbors_sha256': sha(asset / 'neighbors.npz'), 'base_transform': BASE, 'units': UNIT,
                    'context_vocabulary': vocab, 'native_units_manifest_sha256': sha(units.asset / 'manifest.json'),
                    'source_bindings': read(units.asset / 'manifest.json')['source_bindings']}
        for a in (mean, scale, fp):
            a.setflags(write=False)
        return cls(bindings, genes, mean, scale, conditions, fp)

    def action(self, request):
        row = self.conditions.loc[request['condition_id']]
        for name in ('source_id', 'background_id', 'protocol_id'):
            require(row[name] == request[name], 'REQUEST_CONTEXT_MISMATCH')
        require(bool(row['eligible']) and row['chemical_or_experiment_group_id'] == request['chemical_id'], 'CHEMICAL_IDENTITY_MISMATCH')
        require(float(row['dose_um']) == request['dose_uM'], 'DOSE_MISMATCH')
        duration = float(row['duration_h'])
        protocol = request['actual_protocol']
        native_time = protocol.get('source_time_h') if protocol.get('source_time_h') is not None else protocol.get('duration_h')
        require(duration == native_time, 'SOURCE_TIME_MISMATCH')
        fp = self.fingerprints[int(row['fingerprint_row'])].astype(np.float32)
        require(fp.shape == (2048,) and np.isfinite(fp).all() and np.isin(fp, [0, 1]).all() and fp.any(), 'FINGERPRINT_INVALID')
        vocab = self.bindings['context_vocabulary']
        return {'fingerprint': fp, 'dose': request['dose_uM'], 'duration': duration,
                **{name: vocab[name].index(request[name]) for name in vocab}, 'is_vehicle': False}


def load_adapter(root, metadata, coordinates, *, device='cpu'):
    """Load AE and response parameters directly in the declared coordinates."""
    validate_metadata(metadata, coordinates.bindings)
    import torch
    from lung.models import ModelConfig
    from lung.models.latent_staged import SharedAutoencoder, LatentMember
    from .latent_pipeline import LatentMemberAdapter
    from .full_protocol_response import equal_weight, finite_weights
    snapshots = {}
    for name in ('representation', 'response'):
        item = metadata[name]
        path = (Path(root) / item['checkpoint']).resolve()
        require(path.is_relative_to(Path(root).resolve()), 'CHECKPOINT_PATH_OUTSIDE_ROOT')
        require(sha(path) == item['checkpoint_sha256'], 'CHECKPOINT_BYTES_CHANGED')
        saved = torch.load(path, weights_only=True, map_location='cpu')
        actual = saved.get('metadata', {})
        require(actual.get('stage') == item['stage'] and actual.get('data_identity') == item['data_identity'], 'ACTUAL_STAGE_DATA_MISMATCH')
        require(saved.get('counters', {}).get('optimizer_step') == item['optimizer_step'], 'ACTUAL_COUNTER_MISMATCH')
        require(actual.get('native_bindings') == coordinates.bindings, 'ACTUAL_NATIVE_BINDINGS_REQUIRED')
        finite_weights(saved.get('model'))
        if name == 'response':
            finite_weights(saved.get('shared'))
        snapshots[name] = saved
    saved = snapshots['response']
    require(saved['metadata'].get('representation_hash') == metadata['representation']['checkpoint_sha256'], 'ACTUAL_AE_RESPONSE_BINDING_MISMATCH')
    shape = ModelConfig(**saved['metadata']['model_config'])
    require(shape.n_genes == len(coordinates.gene_ids) and shape.reference_mode == 'observed'
            and shape.backbone == 'compact_full_gene', 'ACTUAL_MODEL_SHAPE_MISMATCH')
    for field, name in (('n_sources', 'source_id'), ('n_backgrounds', 'background_id'), ('n_protocols', 'protocol_id')):
        require(getattr(shape, field) == len(coordinates.bindings['context_vocabulary'][name]), 'ACTUAL_CONTEXT_CAPACITY_MISMATCH')
    neighbors_path = Path(root) / '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/neighbors.npz'
    with np.load(neighbors_path, allow_pickle=False) as data:
        neighbors = torch.as_tensor(data['neighbors'].copy(), dtype=torch.long)
    shared = SharedAutoencoder(shape, neighbors)
    shared.load_state_dict(snapshots['representation']['model'], strict=True)
    require(set(saved['shared']) == set(shared.state_dict()) and all(equal_weight(saved['shared'][k], v)
            for k, v in shared.state_dict().items()), 'EMBEDDED_AE_WEIGHTS_MISMATCH')
    member = LatentMember(shape, shared)
    member.load_state_dict(saved['model'], strict=True)
    shared.to(device).eval().requires_grad_(False)
    member.to(device).eval().requires_grad_(False)
    return LatentMemberAdapter(shared, member, shape)


@dataclass(frozen=True)
class VerifiedNativeDescriptor:
    vector: np.ndarray
    feature_names: tuple
    request: dict
    bindings: dict
    execution_kind: str
    cache_identity: str
    reference: dict


def produce(units, coordinates, request, metadata, output, *, role, seed, steps=32,
            microbatch=64, ablation='full_reference', device='cpu'):
    """Public native production: caller supplies metadata, never an adapter.

    Reopen source-bound coordinates before loading matching AE/response weights.
    """
    from lung.data.native_response_units import NativeResponseUnits
    require(type(units) is NativeResponseUnits and type(coordinates) is NativeMetadata,
            'SOURCE_BOUND_NATIVE_INPUTS_REQUIRED')
    require(type(metadata) is dict, 'MODEL_METADATA_REQUIRED')
    # Reopening must preserve the permission already granted to this reader.
    # Request JSON availability never grants a serving role or final-test access.
    require(role in units.allowed_roles, 'READER_ROLE_NOT_AUTHORIZED')
    require(units.request(request['request_id'], role=role) == request, 'FROZEN_REQUEST_MISMATCH')
    reopened = NativeResponseUnits(units.root, units.asset, allowed_roles=(role,), allow_test=role == 'final_test')
    actual = NativeMetadata.from_joint(units.root, reopened)
    require(actual.bindings == coordinates.bindings, 'SOURCE_COORDINATES_CHANGED')
    require(reopened.request(request['request_id'], role=role) == request, 'FROZEN_REQUEST_MISMATCH')
    require(protocol_status(request) == 'native_single_exposure_endpoint_only', 'NATIVE_PROTOCOL_OR_DECODER_PENDING')
    # Keep the declared weight and coordinate identities fixed during generation.
    frozen_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    adapter = load_adapter(units.root, frozen_metadata, actual, device=device)
    return _produce(reopened, actual, request, adapter, frozen_metadata, output, role=role, seed=seed,
                    steps=steps, microbatch=microbatch, ablation=ablation, device=device)


def _produce(units, coordinates, request, adapter, metadata, output, *, role, seed, steps=32,
            microbatch=64, ablation='full_reference', device='cpu'):
    """Full same-role reference mixture; mean ablation uses identical rows/noise."""
    validate_metadata(metadata, coordinates.bindings)
    binding_digest = digest(coordinates.bindings)
    metadata_digest = digest(metadata)
    require(role == request['role'], 'REQUEST_ROLE_MISMATCH')
    require(units.request(request['request_id'], role=role) == request, 'FROZEN_REQUEST_MISMATCH')
    require(protocol_status(request) == 'native_single_exposure_endpoint_only', 'NATIVE_PROTOCOL_OR_DECODER_PENDING')
    require(type(seed) is int and seed >= 0 and type(steps) is int and steps > 0
            and type(microbatch) is int and microbatch > 0, 'SEED_STEPS_BATCH_INVALID')
    require(ablation in ('full_reference', 'same_reference_mean'), 'ABLATION_INVALID')
    action = coordinates.action(request)
    reference = units.reference(request['request_id'], role=role,
        expected_gene_ids=coordinates.gene_ids, max_rows=None)
    require(reference['target_expression_used'] is False and reference['units'] == UNIT, 'REFERENCE_UNITS_OR_TARGET_USE_MISMATCH')
    require(reference.get('selection') == 'full_reference', 'FULL_REFERENCE_REQUIRED')
    rows, ids = reference['reference_rows'], reference['reference_ids']
    weights = np.asarray(reference['reference_weights'], dtype=np.float64)
    require(len(rows) == len(ids) == len(weights) and len(ids) == len(set(ids)) and len(ids) > 0, 'REFERENCE_IDENTITY_INVALID')
    require(all(row['role'] == role and row['cell_id'] == cid and row['condition_id'] in request['physical_reference_ids']
                for row, cid in zip(rows, ids)), 'REFERENCE_SCOPE_INVALID')
    require(np.isfinite(weights).all() and (weights > 0).all() and np.isclose(weights.sum(), 1), 'REFERENCE_WEIGHTS_INVALID')
    for well, specification in request['reference_sampling_weights'].items():
        selected = [i for i, row in enumerate(rows) if row['condition_id'] == well]
        require(len(selected) == specification['cells'], 'FULL_PHYSICAL_WELL_COUNT_MISMATCH')
        probability = specification['well_probability'] * specification['cell_probability_given_well']
        require(all(weights[i] == probability for i in selected), 'REFERENCE_MIXTURE_CHANGED')
    from lung.data.preprocess import standardized
    x = standardized(reference['reference'], coordinates.mean, coordinates.scale)
    require(x.shape == (len(ids), len(coordinates.gene_ids)), 'REFERENCE_AXIS_SHAPE_MISMATCH')
    if ablation == 'same_reference_mean':
        x = np.broadcast_to(weights @ x.astype(np.float64), x.shape).astype(np.float32).copy()
    import torch
    from .response import condition_kwargs, tensor
    from lung.models.objectives import physical_projection
    batch = {**action}
    generator = torch.Generator(device=device).manual_seed(seed)
    # Draw once so both ablations and different microbatch sizes share row noise.
    xt = torch.as_tensor(x, device=device)
    noise = adapter.sample_noise(xt, generator=generator)
    mean, scale = tensor(coordinates.mean, device), tensor(coordinates.scale, device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), microbatch):
            end = min(start + microbatch, len(x))
            condition = adapter.encode_condition(**condition_kwargs(batch, end-start,
                device, elapsed=0., horizon=action['duration']))
            raw = adapter.sample(xt[start:end], condition, noise=noise[start:end], steps=steps)
            projected = physical_projection(raw, mean, scale)
            if not projected.valid.all():
                raise FloatingPointError('INVALID_GENERATED_MASS: no fallback, dropping or fake zero')
            outputs.append(projected.x.cpu().numpy())
    generated = np.concatenate(outputs).astype(np.float64)
    require(generated.shape == x.shape and np.isfinite(generated).all(), 'GENERATED_AXIS_INVALID')
    mean = weights @ generated
    variance = weights @ ((generated - mean) ** 2)
    names = tuple('expression_mean:' + g for g in coordinates.gene_ids) + tuple('expression_variance:' + g for g in coordinates.gene_ids)
    vector = np.concatenate((mean, variance))
    require(digest(coordinates.bindings) == binding_digest and digest(metadata) == metadata_digest,
            'GENERATION_BINDINGS_CHANGED')
    # File identities are checked again before any cache is written.
    from lung.data.native_response_units import NativeResponseUnits
    current_units = NativeResponseUnits(units.root, units.asset,
        allowed_roles=(role,), allow_test=role == 'final_test')
    current = NativeMetadata.from_joint(units.root, current_units)
    require(digest(current.bindings) == binding_digest, 'SOURCE_CHANGED_DURING_GENERATION')
    for item in metadata.values():
        if isinstance(item, dict) and 'checkpoint' in item:
            require(sha(Path(units.root) / item['checkpoint']) == item['checkpoint_sha256'],
                    'MODEL_CHANGED_DURING_GENERATION')
    provenance = {'reference_ids': ids, 'reference_rows': rows, 'reference_weights': weights.tolist(),
                  'reference_semantics': request['reference_semantics'], 'same_cell_fate_supervision': False}
    body = {'schema': SCHEMA, 'producer_version': VERSION, 'request': request, 'bindings': coordinates.bindings,
            'model_metadata': metadata, 'model_metadata_sha256': digest(metadata),
            'role': role, 'seed': seed, 'steps': steps, 'ODE': 'native_adapter_Heun', 'ablation': ablation,
            'reference': provenance, 'feature_names': names, 'vector': vector.tolist(),
            'summary_definition': 'same_reference_weighted_projected_expression_mean_variance',
            'execution_kind': 'native', 'target_expression_used': False,
            'functional_Y_used': False, 'scientific_effect_verified': False}
    body['cache_identity'] = digest(body)
    path = Path(output) / (body['cache_identity'] + '.json')
    write_new(path, body)
    return load_native_descriptor(path, coordinates.bindings)


def load_native_descriptor(path, expected_bindings):
    """Strict cache loading, never caller-vector certification or standard fallback.

    expected_bindings come from the consumer's current data/model plan.
    Content identity checks consistency; it does not establish scientific validity.
    """
    body = read(path)
    fields = {'schema', 'producer_version', 'request', 'bindings', 'model_metadata',
        'model_metadata_sha256', 'role', 'seed', 'steps', 'ODE', 'ablation', 'reference',
        'feature_names', 'vector', 'summary_definition', 'execution_kind',
        'target_expression_used', 'functional_Y_used', 'scientific_effect_verified', 'cache_identity'}
    require(set(body) == fields, 'FIXED_CACHE_FIELDS_MISMATCH')
    require(body.get('schema') == SCHEMA, 'DESCRIPTOR_SCHEMA_MISMATCH')
    require(expected_bindings and body.get('bindings') == expected_bindings, 'DESCRIPTOR_BINDINGS_MISMATCH')
    identity = digest({k: v for k, v in body.items() if k != 'cache_identity'})
    require(identity == body.get('cache_identity'), 'CACHE_IDENTITY_CHANGED')
    kind = body.get('execution_kind')
    require(kind == 'native', 'NATIVE_EXECUTION_REQUIRED')
    validate_metadata(body['model_metadata'], expected_bindings)
    require(body['model_metadata_sha256'] == digest(body['model_metadata']), 'MODEL_METADATA_CHANGED')
    require(body['target_expression_used'] is False and body['functional_Y_used'] is False
            and body['role'] == body['request']['role'], 'TARGET_OR_ROLE_CONTAMINATION')
    require(body['producer_version'] == VERSION and type(body['seed']) is int and body['seed'] >= 0
            and type(body['steps']) is int and body['steps'] > 0 and body['ODE'] == 'native_adapter_Heun'
            and body['ablation'] in ('full_reference', 'same_reference_mean')
            and body['summary_definition'] == 'same_reference_weighted_projected_expression_mean_variance',
            'FIXED_PRODUCTION_CONTRACT_CHANGED')
    ref = body['reference']
    require(set(ref) == {'reference_ids', 'reference_rows', 'reference_weights',
        'reference_semantics', 'same_cell_fate_supervision'} and ref['same_cell_fate_supervision'] is False,
        'REFERENCE_SCHEMA_CHANGED')
    weights = np.asarray(ref['reference_weights'], dtype=np.float64)
    require(len(ref['reference_ids']) == len(ref['reference_rows']) == len(weights)
            and len(weights) > 0 and len(set(ref['reference_ids'])) == len(weights)
            and np.isfinite(weights).all() and (weights > 0).all() and np.isclose(weights.sum(), 1),
            'CACHED_REFERENCE_IDENTITY_INVALID')
    require(protocol_status(body['request']) == 'native_single_exposure_endpoint_only', 'NATIVE_PROTOCOL_OR_DECODER_PENDING')
    vector = np.asarray(body['vector'], dtype=np.float64)
    names = tuple(body['feature_names'])
    require(vector.ndim == 1 and len(vector) == len(names) and len(names) % 2 == 0
            and len(set(names)) == len(names) and np.isfinite(vector).all(), 'FIXED_SUMMARY_INVALID')
    n = len(names) // 2
    require(all(names[i].startswith('expression_mean:') and names[n+i] == names[i].replace('expression_mean:', 'expression_variance:', 1)
            for i in range(n)) and (vector[n:] >= 0).all(), 'FIXED_SUMMARY_SCHEMA_INVALID')
    require(digest(tuple(name.split(':', 1)[1] for name in names[:n])) == expected_bindings['ordered_gene_ids_sha256'], 'SUMMARY_ORDERED_GENE_AXIS_MISMATCH')
    vector.setflags(write=False)
    return VerifiedNativeDescriptor(vector, names, body['request'], body['bindings'], kind, identity, body['reference'])
