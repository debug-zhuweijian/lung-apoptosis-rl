"""Separate Martin measurement heads and complete, single-state decision blocks.

Production encoding resolves typed native-P artifacts internally. Caller feature
vectors, raw bulk RNA and the old standard24 P cache are not accepted inputs.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.multistate_role_plan import MultiStateRolePlan, digest
from ..data.martin_state_function import sha

ALGORITHM = 'multistate-fixed-lambda-ridge-terminal-v1'
CACHE_INDEX_SCHEMA = 'multistate-native-descriptor-index-v1'


def real_array(value):
    """Reject boolean/complex values before NumPy can coerce or discard them."""
    def check(item):
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in 'iuf':
                raise ValueError('REAL_NUMERIC_ARRAY_REQUIRED_NO_BOOL_OR_COMPLEX')
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif (isinstance(item, (bool, np.bool_))
              or not isinstance(item, (int, float, np.integer, np.floating))):
            raise ValueError('REAL_NUMERIC_ARRAY_REQUIRED_NO_BOOL_OR_COMPLEX')
    check(value)
    return np.asarray(value, dtype=np.float64)


def require_plan(source, plan, fold_id, head_id):
    if not isinstance(plan, MultiStateRolePlan) or plan.source is not source:
        raise ValueError('VERIFIED_NEW_MULTISTATE_ROLE_PLAN_REQUIRED')
    plan.verify()
    return plan.identity(fold_id, head_id)


def required_bindings(source, plan, fold_id, head_id, candidate_id):
    identity = require_plan(source, plan, fold_id, head_id)
    c = source.candidates[candidate_id]
    if c['measurement_domain_id'] != head_id:
        raise ValueError('CANDIDATE_MEASUREMENT_HEAD_MISMATCH')
    return {'algorithm_identity': ALGORITHM, 'role_plan_identity': identity,
            'source_manifest_sha256': sha(source.directory / 'manifest.json'),
            'candidate_id': candidate_id, 'measurement_domain_id': head_id,
            'candidate_binding_sha256': digest(c),
            'state_reference_id': c['state_reference_id'],
            'state_reference_binding_sha256': digest(source.references[c['state_reference_id']]),
            'actual_protocol_sha256': digest(c['actual_protocol']),
            'response_request_sha256': digest(c['response_request'])}


def _index(path):
    path = Path(path).resolve()
    index = json.loads(path.read_text(encoding='utf-8-sig'))
    content = {k: v for k, v in index.items() if k != 'content_identity'}
    if (index.get('schema') != CACHE_INDEX_SCHEMA or index.get('content_identity') != digest(content)
            or not isinstance(index.get('descriptors'), dict)):
        raise ValueError('INVALID_NATIVE_DESCRIPTOR_INDEX')
    return path, index


def preflight(source, plan, *, fold_id, head_id, native_cache_index=None):
    """Metadata only; Martin currently has no eligible native bulk72 backend."""
    identity = require_plan(source, plan, fold_id, head_id)
    domain = plan.training_domain(fold_id, head_id)
    if native_cache_index is not None:
        _index(native_cache_index)  # Structure cannot establish producer eligibility.
    return {'status': 'pending_native_backend', 'identity': identity, 'training_domain': domain,
            'counts': {r: len(plan.metadata_units(fold_id, head_id, r))
                       for r in ('fit', 'selection', 'final_test')},
            'missing': ['eligible_native_bulk_source_decoder',
                        'native_target_state_and_full_72h_protocol_response_backend',
                        'registered_compatible_response_weights_and_native_descriptors'],
            'labels_parsed': False, 'model_execution_performed': False,
            'actual_G_fit_completed': False, 'full_scientific_goal_passed': False}


@dataclass(frozen=True)
class EncodedNativeCandidate:
    candidate_id: str
    feature_names: tuple
    vector: np.ndarray
    bindings: dict
    measurement_domain: dict


def _load_producer_descriptor(path, expected_bindings):
    """Deferred import, so metadata CLI never constructs or loads a model."""
    from .native_response_producer import load_native_descriptor
    return load_native_descriptor(path, expected_bindings=expected_bindings)


def _native_bindings(source):
    """No registered Martin bulk decoder/coordinates/weights exist in v1.

    A future version must resolve this from audited source-specific coordinates,
    never from caller-provided cache metadata or the joint single-cell model.
    """
    raise ValueError('NATIVE_BULK72_BACKEND_PENDING')


def _function_request(source, candidate_id):
    c = source.candidates[candidate_id]
    return {'measurement_level': 'bulk', 'state_reference_id': c['state_reference_id'],
            'state_reference_binding_sha256': digest(source.references[c['state_reference_id']]),
            'cell_background': c['cell_background'], 'chemical_id': c['chemical']['chemical_id'],
            'dose_uM': c['nominal_dose_uM'], 'actual_protocol': c['actual_protocol'],
            'target_observation_h': c['actual_protocol']['observation_window_h']}


class NativeMultiStateAdapter:
    def encode(self, candidate_id, *, source, plan, fold_id, head_id, native_cache_index):
        bindings = required_bindings(source, plan, fold_id, head_id, candidate_id)
        native_bindings = _native_bindings(source)
        root, index = _index(native_cache_index)
        item = index['descriptors'].get(candidate_id)
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256', 'bindings', 'request_binding'}
                or item['bindings'] != bindings or item['request_binding'] != _function_request(source, candidate_id)):
            raise ValueError('NATIVE_CANDIDATE_BINDINGS_MISMATCH')
        path = (root.parent / item['path']).resolve()
        if not path.is_relative_to(root.parent) or sha(path) != item['sha256']:
            raise ValueError('NATIVE_DESCRIPTOR_PATH_OR_BYTES_CHANGED')
        descriptor = _load_producer_descriptor(path, native_bindings)
        # The loader owns producer/type/array provenance verification. The bridge
        # independently requires actual target state, protocol and eligible source.
        from .native_response_producer import VerifiedNativeDescriptor
        if (not isinstance(descriptor, VerifiedNativeDescriptor)
                or descriptor.bindings != native_bindings or descriptor.execution_kind != 'native'
                or descriptor.request.get('function_candidate_binding') != item['request_binding']
                or descriptor.request.get('measurement_level') != 'bulk'
                or descriptor.request.get('role_plan_identity') != bindings['role_plan_identity']):
            raise ValueError('NATIVE_P_CAPABILITY_PENDING')
        names = tuple(descriptor.feature_names)
        vector = real_array(descriptor.vector).copy()
        if (not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names)
                or vector.shape != (len(names),) or not np.isfinite(vector).all()):
            raise ValueError('INVALID_TYPED_NATIVE_FEATURE_SCHEMA')
        vector.setflags(write=False)
        return EncodedNativeCandidate(candidate_id, names, vector, bindings,
                                      plan.training_domain(fold_id, head_id))


def encode_units(source, plan, *, fold_id, head_id, units, native_cache_index):
    adapter = NativeMultiStateAdapter()
    encoded = [adapter.encode(u['candidate_id'], source=source, plan=plan, fold_id=fold_id,
                              head_id=head_id, native_cache_index=native_cache_index) for u in units]
    if not encoded or any(e.feature_names != encoded[0].feature_names
                          or e.measurement_domain != encoded[0].measurement_domain for e in encoded):
        raise ValueError('FEATURE_SCHEMA_OR_TRAINING_DOMAIN_CHANGED')
    return encoded


def rank_block(source, plan, *, fold_id, head_id, block_id, role,
               native_cache_index, checkpoint, candidate_ids=None):
    """Single source-defined full block. No caller scorer or feature parameter."""
    from .multistate_endpoint_learning import load_g
    block = plan.decision_block(fold_id, head_id, block_id, role)
    expected = block['candidate_ids']
    if candidate_ids is not None and (len(candidate_ids) != len(set(candidate_ids))
                                     or set(candidate_ids) != set(expected)):
        raise ValueError('COMPLETE_SINGLE_STATE_HEAD_TIME_BLOCK_REQUIRED')
    units = [{'candidate_id': cid} for cid in expected]
    encoded = encode_units(source, plan, fold_id=fold_id, head_id=head_id, units=units,
                           native_cache_index=native_cache_index)
    model, metadata, model_identity = load_g(checkpoint, source=source, plan=plan,
        fold_id=fold_id, head_id=head_id, native_cache_index=native_cache_index)
    if encoded[0].feature_names != tuple(metadata['feature_names']):
        raise ValueError('SCORER_FEATURE_SCHEMA_MISMATCH')
    if block['state_reference_id'] not in metadata['training_domain']['allowed_state_references']:
        raise ValueError('UNDECLARED_SERVING_STATE')
    values = model(np.stack([e.vector for e in encoded]))
    rows = [{'candidate_id': cid, 'value': float(v),
             'rank': 1 + int(np.sum(values > v)), 'tied': bool(np.sum(values == v) > 1)}
            for cid, v in zip(expected, values)]
    return {'schema': 'multistate-native-G-ranking-v1', 'identity': plan.identity(fold_id, head_id),
            'block': block, 'role': role, 'predictions': rows,
            'ranking': sorted(rows, key=lambda r: -r['value']), 'scorer_kind': 'endpoint_G',
            'model_identity': model_identity, 'value_scale': metadata['value_scale'],
            'state_in_fit': block['state_reference_id'] in metadata['training_domain']['fit_state_references'],
            'evidence': 'model_prediction_not_measured_outcome', 'labels_parsed': False,
            'biological_effectiveness_validated': False, 'reward_qualified': False}
