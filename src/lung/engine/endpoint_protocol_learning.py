"""Observation-unit ridge baseline for the new protocol interface.

This is a project-specific G implementation, not evidence of biological benefit.
No legacy endpoint/world/DA-AC code is changed.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .endpoint_protocol_model import ProtocolConditionAdapter, RegisteredProtocolScorer, canonical

SCHEMA = 'protocol-G-ridge-fraction-v1'
LAMBDA = 1.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def bindings(bridge, cache_manifest, role_identity, fit_ids):
    """Serving identities: hashes only, no observation/label deserialization."""
    from .endpoint_protocol_producer import load_standard_cache
    load_standard_cache(cache_manifest, bridge)
    m = json.loads(Path(cache_manifest).read_text(encoding='utf-8-sig'))
    return {'input_manifest_sha256': sha(Path(bridge.directory) / 'manifest.json'),
            'p_cache_manifest_sha256': sha(cache_manifest), 'p_feature_identity': m['feature_identity'],
            'role_contract': role_identity, 'legal_fit_observation_ids': list(fit_ids), 'learning_source_sha256': sha(__file__),
            'adapter_source_sha256': sha(Path(__file__).with_name('endpoint_protocol_model.py'))}


def plan_bindings(bridge, cache_manifest, plan, fold_id, branch='main'):
    """Role IDs are metadata, not outcomes; serving can verify exact membership."""
    b = bindings(bridge, cache_manifest, plan.identity(fold_id, branch),
                 [u['observation_id'] for u in plan.metadata_units(fold_id, branch, role='fit')])
    return {**b, 'legal_selection_observation_ids': [u['observation_id'] for u in
                plan.metadata_units(fold_id, branch, role='selection')],
            'reserved_test_observation_ids': [u['observation_id'] for role in ('final_test', 'external_test')
                for u in plan.metadata_units(fold_id, branch, role=role)]}


def validate_units(units):
    if not units or len({u['observation_id'] for u in units}) != len(units):
        raise ValueError('EMPTY_OR_DUPLICATE_SOURCE_OBSERVATIONS')
    for u in units:
        if (u['role'] != 'fit' or u['weight'] != 1.0
                or u['label_kind'] != 'measured_published_group_mean'
                or not np.isfinite(u['label_fraction']) or not 0 <= u['label_fraction'] <= 1):
            raise ValueError('FIT_SOURCE_OBSERVATION_UNIT_REQUIRED')


def ridge_parameters(x, y):
    """Fixed lambda, fit-only population normalization, unpenalized intercept.

    Dual solve is sized by observations (20 or fewer), not 2k feature columns.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or len(x) < 2 or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('INVALID_FIT_ARRAYS')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale == 0, 1., scale)
    z, intercept = (x - mean) / scale, float(y.mean())
    beta = z.T @ np.linalg.solve(z @ z.T + LAMBDA * np.eye(len(z)), y - intercept)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'beta': beta.tolist(),
            'intercept': intercept}


class FixedRidge:
    def __init__(self, parameters):
        self.mean = np.asarray(parameters['mean'], dtype=float)
        self.scale = np.asarray(parameters['scale'], dtype=float)
        self.beta = np.asarray(parameters['beta'], dtype=float)
        self.intercept = parameters['intercept']
        if (self.mean.ndim != 1 or not self.mean.size or self.scale.shape != self.mean.shape
                or self.beta.shape != self.mean.shape or not np.isfinite(self.intercept)
                or not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.beta))
                or not (self.scale > 0).all()):
            raise ValueError('INVALID_FIXED_PARAMETERS')
        for v in (self.mean, self.scale, self.beta):
            v.setflags(write=False)

    def __call__(self, vectors):
        x = np.asarray(vectors, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('INVALID_SCORER_VECTORS')
        return np.clip(((x - self.mean) / self.scale) @ self.beta + self.intercept, 0., 1.)


def fit_g(bridge, plan, *, fold_id, branch='main', cache_manifest, output):
    """Explicit training entry; caller must obtain units through the role contract.

    All rows are source observations; generated cells never expand Y or weights.
    """
    from ..data.endpoint_protocol_evaluation import EndpointProtocolEvaluationPlan
    if not isinstance(plan, EndpointProtocolEvaluationPlan):
        raise ValueError('VERIFIED_EVALUATION_PLAN_REQUIRED')
    role_identity = plan.identity(fold_id, branch=branch)
    units = plan.fit_units(fold_id, branch=branch)
    validate_units(units)
    output = Path(output)
    if output.exists():
        raise FileExistsError('IMMUTABLE_CHECKPOINT_EXISTS')
    bound = plan_bindings(bridge, cache_manifest, plan, fold_id, branch)
    adapter = ProtocolConditionAdapter()
    encoded = [adapter.encode(bridge.require_response_features(u['candidate_id'],
                       cache_manifest=cache_manifest)) for u in units]
    if len({canonical(e.scopes) for e in encoded}) != 1:
        raise ValueError('HETEROGENEOUS_ENDPOINT_STATE_OR_SCENARIO_BLOCK')
    names = encoded[0].feature_names
    if any(e.feature_names != names for e in encoded):
        raise ValueError('FEATURE_SCHEMA_CHANGED')
    for u, e in zip(units, encoded):
        if any(canonical(u[k]) != canonical(e.scopes[k]) for k in
               ('source_id', 'state_id', 'dose_scenario', 'endpoint_definition')):
            raise ValueError('SUPERVISION_CANDIDATE_SCOPE_MISMATCH')
    if bound['input_manifest_sha256'] != role_identity['input_manifest_sha256']:
        raise ValueError('ROLE_PLAN_INPUT_CHANGED')
    params = ridge_parameters(np.stack([e.vector for e in encoded]),
                              [u['label_fraction'] for u in units])
    meta = {'schema': SCHEMA, 'scorer_kind': 'endpoint_G', 'algorithm': 'fixed_lambda_ridge_baseline',
            'lambda': LAMBDA, 'feature_names': list(names), 'scope': encoded[0].scopes,
            'value_scale': 'terminal_Annexin_fraction', 'bindings': bound,
            'fit_observation_ids': [u['observation_id'] for u in units],
            'fit_label_sources_sha256': identity([{k: u[k] for k in
                ('observation_id', 'source_record', 'label_fraction', 'raw_unit')} for u in units]),
            'weight_per_source_observation': 1, 'normalization_role': 'fit',
            'parameter_selection': 'fixed_lambda_no_selection_or_calibration',
            'synthetic': False, 'actual_fit_completed': True, 'fit_count': len(units),
            'fit_operation': 'closed_form_ridge_solve', 'model_effectiveness_validated': False,
            'reward_qualification': 'pending_independent_measured_validation'}
    checkpoint = {'metadata': meta, 'parameters': params}
    checkpoint['content_identity'] = identity(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(checkpoint, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    return checkpoint


def load_g(path, *, expected_bindings, expected_scope):
    """Only JSON parameters; never loads supervision or outcomes.

    Consistency checks are not cryptographic proof that fitting was executed.
    """
    checkpoint = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(checkpoint) != {'metadata', 'parameters', 'content_identity'}:
        raise ValueError('CHECKPOINT_SCHEMA_INVALID')
    content = {k: checkpoint[k] for k in ('metadata', 'parameters')}
    if checkpoint['content_identity'] != identity(content):
        raise ValueError('CHECKPOINT_CONTENT_CHANGED')
    m = checkpoint['metadata']
    if (m['schema'] != SCHEMA or m['algorithm'] != 'fixed_lambda_ridge_baseline' or m['scorer_kind'] != 'endpoint_G' or m['lambda'] != LAMBDA
            or m['bindings'] != expected_bindings or canonical(m['scope']) != canonical(expected_scope)
            or m['synthetic'] is not False or m['actual_fit_completed'] is not True
            or m['normalization_role'] != 'fit' or m['weight_per_source_observation'] != 1
            or m['fit_count'] < 2 or len(m['fit_observation_ids']) != m['fit_count']
            or len(set(m['fit_observation_ids'])) != m['fit_count']
            or m['fit_observation_ids'] != expected_bindings['legal_fit_observation_ids']
            or m['model_effectiveness_validated'] is not False
            or m['value_scale'] != 'terminal_Annexin_fraction'
            or m['reward_qualification'] != 'pending_independent_measured_validation'):
        raise ValueError('CHECKPOINT_BINDING_OR_QUALIFICATION_INVALID')
    model = FixedRidge(checkpoint['parameters'])
    if len(model.mean) != len(m['feature_names']):
        raise ValueError('CHECKPOINT_FEATURE_DIMENSION_INVALID')
    registered = RegisteredProtocolScorer(model, checkpoint['content_identity'], m['algorithm'],
        tuple(m['feature_names']), m['scope']['endpoint_definition'], m['value_scale'],
        checkpoint['content_identity'], 'endpoint_G')
    return CheckpointBoundProtocolScorer(registered, m['scope'], m['bindings'])


@dataclass(frozen=True)
class CheckpointBoundProtocolScorer:
    """Loaded artifact wrapper; old ranking API rejects this different type."""
    _registered: RegisteredProtocolScorer
    scope: dict
    artifact_bindings: dict

    @property
    def scorer(self):
        return self._registered.scorer

    @property
    def scorer_kind(self):
        return self._registered.scorer_kind

    @property
    def feature_names(self):
        return self._registered.feature_names


def rank_checkpoint_candidates(bridge, candidate_ids, *, checkpoint_scorer, plan,
                               fold_id, branch='main', cache_manifest):
    from .endpoint_protocol_model import rank_protocol_candidates
    if not isinstance(checkpoint_scorer, CheckpointBoundProtocolScorer):
        raise ValueError('CHECKPOINT_BOUND_SCORER_REQUIRED')
    bound = plan_bindings(bridge, cache_manifest, plan, fold_id, branch)
    if bound != checkpoint_scorer.artifact_bindings:
        raise ValueError('CHECKPOINT_RANK_BINDINGS_CHANGED')
    encoded = [ProtocolConditionAdapter().encode(bridge.require_response_features(
        cid, cache_manifest=cache_manifest)) for cid in candidate_ids]
    if not encoded or any(e.scopes != checkpoint_scorer.scope for e in encoded):
        raise ValueError('CHECKPOINT_RANK_SCOPE_MISMATCH')
    result = rank_protocol_candidates(bridge, candidate_ids, scorer=checkpoint_scorer._registered,
                                     cache_manifest=cache_manifest, algorithm='G_enumeration_baseline')
    return {**result, 'record_consistency_verified': True, 'execution_certified': False,
            'biological_effectiveness_validated': False}
