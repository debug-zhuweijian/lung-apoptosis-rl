"""Versioned multi-state, per-assay ridge G; Q qualification remains pending.

Explicit fit entry validates native source eligibility before opening any Y.
No training, weight loading or inference occurs on module import or preflight.
"""
import json
from pathlib import Path

import numpy as np

from ..data.multistate_role_plan import digest, write_new
from ..data.martin_state_function import sha
from . import multistate_endpoint_model as model

SCHEMA = 'multistate-caspase-G-fixed-ridge-v1'
LAMBDA = 1.0


def bindings(source, plan, *, fold_id, head_id, native_cache_index):
    role = model.require_plan(source, plan, fold_id, head_id)
    return {'role_plan_identity': role,
            'native_cache_index_sha256': sha(native_cache_index),
            'native_coordinates': model._native_bindings(source),
            'legal_fit_observation_ids': [u['observation_id'] for u in plan.metadata_units(fold_id, head_id, 'fit')],
            'selection_observation_ids': [u['observation_id'] for u in plan.metadata_units(fold_id, head_id, 'selection')],
            'final_observation_ids': [u['observation_id'] for u in plan.metadata_units(fold_id, head_id, 'final_test')],
            'learning_source_sha256': sha(__file__),
            'adapter_source_sha256': sha(Path(__file__).with_name('multistate_endpoint_model.py'))}


def ridge_parameters(x, y):
    """Population moments from fit only; small-observation dual ridge lambda=1."""
    x, y = model.real_array(x), model.real_array(y)
    if (x.ndim != 2 or len(x) < 2 or x.shape[1] == 0 or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any()):
        raise ValueError('INVALID_SOURCE_FIT_ARRAYS')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale == 0, 1., scale)
    z, intercept = (x - mean) / scale, float(y.mean())
    beta = z.T @ np.linalg.solve(z @ z.T + LAMBDA * np.eye(len(x)), y - intercept)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'beta': beta.tolist(), 'intercept': intercept}


class NonnegativeRidge:
    def __init__(self, parameters):
        self.mean = model.real_array(parameters['mean']).copy()
        self.scale = model.real_array(parameters['scale']).copy()
        self.beta = model.real_array(parameters['beta']).copy()
        self.intercept = parameters['intercept']
        if (self.mean.ndim != 1 or not self.mean.size or self.scale.shape != self.mean.shape
                or self.beta.shape != self.mean.shape or type(self.intercept) not in (int, float)
                or not np.isfinite(self.intercept)
                or not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.beta))
                or not (self.scale > 0).all()):
            raise ValueError('INVALID_FROZEN_G_PARAMETERS')
        for a in (self.mean, self.scale, self.beta):
            a.setflags(write=False)

    def __call__(self, vectors):
        x = model.real_array(vectors)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('INVALID_FROZEN_G_FEATURES')
        result = ((x - self.mean) / self.scale) @ self.beta + self.intercept
        if not np.isfinite(result).all():
            raise ValueError('NONFINITE_TERMINAL_PREDICTION')
        return np.maximum(0., result)


def fit_g(source, plan, *, fold_id, head_id, native_cache_index, output):
    """Explicit future fitting API; no caller labels/vectors/scorer accepted."""
    model.require_plan(source, plan, fold_id, head_id)
    model._native_bindings(source)  # Pending before label parsing or output writes.
    output = Path(output)
    if output.exists():
        raise FileExistsError('IMMUTABLE_G_CHECKPOINT_EXISTS')
    bound = bindings(source, plan, fold_id=fold_id, head_id=head_id, native_cache_index=native_cache_index)
    units = plan.metadata_units(fold_id, head_id, 'fit')
    encoded = model.encode_units(source, plan, fold_id=fold_id, head_id=head_id,
                                 units=units, native_cache_index=native_cache_index)
    labels = plan.fit_units(fold_id, head_id)
    if ([u['observation_id'] for u in labels] != bound['legal_fit_observation_ids']
            or any(u['role'] != 'fit' or u['weight'] != 1 for u in labels)):
        raise ValueError('SOURCE_FIT_MEMBERSHIP_CHANGED')
    params = ridge_parameters(np.stack([e.vector for e in encoded]), [u['value'] for u in labels])
    domain = plan.training_domain(fold_id, head_id)
    if bound != bindings(source, plan, fold_id=fold_id, head_id=head_id, native_cache_index=native_cache_index):
        raise ValueError('SOURCE_ROLE_OR_CACHE_CHANGED_DURING_FIT')
    metadata = {'schema': SCHEMA, 'scorer_kind': 'endpoint_G', 'algorithm_identity': model.ALGORITHM,
                'lambda': LAMBDA, 'training_domain': domain, 'value_scale': domain['value_scale'],
                'feature_names': list(encoded[0].feature_names), 'bindings': bound,
                'fit_observation_ids': bound['legal_fit_observation_ids'], 'weight_per_source_observation': 1,
                'normalization_role': 'fit', 'output_policy': 'nonnegative_no_upper_clip',
                'fit_label_identity': digest(labels), 'actual_fit_completed': True,
                'fit_operation': 'closed_form_source_observation_ridge',
                'reward_qualification': 'pending_independent_selection_validation',
                'biological_effectiveness_validated': False, 'synthetic': False}
    checkpoint = {'metadata': metadata, 'parameters': params}
    checkpoint['content_identity'] = digest(checkpoint)
    write_new(output, checkpoint)
    return checkpoint


def load_g(path, *, source, plan, fold_id, head_id, native_cache_index):
    """JSON only; exact source/role/native/cache/domain and private G identity."""
    bound = bindings(source, plan, fold_id=fold_id, head_id=head_id, native_cache_index=native_cache_index)
    checkpoint = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if set(checkpoint) != {'metadata', 'parameters', 'content_identity'}:
        raise ValueError('INVALID_G_CHECKPOINT_SCHEMA')
    content = {k: checkpoint[k] for k in ('metadata', 'parameters')}
    if checkpoint['content_identity'] != digest(content):
        raise ValueError('FROZEN_G_CONTENT_CHANGED')
    m = checkpoint['metadata']
    names = m.get('feature_names', [])
    if (m.get('schema') != SCHEMA or m.get('scorer_kind') != 'endpoint_G'
            or m.get('algorithm_identity') != model.ALGORITHM or m.get('lambda') != LAMBDA
            or m.get('bindings') != bound or m.get('training_domain') != plan.training_domain(fold_id, head_id)
            or m.get('value_scale') != m['training_domain']['value_scale']
            or m.get('fit_observation_ids') != bound['legal_fit_observation_ids']
            or m.get('normalization_role') != 'fit' or m.get('weight_per_source_observation') != 1
            or m.get('output_policy') != 'nonnegative_no_upper_clip'
            or m.get('actual_fit_completed') is not True or m.get('synthetic') is not False
            or m.get('biological_effectiveness_validated') is not False
            or m.get('reward_qualification') != 'pending_independent_selection_validation'
            or not names or len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError('G_MODEL_BINDINGS_OR_MEASUREMENT_DOMAIN_MISMATCH')
    scorer = NonnegativeRidge(checkpoint['parameters'])
    if len(scorer.mean) != len(names):
        raise ValueError('G_FEATURE_SCHEMA_DIMENSION_CHANGED')
    return scorer, m, checkpoint['content_identity']


def terminal_q_preflight():
    return {'status': 'pending_qualification', 'scorer_kind': 'private_candidate_Q',
            'target': 'single_terminal_reward_no_bootstrap_no_cost',
            'pending': ['eligible_native_bulk72_state_protocol_backend_and_cache',
                        'actual_frozen_multistate_per_measurement_head_G',
                        'predeclared_caspase_scale_selection_rule',
                        'controlled_frozen_selection_predictions_and_independent_measured_validation',
                        'strict_generated_terminal_decision_artifact_bound_to_qualified_G'],
            'old_LOBO_no_selection_role_eligible': False,
            'Q_fit_implemented': False, 'reward_qualified': False,
            'actual_model_execution': False, 'RL_superiority_guaranteed': False}
