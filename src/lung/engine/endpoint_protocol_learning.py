"""Observation-unit ridge baseline for the new protocol interface.

This is a project-specific G implementation, not evidence of biological benefit.
No legacy endpoint/world/DA-AC code is changed. Private terminal Q remains pending
until an independent G reward qualification and decision sample contract exist.
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


def terminal_q_preflight():
    """Do not relabel G as an independently learned Q."""
    return {'status': 'pending', 'scorer_kind': 'private_candidate_Q',
            'missing': ['independently_qualified_frozen_G_reward', 'real_generated_terminal_decision_artifact',
                        'eligible_reward_validation_role_and_predeclared_criterion'],
            'target_definition': 'single_terminal_reward_no_bootstrap_no_cost_penalty',
            'proposed_reward_evidence': 'model_simulated', 'model_execution_performed': False,
            'private_Q_implemented': True, 'validated_reward': False}



Q_SCHEMA = 'protocol-private-terminal-Q-ridge-v1'
DECISION_SCHEMA = 'protocol-terminal-simulated-decisions-v1'


def _rule(value):
    if (set(value) != {'max_mae_fraction', 'require_pairwise_order_match'}
            or type(value['max_mae_fraction']) not in (int, float)
            or not np.isfinite(value['max_mae_fraction']) or not 0 <= value['max_mae_fraction'] <= 1
            or type(value['require_pairwise_order_match']) is not bool):
        raise ValueError('INVALID_PREDECLARED_REWARD_RULE')
    return value


def _passes(rule, evaluation):
    rule = _rule(rule)
    mae = evaluation.get('mean_absolute_error')
    if (type(mae) not in (int, float) or not np.isfinite(mae) or not 0 <= mae <= 1
            or type(evaluation.get('pairwise_order_matches')) is not bool):
        raise ValueError('INVALID_SELECTION_METRICS')
    return (evaluation['role'] == 'selection'
            and evaluation['mean_absolute_error'] <= rule['max_mae_fraction']
            and (not rule['require_pairwise_order_match'] or evaluation['pairwise_order_matches']))


def _read_bound(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('REWARD_VALIDATION_ARTIFACT_CHANGED')
    return json.loads(Path(item['path']).read_text(encoding='utf-8'))


def _qualification(path, g_checkpoint):
    """Serving checks bytes and declarations only; no Y or efficacy certification."""
    q = json.loads(Path(path).read_text(encoding='utf-8'))
    m = g_checkpoint['metadata']
    ids = q.get('reward_validation_observation_ids', [])
    if (q.get('schema') != 'protocol-G-reward-qualification-v2'
            or q.get('role') != 'selection' or q.get('measured_validation_passed') is not True
            or not ids or len(set(ids)) != len(ids)
            or set(ids).intersection(m['fit_observation_ids'] + q['reserved_test_observation_ids'])
            or q.get('g_content_identity') != g_checkpoint['content_identity']
            or q.get('scope') != m['scope'] or q.get('bindings') != m['bindings']
            or ids != m['bindings']['legal_selection_observation_ids']
            or q['reserved_test_observation_ids'] != m['bindings']['reserved_test_observation_ids']):
        raise ValueError('INDEPENDENT_G_REWARD_QUALIFICATION_PENDING')
    rule = _read_bound(q['rule_artifact'])
    frozen = _read_bound(q['frozen_prediction_receipt'])
    evaluation = _read_bound(q['evaluation_artifact'])
    if (frozen['identity'] != m['bindings']['role_contract']
            or evaluation['identity'] != frozen['identity']
            or evaluation['block_id'] != frozen['block_id'] or not _passes(rule, evaluation)):
        raise ValueError('QUALIFICATION_RECEIPT_OR_RULE_MISMATCH')
    from ..data.endpoint_protocol_evaluation import digest
    if (not frozen.get('predictions')
            or frozen.get('predictions_sha256') != digest(frozen['predictions'])
            or not frozen.get('block_id')
            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
                   for v in frozen['predictions'].values())):
        raise ValueError('QUALIFICATION_FROZEN_PREDICTIONS_INVALID')
    for artifact_path, expected_sha in frozen['artifacts'].items():
        if sha(artifact_path) != expected_sha:
            raise ValueError('QUALIFICATION_PREDICTION_ARTIFACT_CHANGED')
    for item in (q['rule_artifact'], q['g_checkpoint'], q['p_cache_manifest']):
        if frozen['artifacts'].get(item['path']) != item['sha256']:
            raise ValueError('QUALIFICATION_REQUIRED_ARTIFACT_MISSING')
    _read_bound(q['g_checkpoint'])
    _read_bound(q['p_cache_manifest'])
    return q


def _selection_prediction(bridge, plan, fold_id, branch, cache_manifest, g_path):
    bound = plan_bindings(bridge, cache_manifest, plan, fold_id, branch)
    checkpoint = json.loads(Path(g_path).read_text(encoding='utf-8'))
    scope = checkpoint['metadata']['scope']
    scorer = load_g(g_path, expected_bindings=bound, expected_scope=scope)
    blocks = [b for b in plan.blocks(fold_id, branch) if b['role'] == 'selection']
    if len(blocks) != 1:
        raise ValueError('ONE_SELECTION_BLOCK_REQUIRED')
    units = plan.metadata_units(fold_id, branch, role='selection')
    encoded = [ProtocolConditionAdapter().encode(bridge.require_response_features(
        u['candidate_id'], cache_manifest=cache_manifest)) for u in units]
    if (not encoded or any(e.scopes != scope or e.feature_names != scorer.feature_names for e in encoded)
            or set(blocks[0]['candidate_ids']) != {u['candidate_id'] for u in units}):
        raise ValueError('SELECTION_SCOPE_OR_CANDIDATES_CHANGED')
    values = scorer.scorer(np.stack([e.vector for e in encoded]))
    return checkpoint, blocks[0], units, {u['candidate_id']: float(v) for u, v in zip(units, values)}


def validate_reward(bridge, plan, *, fold_id, branch='main', cache_manifest,
                    g_path, rule_path, output):
    """Explicit model inference + selection Y evaluation; never final/external Y.

    rule_path is an already saved rule artifact, hashed before model inference.
    Qualification is limited to this selection block, not biological efficacy.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('IMMUTABLE_VALIDATION_OUTPUT_EXISTS')
    rule_sha = sha(rule_path)
    rule = _rule(json.loads(Path(rule_path).read_text(encoding='utf-8')))
    checkpoint, block, units, predictions = _selection_prediction(
        bridge, plan, fold_id, branch, cache_manifest, g_path)
    artifacts = {str(Path(p).resolve()): sha(p) for p in (g_path, cache_manifest, rule_path)}
    if artifacts[str(Path(rule_path).resolve())] != rule_sha:
        raise ValueError('RULE_CHANGED_DURING_VALIDATION')
    frozen = plan.freeze_predictions(fold_id, block['block_id'], predictions,
                                    branch=branch, artifacts=artifacts)
    output.mkdir(parents=True)
    _write_new(output / 'predictions.json', frozen)
    evaluation = plan.evaluate_frozen(frozen, endpoint_definition=checkpoint['metadata']['scope']['endpoint_definition'])
    _write_new(output / 'evaluation.json', evaluation)
    item = lambda p: {'path': str(Path(p).resolve()), 'sha256': sha(p)}
    qualification = {'schema': 'protocol-G-reward-qualification-v2', 'role': 'selection',
        'measured_validation_passed': bool(_passes(rule, evaluation)),
        'g_content_identity': checkpoint['content_identity'], 'scope': checkpoint['metadata']['scope'],
        'bindings': checkpoint['metadata']['bindings'],
        'reward_validation_observation_ids': [u['observation_id'] for u in units],
        'reserved_test_observation_ids': [u['observation_id'] for role in ('final_test', 'external_test')
            for u in plan.metadata_units(fold_id, branch, role=role)],
        'rule_artifact': item(rule_path), 'g_checkpoint': item(g_path),
        'p_cache_manifest': item(cache_manifest), 'frozen_prediction_receipt': item(output / 'predictions.json'),
        'evaluation_artifact': item(output / 'evaluation.json'),
        'execution_certified': False, 'biological_effectiveness_validated': False,
        'qualification_scope': 'limited_selection_block_model_simulation_only'}
    _write_new(output / 'qualification.json', qualification)
    return qualification


def _revalidate_reward(path, g_checkpoint, *, bridge, plan, fold_id, branch, cache_manifest, g_path):
    """Training gate recomputes fixed G and selection measurements, rejects forged metrics."""
    q = _qualification(path, g_checkpoint)
    _check_qualification_roles(q, plan, fold_id, branch)
    _, block, _, predictions = _selection_prediction(bridge, plan, fold_id, branch, cache_manifest, g_path)
    frozen = _read_bound(q['frozen_prediction_receipt'])
    if frozen['predictions'] != predictions or frozen['block_id'] != block['block_id']:
        raise ValueError('QUALIFICATION_NOT_ACTUAL_FIXED_G_PREDICTIONS')
    evaluation = plan.evaluate_frozen(frozen, endpoint_definition=q['scope']['endpoint_definition'])
    if evaluation != _read_bound(q['evaluation_artifact']) or not _passes(_read_bound(q['rule_artifact']), evaluation):
        raise ValueError('QUALIFICATION_NOT_ACTUAL_SELECTION_METRICS')
    if (q['g_checkpoint']['sha256'] != sha(g_path)
            or q['p_cache_manifest']['sha256'] != sha(cache_manifest)):
        raise ValueError('QUALIFICATION_MODEL_OR_CACHE_CHANGED')
    return q


def generate_terminal_decisions(bridge, plan, *, fold_id, branch='main', cache_manifest,
                                g_path, qualification_path, output):
    """Program-generated decisions only, fit candidate block; fixed P/G.

    Every sample terminates after one action. Targets are frozen G estimates,
    not measured rewards. No test Y is read or used as a target here.
    """
    role_identity = plan.identity(fold_id, branch)
    bound = plan_bindings(bridge, cache_manifest, plan, fold_id, branch)
    g_checkpoint = json.loads(Path(g_path).read_text(encoding='utf-8'))
    scope = g_checkpoint['metadata']['scope']
    g = load_g(g_path, expected_bindings=bound, expected_scope=scope)
    qualification = _revalidate_reward(qualification_path, g_checkpoint, bridge=bridge, plan=plan,
        fold_id=fold_id, branch=branch, cache_manifest=cache_manifest, g_path=g_path)
    units = plan.metadata_units(fold_id, branch, role='fit')
    encoded = [ProtocolConditionAdapter().encode(bridge.require_response_features(
        u['candidate_id'], cache_manifest=cache_manifest)) for u in units]
    if not encoded or any(e.scopes != scope or e.feature_names != g.feature_names for e in encoded):
        raise ValueError('Q_LEGAL_FIT_CANDIDATE_SCOPE_MISMATCH')
    vectors = np.stack([e.vector for e in encoded])
    rewards = g.scorer(vectors)
    rows = [{'candidate_id': u['candidate_id'], 'vector': e.vector.tolist(),
             'terminated': True, 'reward': float(r), 'target': float(r), 'weight': 1.}
            for u, e, r in zip(units, encoded, rewards)]
    artifact = {'schema': DECISION_SCHEMA, 'evidence': 'model_simulated', 'scope': scope,
        'bindings': bound, 'feature_names': list(g.feature_names),
        'g_checkpoint': {'path': str(Path(g_path).resolve()), 'sha256': sha(g_path)},
        'qualification': {'path': str(Path(qualification_path).resolve()), 'sha256': sha(qualification_path)},
        'reward_scale': 'terminal_Annexin_fraction', 'bootstrap': False, 'cost_penalty': False,
        'rows': rows, 'candidate_ids': [u['candidate_id'] for u in units]}
    artifact['content_identity'] = identity(artifact)
    _write_new(output, artifact)
    return artifact


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def fit_terminal_q(decision_path, *, bridge, plan, fold_id, branch='main',
                   cache_manifest, output):
    """Independent private parameters; only strict generated artifact accepted."""
    d = json.loads(Path(decision_path).read_text(encoding='utf-8'))
    if d.pop('content_identity') != identity(d):
        raise ValueError('TERMINAL_DECISION_CONTENT_CHANGED')
    bound = plan_bindings(bridge, cache_manifest, plan, fold_id, branch)
    ids = [u['candidate_id'] for u in plan.metadata_units(fold_id, branch, role='fit')]
    if (d['schema'] != DECISION_SCHEMA or d['bindings'] != bound or d['candidate_ids'] != ids
            or d['evidence'] != 'model_simulated' or d['bootstrap'] is not False
            or d['cost_penalty'] is not False or d['reward_scale'] != 'terminal_Annexin_fraction'):
        raise ValueError('TERMINAL_DECISION_CONTRACT_MISMATCH')
    for item in (d['g_checkpoint'], d['qualification']):
        if sha(item['path']) != item['sha256']:
            raise ValueError('FROZEN_REWARD_ARTIFACT_CHANGED')
    checkpoint = json.loads(Path(d['g_checkpoint']['path']).read_text(encoding='utf-8'))
    qualification = _revalidate_reward(d['qualification']['path'], checkpoint, bridge=bridge, plan=plan,
        fold_id=fold_id, branch=branch, cache_manifest=cache_manifest, g_path=d['g_checkpoint']['path'])
    g = load_g(d['g_checkpoint']['path'], expected_bindings=bound, expected_scope=d['scope'])
    encoded = [ProtocolConditionAdapter().encode(bridge.require_response_features(cid,
        cache_manifest=cache_manifest)) for cid in ids]
    if any(e.scopes != d['scope'] or e.feature_names != g.feature_names for e in encoded):
        raise ValueError('TERMINAL_DECISION_SCOPE_CHANGED')
    x = np.stack([e.vector for e in encoded])
    rewards = g.scorer(x)
    expected = [{'candidate_id': cid, 'vector': e.vector.tolist(), 'terminated': True,
                 'reward': float(r), 'target': float(r), 'weight': 1.}
                for cid, e, r in zip(ids, encoded, rewards)]
    validate_decision_rows(d['rows'])
    if d['rows'] != expected or d['feature_names'] != list(g.feature_names):
        raise ValueError('TERMINAL_DECISIONS_NOT_FIXED_G_OUTPUTS')
    result = {'schema': Q_SCHEMA, 'bindings': bound, 'scope': d['scope'],
        'feature_names': d['feature_names'], 'parameters': ridge_parameters(x, rewards),
        'evidence': 'model_simulated', 'actual_fit_completed': True,
        'target': 'terminal_reward_only', 'lambda': LAMBDA,
        'value_scale': d['reward_scale'], 'g_checkpoint': d['g_checkpoint'],
        'qualification': d['qualification'], 'decision_artifact_sha256': sha(decision_path),
        'private_parameters': True, 'biological_effectiveness_validated': False}
    result['content_identity'] = identity(result)
    _write_new(output, result)
    return result


def load_terminal_q(path, *, expected_bindings, expected_scope):
    q = json.loads(Path(path).read_text(encoding='utf-8'))
    content = {k: v for k, v in q.items() if k != 'content_identity'}
    if (q['content_identity'] != identity(content) or q['schema'] != Q_SCHEMA
            or q['bindings'] != expected_bindings or q['scope'] != expected_scope
            or q['actual_fit_completed'] is not True or q['private_parameters'] is not True
            or q['evidence'] != 'model_simulated' or q['target'] != 'terminal_reward_only'
            or q['lambda'] != LAMBDA or q['value_scale'] != 'terminal_Annexin_fraction'
            or q['biological_effectiveness_validated'] is not False):
        raise ValueError('PRIVATE_Q_CHECKPOINT_INVALID')
    for item in (q['g_checkpoint'], q['qualification']):
        if sha(item['path']) != item['sha256']:
            raise ValueError('PRIVATE_Q_FROZEN_G_CHANGED')
    g_checkpoint = json.loads(Path(q['g_checkpoint']['path']).read_text(encoding='utf-8'))
    _qualification(q['qualification']['path'], g_checkpoint)
    g = load_g(q['g_checkpoint']['path'], expected_bindings=expected_bindings, expected_scope=expected_scope)
    if g.feature_names != tuple(q['feature_names']):
        raise ValueError('PRIVATE_Q_FROZEN_G_FEATURE_SCHEMA_CHANGED')
    model = FixedRidge(q['parameters'])
    if len(model.mean) != len(q['feature_names']):
        raise ValueError('PRIVATE_Q_FEATURE_DIMENSION_INVALID')
    registered = RegisteredProtocolScorer(model, q['content_identity'], 'private_terminal_ridge_distillation',
        tuple(q['feature_names']), q['scope']['endpoint_definition'], q['value_scale'],
        q['content_identity'], 'private_candidate_Q')
    return CheckpointBoundProtocolScorer(registered, q['scope'], q['bindings'])



def _check_qualification_roles(qualification, plan, fold_id, branch):
    """Selection authorizes limited simulated reward use; no test calibration."""
    selected = {u['observation_id'] for u in plan.metadata_units(fold_id, branch, role='selection')}
    reserved = {u['observation_id'] for role in ('final_test', 'external_test')
                for u in plan.metadata_units(fold_id, branch, role=role)}
    if (not selected or set(qualification['reward_validation_observation_ids']) != selected
            or set(qualification['reserved_test_observation_ids']) != reserved):
        raise ValueError('QUALIFICATION_SELECTION_OR_RESERVED_ROLE_MISMATCH')





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
    algorithm = ('G_enumeration_baseline' if checkpoint_scorer.scorer_kind == 'endpoint_G'
                 else 'private_candidate_Q')
    result = rank_protocol_candidates(bridge, candidate_ids, scorer=checkpoint_scorer._registered,
                                     cache_manifest=cache_manifest, algorithm=algorithm)
    return {**result, 'record_consistency_verified': True, 'execution_certified': False,
            'biological_effectiveness_validated': False}


def validate_decision_rows(rows):
    keys = {'candidate_id', 'vector', 'terminated', 'reward', 'target', 'weight'}
    if not isinstance(rows, list) or not rows:
        raise ValueError('TERMINAL_DECISION_ROWS_REQUIRED')
    for row in rows:
        if (not isinstance(row, dict) or set(row) != keys
                or type(row['candidate_id']) is not str or not row['candidate_id']
                or row['terminated'] is not True
                or any(type(row[k]) not in (int, float) or not np.isfinite(row[k]) for k in ('reward', 'target', 'weight'))
                or row['weight'] != 1 or not 0 <= row['reward'] <= 1 or row['target'] != row['reward']
                or not isinstance(row['vector'], list) or not row['vector']
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in row['vector'])):
            raise ValueError('TERMINAL_DECISION_ROW_SCHEMA_INVALID')
