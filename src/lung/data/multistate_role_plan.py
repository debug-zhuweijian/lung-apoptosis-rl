"""Independent Martin metadata roles; never rewrites the old LOBO bridge.

This retrospective two-fit/selection/final cycle is not blind validation or
reward qualification. Only the explicit fit API parses functional labels.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .martin_state_function import BACKGROUND_ORDER, DOMAIN_IDS, MartinStateFunctionInputs, sha

SCHEMA = 'martin-multistate-twofit-selection-final-v1'
ROLES = ('fit', 'selection', 'final_test')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def _metadata(source):
    if not isinstance(source, MartinStateFunctionInputs):
        raise ValueError('VERIFIED_MARTIN_SOURCE_REQUIRED')
    on_disk = json.loads((source.directory / 'metadata.json').read_text(encoding='utf-8-sig'))
    for attribute, field, key in (('candidates', 'candidates', 'candidate_id'),
                                  ('references', 'references', 'state_reference_id'),
                                  ('domains', 'measurement_domains', 'measurement_domain_id'),
                                  ('blocks', 'decision_blocks', 'decision_block_id')):
        if getattr(source, attribute) != {r[key]: r for r in on_disk[field]}:
            raise ValueError('MUTATED_SOURCE_METADATA')
    if set(source.domains) != set(DOMAIN_IDS):
        raise ValueError('MARTIN_MEASUREMENT_DOMAINS_CHANGED')
    candidates = list(source.candidates.values())
    if (len(candidates) != 20 or {c['cell_background'] for c in candidates} != set(BACKGROUND_ORDER)
            or any(c['actual_protocol']['observation_window_h'] != 72 for c in candidates)):
        raise ValueError('MARTIN_CANDIDATE_METADATA_CHANGED')
    folds = []
    for i, background in enumerate(BACKGROUND_ORDER):
        selection = BACKGROUND_ORDER[(i + 1) % len(BACKGROUND_ORDER)]
        fit = [b for b in BACKGROUND_ORDER if b not in (background, selection)]
        roles = {'fit': fit, 'selection': [selection], 'final_test': [background]}
        folds.append({'fold_id': 'cycle_final:' + background, 'background_roles': roles,
                      'units': [{'candidate_id': c['candidate_id'], 'observation_id': c['observation_id'],
                                 'measurement_domain_id': c['measurement_domain_id'],
                                 'state_reference_id': c['state_reference_id'],
                                 'cell_background': c['cell_background'],
                                 'decision_block_id': c['decision_block_id'], 'role': role}
                                for role in ROLES for c in candidates if c['cell_background'] in roles[role]]})
    return {'schema': SCHEMA, 'source_manifest_sha256': sha(source.directory / 'manifest.json'),
            'source_metadata_sha256': sha(source.directory / 'metadata.json'),
            'source_observations_sha256': sha(source.directory / 'observations.json'),
            'background_order': list(BACKGROUND_ORDER), 'folds': folds,
            'role_basis': 'metadata_background_cycle_not_outcomes', 'retrospective': True,
            'blind_chronology_certified': False, 'reward_qualified': False,
            'old_LOBO_roles_reused': False}


def build_role_plan(source, output):
    """Only metadata plus hash of existing Y bytes; no Y deserialization."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('IMMUTABLE_ROLE_PLAN_EXISTS')
    plan = _metadata(source)
    output.mkdir(parents=True)
    write_new(output / 'plan.json', plan)
    manifest = {'schema': SCHEMA, 'plan_sha256': sha(output / 'plan.json'),
                'builder_source_sha256': sha(__file__), 'labels_parsed': False,
                'counts_per_fold': {'fit': 10, 'selection': 5, 'final_test': 5}}
    write_new(output / 'manifest.json', manifest)
    return manifest


class MultiStateRolePlan:
    def __init__(self, directory, source):
        self.directory, self.source = Path(directory).resolve(), source
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8-sig'))
        self.metadata = json.loads((self.directory / 'plan.json').read_text(encoding='utf-8-sig'))
        self.verify()

    def verify(self):
        if (self.manifest.get('schema') != SCHEMA
                or self.manifest.get('builder_source_sha256') != sha(__file__)
                or self.manifest.get('plan_sha256') != sha(self.directory / 'plan.json')
                or self.metadata != _metadata(self.source)):
            raise ValueError('ROLE_PLAN_OR_SOURCE_CHANGED')
        return self

    def _fold(self, fold_id):
        self.verify()
        matches = [f for f in self.metadata['folds'] if f['fold_id'] == fold_id]
        if len(matches) != 1:
            raise ValueError('UNKNOWN_MULTISTATE_FOLD')
        return matches[0]

    def metadata_units(self, fold_id, head_id, role):
        if role not in ROLES or head_id not in self.source.domains:
            raise ValueError('UNKNOWN_MULTISTATE_ROLE_OR_HEAD')
        return deepcopy([u for u in self._fold(fold_id)['units']
                         if u['role'] == role and u['measurement_domain_id'] == head_id])

    def identity(self, fold_id, head_id):
        fold = self._fold(fold_id)
        if head_id not in self.source.domains:
            raise ValueError('UNKNOWN_MEASUREMENT_HEAD')
        return {'schema': SCHEMA, 'plan_sha256': self.manifest['plan_sha256'],
                'source_manifest_sha256': self.metadata['source_manifest_sha256'],
                'fold_id': fold['fold_id'], 'head_id': head_id}

    def training_domain(self, fold_id, head_id):
        units = self.metadata_units(fold_id, head_id, 'fit')
        rows = [self.source.candidates[u['candidate_id']] for u in units]
        declared = self.source.domains[head_id]
        return {'measurement_domain_id': head_id, 'head_id': head_id, 'source_id': 'Martin2022',
                'scenario': 'source_nominal_monotherapy', 'terminal_time_h': 72,
                'endpoint_definition': deepcopy(declared['endpoint_definition']),
                'value_scale': declared['value_scale'], 'output_policy': 'nonnegative_no_upper_clip',
                'fit_state_references': sorted({r['state_reference_id'] for r in rows}),
                'allowed_state_references': sorted(declared['allowed_state_references']),
                'held_out_state_status': 'predeclared_background_extrapolation_not_validated'}

    def decision_block(self, fold_id, head_id, block_id, role):
        allowed = {u['candidate_id'] for u in self.metadata_units(fold_id, head_id, role)}
        block = self.source.blocks.get(block_id)
        if (block is None or block['measurement_domain_id'] != head_id
                or not set(block['candidate_ids']).issubset(allowed)):
            raise ValueError('DECISION_BLOCK_ROLE_OR_HEAD_MISMATCH')
        self.source.decision_inputs(block['candidate_ids'])
        return deepcopy(block)

    def fit_units(self, fold_id, head_id):
        """Load one published mean per source observation, solely for new fit roles."""
        units = self.metadata_units(fold_id, head_id, 'fit')
        values = json.loads((self.source.directory / 'observations.json').read_text(encoding='utf-8-sig'))
        if (len(values) != 20 or len({v['observation_id'] for v in values}) != 20
                or len({v['candidate_id'] for v in values}) != 20):
            raise ValueError('DUPLICATE_OR_MISSING_SOURCE_OBSERVATIONS')
        indexed = {v['observation_id']: v for v in values}
        result = []
        for unit in units:
            value = indexed.get(unit['observation_id'])
            if (value is None or value['candidate_id'] != unit['candidate_id']
                    or value['weight'] != 1
                    or value['value_scale'] != self.source.domains[head_id]['value_scale']
                    or value['label_kind'] != 'digitized_published_group_mean_not_raw_replicate'):
                raise ValueError('FIT_SOURCE_OBSERVATION_UNIT_REQUIRED')
            result.append({**deepcopy(value), **unit})
        return result

    def freeze_predictions(self, fold_id, head_id, block_id, role, predictions, artifacts):
        """Consistency snapshot only; caller predictions never grant native qualification."""
        import math
        block = self.decision_block(fold_id, head_id, block_id, role)
        if (set(predictions) != set(block['candidate_ids'])
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in predictions.values())
                or not artifacts or any(sha(p) != h for p, h in artifacts.items())):
            raise ValueError('INVALID_COMPLETE_FROZEN_PREDICTIONS')
        content = {'identity': self.identity(fold_id, head_id), 'block': block, 'role': role,
                   'predictions': deepcopy(predictions), 'artifacts': deepcopy(artifacts),
                   'execution_certified': False, 'reward_qualified': False,
                   'evaluation_entry': 'pending_native_controlled_selection_final_adapter'}
        return {**content, 'content_identity': digest(content)}
