"""Explicit protocol vectors and independent G enumeration / private Q calls.

No model is constructed, fit, or loaded here. Scorers must declare their input
feature schema and terminal Annexin definition. Standard P history is provenance,
never the candidate's physical treatment history.
"""
import json
from dataclasses import dataclass

import numpy as np

from ..data.endpoint_protocol_inputs import validate_protocol
from .config import StageUnavailable


CATEGORIES = {
    'washout_status': ('not_applicable_no_challenge_withdrawal', 'reported_media_decant_rinse_fresh_media'),
    'prior_drug_retention': ('not_applicable', 'removed', 'retained', 'not_reported'),
    'floating_at_washout_recovered': ('not_applicable', 'yes', 'no', 'not_reported'),
    'state_kind': ('source_defined_group_context', 'author_reported_group_state'),
    'vessel': ('T25',),
    'induction_compound': ('cisplatin', 'none'),
}
NUMERIC_COVARIATES = (
    ('nominal_seeded_cells', 'cells'), ('nominal_seeded_cells_per_well', 'cells_per_well'),
    ('preculture_h', 'h'), ('hours_after_replating_before_challenge', 'h'),
    ('induction_nominal_uM', 'uM'), ('induction_days', 'days'),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def scope_for(row):
    # Arm is intentionally free: pulse and continuous belong to the same block.
    return {key: row[key] for key in ('source_id', 'cell_line', 'state_id', 'state_kind',
                                      'dose_scenario', 'endpoint_definition')}


@dataclass(frozen=True)
class EncodedProtocol:
    feature_names: tuple
    vector: np.ndarray
    scopes: dict
    provenance: dict


class ProtocolConditionAdapter:
    """Fixed units, categorical unknown indicators and explicit missing masks."""
    def encode(self, input_packet):
        row = input_packet['candidate_record']
        validate_protocol(row)
        p, cov = row['protocol'], row['state_covariates']
        descriptor = input_packet.get('p_descriptor')
        if descriptor is None:
            raise StageUnavailable('PROTOCOL_P_CACHE_REQUIRED')
        names, values = [], []

        def numeric(name, value, unit):
            known = value is not None
            if known and (not isinstance(value, (int, float)) or not np.isfinite(value)):
                raise ValueError('INVALID_PROTOCOL_NUMBER: ' + name)
            names.extend((name + '[' + unit + ']', name + '.known'))
            values.extend((float(value) if known else 0., float(known)))

        numeric('actual.dose', row['dose_uM'], 'uM')
        for field in ('exposure_duration_h', 'washout_at_h', 'drug_free_recovery_h', 'terminal_time_h'):
            numeric('actual.' + field, p[field], 'h')
        numeric('actual.media_event_count', len(p['media_events']), 'count')
        for key, unit in NUMERIC_COVARIATES:
            numeric('source.' + key, cov.get(key), unit)
        categories = {**{k: p[k] for k in ('washout_status', 'prior_drug_retention',
                                          'floating_at_washout_recovered')},
                      'state_kind': row['state_kind'], 'vessel': cov.get('vessel'),
                      'induction_compound': ('none' if 'induction_compound' in cov
                                             and cov['induction_compound'] is None
                                             else cov.get('induction_compound'))}
        for key, vocabulary in CATEGORIES.items():
            value = categories[key]
            for item in (*vocabulary, 'unknown'):
                names.append('category.' + key + '=' + item)
                values.append(float(value == item if item != 'unknown' else value not in vocabulary))
        # Source-defined state remains a semantic identity even without RNA.
        # The vocabulary is fixed for a registered scorer via its feature schema.
        state_vocabulary = ('a549_parental_standard_culture', 'nonsenescent', 'cisplatin_induced_senescent')
        for state in (*state_vocabulary, 'unknown'):
            names.append('source.state_id=' + state)
            values.append(float(row['state_id'] == state if state != 'unknown' else row['state_id'] not in state_vocabulary))
        fingerprint = np.asarray(input_packet['chemical_fingerprint'], dtype=float)
        if fingerprint.shape != (2048,) or not np.isfinite(fingerprint).all():
            raise ValueError('INVALID_FINGERPRINT')
        names.extend('chemical.morgan2048.' + str(i) for i in range(2048))
        values.extend(fingerprint.tolist())
        for group in ('population', 'bio', 'context'):
            array = np.asarray(descriptor[group], dtype=float)
            if (array.ndim != 1 or not array.size or not np.isfinite(array).all()
                    or (group == 'bio' and array.size != 27)
                    or (group == 'context' and array.size % 3 != 0)):
                raise ValueError('INVALID_STANDARD_P_' + group.upper())
            names.extend('standardP.' + group + '.' + str(i) for i in range(array.size))
            values.extend(array.tolist())
        vector = np.asarray(values, dtype=np.float64)
        vector.setflags(write=False)
        return EncodedProtocol(tuple(names), vector, scope_for(row),
            {key: descriptor.get(key) for key in ('request_id', 'compound_id', 'dose_uM',
              'reference_id', 'feature_identity', 'reference_identity', 'reference_protocol_arm', 'reference_terminal_time_h', 'history', 'elapsed', 'horizon')})


@dataclass(frozen=True)
class RegisteredProtocolScorer:
    """A caller-supplied scorer; registration does not certify training."""
    scorer: object
    model_id: str
    algorithm_identity: str
    feature_names: tuple
    endpoint_definition: dict
    value_scale: str
    evidence_identity: str
    scorer_kind: str

    def __post_init__(self):
        identities = (self.model_id, self.algorithm_identity, self.evidence_identity)
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError('SCORER_IDENTITY_REQUIRED')
        if (not isinstance(self.feature_names, tuple) or not self.feature_names
                or any(not isinstance(name, str) or not name.strip() for name in self.feature_names)
                or len(set(self.feature_names)) != len(self.feature_names)):
            raise ValueError('INVALID_SCORER_FEATURE_NAMES')
        if not callable(self.scorer):
            raise ValueError('SCORER_CALLABLE_REQUIRED')
        if self.scorer_kind != 'endpoint_G':
            raise ValueError('INVALID_SCORER_KIND')


def rank_protocol_candidates(input_bridge, candidate_ids, *, scorer=None,
                             cache_manifest=None, algorithm='G_enumeration_baseline', adapter=None):
    """Call actual vectors; no observed outcomes or cost tie-breakers are read."""
    if algorithm != 'G_enumeration_baseline':
        raise ValueError('UNKNOWN_PROTOCOL_ALGORITHM')
    if scorer is not None:
        if not isinstance(scorer, RegisteredProtocolScorer) or scorer.scorer_kind != 'endpoint_G':
            raise ValueError('SCORER_KIND_ALGORITHM_MISMATCH')
    if scorer is None or cache_manifest is None:
        return {'status': 'pending_model' if scorer is None else 'pending_standard_reference_cache',
                'algorithm': algorithm, 'predictions': [], 'ranking': []}
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('EMPTY_OR_DUPLICATE_CANDIDATES')
    adapter = adapter or ProtocolConditionAdapter()
    packets = [input_bridge.require_response_features(cid, cache_manifest=cache_manifest)
               for cid in candidate_ids]
    encoded = [adapter.encode(packet) for packet in packets]
    scopes = [canonical(e.scopes) for e in encoded]
    if len(set(scopes)) != 1:
        raise ValueError('HETEROGENEOUS_ENDPOINT_STATE_OR_SCENARIO_BLOCK')
    if scorer.value_scale not in ('terminal_Annexin_fraction', 'terminal_Annexin_percent'):
        raise ValueError('UNSUPPORTED_TERMINAL_VALUE_SCALE')
    if canonical(scorer.endpoint_definition) != canonical(encoded[0].scopes['endpoint_definition']):
        raise ValueError('SCORER_ENDPOINT_DEFINITION_MISMATCH')
    if any(e.feature_names != scorer.feature_names for e in encoded):
        raise ValueError('SCORER_FEATURE_SCHEMA_MISMATCH')
    predicted = np.asarray(scorer.scorer(np.stack([e.vector for e in encoded])), dtype=float)
    upper = 1 if scorer.value_scale.endswith('fraction') else 100
    if predicted.shape != (len(packets),) or not np.isfinite(predicted).all() or not ((predicted >= 0) & (predicted <= upper)).all():
        raise ValueError('INVALID_TERMINAL_SCORES')
    records = []
    for packet, value, features in zip(packets, predicted, encoded):
        row = packet['candidate_record']
        records.append({'candidate_id': row['candidate_id'], 'compound_id': row['chemical']['compound_id'],
            'source_drug_id': row['source_drug_id'], 'dose_uM': row['dose_uM'],
            'protocol': row['protocol'], 'state_id': row['state_id'], 'dose_scenario': row['dose_scenario'],
            'value': float(value), 'value_scale': scorer.value_scale, 'scope': features.scopes,
            'model_id': scorer.model_id, 'algorithm': algorithm,
            'algorithm_identity': scorer.algorithm_identity, 'evidence_identity': scorer.evidence_identity,
            'scorer_kind': scorer.scorer_kind, 'scorer_training_verified': False,
            'standard_P_provenance': features.provenance})
    ranking = sorted(records, key=lambda r: -r['value'])
    for record in ranking:
        record['rank'] = 1 + sum(value > record['value'] for value in predicted)
        record['tied'] = sum(value == record['value'] for value in predicted) > 1
    return {'status': 'scored_supplied_scorer', 'algorithm': algorithm,
            'predictions': records, 'ranking': ranking,
            'model_effectiveness_validated': False, 'scorer_training_verified': False}


def enumerate_with_g(*args, **kwargs):
    return rank_protocol_candidates(*args, **kwargs, algorithm='G_enumeration_baseline')
