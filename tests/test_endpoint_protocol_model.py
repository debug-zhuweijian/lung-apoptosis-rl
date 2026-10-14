"""Pure-array mock scorers prove routing, not trained-model performance."""
import copy
from pathlib import Path

import numpy as np
import pytest

from src.lung.engine.config import StageUnavailable
from src.lung.engine.endpoint_protocol_features_v2 import EndpointProtocolFeatureInputsV2
from src.lung.engine.endpoint_protocol_model import (
    ProtocolConditionAdapter, RegisteredProtocolScorer, enumerate_with_g, rank_with_candidate_q,
)


@pytest.fixture
def bridge():
    return EndpointProtocolFeatureInputsV2.load()


def descriptor(packet):
    row = packet['candidate_record']
    return {'request_id': row['p_request_id'], 'compound_id': row['chemical']['compound_id'],
            'dose_uM': row['dose_uM'], 'population': [1., 2.], 'bio': [0.] * 27,
            'context': [0.] * 6, 'history': [99., 24.], 'elapsed': 24., 'horizon': 24.,
            'feature_identity': 'pytest_mock_not_real_prediction'}


def mocked_cache(monkeypatch, bridge):
    import src.lung.engine.endpoint_protocol_producer as producer
    packets = [bridge.model_input(cid) for cid in bridge.candidate_ids]
    cache = {p['candidate_record']['p_request_id']: descriptor(p) for p in packets}
    monkeypatch.setattr(producer, 'load_standard_cache', lambda path, input_bridge: cache)


def registration(packet, capture, algorithm='mock_G'):
    encoded = ProtocolConditionAdapter().encode(packet)
    def score(vectors):
        capture.append(vectors.copy())
        return np.full(len(vectors), 0.5)
    return RegisteredProtocolScorer(score, 'pytest_only', algorithm, encoded.feature_names,
        packet['candidate_record']['endpoint_definition'], 'terminal_Annexin_fraction', 'mock_not_trained',
        'private_candidate_Q' if algorithm == 'mock_private_Q' else 'endpoint_G')


def test_real_defaults_do_not_predict(bridge):
    assert enumerate_with_g(bridge, bridge.candidate_ids)['predictions'] == []
    assert bridge.model_input(bridge.candidate_ids[0])['p_descriptor'] is None
    with pytest.raises(StageUnavailable, match='P_CACHE_REQUIRED'):
        bridge.require_response_features(bridge.candidate_ids[0])


def test_mock_cache_and_both_actual_calls_keep_pulse_distinct(bridge, monkeypatch):
    mocked_cache(monkeypatch, bridge)
    ids = bridge.candidate_ids[:20]
    packet = bridge.require_response_features(ids[0], cache_manifest='mock')
    seen_g, seen_q = [], []
    g = enumerate_with_g(bridge, ids, scorer=registration(packet, seen_g), cache_manifest='mock')
    q = rank_with_candidate_q(bridge, ids, scorer=registration(packet, seen_q, 'mock_private_Q'), cache_manifest='mock')
    assert np.array_equal(seen_g[0], seen_q[0])
    assert np.unique(seen_g[0], axis=0).shape[0] == 20
    assert all(r['rank'] == 1 and r['tied'] for r in g['ranking'])
    assert [r['candidate_id'] for r in g['ranking']] == list(ids)
    assert q['algorithm'] != g['algorithm']
    names = registration(packet, []).feature_names
    assert not any('history' in n or 'elapsed' in n or 'observation' in n or 'labels_seen' in n for n in names)
    assert seen_g[0][0, names.index('actual.exposure_duration_h[h]')] in (6, 24)


def test_source_covariates_and_unknown_masks_enter_vector(bridge):
    packet = bridge.model_input(bridge.candidate_ids[0])
    packet['p_descriptor'] = descriptor(packet)
    original = ProtocolConditionAdapter().encode(packet)
    changed = copy.deepcopy(packet)
    changed['candidate_record']['state_covariates']['nominal_seeded_cells'] = None
    changed['candidate_record']['state_covariates']['vessel'] = 'unreported_vessel'
    changed['candidate_record']['state_id'] = 'nonsenescent'
    changed['candidate_record']['protocol']['prior_drug_retention'] = 'unreported'
    encoded = ProtocolConditionAdapter().encode(changed)
    assert not np.array_equal(original.vector, encoded.vector)
    assert encoded.vector[encoded.feature_names.index('source.nominal_seeded_cells.known')] == 0
    assert encoded.vector[encoded.feature_names.index('category.vessel=unknown')] == 1
    assert encoded.vector[encoded.feature_names.index('source.state_id=nonsenescent')] == 1


def test_endpoint_or_nav_scenarios_cannot_mix(bridge, monkeypatch):
    mocked_cache(monkeypatch, bridge)
    packet = bridge.require_response_features(bridge.candidate_ids[0], cache_manifest='mock')
    with pytest.raises(ValueError, match='HETEROGENEOUS'):
        enumerate_with_g(bridge, [bridge.candidate_ids[0], bridge.candidate_ids[-1]],
                         scorer=registration(packet, []), cache_manifest='mock')
    nav = bridge.candidate_ids[20:]
    packet = bridge.require_response_features(nav[0], cache_manifest='mock')
    with pytest.raises(ValueError, match='HETEROGENEOUS'):
        enumerate_with_g(bridge, nav, scorer=registration(packet, []), cache_manifest='mock')


def test_no_observations_read_in_actual_mock_routing(monkeypatch):
    original = Path.read_text
    def guard(path, *args, **kwargs):
        assert path.name not in ('observations.json', 'observation_mapping.json', 'reference_observations.json')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guard)
    bridge = EndpointProtocolFeatureInputsV2.load()
    mocked_cache(monkeypatch, bridge)
    packet = bridge.require_response_features(bridge.candidate_ids[0], cache_manifest='mock')
    assert enumerate_with_g(bridge, bridge.candidate_ids[:20], scorer=registration(packet, []),
                            cache_manifest='mock')['status'] == 'scored_supplied_scorer'


@pytest.mark.parametrize('field', ['model_id', 'algorithm_identity', 'evidence_identity'])
def test_empty_scorer_identity_rejected(bridge, field):
    from dataclasses import replace
    packet = bridge.model_input(bridge.candidate_ids[0])
    packet['p_descriptor'] = descriptor(packet)
    with pytest.raises(ValueError, match='IDENTITY_REQUIRED'):
        replace(registration(packet, []), **{field: '  '})


@pytest.mark.parametrize('names', [(), ('x', 'x'), ('',), ('x', 2)])
def test_invalid_feature_names_rejected(bridge, names):
    from dataclasses import replace
    packet = bridge.model_input(bridge.candidate_ids[0])
    packet['p_descriptor'] = descriptor(packet)
    with pytest.raises(ValueError, match='FEATURE_NAMES'):
        replace(registration(packet, []), feature_names=names)


def test_scorer_kind_mismatch_and_noncallable_rejected(bridge):
    from dataclasses import replace
    packet = bridge.model_input(bridge.candidate_ids[0])
    packet['p_descriptor'] = descriptor(packet)
    g = registration(packet, [])
    q = registration(packet, [], 'mock_private_Q')
    with pytest.raises(ValueError, match='CALLABLE_REQUIRED'):
        replace(g, scorer=None)
    with pytest.raises(ValueError, match='KIND_ALGORITHM_MISMATCH'):
        rank_with_candidate_q(bridge, bridge.candidate_ids[:20], scorer=g, cache_manifest='mock')
    with pytest.raises(ValueError, match='KIND_ALGORITHM_MISMATCH'):
        enumerate_with_g(bridge, bridge.candidate_ids[:20], scorer=q, cache_manifest='mock')


def test_explicit_no_induction_differs_from_unreported_source(bridge, monkeypatch):
    mocked_cache(monkeypatch, bridge)
    nav_id = next(cid for cid in bridge.candidate_ids
                  if bridge.model_input(cid)['candidate_record']['state_id'] == 'nonsenescent')
    nav = ProtocolConditionAdapter().encode(
        bridge.require_response_features(nav_id, cache_manifest='mock'))
    viel = ProtocolConditionAdapter().encode(
        bridge.require_response_features(bridge.candidate_ids[0], cache_manifest='mock'))
    none = nav.feature_names.index('category.induction_compound=none')
    unknown = nav.feature_names.index('category.induction_compound=unknown')
    assert nav.vector[none] == 1 and nav.vector[unknown] == 0
    assert viel.vector[none] == 0 and viel.vector[unknown] == 1
    assert nav.vector[nav.feature_names.index('source.induction_nominal_uM.known')] == 0
    assert nav.vector[nav.feature_names.index('source.induction_days.known')] == 0
