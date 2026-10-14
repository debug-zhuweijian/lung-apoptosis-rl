"""Source metadata/fixed tensors only: no real model, PT or outcome read."""
import pytest
import numpy as np
from lung.data import full_protocol_request_inputs as module


def test_real_union_identity_scope_without_model():
    rows, context = module.reconstruct()
    assert len(rows) == 32
    assert sum(r['base_kwargs'] is not None for r in rows) == 20
    assert sum(r['request_status'] == 'pending_physical_input' for r in rows) == 8
    assert sum(r['request_status'] == 'pending_bulk_reference_context_decoder' for r in rows) == 4
    assert len(context['reference_rows']) == 2431
    assert context['model_context_indexes'] == {'source_id':0,'background_id':0,'protocol_id':1}
    assert len({r['observation_id'] for r in rows if r['observation_id'] is not None}) == 24


def test_real_pulse_kwargs_and_masks_preserve_common_clock():
    reader = module.FullProtocolRequestInputs()
    rows, _ = module.reconstruct()
    cid = next(r['candidate_id'] for r in rows if r['base_kwargs'] is not None
               and r['base_kwargs']['duration'] == 6)
    kwargs, physical = reader.numeric_kwargs(cid,batch_size=2)
    assert kwargs['fingerprint'].shape == (2,2048)
    assert np.all(kwargs['duration'] == 6) and np.all(kwargs['horizon'] == 24)
    assert np.all(kwargs['elapsed'] == 0) and physical.lengths.tolist() == [4,4]
    assert not physical.known[:,1,4].any()  # source wash count unknown
    assert not physical.known[:,:,5].any() # no fabricated retained-drug flag
    assert physical.endpoint_hours.tolist() == [24,24]


def test_candidate_request_is_fresh_not_mutable_authority():
    reader = module.FullProtocolRequestInputs()
    rows, _ = module.reconstruct()
    cid = rows[0]['candidate_id']
    changed = reader.request(cid)
    changed['base_kwargs']['duration'] = 999
    assert reader.request(cid)['base_kwargs']['duration'] != 999


def test_unknown_version_and_caller_array_rejected():
    with pytest.raises(ValueError,match='VERSION'):
        module.FullProtocolRequestInputs('unregistered')
    with pytest.raises(ValueError,match='CANDIDATE_ID'):
        module.FullProtocolRequestInputs().request(np.zeros(2048))


def test_pinned_source_mismatch_rejected_without_self_update(monkeypatch):
    monkeypatch.setattr(module,'sha',lambda path:'0'*64)
    with pytest.raises(ValueError,match='PINNED_SOURCE_CHANGED'):
        module.reconstruct()


def test_pending_nav_and_bothe_not_silently_removed():
    rows,_ = module.reconstruct()
    reader = module.FullProtocolRequestInputs()
    for kind in ('pending_physical_input','pending_bulk_reference_context_decoder'):
        cid = next(r['candidate_id'] for r in rows if r['request_status'] == kind)
        assert reader.request(cid)['request_status'] == kind
        with pytest.raises(ValueError,match='PENDING_BASE_KWARGS'):
            reader.numeric_kwargs(cid)


@pytest.mark.parametrize('key,value', [
    ('preparation_grants_model_or_scientific_qualification',True),
    ('P_outputs',10), ('G_outputs',True), ('Q_outputs',0.),
    ('PT_loads',False), ('union_candidates',31), ('not_32_independent_Y',1),
    ('reference_transfer','validated'),
])
def test_forged_manifest_declarations_rejected(key,value):
    manifest = {**module.DECLARATIONS,'model_context_indexes':{'source_id':0,'background_id':0,'protocol_id':1}}
    manifest[key] = value
    with pytest.raises(ValueError,match='DECLARATION_CHANGED'):
        module.validate_declarations(manifest,{'model_context_indexes':{'source_id':0,'background_id':0,'protocol_id':1}})


def test_load_rejects_original_manifest_claim_attack(monkeypatch):
    original_read = module.read
    def forged_read(path):
        value = original_read(path)
        if path.name == 'manifest.json' and path.parent.name == module.VERSION:
            value['P_outputs'] = 10
            value['preparation_grants_model_or_scientific_qualification'] = True
        return value
    monkeypatch.setattr(module,'read',forged_read)
    with pytest.raises(ValueError,match='DECLARATION_CHANGED'):
        module.FullProtocolRequestInputs.load()
