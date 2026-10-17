"""Native-P numerical, metadata and role checks on isolated fixtures."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from lung.engine import native_response_producer as p


def fixture_values():
    genes = ('g1', 'g2')
    vocab = {'source_id': ['s'], 'background_id': ['b'], 'protocol_id': ['sciplex_author_reported_endpoint_v1']}
    bindings = {'data_identity': 'data', 'gene_axis_sha256': 'gene',
        'ordered_gene_ids_sha256': p.digest(genes), 'normalization_sha256': 'norm',
        'context_vocabulary': vocab,
        'base_transform': p.BASE, 'units': p.UNIT}
    stages = {name: {'stage': stage, 'optimizer_step': 2, 'data_identity': 'data',
        'gene_axis_sha256': 'gene', 'normalization_sha256': 'norm', 'context_vocabulary': vocab,
        'base_transform': p.BASE, 'units': p.UNIT, 'synthetic': False, 'checkpoint': name + '.pt', 'checkpoint_sha256': letter * 64}
        for name, stage, letter in (('representation', 'latent_ae', 'a'), ('response', 'latent_cfm', 'b'))}
    stages['response']['representation_checkpoint_sha256'] = 'a' * 64
    metadata = {'bindings': bindings, **stages}
    req = {'request_id': 'native:c', 'condition_id': 'c', 'role': 'fit', 'measurement_level': 'single_cell',
        'source_id': 's', 'background_id': 'b', 'protocol_id': vocab['protocol_id'][0],
        'chemical_id': 'chem', 'dose_uM': 1.0, 'actual_protocol': {'duration_h': 72, 'source_time_h': 72,
        'time_semantics': 'author_reported_endpoint', 'washout_h': None, 'prior_drug': None},
        'physical_reference_ids': ['w1', 'w2'], 'reference_semantics': 'unpaired_contemporaneous',
        'reference_sampling_weights': {'w1': {'cells': 2, 'well_probability': .5, 'cell_probability_given_well': .5},
                                      'w2': {'cells': 1, 'well_probability': .5, 'cell_probability_given_well': 1.}}}
    table = pd.DataFrame([{'condition_id': 'c', 'source_id': 's', 'background_id': 'b',
        'protocol_id': req['protocol_id'], 'eligible': True, 'chemical_or_experiment_group_id': 'chem',
        'dose_um': 1., 'duration_h': 72., 'fingerprint_row': 0}]).set_index('condition_id')
    coords = p.NativeMetadata(bindings, genes, np.zeros(2), np.ones(2), table, np.ones((1, 2048)))
    return req, coords, metadata


class Units:
    def __init__(self):
        self.calls = []

    def read(self, *args, **kwargs):
        raise AssertionError('target reader must never be used')

    def request(self, request_id, *, role):
        request, _, _ = fixture_values()
        assert request_id == request['request_id'] and role == request['role']
        return request

    def reference(self, request_id, **kwargs):
        self.calls.append((request_id, kwargs))
        return {'reference': sparse.csr_matrix([[1., 3.], [3., 1.], [1., 1.]]),
            'reference_ids': ['r1', 'r2', 'r3'], 'reference_weights': [.25, .25, .5],
            'reference_rows': [{'cell_id': c, 'role': 'fit', 'condition_id': w,
                'matrix_path': 'source.npz', 'matrix_row': i}
                for i, (c, w) in enumerate(zip(('r1', 'r2', 'r3'), ('w1', 'w1', 'w2')))],
            'target_expression_used': False, 'units': p.UNIT, 'selection': 'full_reference'}


def generation_fixture(tmp_path, monkeypatch):
    from lung.data import native_response_units
    request, coordinates, metadata = fixture_values()
    for name in ('representation', 'response'):
        path = tmp_path / metadata[name]['checkpoint']
        path.write_bytes(b'numerical-pipeline-fixture')
        metadata[name]['checkpoint_sha256'] = p.sha(path)
    metadata['response']['representation_checkpoint_sha256'] = metadata['representation']['checkpoint_sha256']
    units = Units()
    units.root, units.asset = tmp_path, tmp_path
    monkeypatch.setattr(native_response_units, 'NativeResponseUnits', lambda *a, **k: units)
    monkeypatch.setattr(p.NativeMetadata, 'from_joint', lambda *a: coordinates)
    return request, coordinates, metadata, units


class Adapter:
    def __init__(self):
        self.references, self.conditions, self.noises = [], [], []

    def sample_noise(self, reference, generator):
        return torch.randn(reference.shape, generator=generator)

    def encode_condition(self, **kwargs):
        self.conditions.append(kwargs)
        return kwargs

    def sample(self, reference, condition, *, noise, steps):
        self.references.append(reference.clone())
        self.noises.append(noise.clone())
        # Nonlinear reference dependence proves the mean ablation differs.
        return reference.square() / 10 + noise * .001


def test_full_reference_and_same_reference_mean(tmp_path, monkeypatch):
    req, coords, metadata, units = generation_fixture(tmp_path, monkeypatch)
    adapters = [Adapter(), Adapter()]
    outputs = []
    for ablation, adapter in zip(('full_reference', 'same_reference_mean'), adapters):
        units.calls.clear()
        outputs.append(p._produce(units, coords, req, adapter, metadata,
            tmp_path / ablation, role='fit', seed=19, microbatch=2, ablation=ablation))
        assert units.calls == [('native:c', {'role': 'fit', 'expected_gene_ids': ('g1', 'g2'), 'max_rows': None})]
        assert all(torch.all(c['duration'] == 72) and torch.all(c['horizon'] == 72)
            and torch.all(c['elapsed'] == 0) and c['fingerprint'].shape[-1] == 2048
            for c in adapter.conditions)
        assert outputs[-1].execution_kind == 'native'
        assert not outputs[-1].vector.flags.writeable
    x = torch.cat(adapters[0].references)
    mean = torch.as_tensor(np.array([.25, .25, .5]) @ x.numpy(), dtype=x.dtype)
    assert torch.allclose(torch.cat(adapters[1].references), mean.expand_as(x))
    assert torch.equal(torch.cat(adapters[0].noises), torch.cat(adapters[1].noises))
    assert not np.allclose(outputs[0].vector, outputs[1].vector)
    path = next((tmp_path / 'full_reference').glob('*.json'))
    assert p.load_native_descriptor(path, coords.bindings).cache_identity == outputs[0].cache_identity


@pytest.mark.parametrize('mutation', ['stage', 'counter', 'data', 'gene', 'norm', 'context', 'representation'])
def test_metadata_guards(mutation):
    _, coords, metadata = fixture_values()
    m = copy.deepcopy(metadata)
    key, value = {'stage': ('stage', 'response'), 'counter': ('optimizer_step', True),
        'data': ('data_identity', 'other'), 'gene': ('gene_axis_sha256', 'other'),
        'norm': ('normalization_sha256', 'other'), 'context': ('context_vocabulary', {}),
        'representation': ('representation_checkpoint_sha256', 'other')}[mutation]
    m['response'][key] = value
    with pytest.raises(ValueError):
        p.validate_metadata(m, coords.bindings)


def test_preflight_accepts_matching_metadata_without_loading_weights(monkeypatch, tmp_path):
    req, coords, metadata = fixture_values()
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('PT opened'))
    result = p.preflight([req], coords.bindings, metadata)
    assert not result['forward_called'] and result['matching_metadata_valid']


@pytest.mark.parametrize('field,value', [('measurement_level', 'bulk'), ('protocol_id', 'pulse6_recovery18')])
def test_unsupported_domain_protocol_before_reference(tmp_path, field, value):
    req, coords, metadata = fixture_values()
    req[field] = value
    units = Units()
    units.request = lambda *a, **k: req
    with pytest.raises(ValueError, match='PROTOCOL_OR_DECODER_PENDING'):
        p._produce(units, coords, req, Adapter(), metadata, tmp_path, role='fit', seed=0)
    assert units.calls == []


def test_cache_tamper_and_source_binding(tmp_path, monkeypatch):
    req, coords, metadata, units = generation_fixture(tmp_path, monkeypatch)
    p._produce(units, coords, req, Adapter(), metadata, tmp_path, role='fit', seed=1)
    path = next(f for f in tmp_path.glob('*.json') if not f.name.endswith('.receipt.json'))
    with pytest.raises(ValueError, match='BINDINGS'):
        p.load_native_descriptor(path, {**coords.bindings, 'data_identity': 'changed'})
    body = json.loads(path.read_text())
    body['vector'][0] += 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='CACHE_IDENTITY'):
        p.load_native_descriptor(path, coords.bindings)


def test_arbitrary_caller_backend_rejected(tmp_path):
    req, coords, _ = fixture_values()
    with pytest.raises(ValueError, match='SOURCE_BOUND_NATIVE_INPUTS'):
        p.produce(Units(), coords, req, object(), tmp_path, role='fit', seed=0)


def test_bindings_cannot_change_during_generation(tmp_path):
    req, coords, metadata = fixture_values()
    class MutationAdapter(Adapter):
        def sample(self, *args, **kwargs):
            result = super().sample(*args, **kwargs)
            coords.bindings['data_identity'] = 'changed'
            return result
    with pytest.raises(ValueError, match='GENERATION_BINDINGS_CHANGED'):
        p._produce(Units(), coords, req, MutationAdapter(), metadata,
            tmp_path, role='fit', seed=0)
    assert list(tmp_path.iterdir()) == []


def test_production_does_not_upgrade_fit_reader_to_final_test(tmp_path, monkeypatch):
    from lung.data.native_response_units import NativeResponseUnits
    req, coords, metadata = fixture_values()
    req['role'] = 'final_test'
    units = object.__new__(NativeResponseUnits)
    units.allowed_roles = frozenset(('fit',))
    monkeypatch.setattr(p, 'load_adapter', lambda *a, **k: pytest.fail('model loader called'))
    with pytest.raises(ValueError, match='READER_ROLE_NOT_AUTHORIZED'):
        p.produce(units, coords, req, metadata, tmp_path, role='final_test', seed=0)
    assert list(tmp_path.iterdir()) == []
