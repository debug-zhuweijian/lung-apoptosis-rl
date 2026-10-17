"""Synthetic optimization and locally constructed tmp_path checkpoints only."""
from dataclasses import replace

import pytest
import torch

from lung.engine import martin_source_learning as m
from tests.test_martin_source_model import synthetic_batch


def packet(domain, *, role='fit', part='reference'):
    batch = synthetic_batch(domain, part)
    if role == 'selection':
        batch = replace(batch, role=role, sample_ids=('SYNTHETIC_SAMPLE_C', 'SYNTHETIC_SAMPLE_D'),
                        background='HCC827', request_id='SYNTHETIC_SELECTION')
    return batch


def datasets():
    fit = m.collect_observations([packet(d) for d in m.DOMAINS], role='fit', contract=packet('log2TPM').contract)
    selection = m.collect_observations([packet(d, role='selection') for d in m.DOMAINS], role='selection', contract=fit.contract)
    return fit, selection


def settings(epochs=2):
    return {'width': 8, 'epochs': epochs, 'batch': 1, 'lr': 1e-4, 'weight_decay': 1e-4,
            'grad_clip': 1., 'seed': 12, 'device': 'cpu', 'dtype': 'float32',
            'early_stopping': {'enabled': True, 'patience': 20, 'min_delta': 1e-4, 'min_updates': 1000}}


def test_reference_request_duplicates_do_not_change_sample_weight():
    packets = [packet(d) for d in m.DOMAINS]
    fit = m.collect_observations(packets + packets * 3, role='fit', contract=packets[0].contract)
    assert fit.counts['unique_GSM'] == 2 and fit.counts['domain_views_per_GSM'] == 2
    assert fit.counts['unique_reference_GSM'] == 2
    assert fit.counts['reference_domain_request_rows_before_dedup'] == 16
    assert fit.identity == datasets()[0].identity


def test_same_reference_in_different_candidate_requests_is_same_observation():
    original = [packet(d) for d in m.DOMAINS]
    other = [replace(p, request_id='OTHER_CANDIDATE', protocol='Martin2022_RNA_DTP') for p in original]
    fit = m.collect_observations(original + other, role='fit', contract=original[0].contract)
    assert fit.counts['unique_GSM'] == 2 and fit.identity == datasets()[0].identity


def test_conflicting_observed_expression_cannot_be_deduplicated():
    packets = [packet(d) for d in m.DOMAINS]
    b = packets[0]
    raw = b.raw_values.clone()
    raw[0, 0] += 1
    changed = replace(b, raw_values=raw, model_values=(raw-b.offset)/b.scale)
    with pytest.raises(ValueError, match='DUPLICATE_GSM'):
        m.collect_observations(packets + [changed], role='fit', contract=b.contract)


def test_target_protocol_context_conflict_cannot_be_deduplicated():
    packets = [packet(d, part='target') for d in m.DOMAINS]
    b = packets[0]
    changed = replace(b, protocol='Martin2022_RNA_DTP')
    with pytest.raises(ValueError, match='DUPLICATE_GSM'):
        m.collect_observations(packets + [changed], role='fit', contract=b.contract)


def test_selection_and_final_are_never_fit_samples():
    fit, selection = datasets()
    assert not set(fit.sample_ids) & set(selection.sample_ids)
    with pytest.raises(ValueError, match='COLLECTION_ROLE'):
        m.collect_observations([packet('log2TPM', role='selection')], role='fit', contract=fit.contract)
    with pytest.raises(ValueError, match='FINAL_NOT_ENABLED'):
        m.collect_observations([], role='final_test', contract=fit.contract)


def test_preflight_does_not_construct_model_load_pt_or_optimize(monkeypatch):
    fit, selection = datasets()
    def forbidden(*args, **kwargs):
        pytest.fail('Preflight cannot execute a model or load weights')
    monkeypatch.setattr(m, 'MartinSourceModel', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    report = m.preflight(fit, selection, settings())
    assert report['real_model_forwards'] == report['PT_loads'] == report['optimizer_steps'] == 0
    assert report['final_RNA_enabled'] is False


def test_settings_inherit_current_project_optimizer_and_stopper():
    from lung.engine.config import load_config
    cfg = load_config('configs/modular_research.yaml')
    value = m.settings_from_config(cfg)
    assert value['lr'] == cfg['research']['latent']['ae']['lr']
    assert value['weight_decay'] == cfg['response']['weight_decay']
    assert value['epochs'] == cfg['modular']['stages']['ae']['epochs']
    assert value['early_stopping'] == cfg['early_stopping']


def test_synthetic_fit_selection_identity_and_output(tmp_path, cpu_only):
    fit, selection = datasets()
    result = m.fit_source(fit, selection, settings(), tmp_path/'source', synthetic=True)
    assert result['optimizer_steps'] == 4 and result['epochs_completed'] == 2
    assert len(result['history']) == 2 and result['final_RNA_used'] is False
    assert result['scientific_qualified'] is False and result['synthetic'] is True
    saved = torch.load(tmp_path/'source/final.pt', weights_only=True)
    assert saved['metadata']['training_role'] == 'fit'
    assert saved['metadata']['model_identity']['input_contract']['fit_sample_ids'] == list(fit.sample_ids)
    assert 'optimizer' not in saved and 'rng' not in saved
    assert saved['metadata']['final_RNA_used'] is False


def test_selection_only_no_grad_and_never_optimizer_data(tmp_path, cpu_only, monkeypatch):
    fit, selection = datasets()
    seen = []
    original = m._loss
    def capture(model, data, ids):
        seen.append((data.role, torch.is_grad_enabled(), list(ids)))
        return original(model, data, ids)
    monkeypatch.setattr(m, '_loss', capture)
    m.fit_source(fit, selection, settings(1), tmp_path/'source', synthetic=True)
    assert all(grad is True and set(ids) <= set(fit.sample_ids) for role, grad, ids in seen if role=='fit')
    assert all(grad is False and set(ids) == set(selection.sample_ids) for role, grad, ids in seen if role=='selection')


def test_existing_output_rejected(tmp_path, cpu_only, monkeypatch):
    fit, selection = datasets()
    out = tmp_path/'source'
    m.fit_source(fit, selection, settings(1), out, synthetic=True)
    with pytest.raises(ValueError, match='NEW_SOURCE_LEARNING_EXPERIMENT'):
        m.fit_source(fit, selection, settings(1), out, synthetic=True)


def test_no_new_selection_role_is_invented_for_lobo(tmp_path):
    with pytest.raises(ValueError, match='EXISTING_CYCLE_PLAN'):
        m.prepare_fold(tmp_path, system='lobo', fold_id='anything')
