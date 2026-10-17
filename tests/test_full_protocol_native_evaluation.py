"""Tiny synthetic boundary tests; actual final RNA/PT/model runs are forbidden."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from lung.engine import full_protocol_native_evaluation as module
from lung.engine.checkpoint import atomic_json
from lung.engine.config import load_config


def records_fixture():
    def request(cid, role, chemical, group):
        return {'condition_id': cid, 'request_id': 'native:' + cid, 'role': role, 'fold_id': None,
            'measurement_level': 'single_cell', 'gene_axis_id': 'joint_ensembl_32461', 'unique_functional_Y': None,
            'units': 'source_raw_counts_no_new_normalization', 'decoder_domain': 'singlecell_raw_counts_common_ensembl',
            'reference_semantics': 'contemporaneous_vehicle_not_same_cell_t0', 'dose_uM': 1.,
            'source_id': 'synthetic-source', 'background_id': 'A549-fixture', 'protocol_id': 'fixture',
            'chemical_id': chemical, 'split_group_id': group, 'outer_split': 'test' if role == 'final_test' else 'train',
            'cell_index_table': module.JOINT + '/test.parquet', 'physical_reference_ids': ['control'],
            'reference_sampling_weights': {'control': {'cells': 10, 'well_probability': 1., 'cell_probability_given_well': .1}},
            'target_cells': 10, 'actual_protocol': {'duration_h': 24., 'exposure_duration_unknown': True,
                'exposure_duration_h': None, 'prior_drug_unknown': True, 'prior_drug': None,
                'washout_unknown': True, 'washout_h': None, 'withdrawal_unknown': True}}
    records = [request('fit', 'fit', 'fit-chemical', 'fit-group'), request('final', 'final_test', 'E-chemical', 'E-group')]
    conditions = pd.DataFrame([{'condition_id': r['condition_id'], 'role': r['role'], 'outer_split': r['outer_split'],
        'condition_type': 'treated', 'eligible': True, 'source_id': r['source_id'], 'background_id': r['background_id'],
        'protocol_id': r['protocol_id'], 'dose_um': 1., 'duration_h': 24., 'plate': 'p',
        'chemical_or_experiment_group_id': r['chemical_id'], 'split_group_id': r['split_group_id'],
        'drug_id': r['chemical_id']} for r in records] + [{'condition_id': 'control', 'condition_type': 'control',
        'eligible': True, 'role': None, 'outer_split': None, 'source_id': 'synthetic-source',
        'background_id': 'A549-fixture', 'protocol_id': 'fixture', 'duration_h': 24., 'plate': 'p'}])
    counts = pd.DataFrame([{'role': r['role'], 'condition_id': r['condition_id'], 'n_cells': 10} for r in records]
                         + [{'role': 'final_test', 'condition_id': 'control', 'n_cells': 10}])
    links = pd.DataFrame([{'condition_id': 'final', 'control_condition_id': 'control'}])
    splits = pd.DataFrame([{'drug_id': r['chemical_id'], 'role': r['role'], 'split_group_id': r['split_group_id']} for r in records])
    return records, conditions, counts, links, splits


def test_final_role_only_unknown_history_and_control_role_cells():
    final, gaps = module.compile_records(*records_fixture())
    assert len(final) == 1 and not gaps and final[0]['role'] == 'final_test'
    assert final[0]['evaluation_ready'] and final[0]['actual_protocol']['exposure_duration_h'] is None
    scope = final[0]['holdout_scope']
    assert scope['chemical_unseen_from_fit_and_selection']
    assert not scope['dose_value_unseen_within_source'] and not scope['dose_holdout_at_seen_chemical']
    assert not scope['observation_hours_unseen_within_source'] and not scope['full_exposure_history_holdout_identifiable']


@pytest.mark.parametrize('mutation', ['chemical_leak', 'wrong_role', 'observed_hours', 'exposure_known', 'fake_Y', 'dose_bool', 'wrong_units'])
def test_role_protocol_or_domain_mutations_rejected(mutation):
    records, conditions, counts, links, splits = records_fixture()
    final = records[1]
    if mutation == 'chemical_leak': final['chemical_id'] = 'fit-chemical'
    elif mutation == 'wrong_role': conditions.loc[conditions.condition_id.eq('final'), 'role'] = 'selection'
    elif mutation == 'observed_hours': final['actual_protocol']['duration_h'] = 72.
    elif mutation == 'exposure_known': final['actual_protocol']['exposure_duration_unknown'] = False
    elif mutation == 'fake_Y': final['unique_functional_Y'] = .5
    elif mutation == 'dose_bool': final['dose_uM'] = True
    else: final['units'] = 'TPM'
    with pytest.raises(ValueError):
        module.compile_records(records, conditions, counts, links, splits)


def test_missing_same_role_control_is_a_source_gap_not_fake_zero():
    records, conditions, counts, links, splits = records_fixture()
    counts = counts.loc[counts.condition_id.ne('control')].copy()
    final, gaps = module.compile_records(records, conditions, counts, links, splits)
    assert not final[0]['evaluation_ready'] and gaps[0]['gap'] == 'no_original_final_role_control_cells'


def test_missing_fit_context_baseline_remains_pending():
    fixture = records_fixture()
    fixture[0][0]['actual_protocol']['duration_h'] = 72.
    final, _ = module.compile_records(*fixture)
    assert final[0]['fit_mean_baseline_status'] == 'pending_no_fit_context_targets'
    assert not final[0]['holdout_scope']['time_holdout_at_seen_chemical']
    assert final[0]['holdout_scope']['observation_hours_unseen_within_source']


def test_default_actual_metadata_never_opens_E_or_PT(monkeypatch):
    monkeypatch.setattr(module, 'EngineData', lambda *a, **k: pytest.fail('default cannot open a role reader'))
    monkeypatch.setattr(module, 'load_checkpoint', lambda *a, **k: pytest.fail('default cannot load PT'))
    monkeypatch.setattr(module.torch, 'load', lambda *a, **k: pytest.fail('default cannot load PT'))
    view = module.NativeFinalMetadata(load_config('configs/modular_research.yaml'))
    result = view.report()
    assert result['conditions'] == 367 and result['independent_split_groups'] == 52
    assert result['holdout_counts']['chemical_unseen_from_fit_and_selection'] == 367
    assert result['holdout_counts']['dose_holdout_at_seen_chemical'] == 0
    assert result['holdout_counts']['time_holdout_at_seen_chemical'] == 0
    assert not result['final_RNA_read'] and result['PT_loads'] == result['model_forwards'] == 0
    assert not result['source_gaps'] and result['preprocessing_role'] == result['bandwidth_role'] == 'fit'
    assert {r['role'] for r in view.records} == {'final_test'}
    assert all(r['registered_chemical_identity']['experimental_material_verified'] is False for r in view.records)


def test_partial_batch_and_synthetic_microbatch_generation_preserve_unknowns(monkeypatch):
    request = records_fixture()[0][1]
    data = SimpleNamespace(requests={'final': request}, action_features=lambda cid:
        {'fingerprint': np.ones(2048), 'dose': 1., 'duration': 24., 'source_id': 0,
         'background_id': 0, 'protocol_id': 0, 'is_vehicle': False})
    data.batch = lambda cid, n, device: module.learning.NativeProtocolData.batch(data, cid, n, device)
    data.data = data
    calls = []
    def sample(h, physical, **numeric):
        assert physical.endpoint_hours.tolist() == [24.] * len(h)
        assert physical.known[:, :, 0].all() and not physical.known[:, :, 1:].any()
        assert torch.isnan(physical.values[:, :, 1:]).all()
        calls.append(len(h))
        return h + numeric['noise'] * 0
    model = SimpleNamespace(shared=SimpleNamespace(encode_h=lambda x: (None, x), config=SimpleNamespace(n_genes=3, d_model=3)),
        full=SimpleNamespace(sample_latent=sample, member=SimpleNamespace(decode=lambda z: z)),
        mean=torch.ones(3), scale=torch.ones(3), reference_contract=module.producer.reference_contract('observed'))
    monkeypatch.setattr(module.producer, 'validate_batch', lambda *a: pytest.fail('native partial must not invoke Viel validation'))
    x = torch.ones((5, 3))
    result = module.generate(model, data, 'final', x, {'microbatch': 2, 'device': 'cpu', 'steps': 1}, torch.zeros_like(x))
    assert result.shape == (5, 3) and torch.isfinite(result).all() and calls == [2, 2, 1]


def test_nonfinite_model_output_refused_without_row_drop():
    data = SimpleNamespace(batch=lambda *a: (None, SimpleNamespace(known=torch.tensor([[[True, *[False]*7]]]),
        values=torch.tensor([[[1., *[float('nan')]*7]]]))))
    model = SimpleNamespace(shared=SimpleNamespace(encode_h=lambda x: (None, x), config=SimpleNamespace(n_genes=3, d_model=3)),
        full=SimpleNamespace(sample_latent=lambda *a, **k: torch.full((1, 3), float('nan')),
            member=SimpleNamespace(decode=lambda z: z)), mean=torch.ones(3), scale=torch.ones(3),
            reference_contract=module.producer.reference_contract('observed'))
    data.batch = lambda *a: ({}, SimpleNamespace(known=torch.tensor([[[True, *[False]*7]]]),
        values=torch.tensor([[[1., *[float('nan')]*7]]])))
    with pytest.raises(ValueError, match='NO_ROW_DROP'):
        module.generate(model, data, 'final', torch.ones((1, 3)), {'microbatch': 1, 'device': 'cpu', 'steps': 1}, torch.zeros((1, 3)))


def test_standardized_metrics_are_empirical_V_and_unmodified_mean_error():
    x = torch.tensor([[1., 0.], [2., 1.]])
    result = module.metrics(x, x, 1.)
    assert result['standardized_mean_mse'] == 0 and result['standardized_population_mmd_v'] == pytest.approx(0.)
    other = x + 1
    result = module.metrics(x, other, 1.)
    assert result['standardized_mean_mse'] == pytest.approx(1.) and result['standardized_population_mmd_v'] > 0


@pytest.mark.parametrize('variant', ['observed', 'zero_h', 'mean_h'])
def test_trained_reference_variant_applies_to_whole_draw_before_microbatch(variant):
    request = records_fixture()[0][1]
    data = SimpleNamespace(requests={'final': request}, action_features=lambda cid:
        {'fingerprint': np.ones(2048), 'dose': 1., 'duration': 24., 'source_id': 0,
         'background_id': 0, 'protocol_id': 0, 'is_vehicle': False})
    data.data = data
    data.batch = lambda cid, n, device: module.learning.NativeProtocolData.batch(data, cid, n, device)
    encodes, h_parts = [], []
    def encode(x):
        encodes.append(x.clone()); return None, x.square()
    def sample(h, physical, **numeric):
        h_parts.append(h.clone()); return h
    model = SimpleNamespace(shared=SimpleNamespace(encode_h=encode, config=SimpleNamespace(n_genes=3, d_model=3)),
        full=SimpleNamespace(sample_latent=sample, member=SimpleNamespace(decode=lambda z: z)),
        mean=torch.ones(3), scale=torch.ones(3), reference_contract=module.producer.reference_contract(variant))
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    module.generate(model, data, 'final', x, {'microbatch': 2, 'device': 'cpu', 'steps': 1}, torch.zeros_like(x))
    actual = torch.cat(h_parts)
    expected = x.square() if variant == 'observed' else torch.zeros_like(x) if variant == 'zero_h' else x.square().mean(0).expand_as(x)
    assert torch.equal(actual, expected)
    if variant == 'zero_h': assert not encodes
    if variant == 'mean_h': assert not torch.allclose(actual, x.mean(0).square().expand_as(x))


def test_fit_baseline_reads_T_targets_only_with_seeded_ids(tmp_path, monkeypatch):
    request = records_fixture()[0][0]
    rows = pd.DataFrame({'cell_id': ['fit-cell-1', 'fit-cell-2'], 'role': ['fit', 'fit']})
    calls = []
    def read(frame):
        calls.append(frame.cell_id.tolist())
        return sparse.csr_matrix(np.tile([1., 2., 3.], (len(frame), 1)))
    reader = SimpleNamespace(allowed_roles=frozenset({'fit'}), allow_test=False,
        role_cells=lambda cid, role: rows.copy(), rows=SimpleNamespace(read=read))
    def open_fit(config, roles, allow_test):
        assert roles == ('T',) and allow_test is False
        return SimpleNamespace(identity='fixture', reader=reader)
    monkeypatch.setattr(module, 'EngineData', open_fit)
    monkeypatch.setattr(module, 'allowed_fit_target_ids', lambda view:
        ({'fit': frozenset(rows.cell_id)}, {'path': 'synthetic-fit-index', 'sha256': 'fixture'}))
    original_sha = module.sha
    monkeypatch.setattr(module, 'sha', lambda p: 'fixture' if p == 'synthetic-fit-index' else original_sha(p))
    view = SimpleNamespace(config={}, identity='fixture', fit_records=[request], records=[request],
        mu=np.zeros(3), scale=np.ones(3), n_genes=3)
    result = module.fit_context_means(view, {'seed': 4, 'fit_mean_samples_per_condition': 2}, tmp_path)
    assert len(result) == 1 and all(set(ids) <= {'fit-cell-1', 'fit-cell-2'} for ids in calls)
    receipt = module.read(tmp_path / 'fit_baseline_receipts.json')[0]
    assert not receipt['final_or_selection_rows_used'] and receipt['sampled_fit_rows'][0]['role'] == 'fit'


def test_namespace_metadata_rejection_precedes_any_PT(tmp_path, monkeypatch):
    run = tmp_path / 'experiments/P'; assets = run.with_name('P-assets'); assets.mkdir(parents=True); run.mkdir()
    for path in (run / 'model_declaration.json', run / 'run_manifest.json', assets / 'run_manifest.json'):
        path.write_text('{}', encoding='utf8')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.producer, 'bound_inputs', lambda: (None, None, None, {}))
    monkeypatch.setattr(module.torch, 'load', lambda *a, **k: pytest.fail('must reject before PT'))
    monkeypatch.setattr(module, 'load_checkpoint', lambda *a, **k: pytest.fail('must reject before PT'))
    with pytest.raises(ValueError, match='MODEL_BINDINGS_MISMATCH'):
        module.completed_namespace(SimpleNamespace(), run)


def test_missing_real_namespace_preserves_failure_before_final_RNA(tmp_path, monkeypatch):
    view = SimpleNamespace(config={'population': 2, 'response': {'inference_ode_steps': 1}, 'seed': 1, 'device': 'cpu'},
        records=[{'evaluation_ready': True}], report=lambda: {'synthetic_fixture': True})
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'NativeFinalMetadata', lambda *a: view)
    monkeypatch.setattr(module, 'completed_namespace', lambda *a: (_ for _ in ()).throw(ValueError('pending_model_namespace')))
    monkeypatch.setattr(module, 'FinalRNAData', lambda *a: pytest.fail('no E access before model completion'))
    with pytest.raises(ValueError, match='pending_model_namespace'):
        module.evaluate({}, tmp_path / 'experiments/missingP', tmp_path / 'experiments/output')
    receipt = module.read(tmp_path / 'experiments/output/failure.json')
    assert receipt['completed_evaluation'] is False and receipt['optimizer_steps'] == 0


def test_fixed_settings_reject_boolean_sample_size_and_preserve_config():
    config = {'population': 2, 'response': {'inference_ode_steps': 1}, 'seed': 1, 'device': 'cpu'}
    prior = deepcopy(config)
    value = module.evaluation_settings(config, reference_mean_sensitivity=True)
    assert config == prior and value['reference_mean_sensitivity']
    config['population'] = True
    with pytest.raises(ValueError, match='SETTINGS'):
        module.evaluation_settings(config)


def test_aggregate_weights_original_components_not_condition_or_cell_count():
    rows = [{'split_group_id': group, 'methods': {'full_P': {'standardized_mean_mse': score, 'standardized_population_mmd_v': score}}}
            for group, score in [('component-A', 0.), ('component-A', 0.), ('component-B', 3.)]]
    result = module.aggregate(rows)
    assert result['conditions'] == 3 and result['split_components'] == 2
    assert result['group_equal_metrics']['full_P']['standardized_mean_mse'] == 1.5


@pytest.mark.parametrize('seed', [True, -900, -1, 1.5])
def test_original_seed_is_checked_before_offset(seed):
    with pytest.raises(ValueError, match='ORIGINAL_EVALUATION_SEED'):
        module.evaluation_settings({'population': 2, 'response': {'inference_ode_steps': 1}, 'seed': seed, 'device': 'cpu'})


def comparison_fixture(tmp_path, monkeypatch):
    """Three tiny fabricated metadata packs for boundary CI, no real weights/RNA."""
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    config = module.learning.joint_config(load_config('configs/modular_research.yaml'))
    config['population'] = 2; config['device'] = 'cpu'
    records = []
    for cid, group in [('E1', 'component-A'), ('E2', 'component-A'), ('E3', 'component-B')]:
        row = deepcopy(records_fixture()[0][1]); row.update(condition_id=cid, split_group_id=group)
        records.append(row)
    joint = tmp_path / module.JOINT; joint.mkdir(parents=True)
    pd.DataFrame({'cell_id': ['fit-only-id'], 'condition_id': ['fit'], 'role': ['fit']}).to_parquet(joint / 'train.parquet', index=False)
    fit_index = {'path': str(joint / 'train.parquet'), 'sha256': module.sha(joint / 'train.parquet')}
    view = SimpleNamespace(root=tmp_path, manifest={'files': {'train.parquet': {'sha256': fit_index['sha256']}}},
        config=config, identity='fixture-data', protocol_identity='fixture-protocol', records=records,
        context={'gene_ids': ['fixture-G1', 'fixture-G2', 'fixture-G3']}, n_genes=3, mu=np.zeros(3), scale=np.ones(3),
        fit_records=[records_fixture()[0][0]], report=lambda: {'synthetic_fixture_only': True}, bindings={},
        cell_ids_by_condition={'control': {'control-1', 'control-2'}, **{r['condition_id']:
            {r['condition_id'] + '-target-1', r['condition_id'] + '-target-2'} for r in records}})
    monkeypatch.setattr(module, 'NativeFinalMetadata', lambda *a: view)
    origin = tmp_path / 'experiments/shared-AE'; origin.mkdir(parents=True)
    ae = {'schema': module.learning.AE_RESULT_SCHEMA, 'stage': 'latent_ae', 'data_identity': view.identity,
        'training_completion': {'status': 'complete'}, 'weights': str(origin / 'weights/latent_ae/best.pt'),
        'final_checkpoint': str(origin / 'weights/latent_ae/final.pt'), 'checkpoint_sha256': 'a'*64, 'final_checkpoint_sha256': 'd'*64}
    module.write(origin / 'AE_result.json', ae)
    module.write(origin / 'run_manifest.json', {'schema': module.learning.SCHEMA + ':joint-AE',
        'config': module.learning.joint_ae_config(config), 'data_identity': view.identity})
    directories = []
    for variant, score in [('observed', 0.), ('zero_h', 1.), ('mean_h', 2.)]:
        run = tmp_path / ('experiments/P-' + variant); assets = run.with_name(run.name + '-assets')
        directory = tmp_path / ('experiments/E-' + variant)
        for p in (run, assets, directory): p.mkdir()
        current_config = deepcopy(config)
        current_config.setdefault('research', {}).setdefault('full_protocol_response', {})['reference_variant'] = variant
        train_settings = module.learning.settings_from_config(current_config)
        contract = module.producer.reference_contract(variant); bindings = {'data_identity': view.identity}
        meta = {'stage': module.producer.STAGE, 'synthetic': False, 'data_identity': view.identity,
            'full_response_bindings': bindings, 'representation_hash': 'b'*64, 'reference_contract': contract,
            'training_protocol_coverage': module.learning.COVERAGE, 'model_config': {'latent_dim': 2}}
        module.write(run / 'run_manifest.json', {'schema': module.learning.SCHEMA, 'metadata': meta,
            'settings': train_settings, 'protocol_identity': view.protocol_identity, 'cache_encoder': 'fixture-cache'})
        module.write(assets / 'run_manifest.json', {'config': current_config, 'bindings': bindings, 'protocol_identity': view.protocol_identity,
            'AE_declaration': ae, 'settings': train_settings})
        rep = {'stage': 'latent_ae', 'synthetic': False, 'data_identity': view.identity, 'optimizer_step': 1,
            'checkpoint': str(assets / 'representation_weights.pt'), 'checkpoint_sha256': 'b'*64}
        declaration = {'bindings': bindings, 'representation': rep, 'response': {'stage': module.producer.STAGE,
            'synthetic': False, 'data_identity': view.identity, 'optimizer_step': 1, 'checkpoint': str(run / 'response_weights.pt'),
            'checkpoint_sha256': module.config_hash(variant), 'representation_hash': 'b'*64,
            'training_protocol_coverage': module.learning.COVERAGE, 'reference_contract': contract}}
        module.write(run / 'model_declaration.json', declaration); module.write(assets / 'representation_declaration.json', rep)
        json_paths = [run / 'model_declaration.json', run / 'run_manifest.json', assets / 'run_manifest.json',
            assets / 'representation_declaration.json', origin / 'AE_result.json', origin / 'run_manifest.json']
        model_hashes = {str(p): module.sha(p) for p in json_paths}
        model_hashes.update({str(run / 'final.pt'): 'e'*64, str(run / 'response_weights.pt'): module.config_hash(variant),
            str(assets / 'representation_weights.pt'): 'b'*64, ae['weights']: 'a'*64, ae['final_checkpoint']: 'd'*64})
        settings = module.evaluation_settings(config); key = module.config_hash(list(module.context_key(records[0])))
        np.save(directory / ('fit-mean-' + key + '.npy'), np.full(3, .5, np.float32), allow_pickle=False)
        baseline = [{'context': list(module.context_key(records[0])), 'status': 'actual_fit_context_target_mean', 'conditions': 1,
            'samples_per_condition': 32, 'sampled_fit_rows': [{'condition_id': 'fit', 'cell_ids': ['fit-only-id']*32, 'role': 'fit'}],
            'sampling': 'equal fit target-condition means; uniform cells with replacement', 'fit_mean_key': key,
            'original_fit_cell_index': fit_index,
            'array_sha256': module.sha(directory / ('fit-mean-' + key + '.npy')), 'final_or_selection_rows_used': False}]
        module.write(directory / 'fit_baseline_receipts.json', baseline)
        plan = {'schema': module.SCHEMA, 'metadata_view': view.report(), 'settings': settings, 'response_run': str(run),
            'reference_contract_status': 'actual_completed_matching_model_verified', 'trained_reference_contract': contract,
            'independently_trained_zero_h_model': variant == 'zero_h', **module.NO_EFFECT_CLAIMS}
        module.write(directory / 'plan.json', plan); module.write(directory / 'request.json', {'synthetic_fixture_only': True})
        module.write(directory / 'gene_ids.json', view.context['gene_ids'])
        rows = []
        for r in records:
            cid = r['condition_id']; values = {'full_P': score, 'reference': 1., 'fit_context_mean': .5, 'observed': 0.}
            arrays = {name + suffix: np.full(3, 0. if suffix.startswith('_variance') else value, np.float32)
                      for name, value in values.items() for suffix in ('_mean_standardized', '_mean_log1p_library10000', '_variance_standardized')}
            name = 'gene-moments-' + module.config_hash(cid) + '.npz'; np.savez_compressed(directory / name, **arrays)
            row = {'condition_id': cid, 'chemical_id': r['chemical_id'], 'split_group_id': r['split_group_id'], 'source_id': r['source_id'],
                'dose_uM': r['dose_uM'], 'holdout_scope': {}, 'observation_hours': 24., 'raw_source_units': r['units'],
                'metric_domain': module.producer.DOMAIN, 'trained_reference_contract': contract, 'fit_mean_status': 'actual_fit_only_mean',
                'methods': {method: {'standardized_mean_mse': value**2, 'standardized_population_mmd_v': value**2}
                            for method, value in values.items() if method != 'observed'},
                'reference_cell_ids': ['control-1', 'control-2'], 'target_cell_ids': [cid + '-target-1', cid + '-target-2'],
                'sampling_seed': module.stream_seed(settings['seed'], 'E-response', cid), 'noise_seed': module.stream_seed(settings['seed'], 'P-noise', cid),
                'same_cell_t0': False, 'individual_pairing': False, 'state_increment_conclusion': False,
                'sensitivity_is_independent_trained_stateless_baseline': False,
                'gene_moments_file': name, 'gene_moments_sha256': module.sha(directory / name)}
            r['holdout_scope'] = {}; module.write(directory / ('condition-' + module.config_hash(cid) + '.json'), row); rows.append(row)
        result = {'schema': module.SCHEMA, 'status': 'complete_actual_native_final_expression_evaluation', 'synthetic': False,
            'final_RNA_read': True, 'data_identity': view.identity, 'optimizer_steps': 0, 'test_used_for_selection': False,
            'normalization_or_bandwidth_refitted': False, 'T_V_representation_cache_constructed': False, 'functional_Y_read': False,
            'observed_targets_used_only_for_metrics': True, 'conditions': 3, 'actual_P_population_predictions': 3,
            'settings': settings, 'trained_reference_contract': contract, 'independently_trained_zero_h_model': variant == 'zero_h',
            'frozen_model_hashes': model_hashes, 'independent_split_groups': 2, 'group_equal_metrics': module.aggregate(rows)['group_equal_metrics'],
            'inference_sensitivity_is_state_increment_experiment': False, **module.NO_EFFECT_CLAIMS}
        result['files'] = {p.name: module.sha(p) for p in directory.iterdir()}
        module.write(directory / 'result.json', result); directories.append(directory)
    return view, directories


def reclose_fixture(directory):
    """Rehash a deliberately edited toy file to reach semantic rejection."""
    result = module.read(directory / 'result.json')
    result['files'] = {p.name: module.sha(p) for p in directory.iterdir() if p.name != 'result.json'}
    atomic_json(directory / 'result.json', result)


def test_three_synthetic_result_readbacks_compare_without_PT_RNA_or_models(tmp_path, monkeypatch):
    view, directories = comparison_fixture(tmp_path, monkeypatch)
    original_sha = module.sha
    monkeypatch.setattr(module, 'sha', lambda p: pytest.fail('comparison may not hash PT bytes') if str(p).endswith('.pt') else original_sha(p))
    monkeypatch.setattr(module.torch, 'load', lambda *a, **k: pytest.fail('comparison cannot load PT'))
    monkeypatch.setattr(module, 'EngineData', lambda *a, **k: pytest.fail('comparison cannot open any RNA role'))
    result = module.compare({}, directories)
    assert result['three_variant_comparison_complete'] and result['PT_loads'] == result['model_forwards'] == 0
    assert result['comparison']['conditions'] == 3 and result['comparison']['split_components'] == 2
    assert result['comparison']['group_equal_metrics']['observed_minus_zero_h']['standardized_mean_mse'] == -1
    assert not result['state_increment_validated'] and not result['statistical_CI_or_test']
    assert not result['RNA_values_read_during_comparison']


@pytest.mark.parametrize('mutation', ['ids', 'seed', 'metric_name', 'moment_hash', 'role', 'E_id_in_fit', 'standardized_units', 'training_lr', 'AE_origin'])
def test_rehashed_comparison_schema_or_information_changes_are_rejected(tmp_path, monkeypatch, mutation):
    view, directories = comparison_fixture(tmp_path, monkeypatch)
    directory = directories[1]; path = directory / ('condition-' + module.config_hash('E1') + '.json')
    row = module.read(path)
    if mutation == 'ids': row['reference_cell_ids'] = ['control-2', 'control-1']
    elif mutation == 'seed': row['sampling_seed'] += 1
    elif mutation == 'metric_name': row['methods']['full_P']['raw_mean_mse'] = row['methods']['full_P'].pop('standardized_mean_mse')
    elif mutation == 'moment_hash': row['gene_moments_sha256'] = 'f'*64
    elif mutation == 'standardized_units': row['raw_source_units'] = 'TPM'
    elif mutation in ('role', 'E_id_in_fit'):
        base = module.read(directory / 'fit_baseline_receipts.json')
        if mutation == 'role': base[0]['sampled_fit_rows'][0]['role'] = 'final_test'
        else: base[0]['sampled_fit_rows'][0]['cell_ids'] = ['E1-target-1'] * 32
        atomic_json(directory / 'fit_baseline_receipts.json', base)
    else:
        result = module.read(directory / 'result.json'); plan = module.read(directory / 'plan.json')
        run = module.Path(plan['response_run']); assets = run.with_name(run.name + '-assets')
        training = module.read(assets / 'run_manifest.json')
        if mutation == 'training_lr':
            training['settings']['lr'] *= 2
        else:
            training['AE_declaration']['weights'] = str(tmp_path / 'experiments/different-AE/weights/latent_ae/best.pt')
        atomic_json(assets / 'run_manifest.json', training)
        result['frozen_model_hashes'][str(assets / 'run_manifest.json')] = module.sha(assets / 'run_manifest.json')
        atomic_json(directory / 'result.json', result)
    atomic_json(path, row); reclose_fixture(directory)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.compare({}, directories)


def test_missing_actual_comparison_outputs_remain_pending_without_any_PT(tmp_path, monkeypatch):
    view, directories = comparison_fixture(tmp_path, monkeypatch)
    missing = tmp_path / 'experiments/missing-actual-final'
    monkeypatch.setattr(module, 'read_completed_evaluation', lambda *a: pytest.fail('missing outputs must not open evaluator packs'))
    result = module.compare({}, [directories[0], directories[1], missing], tmp_path / 'experiments/comparison-pending')
    assert result['status'].startswith('pending') and not result['three_variant_comparison_complete']
    assert len(result['missing_actual_outputs']) == 2 and result['PT_loads'] == result['model_forwards'] == 0


def test_comparison_rejects_PT_named_report_before_hashing_it(tmp_path, monkeypatch):
    view, directories = comparison_fixture(tmp_path, monkeypatch)
    result = module.read(directories[0] / 'result.json'); result['files']['caller.pt'] = 'f'*64
    atomic_json(directories[0] / 'result.json', result)
    original_sha = module.sha
    monkeypatch.setattr(module, 'sha', lambda p: pytest.fail('PT file must be refused before hash') if str(p).endswith('.pt') else original_sha(p))
    with pytest.raises(ValueError, match='REPORT_ARRAY_FILES_ONLY_NO_PT'):
        module.compare({}, directories)


def test_paired_variant_summary_keeps_original_component_equal_weight(tmp_path, monkeypatch):
    view, directories = comparison_fixture(tmp_path, monkeypatch)
    runs = [module.read_completed_evaluation(view, p) for p in directories]
    runs[0]['rows'][2]['methods']['full_P'] = {'standardized_mean_mse': 3., 'standardized_population_mmd_v': 3.}
    result = module.paired_variant_comparison(runs)
    assert result['group_equal_metrics']['observed']['standardized_mean_mse'] == 1.5
    assert result['group_equal_metrics']['observed_minus_zero_h']['standardized_mean_mse'] == .5
    assert result['per_split_component_metrics']['component-B']['observed_minus_zero_h']['standardized_mean_mse'] == 2.
