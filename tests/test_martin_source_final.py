"""Three-gene synthetic fixtures; metadata preflight never reads real RNA/PT.

The positive frozen-checkpoint fixture deliberately constructs matching local
metadata under a mocked source factory. It is no real source/model qualification.
"""
from dataclasses import fields, replace
import numpy as np
import pytest
import torch

from lung.data.martin_source_inputs import canonical, digest, sha
from lung.engine import martin_source_final as f
from lung.engine import martin_source_learning as learning
from tests.test_martin_source_learning import datasets, settings
from tests.test_martin_source_model import synthetic_batch


class SyntheticContext:
    def __init__(self, root):
        self.root = root.resolve()
        fit, selection = datasets()
        self.contract, self.gene_ids = fit.contract, synthetic_batch().gene_ids
        self.fold_id = self.contract['fold_id']
        self.roles = {'final_test': [{'request_id': 'FINAL_REQUEST_A'}, {'request_id': 'FINAL_REQUEST_B'}]}
        self.ids = {'final_test': ['FINAL_E', 'FINAL_F', 'FINAL_G', 'FINAL_H', 'FINAL_I', 'FINAL_J']}
        self.counts = {'fit': fit.counts, 'selection': selection.counts,
            'final_test': {'unique_GSM': 6, 'unique_reference_GSM': 2, 'unique_target_GSM': 4,
                'domain_views_per_GSM': 2, 'reference_domain_request_rows_before_dedup': 8,
                'target_domain_request_rows_before_dedup': 8, 'not_twice_as_many_biological_replicates': True}}
        self.scope = {'schema': f.SCHEMA, 'role': 'final_test', 'system': 'cycle', 'fold_id': self.fold_id,
            'input_contract_sha256': digest(self.contract), 'final_sample_ids': self.ids['final_test'],
            'request_ids': ['FINAL_REQUEST_A', 'FINAL_REQUEST_B'], 'statistic_refitting': False, 'model_selection': False}
        self.closed = self.checked = False

    def batch(self, request, domain, part):
        original = synthetic_batch(domain, part)
        ids = ('FINAL_E', 'FINAL_F') if part == 'reference' else (
            ('FINAL_G', 'FINAL_H') if request['request_id'] == 'FINAL_REQUEST_A' else ('FINAL_I', 'FINAL_J'))
        value = {key.name: getattr(original, key.name) for key in fields(original)}
        value.update(role='final_test', sample_ids=ids, request_id=request['request_id'],
            protocol='Martin2022_RNA_DTP' if request['request_id'] == 'FINAL_REQUEST_B' else original.protocol,
            final_scope_json=canonical(self.scope))
        return f.FinalSourceBatch(**value)

    def close(self):
        self.closed = True

    def verify_source_bytes(self):
        self.checked = True


def synthetic_run(tmp_path, *, synthetic=False):
    context = SyntheticContext(tmp_path)
    fit, selection = datasets()
    directory = tmp_path / 'experiments' / 'synthetic_completed'
    # All values are synthetic. False merely exercises the production metadata
    # checks with this mocked three-gene source, never the registered real source.
    learning.fit_source(fit, selection, settings(1), directory, synthetic=synthetic)
    return directory, context


def test_real_final_metadata_preflight_does_not_read_rna_statistics_or_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Metadata preflight must not read any RNA/coordinate arrays or execute/load models')
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(f, 'MartinSourceModel', forbidden)
    report = f.final_preflight(f.ROOT, fold_id='cycle_final:HCC2935')
    assert report['final_counts']['unique_GSM'] == 15
    assert report['final_counts']['unique_reference_GSM'] == 3
    assert len(report['scope']['request_ids']) == 4
    assert report['final_RNA_values_read'] is report['fixed_fit_statistics_values_read'] is False
    assert report['PT_loads'] == report['model_forwards'] == report['optimizer_steps'] == 0


@pytest.mark.parametrize('domain', f.DOMAINS)
@pytest.mark.parametrize('part', ('reference', 'target'))
def test_final_batches_keep_true_role_units_nan_and_source_clock(tmp_path, domain, part):
    context = SyntheticContext(tmp_path)
    batch = context.batch(context.roles['final_test'][0], domain, part)
    batch.validate()
    assert batch.role == 'final_test' and batch.units == context.contract['units'][domain]
    assert torch.isnan(batch.raw_values[~batch.observed_mask]).all()
    assert (batch.measurement_clock == (24 if part == 'reference' else 576)).all()
    if domain == 'estimated_counts':
        assert (batch.raw_values[batch.observed_mask] % 1 != 0).any()
    else:
        assert (batch.raw_values[batch.observed_mask] < 0).any()


@pytest.mark.parametrize('change', (
    {'role': 'fit'}, {'sample_ids': ('SYNTHETIC_SAMPLE_A', 'FINAL_F')},
    {'fold_id': 'OTHER_FOLD'}, {'request_id': 'OTHER_REQUEST'}))
def test_final_scope_rejects_role_sample_fold_and_request_relabeling(tmp_path, change):
    context = SyntheticContext(tmp_path)
    batch = context.batch(context.roles['final_test'][0], 'log2TPM', 'reference')
    with pytest.raises(ValueError, match='FINAL_SCOPE'):
        replace(batch, **change).validate()


@pytest.mark.parametrize('part', ('reference', 'target'))
def test_final_clock_cannot_be_shifted(tmp_path, part):
    context = SyntheticContext(tmp_path)
    batch = context.batch(context.roles['final_test'][0], 'log2TPM', part)
    with pytest.raises(ValueError, match='FINAL_SOURCE_CLOCK'):
        replace(batch, measurement_clock=batch.measurement_clock + 1000).validate()


def test_old_final_reader_and_fit_collection_remain_closed():
    with pytest.raises(ValueError, match='FINAL_NOT_ENABLED'):
        learning.collect_observations([], role='final_test', contract=synthetic_batch().contract)
    with pytest.raises(ValueError, match='FINAL.*NOT_ENABLED'):
        replace(synthetic_batch(), role='final_test').validate()


def test_completion_proof_is_emitted_and_synthetic_run_rejected_before_pt(tmp_path, cpu_only, monkeypatch):
    directory, context = synthetic_run(tmp_path, synthetic=True)
    result = f.read(directory / 'result.json')
    assert result['completion'] == {'status': 'complete', 'reason': 'max_epochs'}
    assert result['selected_checkpoint']['sha256'] == sha(directory / 'best.pt')
    def forbidden(*args, **kwargs):
        pytest.fail('Synthetic run must reject before loading checkpoints')
    monkeypatch.setattr(f, 'load_checkpoint', forbidden)
    with pytest.raises(ValueError, match='FINAL_COMPLETED'):
        f.load_frozen_model(directory, context, 'cpu')


@pytest.mark.parametrize('surface', ('dependencies', 'best_bytes', 'incomplete', 'counts'))
def test_final_completion_metadata_drift_rejects_before_pt(tmp_path, cpu_only, monkeypatch, surface):
    directory, context = synthetic_run(tmp_path)
    if surface == 'dependencies':
        monkeypatch.setattr(learning, 'execution_dependencies', lambda: {'changed': True})
    elif surface == 'best_bytes':
        with (directory / 'best.pt').open('ab') as stream:
            stream.write(b'SYNTHETIC_BYTE_DRIFT')
    else:
        result = f.read(directory / 'result.json')
        if surface == 'incomplete':
            result['completion']['status'] = 'running'
        else:
            result['fit_counts']['unique_GSM'] += 1
        learning.atomic_json(directory / 'result.json', result)
    def forbidden(*args, **kwargs):
        pytest.fail('Metadata drift must reject before PT or model construction')
    monkeypatch.setattr(f, 'load_checkpoint', forbidden)
    monkeypatch.setattr(f, 'MartinSourceModel', forbidden)
    with pytest.raises(ValueError):
        f.load_frozen_model(directory, context, 'cpu')


def test_final_actual_last_counter_must_match_completion(tmp_path, cpu_only):
    directory, context = synthetic_run(tmp_path)
    last = learning.load_checkpoint(directory / 'last.pt')
    last['counters']['optimizer_step'] += 1
    learning.atomic_save(directory / 'last.pt', last)
    result = f.read(directory / 'result.json')
    result['last_checkpoint']['sha256'] = sha(directory / 'last.pt')
    learning.atomic_json(directory / 'result.json', result)
    with pytest.raises(ValueError, match='FINAL_ACTUAL_COMPLETED_LAST'):
        f.load_frozen_model(directory, context, 'cpu')


@pytest.mark.parametrize('surface', ('selected_history', 'selected_cursor_bool'))
def test_final_selected_snapshot_must_match_its_saved_selection_epoch(tmp_path, cpu_only, surface):
    directory, context = synthetic_run(tmp_path)
    selected = learning.load_checkpoint(directory / 'best.pt')
    if surface == 'selected_history':
        selected['extra']['history'] = []
    else:
        selected['extra']['cursor'] = False
    learning.atomic_save(directory / 'best.pt', selected)
    result = f.read(directory / 'result.json')
    result['selected_checkpoint']['sha256'] = sha(directory / 'best.pt')
    learning.atomic_json(directory / 'result.json', result)
    with pytest.raises(ValueError, match='FINAL_ACTUAL_SELECTED_HISTORY'):
        f.load_frozen_model(directory, context, 'cpu')


def test_final_uses_frozen_fit_weights_and_deduplicated_true_final_units(tmp_path, cpu_only, monkeypatch):
    directory, context = synthetic_run(tmp_path)
    monkeypatch.setattr(f, 'FinalSourceInputs', lambda root, fold_id: context)
    def forbidden(*args, **kwargs):
        pytest.fail('Final evaluation cannot fit, prepare fit/selection or construct optimizer')
    monkeypatch.setattr(learning, 'fit_source', forbidden)
    monkeypatch.setattr(learning, 'prepare_fold', forbidden)
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    model, _ = f.load_frozen_model(directory, context, 'cpu')
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    before = {name: sha(directory / name) for name in ('best.pt', 'last.pt', 'result.json', 'run_manifest.json')}
    report = f.evaluate_final(tmp_path, fold_id=context.fold_id, trained_run=directory,
        output=tmp_path / 'experiments' / 'synthetic_final')
    assert context.closed and context.checked
    assert report['optimizer_steps'] == 0 and report['model_selection'] is False
    assert len(report['records']) == 12 and all(r['role'] == 'final_test' for r in report['records'])
    assert set(report['by_domain']) == set(f.DOMAINS)
    assert all(sum(r['domain'] == d for r in report['records']) == 6 for d in f.DOMAINS)
    assert report['scientific_qualified'] is False
    assert before == report['frozen_assets_sha256'] == {name: sha(directory / name) for name in before}


def test_final_fp_binding_and_source_domain_transform_fixture(tmp_path, monkeypatch):
    # Exercising the standalone actual reader on synthetic array files avoids
    # reopening the registered real final RNA during this authorized CI.
    context = object.__new__(f.FinalSourceInputs)
    context.root = tmp_path
    fp_name = (f.Path(f.BASE) / 'chemistry/morgan2048_chiral.npy').as_posix()
    path = tmp_path / fp_name
    path.parent.mkdir(parents=True)
    np.save(path, np.asarray([[1, 0] + [0] * 2046], dtype=np.uint8))
    context.manifests = {'observed-protocol-inputs-v1': {'source_bindings': {fp_name: sha(path)}}}
    fields = dict(zip(f.SCALAR_FIELDS, [500, 0, 24, 24, None, None, None, None]))
    group = {'endpoint_hours': 24., 'events': [{'event_type': 'drug_interval', 'dose_unit': 'nM',
        'vehicle': False, 'values': fields, 'fingerprint': 'synthetic'}]}
    physical = context.physical(group)
    physical.validate()
    assert physical.fingerprint[0, 0, 0] == 1
