"""Constructed tensors only; this module must never forward real RNA or load PT."""
from dataclasses import replace

import pytest
import torch

from lung.data.martin_source_inputs import (BACKGROUNDS, DOMAINS, META_FIELDS, PROTOCOLS,
                                          SCHEMA, MartinSourceBatch, MartinSourceInputs,
                                          canonical, digest)
from lung.models.full_protocol import FullPhysicalProtocolBatch
from lung.models.martin_source import MartinSourceModel


def synthetic_physical():
    values = torch.zeros((2, 4, 8))
    known = torch.zeros_like(values, dtype=torch.bool)
    values[:, 0, :4] = torch.tensor([500., 0., 504., 504.])
    known[:, 0, :4] = True
    for index in (1, 2):
        values[:, index, 1:3] = 504
        known[:, index, 1:3] = True
    values[:, 1, 4], known[:, 1, 4] = 2, True
    values[:, 2, 6], known[:, 2, 6] = 1, True
    values[:, 3, [1, 2, 7]] = torch.tensor([504., 576., 72.])
    known[:, 3, [1, 2, 7]] = True
    fp = torch.zeros((2, 4, 2048))
    fp[:, 0, :2] = 1
    fp_known = torch.tensor([[True, False, False, False]]).repeat(2, 1)
    return FullPhysicalProtocolBatch(fp, fp_known, torch.zeros((2, 4), dtype=torch.bool),
                                    torch.tensor([[0, 2, 4, 3]]).repeat(2, 1),
                                    torch.tensor([[2, 0, 0, 0]]).repeat(2, 1), values, known,
                                    torch.full((2,), 4), torch.full((2,), 576.), torch.ones(2, dtype=torch.bool))


def synthetic_batch(domain='log2TPM', part='target'):
    genes = ('ENSG_TEST_1', 'ENSG_TEST_2', 'ENSG_TEST_ABSENT')
    mask = torch.tensor([[True, True, False]]).repeat(2, 1)
    raw = torch.tensor([[-4., 2., torch.nan], [0., 3., torch.nan]], dtype=torch.float64)
    offset, scale = torch.tensor([0., 2., torch.nan], dtype=torch.float64), torch.tensor([2., 1., torch.nan], dtype=torch.float64)
    if domain == 'estimated_counts':
        raw = torch.tensor([[.25, 20.5, torch.nan], [0., 8.25, torch.nan]], dtype=torch.float64)
        offset, scale = torch.tensor([0., 0., torch.nan], dtype=torch.float64), torch.tensor([1., 1., torch.nan], dtype=torch.float64)
    transformed = (raw - offset) / scale if domain == 'log2TPM' else raw.log1p()
    meta = torch.full((2, len(META_FIELDS)), torch.nan, dtype=torch.float64)
    meta[:, :2], meta[:, -1] = torch.tensor([21., 24.]), 0.
    contract = {'schema': SCHEMA, 'source_id': 'GSE193258', 'gene_ids_sha256': digest(genes),
                'domains': list(DOMAINS), 'backgrounds': list(BACKGROUNDS), 'protocols': list(PROTOCOLS),
                'units': {'log2TPM': 'synthetic_log2TPM', 'estimated_counts': 'synthetic_estimate'},
                'system': 'cycle', 'fold_id': 'synthetic_only', 'coordinates_manifest_sha256': 'synthetic',
                'fit_sample_ids': ['SYNTHETIC_SAMPLE_A', 'SYNTHETIC_SAMPLE_B'],
                'selection_sample_ids': ['SYNTHETIC_SAMPLE_C', 'SYNTHETIC_SAMPLE_D'],
                'same_cell_t0': False, 'functional_Y': False, 'scientific_qualified': False}
    return MartinSourceBatch(raw, transformed, mask, offset, scale,
                             torch.full((2,), 576. if part == 'target' else 24., dtype=torch.float64),
                             meta, torch.isfinite(meta), synthetic_physical(), domain, contract['units'][domain], genes,
                             ('SYNTHETIC_SAMPLE_A', 'SYNTHETIC_SAMPLE_B'), 'GSE193258', 'H1975',
                             'Martin2022_RNA_long_washout', part, 'SYNTHETIC_REQUEST', 'cycle',
                             'synthetic_only', 'fit', canonical(contract))


@pytest.fixture
def source_model(cpu_only):
    torch.manual_seed(23)
    batch = synthetic_batch()
    return MartinSourceModel(batch.gene_ids, batch.contract, width=8)


@pytest.mark.parametrize('domain', DOMAINS)
@pytest.mark.parametrize('part', ('reference', 'target'))
def test_synthetic_domain_encode_decode_loss(source_model, domain, part):
    batch = synthetic_batch(domain, part)
    output = source_model(batch)
    assert source_model.encode(batch).shape == (2, 8)
    assert output['source_values'].dtype == torch.float64
    assert torch.isnan(output['source_values'][~batch.observed_mask]).all()
    assert torch.isfinite(output['source_values'][batch.observed_mask]).all()
    assert torch.isfinite(source_model.observation_loss(output, batch))
    assert output['scientific_qualified'] is False
    if domain == 'estimated_counts':
        assert (batch.raw_values[batch.observed_mask] % 1 != 0).any()
        assert (output['source_values'][batch.observed_mask] >= 0).all()
    else:
        assert (batch.raw_values[batch.observed_mask] < 0).any()


def test_loss_excludes_missing_and_uses_observed_zero(source_model):
    batch = synthetic_batch()
    output = source_model(batch)
    baseline = source_model.observation_loss(output, batch)
    ignored = output['model_values'].clone()
    ignored[:, -1] = torch.nan
    assert torch.equal(source_model.observation_loss({**output, 'model_values': ignored}, batch), baseline)
    changed = output['model_values'].clone()
    changed[1, 0] += 4
    assert not torch.equal(source_model.observation_loss({**output, 'model_values': changed}, batch), baseline)


def test_missing_cannot_become_measured_zero(source_model):
    batch = synthetic_batch()
    raw = batch.raw_values.clone()
    raw[:, -1] = 0
    with pytest.raises(ValueError, match='MISSING_IS_NAN'):
        source_model(replace(batch, raw_values=raw))


def test_encoder_consumes_mask(source_model):
    batch = synthetic_batch()
    observed = replace(batch, raw_values=batch.raw_values.nan_to_num(),
                       model_values=batch.model_values.nan_to_num(), observed_mask=torch.ones_like(batch.observed_mask),
                       offset=batch.offset.nan_to_num(), scale=batch.scale.nan_to_num(nan=1.))
    assert not torch.equal(source_model.encode(batch), source_model.encode(observed))


@pytest.mark.parametrize('change', ('fingerprint', 'dose', 'unit', 'wash', 'recovery', 'unknown_retention', 'order'))
def test_target_consumes_reported_protocol_and_masks(source_model, change):
    batch = synthetic_batch()
    p = batch.physical
    if change == 'fingerprint':
        fp = p.fingerprint.clone()
        fp[:, 0, 2] = 1
        modified = replace(p, fingerprint=fp)
    elif change == 'unit':
        units = p.dose_unit.clone()
        units[:, 0] = 1
        modified = replace(p, dose_unit=units)
    elif change == 'order':
        indexes = torch.tensor([0, 2, 1, 3])
        modified = replace(p, fingerprint=p.fingerprint[:, indexes], fingerprint_known=p.fingerprint_known[:, indexes],
                           vehicle=p.vehicle[:, indexes], event_type=p.event_type[:, indexes], dose_unit=p.dose_unit[:, indexes],
                           values=p.values[:, indexes], known=p.known[:, indexes])
    else:
        values, known = p.values.clone(), p.known.clone()
        if change == 'dose':
            values[:, 0, 0] = 400
        elif change == 'wash':
            values[:, 1, 4] = 1
        elif change == 'recovery':
            values[:, 3, 2], values[:, 3, 7] = 600, 96
        elif change == 'unknown_retention':
            values[:, 3, 5], known[:, 3, 5] = 0, True
        modified = replace(p, values=values, known=known,
                           endpoint_hours=torch.full((2,), 600.) if change == 'recovery' else p.endpoint_hours)
    changed_batch = replace(batch, physical=modified,
                            measurement_clock=modified.endpoint_hours.to(torch.float64) if change == 'recovery' else batch.measurement_clock)
    assert not torch.equal(source_model.protocol_context(batch), source_model.protocol_context(changed_batch))


def test_unknown_physical_placeholders_cannot_change_context(source_model):
    batch = synthetic_batch()
    p = batch.physical
    values = torch.where(p.known, p.values, torch.full_like(p.values, torch.nan))
    fp = torch.where(p.fingerprint_known[..., None], p.fingerprint, torch.full_like(p.fingerprint, 900.))
    modified = replace(batch, physical=replace(p, values=values, fingerprint=fp))
    torch.testing.assert_close(source_model.protocol_context(batch), source_model.protocol_context(modified), rtol=0, atol=0)


def test_reference_uses_own_clock_and_no_target_protocol_reconstruction(source_model):
    batch = synthetic_batch(part='reference')
    values = batch.physical.values.clone()
    values[:, 0, 0] = 400
    modified = replace(batch, physical=replace(batch.physical, values=values))
    assert batch.measurement_clock.tolist() == [24., 24.]
    torch.testing.assert_close(source_model(batch)['model_values'], source_model(modified)['model_values'], rtol=0, atol=0)
    output = source_model.decode(source_model.encode(batch), batch, as_target=True)
    with pytest.raises(ValueError, match='OUTPUT_SUPERVISION_IDENTITY'):
        source_model.observation_loss(output, batch)


@pytest.mark.parametrize('change', ({'role': 'final_test'}, {'background': 'A549'},
                                  {'source_id': 'Tahoe-100M'}, {'fold_id': 'other'},
                                  {'gene_ids': ('changed', 'ENSG_TEST_2', 'ENSG_TEST_ABSENT')},
                                  {'domain': 'UMI_counts'}))
def test_identity_and_final_mismatch_rejected(source_model, change):
    with pytest.raises(ValueError):
        source_model(replace(synthetic_batch(), **change))


def test_final_reader_never_opens_real_files(tmp_path):
    with pytest.raises(PermissionError, match='FINAL_NOT_ENABLED'):
        MartinSourceInputs(tmp_path, allowed_roles=('final_test',))


def test_checkpoint_requires_exact_new_namespace_and_fold(source_model):
    metadata = {'model_identity': source_model.identity, 'training_role': 'fit', 'final_RNA_used': False}
    assert source_model.validate_checkpoint_metadata(metadata) is None
    identity = {**source_model.identity, 'parameter_identity': 'latent_ae'}
    with pytest.raises(ValueError, match='CHECKPOINT_DOMAIN_FOLD_AXIS'):
        source_model.validate_checkpoint_metadata({**metadata, 'model_identity': identity})
    with pytest.raises(ValueError, match='REGISTERED_SOURCE_METADATA'):
        source_model.load_state_dict({})


def test_decoder_log2_outlet_is_not_softplus_or_count_rounding(source_model):
    batch = synthetic_batch()
    offset = torch.tensor([-1000., -1000., torch.nan], dtype=torch.float64)
    raw = batch.model_values * batch.scale + offset
    negative = replace(batch, raw_values=raw, offset=offset)
    output = source_model(negative)
    assert (output['source_values'][negative.observed_mask] < 0).all()


@pytest.mark.parametrize('part', ('reference', 'target'))
def test_observation_clock_cannot_be_relabelled(source_model, part):
    batch = synthetic_batch(part=part)
    with pytest.raises(ValueError, match='OWN_CLOCK|OBSERVATION_CLOCK'):
        source_model(replace(batch, measurement_clock=batch.measurement_clock + 1000))


def test_fit_gsm_cannot_be_relabelled(source_model):
    with pytest.raises(ValueError, match='ROLE_GSM_MEMBERSHIP'):
        source_model(replace(synthetic_batch(), sample_ids=('SYNTHETIC_SAMPLE_C', 'SYNTHETIC_SAMPLE_D')))


def test_loss_cannot_exchange_same_fold_request_or_sample(source_model):
    batch = synthetic_batch()
    output = source_model(batch)
    for changed in (replace(batch, request_id='OTHER_REQUEST'), replace(batch, sample_ids=tuple(reversed(batch.sample_ids)))):
        with pytest.raises(ValueError, match='OUTPUT_REQUEST_SAMPLE_CONTEXT'):
            source_model.observation_loss(output, changed)


def test_target_decoder_uses_target_clock_even_for_reference_inputs(source_model):
    reference, target = synthetic_batch(part='reference'), synthetic_batch(part='target')
    # Same source/background/domain and physical protocol: the target-side
    # context is identical, while their source E clocks remain different.
    torch.testing.assert_close(source_model.protocol_context(reference), source_model.protocol_context(target), rtol=0, atol=0)
    assert not torch.equal(source_model._context(reference), source_model._context(target))
