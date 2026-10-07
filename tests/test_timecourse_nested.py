"""Actual tiny synthetic P/B/K/U training, isolation, calibration, and recovery."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.timecourse.data import TimeData
from lung.timecourse.nested import run_nested, FrozenTimePredictor
from lung.timecourse.auxiliary import calibrate, population_error, fit_regressor, infer
from lung.models.objectives import rbf_mmd
from test_timecourse_data_response import tiny_config


def fixture_data(change=1.):
    rows, counts = [], []
    drugs = ['S0000', 'd0', 'd1', 'd2', 'd3', 'c', 'e']
    for time in (24, 72):
        for drug in drugs:
            for _ in range(3):
                rows.append(dict(catalog_number=drug, time_point=time, dose_uM=int(drug != 'S0000'),
                                 replicate='rep1', culture_plate=f'p{time}', author_hash_qc=True))
                index = drugs.index(drug)
                value = np.array([10000 + index, 10020 - index, 10030 + index, 10040.])
                if time == 72 and drug in ('d0', 'd1', 'c', 'e'):
                    value[0] *= change
                counts.append(value)
    return TimeData.from_arrays(counts, pd.DataFrame(rows), ['a', 'b', 'c', 'd'],
                               {drug: np.zeros(2048) for drug in drugs}, fit_drugs=[],
                               allowed_target_drugs=[], synthetic=True,
                               drug_groups={'D': drugs[1:5], 'C': ['c'], 'E': ['e']})


def test_error_and_group_rank():
    x = np.array([[0., 1.], [1., 0.]])
    assert population_error(x, x)['error'] == pytest.approx(0.)
    assert population_error(x, x + 1)['mean_mse'] == pytest.approx(1.)
    assert population_error(x, x[:1])['error'] is None
    assert population_error(x, x[:0])['status'] == 'empty'
    rows = [{'drug': str(i), 'error': float(i+1), 'u': 1.} for i in range(9)]
    result = calibrate(rows, eps=0.)
    assert result['rank'] == 9 and result['quantile'] == 9.
    assert calibrate(rows[:8])['status'] == 'unbounded'
    assert calibrate(rows + [dict(drug='0', error=20., u=1.)], eps=0.)['quantile'] == 20.


def test_error_matches_training_kernel_and_gene_repetition():
    rng = np.random.default_rng(123)
    x, y = rng.normal(size=(67, 5)), rng.normal(size=(71, 5)) + .3
    bandwidth = .7
    measured = population_error(x, y, bandwidth)
    expected = float(rbf_mmd(torch.from_numpy(x), torch.from_numpy(y), bandwidth))
    assert measured['mmd_v'] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # Independent direct five-scale expression, not a second wrapper around the kernel.
    def reference_kernel(a, b):
        distance = ((a[:, None, :] - b[None, :, :]) ** 2).mean(-1)
        return np.exp(-distance[..., None] / (2 * bandwidth * np.array([.25, .5, 1., 2., 4.]))).mean(-1)
    direct = reference_kernel(x, x).mean() + reference_kernel(y, y).mean() - 2 * reference_kernel(x, y).mean()
    assert measured['mmd_v'] == pytest.approx(direct, rel=1e-12, abs=1e-12)
    repeated = population_error(np.tile(x, (1, 7)), np.tile(y, (1, 7)), bandwidth)
    assert repeated['mmd_v'] == pytest.approx(measured['mmd_v'], abs=1e-12)
    assert repeated['mean_mse'] == pytest.approx(measured['mean_mse'], abs=1e-12)


def test_auxiliary_interruption_resume(tmp_path):
    torch.set_num_threads(1)
    x, y = np.arange(12, dtype=np.float32).reshape(4, 3)/10, np.ones((4, 2), dtype=np.float32)
    cfg = dict(epochs=3, device='cpu', hidden=3, seed=9)
    def stop(role, epoch, total):
        if epoch == 1:
            raise RuntimeError('interrupt')
    with pytest.raises(RuntimeError, match='interrupt'):
        fit_regressor(x, y, tmp_path/'resume.pt', cfg, {'role': 'B'}, progress=stop)
    resumed = fit_regressor(x, y, tmp_path/'resume.pt', cfg, {'role': 'B'}, resume=True)
    direct = fit_regressor(x, y, tmp_path/'direct.pt', cfg, {'role': 'B'})
    np.testing.assert_array_equal(infer(resumed, x[0]), infer(direct, x[0]))
    with pytest.raises(ValueError, match='IDENTITY'):
        fit_regressor(x + 1, y, tmp_path/'resume.pt', cfg, {'role': 'B'}, resume=True)


def test_actual_nested_isolation_and_frozen_prediction(tmp_path):
    torch.set_num_threads(1)
    config = {**tiny_config(), 'time_auxiliary': {'epochs': 2, 'hidden': 4, 'population': 2,
                                                'outer_folds': 2, 'inner_folds': 2}}
    data, changed = fixture_data(), fixture_data(1.001)
    # Replace only the disk source constructor; all preprocessing and models execute.
    def bind(source):
        return lambda root, config, fit_drugs, allowed_target_drugs, role: source.view(fit_drugs, allowed_target_drugs, role)
    with patch('lung.timecourse.nested.TimeData', side_effect=bind(data)):
        def interrupt_fold(phase, current, total):
            if phase == 'time_nested_outer_folds' and current == 1:
                raise RuntimeError('fold interruption')
        with pytest.raises(RuntimeError, match='fold interruption'):
            run_nested(tmp_path, config, tmp_path/'first', progress=interrupt_fold)
        fold_before = {str(p): p.stat().st_mtime_ns for p in (tmp_path/'first/outer0').rglob('*.pt')}
        def interrupt_u(phase, current, total):
            if phase == 'final/U' and current == 1:
                raise RuntimeError('U interruption')
        with pytest.raises(RuntimeError, match='U interruption'):
            run_nested(tmp_path, config, tmp_path/'first', resume=True, progress=interrupt_u)
        result = run_nested(tmp_path, config, tmp_path/'first', resume=True)
        assert fold_before == {str(p): p.stat().st_mtime_ns for p in (tmp_path/'first/outer0').rglob('*.pt')}
        before = {p: Path(p).stat().st_mtime_ns for p in result['artifacts']}
        resumed = run_nested(tmp_path, config, tmp_path/'first', resume=True)
        assert before == {p: Path(p).stat().st_mtime_ns for p in resumed['artifacts']}
        with patch('lung.timecourse.nested.implementation_identity', return_value='changed-shared-code'):
            with pytest.raises(ValueError, match='NESTED_RESUME_IDENTITY_MISMATCH'):
                run_nested(tmp_path, config, tmp_path/'first', resume=True)
        predictor = FrozenTimePredictor(tmp_path, config, result)
        cid = data.condition_ids(drugs=['e'])[0]
        with patch.object(TimeData, 'sample_target', side_effect=AssertionError('target read at inference')):
            prediction = predictor.predict(cid, sample_count=2, seed=4)
        assert prediction['projected'].shape == (2, 4)
        assert prediction['uncertainty']['status'] == 'unbounded'
        fixed = data.sample_controls(cid, 2, np.random.default_rng(3))['control']
        rng_before = torch.get_rng_state().clone()
        with patch('torch.load', side_effect=AssertionError('frozen chain reloaded')):
            with patch.object(TimeData, 'sample_controls', side_effect=AssertionError('fixed state resampled')):
                anchored = predictor.predict(cid, sample_count=2, seed=4, reference=fixed)
                again = predictor.predict(cid, sample_count=2, seed=4, reference=fixed)
        assert torch.equal(rng_before, torch.get_rng_state())
        for key in ('raw', 'corrected', 'projected'):
            assert isinstance(anchored[key], torch.Tensor)
            torch.testing.assert_close(anchored[key], again[key], rtol=0, atol=0)
        assert anchored['uncertainty'] == again['uncertainty']
    with patch('lung.timecourse.nested.TimeData', side_effect=bind(changed)):
        run_nested(tmp_path, config, tmp_path/'changed')
    # d0/d1 are outer-fold zero's target only: every fitted P/B/K parameter is identical.
    paths = list((tmp_path/'first/outer0').rglob('*.pt'))
    for first in paths:
        other = tmp_path/'changed'/first.relative_to(tmp_path/'first')
        left, right = (torch.load(p, weights_only=False) for p in (first, other))
        if first.name == 'inner_features.pt':
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a['features'], b['features'])
                np.testing.assert_array_equal(a['residual'], b['residual'])
        elif first.name != 'completed.pt':
            for key, expected in left['model'].items():
                actual = right['model'][key]
                if expected.is_sparse:
                    assert torch.equal(actual.coalesce().indices(), expected.coalesce().indices()), key
                    assert torch.equal(actual.coalesce().values(), expected.coalesce().values()), key
                else:
                    assert torch.equal(actual, expected), key
    audit = json.loads(Path(result['audit']).read_text())
    for model in audit['models']:
        assert not set(model['fit_drugs']) & {'c', 'e'}
        assert not set(model['fit_drugs']) & set(model['excluded_drugs'])
    assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in result['artifacts'])
    assert list((tmp_path/'first').rglob('*.metrics.csv'))
    assert list((tmp_path/'first').rglob('events.out.tfevents.*'))
