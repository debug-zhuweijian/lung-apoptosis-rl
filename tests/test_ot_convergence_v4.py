"""Regression for slow log-Sinkhorn convergence, with the original OT objective."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.models.objectives import entropic_ot, mean_squared_distance


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def epoch_two_second_draw():
    # SyntheticData's second condition population at member 0, epoch index 1.
    rng = np.random.default_rng(20260924)
    for _ in range(2):
        x = rng.normal(0, .2, (4, 8)).astype(np.float32)
        y = rng.normal(.2, .2, (4, 8)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def test_slow_fixture_continues_same_objective_to_original_tolerance():
    x,y = epoch_two_second_draw()
    with pytest.raises(RuntimeError, match='row L1=8.1e-05'):
        entropic_ot(x,y,max_iter=1000)
    plan,info = entropic_ot(x,y,return_info=True)
    assert 1000 < info['iterations'] <= 10000
    assert info['iterations'] == 2307
    assert info['row_l1'] <= 1e-6 and info['column_l1'] <= 1e-6
    assert torch.isfinite(plan).all() and (plan > 0).all()
    cost = mean_squared_distance(x.double(),y.double())
    # log P = -C/(median(C)*epsilon) + row potential + column potential.
    # This checks preservation of the Gibbs objective, not just feasible marginals.
    potentials = plan.log() + cost / cost.quantile(.5).clamp_min(1e-6) / .05
    separable = potentials[:, :1] + potentials[:1, :] - potentials[0,0]
    torch.testing.assert_close(potentials,separable,rtol=0,atol=1e-12)
    explicit = entropic_ot(x,y,max_iter=info['iterations'])
    assert torch.equal(plan,explicit)
    with pytest.raises(RuntimeError, match='OT_NUMERICAL_FAILURE'):
        entropic_ot(x,y,max_iter=info['iterations']-1)
    print('original_cap=1000 failed; default_cap=10000', info)


def test_fast_problem_and_explicit_iteration_failure_unchanged():
    rng=np.random.default_rng(72)
    x,y=torch.tensor(rng.normal(size=(5,8))),torch.tensor(rng.normal(size=(4,8)))
    before,old_info=entropic_ot(x,y,max_iter=1000,return_info=True)
    after,new_info=entropic_ot(x,y,return_info=True)
    assert torch.equal(before,after) and old_info==new_info
    with pytest.raises(RuntimeError, match='OT_NUMERICAL_FAILURE'):
        entropic_ot(x,y,max_iter=1,tolerance=1e-18)
    print('fast_input_bitwise_unchanged',new_info)
