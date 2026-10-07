"""OT-only regression from the CUDA BF16 pipeline's printed failing pools."""

import math
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models.objectives import entropic_ot, mean_squared_distance


def failing_pools(device):
    # Four-decimal values printed by runtime-agent-1.log; synthetic, not real cells.
    source = [[.0582, -.0629, .2331, .1359], [.0921, -.3373, .0843, .3325],
              [.0131, -.0125, .3374, .1548], [.0921, -.3373, .0843, .3325]]
    target = [[.0794, -.1593, .1771, .2193], [.0794, -.1593, .1771, .2193],
              [.0661, -.3250, .0772, .3544], [.0371, -.0151, .3639, .0925]]
    return torch.tensor(source, device=device), torch.tensor(target, device=device)


def ordinary_sinkhorn(source, target, count):
    cost = mean_squared_distance(source.double(), target.double())
    kernel = -cost / cost.quantile(.5).clamp_min(1e-6) / .05
    u, v = cost.new_zeros(len(source)), cost.new_zeros(len(target))
    for _ in range(count):
        u = -math.log(len(source)) - torch.logsumexp(kernel + v[None], 1)
        v = -math.log(len(target)) - torch.logsumexp(kernel + u[:, None], 0)
    return (kernel + u[:, None] + v[None]).exp()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runtime_pool_converges_at_original_objective(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source, target = failing_pools(device)
    baseline = ordinary_sinkhorn(source, target, 10000)
    baseline_error = (baseline.sum(1) - .25).abs().sum().item()
    assert baseline_error > 1e-6
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        plan, info = entropic_ot(source, target, return_info=True)
    assert plan.dtype == torch.float64 and torch.isfinite(plan).all() and (plan > 0).all()
    assert 3000 < info["iterations"] < 3100
    assert info["row_l1"] <= 1e-6 and info["column_l1"] <= 1e-6
    cost = mean_squared_distance(source.double(), target.double())
    potentials = plan.log() + cost / cost.quantile(.5).clamp_min(1e-6) / .05
    torch.testing.assert_close(potentials, potentials[:, :1] + potentials[:1, :] - potentials[0, 0],
                               rtol=0, atol=1e-12)
    assert torch.equal(plan, entropic_ot(source, target, max_iter=info["iterations"]))
    with pytest.raises(RuntimeError, match="OT_NUMERICAL_FAILURE"):
        entropic_ot(source, target, max_iter=info["iterations"] - 1)
    with pytest.raises(RuntimeError, match="OT_NUMERICAL_FAILURE"):
        entropic_ot(source, target, max_iter=1)
    print(device, "ordinary_10000_row_l1", baseline_error, "guarded_newton", info)


def test_invalid_newton_direction_cannot_bypass_failure():
    source, target = failing_pools("cpu")
    with patch("torch.linalg.solve_ex", return_value=(torch.full((3,), float("nan"),
                                                                  dtype=torch.float64), torch.tensor(0))):
        with pytest.raises(RuntimeError, match="OT_NUMERICAL_FAILURE"):
            entropic_ot(source, target, max_iter=3001)


def test_ascent_direction_is_rejected_by_bounded_backtracking():
    source, target = failing_pools("cpu")
    solve = torch.linalg.solve_ex

    def ascent(*args):
        direction, status = solve(*args)
        return -direction, status

    with patch("torch.linalg.solve_ex", side_effect=ascent) as mock:
        with pytest.raises(RuntimeError, match="OT_NUMERICAL_FAILURE"):
            entropic_ot(source, target, max_iter=3001)
        assert mock.call_count == 1
