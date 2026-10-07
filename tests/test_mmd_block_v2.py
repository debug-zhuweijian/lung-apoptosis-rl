"""Small exact synthetic checks; no biological or training claims."""

import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models.objectives import rbf_kernel, rbf_mmd, shared_control_mmd
from lung.engine.expression_diagnostics import FrozenExpressionDiagnostics


def test_exact_shared_control_counterexample():
    old, block = [], []
    for xa, ya in itertools.product(itertools.product([0., 1.], repeat=2), repeat=2):
        x, y = (torch.tensor(a, dtype=torch.float64)[:, None] for a in (xa, ya))
        old.append(rbf_mmd(x.repeat(5, 1), y, 1., unbiased=True))
        block.append(shared_control_mmd([x]*5, y, 1., block_size=1))
    assert float(torch.stack(old).mean()) == pytest.approx(.0990647524543693, abs=1e-14)
    assert abs(float(torch.stack(block).mean())) < 1e-14
    assert min(float(v) for v in block) < 0


def test_single_member_and_identical_shared_members():
    x = torch.tensor([[0., 1.], [1., 0.], [2., 1.]], dtype=torch.float64)
    y = x[:2]
    expected = rbf_mmd(x, y, .7, unbiased=True)
    for count in (1, 2, 5):
        for size in (1, 2, 128):
            actual = shared_control_mmd([x]*count, y, .7, block_size=size)
            assert actual.dtype == torch.float64 and actual.device == x.device
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)


def test_independent_population_reference_and_permutations():
    generator = torch.Generator().manual_seed(32)
    members = [torch.randn(3, 4, generator=generator, dtype=torch.float64) for _ in range(3)]
    y = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    flat = torch.cat(members)
    ids = torch.arange(3).repeat(3)
    xx, yy = rbf_kernel(flat, flat, .8), rbf_kernel(y, y, .8)
    expected = xx[ids[:, None] != ids[None]].mean() + (yy.sum()-yy.trace())/20 - 2*rbf_kernel(flat, y, .8).mean()
    for values in (members, [v[[2, 0, 1]] for v in members[::-1]]):
        torch.testing.assert_close(shared_control_mmd(values, y, .8, block_size=2), expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("kind", ["empty", "one_control", "one_observed", "shape", "genes", "nan", "dtype", "zero_genes", "block"])
def test_invalid_boundaries(kind):
    x = torch.ones(3, 2, dtype=torch.float64)
    members, y, size = [x], x.clone(), 2
    if kind == "empty": members = []
    if kind == "one_control": members = [x[:1]]
    if kind == "one_observed": y = y[:1]
    if kind == "shape": members = [x, x[:2]]
    if kind == "genes": y = y[:, :1]
    if kind == "nan": y[0, 0] = float("nan")
    if kind == "dtype": y = y.float()
    if kind == "zero_genes": members, y = [x[:, :0]], y[:, :0]
    if kind == "block": size = 0
    with pytest.raises(ValueError):
        shared_control_mmd(members, y, 1., block_size=size)


@pytest.mark.parametrize("gate", [False, True])
def test_diagnostics_reports_population_dependence(gate):
    diagnostic = FrozenExpressionDiagnostics.__new__(FrozenExpressionDiagnostics)
    diagnostic.data, diagnostic.device = None, "cpu"
    diagnostic.reference = SimpleNamespace(encode_condition=lambda **kwargs: torch.zeros(1, 2))
    diagnostic.gates, diagnostic.calibration = {"domains": {}}, None
    diagnostic.modules = {"pool": lambda z: z.mean(0),
                          "backward": SimpleNamespace(diagnostic=lambda *args: torch.tensor(0.)),
                          "correction": lambda raw, *args, gate: raw + (raw.mean(0) if gate else 0)}
    x = torch.tensor([[1., 2.], [3., 1.]])
    batch = {"source_id": "s", "background_id": "b", "protocol_id": "p"}
    with patch("lung.engine.expression_diagnostics.projected", lambda x, _: x), \
         patch("lung.engine.expression_diagnostics.encode_vectors", lambda _, x, device: torch.as_tensor(x)), \
         patch("lung.engine.expression_diagnostics.condition_kwargs", return_value={}), \
         patch("lung.engine.expression_diagnostics.gate_for", return_value=gate):
        output, baseline, report = diagnostic.apply([x, x+1], x, batch)
    assert report["population_correction_applied"] is gate
    assert report["mmd_estimator"] == ("empirical_population_V" if gate else "equal_member_shared_control_block_U")
    assert report["member_count"] == 2 and report["control_block_count"] == 2
    assert report["member_major_order"] is True
    torch.testing.assert_close(output, baseline + (baseline.mean(0) if gate else 0))


def test_population_mean_v_metric_never_claims_u():
    from lung.engine.pilot_response import mixture_metrics
    from lung.models.objectives import physical_projection
    x = torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64)
    mean, scale = torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    y = physical_projection(x, mean, scale).x
    result = mixture_metrics([x]*5, x, y, mean, scale, 1., population_dependent=True, block_size=1)
    assert result["mmd_u"] is None
    assert result["mmd_estimator"] == "empirical_population_V"
    assert result["na_reasons"]["mmd_u"] == "population_dependent_reference"
    assert result["empirical_population_mmd_v"] == pytest.approx(float(rbf_mmd(y.repeat(5, 1), y, 1.)), abs=1e-14)


def test_diagnostics_without_correction_preserves_block_metadata():
    diagnostic = FrozenExpressionDiagnostics({}, None, "cpu", None)
    x = torch.ones(2, 3)
    with patch("lung.engine.expression_diagnostics.projected", lambda x, _: x):
        output, baseline, report = diagnostic.apply([x]*5, x, {})
        with pytest.raises(ValueError):
            diagnostic.apply([x[:1]], x, {})
    assert report["population_correction_applied"] is False
    assert report["member_count"] == 5 and report["member_population_size"] == 2
    torch.testing.assert_close(output, torch.cat([x]*5))
    torch.testing.assert_close(baseline, output)
