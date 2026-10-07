"""Small numerical stubs test audit persistence; these are not real audit evidence."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import diagnostics
from lung.engine.modular_budget import StagePaused


class Monitor:
    def __init__(self, directory, limit=None):
        self.experiment, self.limit, self.calls = directory, limit, 0

    def stop_check(self, *_):
        self.calls += 1
        return self.limit is not None and self.calls > self.limit

    def progress(self, iterable, **_):
        return iterable


def test_ode_audit_resumes_saved_samples_without_lowering_protocol(tmp_path, monkeypatch):
    calls = []
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"unit-test-only")
    data = SimpleNamespace(identity="unit-test", n_genes=2, mu=np.zeros(2), scale=np.ones(2),
        condition_ids=lambda role: ["V-condition"],
        sample_controls=lambda *args: {"control": np.ones((1, 2), np.float32)})
    class Model:
        def encode_condition(self, **_):
            return None

        def sample(self, *args, **kwargs):
            calls.append(kwargs["steps"])
            return torch.ones(1, 2)
    monkeypatch.setattr(diagnostics, "load_response", lambda *args: Model())
    monkeypatch.setattr(diagnostics, "condition_kwargs", lambda *args: {})
    monkeypatch.setattr(diagnostics, "response_noise", lambda *args: torch.zeros(1, 2))
    monkeypatch.setattr(diagnostics, "physical_projection", lambda x, *args: SimpleNamespace(y=x, valid=torch.ones(1, dtype=torch.bool)))
    config = {"device": "cpu", "synthetic": False, "response": {"inference_ode_steps": 128}}
    pipeline = {"response": [str(weight)]}
    with pytest.raises(StagePaused):
        diagnostics.resolve_ode_steps(data, config, Monitor(tmp_path, 2), pipeline)
    assert len(calls) == 2
    assert not (tmp_path / "ode_audit.json").exists()
    result = diagnostics.resolve_ode_steps(data, config, Monitor(tmp_path), pipeline)
    assert len(calls) == 512  # 256 at 128 steps + 256 at 256 steps; saved 2 reused.
    import json
    report = json.loads(Path(result["ode_audit"]).read_text())
    assert report["n_fixed_V_noise"] == 256 and report["tolerance"] == .001
    assert report["selected_steps"] == 128 and report["passed"]


def test_population_audit_resumes_repeat_boundary_and_keeps_1024(tmp_path, monkeypatch):
    from lung.engine import auxiliary
    monkeypatch.setattr(auxiliary, "calibration_audit_binding", lambda *args: {"test": "unit-only"})
    calls = []
    class World:
        config = {"synthetic": False}
        data = SimpleNamespace()
        biological_identity = "unit-test-world"
        registry = SimpleNamespace(candidates=[SimpleNamespace(action_id="a")])
        protocol = SimpleNamespace(position=lambda registry, prefix: (None, None, bool(prefix)),
                                   legal_mask=lambda registry, prefix: torch.tensor([True]))
        def roots(self, role):
            assert role == "V"
            return [SimpleNamespace(root_id="V-root")]

        def score_one(self, root, sequence, seed, n):
            calls.append((seed, n))
            return .4
    with pytest.raises(StagePaused):
        diagnostics.population_diagnostic(World(), Monitor(tmp_path, 2), {})
    assert len(calls) == 6
    assert not (tmp_path / "population_resolution.json").exists()
    pipeline = diagnostics.population_diagnostic(World(), Monitor(tmp_path), {})
    assert len(calls) == 1024 * 3
    assert len(set(calls)) == len(calls)
    import json
    report = json.loads(Path(pipeline["numerical_audit"]).read_text())
    assert report["repeats"] == 1024 and report["sizes"] == [512, 1024, 2048]
    assert report["adjacent_tolerance"] == .01 and report["passed"]
