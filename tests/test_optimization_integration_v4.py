"""Integration checks for optional comparisons and convergence-aware artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung.engine.config import ROOT, load_config
from lung.engine.inference_noise import response_noise
from lung.engine.optimization import candidate_names, candidate_overrides
from lung.engine.suite_plan import task_graph, RESUMABLE
from lung.engine.suite_training import time_config, pipeline_result
from lung.engine.synthetic import smoke_config, SyntheticData


def test_default_graph_stays_19_and_candidates_are_explicit():
    config = load_config()
    assert candidate_names(config) == []
    assert len(task_graph(config)) == 19
    config["suite"]["response_candidates"] = ["cfm_then_full", "one_step_then_full", "latent_flow"]
    graph = task_graph(config)
    assert len(graph) == 23
    for name in config["suite"]["response_candidates"]:
        key = "response_" + name
        assert graph[key] == ["prepare"]
        assert key in graph["optimization_compare"]
        assert key in graph["response_final_test"]
        assert key in RESUMABLE
        assert key not in graph["tahoe_auxiliary"]


@pytest.mark.parametrize("names", [["unknown"], ["latent_flow", "latent_flow"], "latent_flow"])
def test_candidate_names_fail_closed(names):
    with pytest.raises(ValueError):
        candidate_names({"suite": {"response_candidates": names}})


def test_comparison_overrides_do_not_reduce_members_genes_or_sampling():
    config = load_config()
    before = deepcopy(config)
    for name in ("cfm_then_full", "one_step_then_full", "latent_flow"):
        result = candidate_overrides(name, config)["response"]
        assert not {"members", "mmd_samples", "cfm_samples", "train_ode_steps"}.intersection(result)
        assert "n_genes" not in result["model"]
        assert result["mmd_weight"] == 1.
    assert config == before


def test_noise_is_drawn_in_declared_space_and_legacy_rng_is_unchanged():
    reference = torch.zeros(3, 8)
    seed = 81
    actual = response_noise(object(), reference, torch.Generator().manual_seed(seed))
    assert torch.equal(actual, torch.randn(reference.shape, generator=torch.Generator().manual_seed(seed)))
    class Latent:
        def sample_noise(self, ref, generator=None):
            return torch.randn((len(ref), 2), generator=generator, device=ref.device, dtype=ref.dtype)
    assert response_noise(Latent(), reference, torch.Generator().manual_seed(seed)).shape == (3, 2)


def test_time_epochs_use_explicit_stage_setting_or_global_fallback(tmp_path):
    config = smoke_config(load_config())
    config["epochs"] = 7
    context = {"root": ROOT, "config": config, "suite_dir": tmp_path, "job_dir": tmp_path / "jobs" / "nested"}
    assert time_config(context)["timecourse"]["epochs"] == 7
    config["timecourse"] = {"epochs": 4}
    assert time_config(context)["timecourse"]["epochs"] == 4


def test_completion_survives_pipeline_result(tmp_path):
    path = tmp_path / "pipeline.json"
    completion = {"response": [{"status": "complete", "reason": "validation_plateau",
                               "actual_epochs": 17, "target_epochs": 1000}]}
    path.write_text(json.dumps({"artifacts": {"_completion": completion}, "config": {},
                                "data_identity": "fixture", "synthetic": True}), encoding="utf-8")
    assert pipeline_result(path)["training_completion"] == completion


def test_optional_comparison_never_opens_final_test_role(tmp_path, monkeypatch):
    from lung.engine import data as data_module, evaluate, optimization, response
    config = smoke_config(load_config())
    observed = []
    monkeypatch.setattr(data_module, "open_data", lambda cfg, stage, prepare=False: SyntheticData(cfg))
    monkeypatch.setattr(response, "load_response", lambda *args: object())
    def metric(models, data, cfg, output, *, role):
        observed.append(role)
        return {"role": role, "model_updates": 0}
    monkeypatch.setattr(evaluate, "evaluate_expression", metric)
    weights = tmp_path / "fixture.pt"
    weights.write_bytes(b"immutable")
    context = {"config": config, "job_dir": tmp_path, "synthetic": True,
               "dependencies": {"response_latent_flow": {"original_config": config,
                    "pipeline_artifacts": {"response": [str(weights)]}}}}
    result = optimization.compare_responses(context)
    report = json.loads(Path(result["report"]).read_text())
    assert observed == ["V"]
    assert report["automatic_promotion"] is False
    assert weights.read_bytes() == b"immutable"


def test_auxiliary_stops_on_validation_and_saves_honest_completion(tmp_path):
    from types import SimpleNamespace
    from lung.engine.auxiliary import fit_epochs
    from lung.engine.checkpoint import load_checkpoint
    from lung.engine.training_control import checkpoint_completion
    torch.set_num_threads(1)
    model = torch.nn.Linear(1, 1)
    config = {"epochs": 10, "aux_lr": 0.01, "aux_batch": 1,
              "early_stopping": {"enabled": True, "patience": 2}, "save_period": 10}
    monitor = SimpleNamespace(weights=tmp_path, progress=lambda it, **kwargs: it, record=lambda *a, **kw: None)
    loss = lambda net, value: net(torch.tensor([[value]])).square().mean()
    selected = fit_epochs("pool", model, [1.], loss, SimpleNamespace(identity="data"), config,
                          monitor, "identity", {}, validation=lambda net: {"val/loss": 1.})
    saved = load_checkpoint(Path(selected).parent / "last.pt")
    completion = checkpoint_completion(saved, 10)
    assert completion["actual_epochs"] == 3
    assert completion["reason"] == "validation_plateau"
    assert load_checkpoint(selected)["epoch"] == 1


def test_auxiliary_without_selection_does_not_stop_using_train_loss(tmp_path):
    from types import SimpleNamespace
    from lung.engine.auxiliary import fit_epochs
    from lung.engine.checkpoint import load_checkpoint
    model = torch.nn.Linear(1, 1)
    config = {"epochs": 4, "early_stopping": {"enabled": True, "patience": 1}}
    monitor = SimpleNamespace(weights=tmp_path, progress=lambda it, **kwargs: it, record=lambda *a, **kw: None)
    selected = fit_epochs("backward", model, [1.], lambda net, v: net(torch.tensor([[v]])).square().mean(),
                          SimpleNamespace(identity="data"), config, monitor, "identity", {})
    saved = load_checkpoint(selected)
    assert saved["epoch"] == 4
    assert saved["extra"]["selection_available"] is False
