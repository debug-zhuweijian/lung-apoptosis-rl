"""Bounded CUDA execution proof on synthetic counts, never real training data."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lung.timecourse.data import TimeData
from lung.timecourse.response import train_response, predict_response
from lung.timecourse.decision_world import TimeDecisionWorld
from lung.timecourse.decision_audit import audit_population
from lung.timecourse.decisions import _train
from test_timecourse_data_response import tiny_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_response_and_both_policies(tmp_path):
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(20260926)
    rows, counts = [], []
    for time in (24, 72):
        for drug in ("S0000", "d"):
            for _ in range(128):
                rows.append(dict(catalog_number=drug, time_point=time,
                    dose_uM=0 if drug == "S0000" else 1, replicate="rep1",
                    culture_plate=f"plate{time}", author_hash_qc=True))
                counts.append(rng.poisson(2., 64))
    data = TimeData.from_arrays(counts, rows, [f"g{i}" for i in range(64)],
        {d: np.zeros(2048) for d in ("S0000", "d")}, fit_drugs=["d"],
        allowed_target_drugs=["d"], synthetic=True,
        drug_groups={"D": ["d"], "C": [], "E": []})
    config = tiny_config()
    config.update(epochs=1, device="cuda:0")
    config["response"].update(ot_samples=16, cfm_samples=4, mmd_samples=4)
    config["response"]["model"].update(n_genes=64, query_chunk=64)
    result = train_response(data, config, tmp_path / "response")
    saved = torch.load(result["checkpoints"][0], weights_only=False)
    assert saved["epoch"] == 1
    assert any(value.is_cuda for value in saved["model"].values())

    class Predictor:
        def __init__(self):
            self.data = data

        def predict(self, cid, sample_count, seed, reference=None):
            return {"projected": predict_response(result["checkpoints"], data,
                cid, sample_count, seed, "cuda:0", reference=reference, as_tensor=True)}

    class Proxy:
        identity = "synthetic_gpu_execution_proxy"
        device = torch.device("cuda:0")

        def __call__(self, samples):
            assert isinstance(samples, torch.Tensor) and samples.is_cuda
            return float(torch.sigmoid(samples.mean().clamp(-30, 30)))

    config.update(population=2, projection_width=4, scoring_repeats=1,
        population_audit={"repeats": 2, "tolerance": .5},
        daac={"episodes_per_epoch": 4, "warmup_transitions": 1, "batch_size": 2,
              "actor_delay": 1, "replay_capacity": 16},
        dlpa={"roots_per_epoch": 1, "iterations": 1, "candidates": 4,
              "elites": 2, "distill_updates": 1, "batch_size": 2})
    world = TimeDecisionWorld(Predictor(), Proxy(), data.condition_ids(drugs=["d"]),
        ("rep1", "plate72"), "T", "synthetic_cuda_probe", config)
    audit_population(world, config, tmp_path / "population_audit")
    for stage in ("daac", "dlpa"):
        actor, _, counters = _train(world, config, stage, 3, tmp_path / stage)
        assert next(actor.parameters()).is_cuda
        assert counters["updates"] > 0
        assert counters["parameter_delta_this_invocation"] > 0
    assert torch.cuda.max_memory_allocated() > 0
