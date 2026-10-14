"""Tiny pipeline regressions; synthetic cells only, no real training."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung.engine.checkpoint import load_checkpoint
from lung.engine.latent_pipeline import (
    load_latent_adapter, run_latent_ae, run_latent_cache, run_latent_cfm, run_latent_members,
)
from lung.engine.representation_cache import load_representation_cache
from lung.engine.response import condition_kwargs


class TinyData:
    """In-memory reader contract with disjoint fit/selection cells."""

    n_genes = 8
    identity = "SYNTHETIC-CUDA-BF16-RUNTIME-v1"
    bandwidth = 1.0
    manifest = {"synthetic": True, "signature": identity}

    def __init__(self):
        self.mu = np.full(self.n_genes, 5.0, np.float32)
        self.scale = np.ones(self.n_genes, np.float32)
        self.neighbors = np.array([[i, (i + 1) % self.n_genes]
                                   for i in range(self.n_genes)])
        self.reader = self.rows = self
        self.controls = pd.DataFrame([
            {"condition_id": f"{role}-{i}", "control_condition_id": f"{role}-control"}
            for role in ("T", "V") for i in range(2)
        ])
        rng = np.random.default_rng(712)
        self.cells, self.expression = {}, {}
        for role in ("fit", "selection"):
            for condition in self.condition_ids(role, include_controls=True):
                ids = [f"{condition}-cell-{i}" for i in range(4)]
                self.cells[role, condition] = pd.DataFrame({"cell_id": ids})
                for cell in ids:
                    self.expression[cell] = rng.normal(0.2, 0.1, self.n_genes).astype(np.float32)

    def condition_ids(self, role, include_controls=False):
        prefix = {"T": "T", "V": "V", "fit": "T", "selection": "V"}[role]
        return [f"{prefix}-{i}" for i in range(2)] + (
            [f"{prefix}-control"] if include_controls else [])

    def role_cells(self, condition, role):
        return self.cells[role, condition]

    def read(self, rows):
        return np.stack([self.expression[cell] for cell in rows["cell_id"]])

    def normalize(self, counts):
        return counts

    def iter_response_epoch(self, role, seed):
        return iter(np.random.default_rng(seed).permutation(self.condition_ids(role)).tolist())

    def action_features(self, condition):
        fingerprint = np.zeros(2048, np.float32)
        fingerprint[int(condition[-1])] = 1
        return {"fingerprint": fingerprint, "dose": float(int(condition[-1]) + 1),
                "duration": 24.0, "background_id": 0, "source_id": 0, "protocol_id": 0}

    def sample_response(self, condition, n, rng, role):
        assert condition in self.condition_ids(role)
        cache_role = {"T": "fit", "V": "selection"}[role]
        control = self.cells[cache_role, f"{role}-control"].iloc[rng.integers(4, size=n)]
        target = self.cells[cache_role, condition].iloc[rng.integers(4, size=n)]
        return {**self.action_features(condition), "control": self.read(control),
                "target": self.read(target), "control_cell_ids": control.cell_id.to_numpy(),
                "target_cell_ids": target.cell_id.to_numpy()}


class Monitor:
    def __init__(self, root):
        self.weights = root / "weights"
        self.experiment = root / "experiment"
        self.records = []

    def say(self, message):
        pass

    def record(self, stage, epoch, metrics):
        assert all(np.isfinite(value) for value in metrics.values())
        self.records.append({"stage": stage, "epoch": epoch, "metrics": metrics})


class Budget:
    def __init__(self, stop_at=None):
        self.stop_at = stop_at
        self.exhausted = False

    def check(self, step, epoch):
        self.exhausted = self.stop_at is not None and step >= self.stop_at
        return self.exhausted


def _assert_optimizer(saved, expected_step):
    states = saved["optimizer"]["state"]
    assert states
    assert all(int(state["step"]) == expected_step for state in states.values())
    for state in states.values():
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].dtype == torch.float32
            assert torch.isfinite(state[key]).all()
    assert any(torch.count_nonzero(state["exp_avg"]) > 0 for state in states.values())


def _equal(left, right):
    if left.is_sparse:
        return (left.shape == right.shape
                and torch.equal(left.coalesce().indices(), right.coalesce().indices())
                and torch.equal(left.coalesce().values(), right.coalesce().values()))
    return torch.equal(left, right)


@pytest.mark.parametrize("stop_at", [1, 2], ids=["mid_epoch", "end_cursor"])
def test_cpu_cfm_resume_keeps_complete_epoch_aggregate(tmp_path, stop_at):
    config = {
        "device": "cpu", "seed": 712, "epochs": 2, "early_stopping": False,
        "research": {"precision": {"mode": "fp32"},
                     "latent": {"d_model": 8, "latent_dim": 4, "coupling_space": "latent",
                                "ae": {"batch": 4}, "refine": {"enabled": False}}},
        "response": {"lr": 1e-3, "min_lr": 0., "weight_decay": 0.01, "grad_clip": 1.,
                     "conditions_per_update": 1, "ot_samples": 4, "cfm_samples": 4,
                     "microbatch": 2, "mmd_samples": 4, "train_ode_steps": 2,
                     "inference_ode_steps": 2},
    }

    class CFMMonitor(Monitor):
        def record(self, stage, epoch, metrics):
            # B3 records no MMD loss until refinement is enabled.
            super().record(stage, epoch, {key: value for key, value in metrics.items()
                                         if value is not None})

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        data, upstream = TinyData(), Monitor(tmp_path / "upstream")
        ae = run_latent_ae(config, upstream, data)
        cache = run_latent_cache(config, upstream, data)
        cfm_config = {**config, "research": {**config["research"], "latent": {
            **config["research"]["latent"], "ae_checkpoint": ae["weights"],
            "cache_dir": cache["directory"]}}}
        complete = CFMMonitor(tmp_path / "complete")
        run_latent_cfm(cfm_config, complete, data)
        expected = load_checkpoint(complete.weights / "latent_cfm" / "member_1" / "last.pt")

        resumed = CFMMonitor(tmp_path / "resumed")
        run_latent_cfm(cfm_config, resumed, data, Budget(stop_at))
        path = resumed.weights / "latent_cfm" / "member_1" / "last.pt"
        partial = load_checkpoint(path)
        assert partial["epoch"] == 0
        assert partial["extra"]["resume"]["next_group_start"] == stop_at
        assert partial["extra"]["resume"]["aggregate"]["cfm_loss"] > 0
        assert not resumed.records

        run_latent_cfm(cfm_config, resumed, data)
        actual = load_checkpoint(path)
        assert len(resumed.records) == len(complete.records) == 2
        for restored, full in zip(resumed.records, complete.records):
            assert restored["epoch"] == full["epoch"]
            for name in ("train/cfm_loss", "train/loss"):
                assert restored["metrics"][name] == full["metrics"][name]
        assert actual["counters"] == expected["counters"] == {"optimizer_step": 4}
        assert actual["extra"]["early_stop"] == expected["extra"]["early_stop"]
        for name in expected["model"]:
            assert _equal(actual["model"][name], expected["model"][name])
        for key, state in expected["optimizer"]["state"].items():
            for name, value in state.items():
                restored = actual["optimizer"]["state"][key][name]
                assert _equal(restored, value) if torch.is_tensor(value) else restored == value
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_shared_cache_five_members_resume_and_generate(tmp_path):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Native CUDA BF16 support required")
    config = {
        "device": "cuda:0", "seed": 712, "epochs": 1, "early_stopping": False,
        "research": {"precision": {"mode": "bf16"},
                     "latent": {"d_model": 8, "latent_dim": 4, "coupling_space": "latent",
                                "ae": {"batch": 4}, "refine": {"enabled": False}}},
        "response": {"lr": 1e-3, "weight_decay": 0.01, "grad_clip": 1.0,
                     "members": 5, "member_workers": 1, "conditions_per_update": 1,
                     "ot_samples": 4, "microbatch": 2, "mmd_samples": 4,
                     "inference_ode_steps": 2},
    }
    data, monitor = TinyData(), Monitor(tmp_path)
    observations = {"ae": 0, "members": 0}
    stage = "ae"

    class ObserveBF16(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            # vmap can add an FP32 bias after a BF16 matmul, so the Linear
            # output dtype alone does not identify the matrix compute dtype.
            if (func._schema.name in {"aten::mm", "aten::bmm", "aten::addmm"}
                    and output.is_cuda and output.dtype == torch.bfloat16):
                observations[stage] += 1
            return output

    old_threads = torch.get_num_threads()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    observer = ObserveBF16()
    observer.__enter__()
    try:
        torch.set_num_threads(1)
        ae = run_latent_ae(config, monitor, data, Budget())
        ae_saved = load_checkpoint(ae["weights"])
        assert ae_saved["counters"]["optimizer_step"] == 2
        _assert_optimizer(ae_saved, 2)
        assert any(not _equal(value, ae_saved["shared"][name])
                   for name, value in ae_saved["model"].items())
        cache_result = run_latent_cache(config, monitor, data, Budget())
        cache = load_representation_cache(cache_result["directory"], "cuda:0",
                                          expected_data_identity=data.identity)
        assert cache.manifest["status"] == "complete"
        assert cache.summary() == {role: {"cells": 12, "conditions": 3}
                                   for role in ("fit", "selection")}
        assert set(cache.cell_index["fit"]).isdisjoint(cache.cell_index["selection"])
        assert all(h.is_cuda and torch.isfinite(h).all() for h in cache.h.values())
        assert run_latent_cache(config, monitor, data, Budget())["status"] == "reused"

        stage = "members"
        first = run_latent_members(config, monitor, data, Budget(stop_at=1))
        assert len(first["weights"]) == 5
        before = [load_checkpoint(path) for path in first["weights"]]
        for index, saved in enumerate(before):
            assert saved["metadata"]["member"] == index
            assert saved["counters"]["optimizer_step"] == 1
            assert saved["extra"]["budget_exhausted"] is True
            assert saved["extra"]["resume"]["next_group_start"] == 1
            assert len(saved["extra"]["resume"]["torch_rngs"]) == 5
            assert len(saved["extra"]["resume"]["numpy_rngs"]) == 5
            assert saved["rng"]["cuda"]
            _assert_optimizer(saved, 1)

        resumed = run_latent_members(config, monitor, data, Budget())
        assert len(set(resumed["weights"])) == 5
        after = [load_checkpoint(path) for path in resumed["weights"]]
        for index, (old, saved) in enumerate(zip(before, after)):
            assert Path(resumed["weights"][index]).name == "best.pt"
            assert saved["epoch"] == 1 and saved["counters"]["optimizer_step"] == 2
            assert saved["identity"] == old["identity"]
            _assert_optimizer(saved, 2)
            for name, value in saved["model"].items():
                assert torch.isfinite(value).all()
                if name.startswith("decoder."):
                    assert torch.equal(value, old["model"][name])
            for prefix in ("velocity_model.", "condition_encoder."):
                assert any(not torch.equal(value, old["model"][name])
                           for name, value in saved["model"].items() if name.startswith(prefix))
            for name, value in saved["shared"].items():
                assert _equal(value, ae_saved["model"][name])
            last = load_checkpoint(Path(resumed["weights"][index]).with_name("last.pt"))
            assert last["counters"]["optimizer_step"] == 2
            assert last["extra"]["resume"]["next_group_start"] == 2
            for name in saved["model"]:
                assert torch.equal(last["model"][name], saved["model"][name])
        assert len({saved["identity"] for saved in after}) == 5
        for left in range(5):
            for right in range(left):
                assert any(not torch.equal(value, after[right]["model"][name])
                           for name, value in after[left]["model"].items()
                           if name.startswith("velocity_model."))

        generated = []
        reference = torch.as_tensor(data.sample_response(
            "V-0", 3, np.random.default_rng(1), "V")["control"], device="cuda:0")
        for path in resumed["weights"]:
            adapter = load_latent_adapter(path, data, "cuda:0")
            assert all(not p.requires_grad for p in adapter.member.parameters())
            condition = adapter.encode_condition(**condition_kwargs(data.action_features("V-0"), 3, "cuda:0"))
            noise = adapter.sample_noise(reference, torch.Generator(device="cuda:0").manual_seed(912))
            sample = adapter.sample(adapter.condition_reference(reference), condition, noise=noise, steps=2)
            assert sample.shape == (3, 8) and sample.is_cuda and torch.isfinite(sample).all()
            generated.append(sample.cpu())
        torch.save(generated, tmp_path / "generated.pt")
        assert all(count > 0 for count in observations.values())
        (tmp_path / "runtime_evidence.json").write_text(json.dumps({
            "status": "passed", "synthetic_only": True, "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "bf16_matmul_calls": observations,
            "ae_optimizer_steps": 2, "members": 5, "member_optimizer_steps": 2,
            "resumed_from_step": 1, "cache_manifest": cache_result["cache"],
            "weights": resumed["weights"], "records": monitor.records,
            "generation_shape_per_member": [3, 8],
        }, indent=2), encoding="utf-8")
    finally:
        observer.__exit__(*sys.exc_info())
        torch.set_num_threads(old_threads)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
