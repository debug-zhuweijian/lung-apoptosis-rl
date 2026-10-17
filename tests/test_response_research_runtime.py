"""Tiny pipeline regressions; synthetic cells only, no real training."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung.engine.checkpoint import load_checkpoint
from lung.engine.latent_pipeline import (
    load_latent_adapter, run_latent_ae, run_latent_cache, run_latent_cfm,
)


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


def _equal(left, right):
    if left.is_sparse:
        return (left.shape == right.shape
                and torch.equal(left.coalesce().indices(), right.coalesce().indices())
                and torch.equal(left.coalesce().values(), right.coalesce().values()))
    return torch.equal(left, right)


def test_cpu_cfm_restarts_from_scratch_and_exports_loadable_weights(tmp_path):
    config = {
        "device": "cpu", "seed": 712, "epochs": 2, "early_stopping": False,
        "research": {"precision": {"mode": "fp32"},
                     "latent": {"d_model": 8, "latent_dim": 4, "coupling_space": "latent",
                                "ae": {"batch": 4}, "refine": {"enabled": False}}},
        "response": {"lr": 1e-3, "weight_decay": 0.01, "grad_clip": 1.,
                     "conditions_per_update": 1, "ot_samples": 4,
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
        expected_ae = load_checkpoint(ae["weights"])
        Path(ae["weights"]).with_name("last.pt").write_bytes(b"unused legacy snapshot")
        fresh_ae = run_latent_ae(config, upstream, data)
        actual_ae = load_checkpoint(fresh_ae["weights"])
        assert "optimizer" not in actual_ae and "rng" not in actual_ae
        for name in expected_ae["model"]:
            assert _equal(actual_ae["model"][name], expected_ae["model"][name])
        cache = run_latent_cache(config, upstream, data)
        cfm_config = {**config, "research": {**config["research"], "latent": {
            **config["research"]["latent"], "ae_checkpoint": ae["weights"],
            "cache_dir": cache["directory"]}}}
        complete = CFMMonitor(tmp_path / "complete")
        run_latent_cfm(cfm_config, complete, data)
        expected = load_checkpoint(complete.weights / "latent_cfm" / "member_1" / "final.pt")

        restarted = CFMMonitor(tmp_path / "complete")
        path = restarted.weights / "latent_cfm" / "member_1" / "final.pt"
        # A stale training snapshot must never affect a fresh invocation.
        path.with_name("last.pt").write_bytes(b"unused legacy snapshot")
        run_latent_cfm(cfm_config, restarted, data)
        actual = load_checkpoint(path)
        assert len(restarted.records) == len(complete.records) == 2
        for restored, full in zip(restarted.records, complete.records):
            assert restored["epoch"] == full["epoch"]
            for name in ("train/cfm_loss", "train/loss"):
                assert restored["metrics"][name] == full["metrics"][name]
        assert actual["counters"] == expected["counters"] == {"optimizer_step": 4}
        for name in expected["model"]:
            assert _equal(actual["model"][name], expected["model"][name])
        assert "optimizer" not in actual and "rng" not in actual
        adapter = load_latent_adapter(path.with_name("best.pt"), data, "cpu")
        assert adapter.member.config.n_genes == data.n_genes
    finally:
        torch.set_num_threads(previous_threads)
