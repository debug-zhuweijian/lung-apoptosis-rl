"""Explicitly labelled small data for engineering tests, never a biological result."""

from pathlib import Path

import numpy as np

from lung.engine.config import merge


def smoke_config(config):
    return merge(config, {
        "synthetic": True, "epochs": 1, "early_stopping": False,
        "device": "cpu",
        "training_control": {"validation_every_epochs": 1, "validation_every_updates": 0},
        "population": 4, "scoring_repeats": 2,
        "state_batch": 6, "aux_batch": 2, "calibration_steps": 3,
        "potential_train_prefixes": 4, "potential_val_prefixes": 2,
        "potential_continuations": 2, "potential_batch": 2,
        "policy_seeds": [0], "decision_validation_repeats": 2,
        "response": {"members": 2, "conditions_per_update": 2, "ot_samples": 4,
                     "condition_microbatch": 1,
                     "cfm_samples": 4, "mmd_samples": 3, "microbatch": 1,
                     "train_ode_steps": 2, "inference_ode_steps": 2,
                     "model": {"n_genes": 8, "d_model": 8, "n_heads": 2,
                               "encoder_layers": 1, "pad_layers": 1, "query_chunk": 4,
                               "context_dim": 2}},
        "daac": {"episodes_per_epoch": 6, "warmup_transitions": 2, "batch_size": 2,
                 "replay_capacity": 32, "validation_roots": 2},
        "dlpa": {"roots_per_epoch": 2, "distill_updates": 2, "batch_size": 2,
                 "iterations": 2, "candidates": 4, "elites": 2, "validation_roots": 2}})


class SyntheticData:
    synthetic = True

    def __init__(self, config):
        self.config = config
        self.n_genes = config["response"]["model"]["n_genes"]
        # A positive log-expression offset makes the eight-gene fixture physically
        # nondegenerate; invalid generated mass is still a hard error in production.
        self.mu = np.full(self.n_genes, 5., np.float32)
        self.scale = np.ones(self.n_genes, np.float32)
        self.neighbors = np.stack([np.array([i, (i + 1) % self.n_genes, (i + 2) % self.n_genes])
                                   for i in range(self.n_genes)])
        self.bandwidth = 1.
        self.identity = "SYNTHETIC-v2-G" + str(self.n_genes)
        self.manifest = {"synthetic": True, "identity": self.identity}
        self.assets_dir = Path("SYNTHETIC-NO-REAL-DATA")

    def condition_ids(self, role):
        if role not in "TVRGCE":
            raise ValueError("Invalid role")
        return [f"{role}-{i}" for i in range(5 if role == "R" else 2)]

    def sample_response(self, condition_id, n, rng, role):
        if condition_id not in self.condition_ids(role):
            raise ValueError("ROLE_MISMATCH")
        fingerprint = np.zeros(2048, np.float32)
        fingerprint[int(condition_id[-1])] = 1
        return {"control": rng.normal(0, .2, (n, self.n_genes)).astype(np.float32),
                "target": rng.normal(.2, .2, (n, self.n_genes)).astype(np.float32),
                "fingerprint": fingerprint, "dose": float(int(condition_id[-1]) + 1),
                "duration": 24., "source_id": 0, "background_id": 0, "protocol_id": 0,
                "condition_id": condition_id, "group_id": condition_id,
                "condition": {"dose_value": float(int(condition_id[-1]) + 1), "treatment_hours": 24}}

    def iter_response_epoch(self, role, seed):
        rng = np.random.default_rng(seed)
        keys = self.condition_ids(role)
        return iter(rng.choice(keys, size=len(keys), replace=True).tolist())

    def action_features(self, condition_id):
        batch = self.sample_response(condition_id, 1, np.random.default_rng(0), condition_id[0])
        return {key: value for key, value in batch.items() if key not in {"control", "target"}}

    def sample_controls(self, condition_id, n, rng, role):
        batch = self.sample_response(condition_id, n, rng, role)
        return {key: value for key, value in batch.items() if key != "target"}

    def state_arrays(self, fold, role):
        rng = np.random.default_rng({"fit": 1, "selection": 2, "calibration": 3, "test": 4}[role])
        y = np.tile(np.arange(3), 4)
        x = rng.normal(0, .2, (12, self.n_genes)).astype(np.float32) + y[:, None] * .2
        return {"x": x, "labels": y, "cell_ids": [f"{role}-{i}" for i in range(12)]}
