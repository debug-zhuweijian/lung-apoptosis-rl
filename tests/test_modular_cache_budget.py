"""CPU cache transaction tests; tiny data do not certify biological resolution."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung.engine.representation_cache import (
    _manifest_matches, build_representation_cache, encoder_identity,
    ensure_representation_cache, load_representation_cache,
)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_genes=3, d_model=2, latent_dim=1)
        self.projection = torch.nn.Linear(3, 2)
        self.representation_hash = "a" * 64
        self.encoded_cells = 0

    def encode_h(self, expression):
        self.encoded_cells += len(expression)
        return None, self.projection(expression)


class TinyReader:
    def __init__(self):
        self.rows = self
        self.read_ids = []
        self.frames = {}
        for role in ("fit", "selection"):
            for condition in self.condition_ids(role, include_controls=True):
                self.frames[role, condition] = pd.DataFrame(
                    {"cell_id": [f"{condition}-cell-{index}" for index in range(5)]})

    def condition_ids(self, role, include_controls=False):
        return [f"{role}-treated"] + ([f"{role}-control"] if include_controls else [])

    def role_cells(self, condition, role):
        return self.frames[role, condition].copy()

    def read(self, rows):
        self.read_ids.extend(rows.cell_id)
        return np.asarray([[len(cell), int(cell[-1]), 1] for cell in rows.cell_id], dtype=np.float32)

    def normalize(self, counts):
        return counts / 10


@pytest.fixture
def assets():
    torch.manual_seed(312)
    data = SimpleNamespace(identity="tiny-real-reader-contract", manifest={"signature": "normalization-A"},
                           reader=TinyReader())
    return data, TinyEncoder()


def test_cache_build_and_complete_reuse_preserve_cells_and_roles(tmp_path, assets):
    data, shared = assets
    requested = tmp_path / "cache"
    callbacks = []
    directory, status = ensure_representation_cache(
        data, shared, requested, "cpu", batch_size=2,
        progress=lambda *values: callbacks.append(values))
    assert status == "built" and shared.encoded_cells == 20
    assert set(Counter(data.reader.read_ids).values()) == {1}
    assert not (directory / "build_state.json").exists()
    assert callbacks == [(role, index + 1, 2, condition)
                         for role in ("fit", "selection")
                         for index, condition in enumerate(data.reader.condition_ids(role, True))]
    cached = load_representation_cache(directory, "cpu", expected_data_identity=data.identity,
                                      expected_encoder=encoder_identity(shared, shared.config))
    assert cached.summary() == {role: {"cells": 10, "conditions": 2} for role in ("fit", "selection")}
    assert set(cached.cell_index["fit"]).isdisjoint(cached.cell_index["selection"])
    reference_data = SimpleNamespace(identity=data.identity, manifest=data.manifest, reader=TinyReader())
    uninterrupted = tmp_path / "uninterrupted"
    build_representation_cache(reference_data, deepcopy(shared), uninterrupted, "cpu", batch_size=2)
    reference = load_representation_cache(uninterrupted, "cpu")
    for role in ("fit", "selection"):
        torch.testing.assert_close(cached.h[role], reference.h[role], rtol=0, atol=0)
        assert cached.cell_index[role] == reference.cell_index[role]
    assert ensure_representation_cache(data, shared, requested, "cpu", batch_size=2) == (directory, "reused")
    assert shared.encoded_cells == 20


def test_changed_ae_file_with_identical_tensors_has_separate_cache(tmp_path, assets):
    data, shared = assets
    requested = tmp_path / "cache"
    first, status = ensure_representation_cache(data, shared, requested, "cpu", batch_size=2)
    assert status == "built"
    original_manifest = (first / "manifest.json").read_bytes()
    old_identity = encoder_identity(shared, shared.config)
    shared.representation_hash = "b" * 64
    new_identity = encoder_identity(shared, shared.config)
    assert old_identity["weights"] == new_identity["weights"]
    assert old_identity["checkpoint_sha256"] != new_identity["checkpoint_sha256"]
    second, status = ensure_representation_cache(data, shared, requested, "cpu", batch_size=2)
    assert status == "built" and second != first
    assert (first / "manifest.json").read_bytes() == original_manifest
    assert not _manifest_matches(first, data.identity, new_identity)
    with pytest.raises(ValueError, match="CACHE_ENCODER_IDENTITY_MISMATCH"):
        load_representation_cache(first, "cpu", expected_encoder=new_identity)


@pytest.mark.parametrize("field", ["n_genes", "d_model", "latent_dim", "checkpoint_sha256"])
def test_loader_checks_entire_encoder_identity(tmp_path, assets, field):
    data, shared = assets
    manifest = build_representation_cache(data, shared, tmp_path, "cpu")
    wrong = {**manifest["encoder"], field: "wrong"}
    with pytest.raises(ValueError, match="CACHE_ENCODER_IDENTITY_MISMATCH"):
        load_representation_cache(tmp_path, "cpu", expected_encoder=wrong)


@pytest.mark.parametrize("roles", [("calibration",), ("final_test",), ("C",), ("E",),
                                    ("fit", "selection", "calibration"), ("fit", "fit"), ()])
def test_forbidden_roles_never_read_or_publish(tmp_path, assets, roles):
    data, shared = assets
    with pytest.raises(ValueError, match="CACHE_ROLE_FORBIDDEN"):
        ensure_representation_cache(data, shared, tmp_path, "cpu", roles=roles)
    assert not data.reader.read_ids and not list(tmp_path.iterdir())


def test_loader_rejects_forbidden_role_in_manifest(tmp_path, assets):
    data, shared = assets
    manifest = build_representation_cache(data, shared, tmp_path, "cpu", roles=("fit",))
    malformed = {**manifest, "roles": {"calibration": manifest["roles"]["fit"]}}
    (tmp_path / "manifest.json").write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="CACHE_ROLE_FORBIDDEN"):
        load_representation_cache(tmp_path, "cpu")


def test_direct_build_preserves_completed_cache_for_changed_identity(tmp_path, assets):
    data, shared = assets
    build_representation_cache(data, shared, tmp_path, "cpu")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    shared.representation_hash = "b" * 64
    with pytest.raises(ValueError, match="CACHE_TARGET_IDENTITY_MISMATCH"):
        build_representation_cache(data, shared, tmp_path, "cpu")
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_encoder_without_checkpoint_attribute_preserves_legacy_call_contract(tmp_path, assets):
    data, shared = assets
    del shared.representation_hash
    identity = encoder_identity(shared, shared.config)
    assert "checkpoint_sha256" not in identity
    build_representation_cache(data, shared, tmp_path, "cpu", roles=("fit",))
    loaded = load_representation_cache(tmp_path, "cpu", expected_encoder=identity)
    assert loaded.h["fit"].shape == (10, 2)
