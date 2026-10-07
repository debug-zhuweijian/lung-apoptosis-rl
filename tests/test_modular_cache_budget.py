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

from lung.engine.modular_budget import StagePaused
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


def test_budget_pause_commits_partial_chunk_and_resume_never_encodes_it_twice(tmp_path, assets):
    data, shared = assets
    requested = tmp_path / "cache"
    callbacks = []
    with pytest.raises(StagePaused) as stopped:
        ensure_representation_cache(data, shared, requested, "cpu", batch_size=2,
                                    stop_check=lambda step, epoch: step >= 1)
    pause = stopped.value
    state_path = Path(pause.checkpoint_path)
    directory = state_path.parent
    assert pause.stage == "latent_cache" and pause.step == pause.cursor == 1 and pause.epoch == 0
    assert shared.encoded_cells == 2 and len(data.reader.read_ids) == 2
    assert not (directory / "manifest.json").exists()
    with pytest.raises(ValueError, match="CACHE_MANIFEST_MISSING"):
        load_representation_cache(directory, "cpu")
    first_shard = directory / json.loads(state_path.read_text())["roles"]["fit"][0]["file"]
    first_bytes = first_shard.read_bytes()
    first_mtime = first_shard.stat().st_mtime_ns

    resumed, status = ensure_representation_cache(data, shared, requested, "cpu", batch_size=2,
                                                 progress=lambda *values: callbacks.append(values))
    assert resumed == directory and status == "built"
    assert shared.encoded_cells == 20
    assert set(Counter(data.reader.read_ids).values()) == {1}
    assert first_shard.read_bytes() == first_bytes and first_shard.stat().st_mtime_ns == first_mtime
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


def test_exhausted_before_first_chunk_creates_only_resumable_registration(tmp_path, assets):
    data, shared = assets
    with pytest.raises(StagePaused) as stopped:
        build_representation_cache(data, shared, tmp_path, "cpu", stop_check=lambda step, epoch: True)
    assert stopped.value.step == 0 and not list(tmp_path.glob("*.npz"))
    assert not data.reader.read_ids and shared.encoded_cells == 0
    build_representation_cache(data, shared, tmp_path, "cpu")
    assert (tmp_path / "manifest.json").is_file() and shared.encoded_cells == 20


def test_last_chunk_pause_finalizes_without_reencoding_on_resume(tmp_path, assets):
    data, shared = assets
    with pytest.raises(StagePaused):
        build_representation_cache(data, shared, tmp_path, "cpu", batch_size=5,
                                   stop_check=lambda step, epoch: step >= 4)
    assert shared.encoded_cells == 20 and not (tmp_path / "manifest.json").exists()
    build_representation_cache(data, shared, tmp_path, "cpu", batch_size=5)
    assert shared.encoded_cells == 20 and (tmp_path / "manifest.json").is_file()


def test_resume_accepts_actual_partial_directory_without_adding_version_suffix(tmp_path, assets):
    data, shared = assets
    with pytest.raises(StagePaused) as stopped:
        ensure_representation_cache(data, shared, tmp_path / "cache", "cpu", batch_size=2,
                                    stop_check=lambda step, epoch: step >= 1)
    actual = Path(stopped.value.checkpoint_path).parent
    directory, status = ensure_representation_cache(data, shared, actual, "cpu", batch_size=2)
    assert directory == actual and status == "built"
    assert shared.encoded_cells == 20 and set(Counter(data.reader.read_ids).values()) == {1}
    assert list(tmp_path.iterdir()) == [actual]


def test_corrupt_committed_chunk_rejects_resume_before_new_encoding(tmp_path, assets):
    data, shared = assets
    with pytest.raises(StagePaused):
        build_representation_cache(data, shared, tmp_path, "cpu", batch_size=2,
                                   stop_check=lambda step, epoch: step >= 1)
    shard = next(tmp_path.glob("*.npz"))
    shard.write_bytes(shard.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="CACHE_SHARD_MISMATCH"):
        build_representation_cache(data, shared, tmp_path, "cpu", batch_size=2)
    assert shared.encoded_cells == 2 and not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("change", ["data", "normalization", "batch", "roles", "rows", "checkpoint"])
def test_resume_rejects_changed_registration_or_committed_rows(tmp_path, assets, change):
    data, shared = assets
    with pytest.raises(StagePaused):
        build_representation_cache(data, shared, tmp_path, "cpu", batch_size=2,
                                   stop_check=lambda step, epoch: step >= 1)
    kwargs: dict = {"batch_size": 2}
    if change == "data":
        data.identity = "changed-data"
    elif change == "normalization":
        data.manifest = {"signature": "changed-normalization"}
    elif change == "batch":
        kwargs["batch_size"] = 3
    elif change == "roles":
        kwargs["roles"] = ("fit",)
    elif change == "checkpoint":
        shared.representation_hash = "b" * 64
    else:
        key = ("fit", "fit-treated")
        data.reader.frames[key] = data.reader.frames[key].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="CACHE_RESUME_(IDENTITY|ROWS)_MISMATCH"):
        build_representation_cache(data, shared, tmp_path, "cpu", **kwargs)
    assert shared.encoded_cells == 2 and not (tmp_path / "manifest.json").exists()


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
