"""Small disk fixtures exercise real TimeData I/O without real training data."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine.config import file_hash
from lung.timecourse.data import TimeData
from lung.timecourse.protocol import GROUPS, VEHICLE
from tests.test_timecourse_data_response import fixture_data


def disk_fixture(root, *, use_disk_cache=True):
    root = Path(root)
    source, assets = root / "source", root / "assets"
    source.mkdir()
    assets.mkdir()
    fixture = fixture_data()
    replacements = {"d": GROUPS["D"][0], "c": GROUPS["C"][0], "e": GROUPS["E"][0], VEHICLE: VEHICLE}
    cells = fixture._cells.copy()
    cells["catalog_number"] = cells.catalog_number.map(replacements)
    cells["shard_id"] = np.arange(len(cells)) % 4
    cells["row_in_shard"] = cells.groupby("shard_id", sort=False).cumcount()
    shards = {}
    for shard in range(4):
        matrix = fixture._array[np.flatnonzero(cells.shard_id.eq(shard))]
        path = source / f"shard{shard}.npz"
        sparse.save_npz(path, matrix)
        shards[str(shard)] = {"file": path.name, "sha256": file_hash(path), "rows": matrix.shape[0]}
    cells.to_parquet(assets / "cells.parquet", index=False)
    pd.DataFrame({"ensembl_id": fixture.genes}).to_parquet(assets / "genes.parquet", index=False)
    (assets / "fingerprints.json").write_text(json.dumps(
        {replacements[name]: value for name, value in fixture._fingerprints.items()}))
    np.savez(assets / "preprocessing.npz", mu=fixture.mu, scale=fixture.scale,
             neighbors=fixture.neighbors, bandwidth=fixture.bandwidth)
    manifest = {"status": "complete", "identity": fixture.identity,
                "coordinate_identity": fixture.coordinate_identity, "source": str(source), "shards": shards,
                "files": {p.name: file_hash(p) for p in assets.iterdir()}}
    (assets / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"timecourse": {"source": "source", "assets": "assets"},
              "data_loading": {"row_cache": {"use_disk_cache": use_disk_cache}}}
    return config, manifest


def reader(root, config):
    return TimeData(root, config, [GROUPS["D"][0]], [GROUPS["D"][0]], "fit")


def test_default_disk_reads_match_npz_and_reuse_across_views(tmp_path):
    config, manifest = disk_fixture(tmp_path)
    cached = reader(tmp_path, config)
    uncached = reader(tmp_path, {**config, "data_loading": {"row_cache": {"use_disk_cache": False}}})
    assert cached._sparse_rows.max_cache_bytes == 0
    assert cached._sparse_rows.cache_bytes == 0
    original = {name: file_hash(tmp_path / "source" / item["file"]) for name, item in manifest["shards"].items()}
    conditions = cached.condition_ids("T")
    with patch("scipy.sparse.load_npz", wraps=sparse.load_npz) as load:
        first = [cached.sample_response(cid, 7, np.random.default_rng(99), "T") for cid in conditions]
        assert load.call_count == 4
    with patch("scipy.sparse.load_npz", side_effect=AssertionError("warm disk read must not decompress")):
        again = [cached.sample_response(cid, 7, np.random.default_rng(99), "T") for cid in conditions]
        reopened = reader(tmp_path, config)
        reopened.sample_response(conditions[0], 7, np.random.default_rng(99), "T")
    for cid, expected, actual in zip(conditions, first, again):
        baseline = uncached.sample_response(cid, 7, np.random.default_rng(99), "T")
        for key in ("target", "control"):
            np.testing.assert_array_equal(expected[key], actual[key])
            np.testing.assert_array_equal(expected[key], baseline[key])
    evaluation = cached.view([], [GROUPS["E"][0]], "final_test")
    assert evaluation._sparse_rows is cached._sparse_rows
    cid = evaluation.condition_ids()[0]
    with pytest.raises(PermissionError, match="TARGET72_ACCESS_DENIED"):
        cached.sample_target(cid)
    with pytest.raises(PermissionError, match="TARGET72_ACCESS_DENIED"):
        evaluation.sample_target(cid, role="T")
    assert evaluation.sample_target(cid).shape == (3, 4)
    assert original == {name: file_hash(tmp_path / "source" / item["file"]) for name, item in manifest["shards"].items()}


def test_known24_and_holdout_are_unchanged(tmp_path):
    config, _ = disk_fixture(tmp_path)
    cached = reader(tmp_path, config)
    baseline = reader(tmp_path, {**config, "data_loading": {"row_cache": {"use_disk_cache": False}}})
    left = cached.training_view(validation_fraction=.34)
    right = baseline.training_view(validation_fraction=.34)
    for role in ("T", "V"):
        assert left.condition_ids(role) == right.condition_ids(role)
        for cid in left.condition_ids(role):
            np.testing.assert_array_equal(left.sample_target(cid, role=role), right.sample_target(cid, role=role))
    known = cached.training_view(known24_only=True)
    assert (known.conditions.loc[known.condition_ids("T"), "time_point"] == 24).all()
    with pytest.raises(PermissionError, match="PRETRAIN24_ONLY"):
        known.sample_target(cached.condition_ids(time=72)[0], role="T")


@pytest.mark.parametrize("kind", ["hash", "rows", "columns", "mutation"])
def test_real_source_contract_fails_closed(tmp_path, kind):
    config, manifest = disk_fixture(tmp_path)
    path = tmp_path / "source" / "shard0.npz"
    if kind == "hash":
        manifest["shards"]["0"]["sha256"] = "0" * 64
    elif kind == "rows":
        manifest["shards"]["0"]["rows"] += 1
    elif kind == "columns":
        original = sparse.load_npz(path)
        sparse.save_npz(path, sparse.hstack([original, original[:, :1]]).tocsr())
        manifest["shards"]["0"]["sha256"] = file_hash(path)
    (tmp_path / "assets" / "manifest.json").write_text(json.dumps(manifest))
    data = reader(tmp_path, config)
    if kind == "mutation":
        with path.open("ab") as stream:
            stream.write(b"changed after reader construction")
    cid = next(cid for cid, rows in data._rows.items() if rows.shard_id.eq(0).any())
    with pytest.raises(ValueError, match="hash differs|dimensions differ|changed since"):
        data._read(cid, None, np.random.default_rng(0))


def test_escape_is_rejected_and_disabled_cache_keeps_sha_path(tmp_path):
    config, manifest = disk_fixture(tmp_path, use_disk_cache=False)
    data = reader(tmp_path, config)
    assert data._sparse_rows is None
    with patch("lung.timecourse.data.file_hash", wraps=file_hash) as hasher:
        data.sample_target(data.condition_ids()[0])
        assert hasher.call_count > 0
    assert not (tmp_path / "08-datasets" / ".csr-cache-v1").exists()
    manifest["shards"]["0"]["file"] = "../../outside.npz"
    (tmp_path / "assets" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="remain inside"):
        reader(tmp_path, {**config, "data_loading": {"row_cache": {"use_disk_cache": True}}})
