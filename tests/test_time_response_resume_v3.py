"""CPU update-boundary recovery checks on actual synthetic response training."""
import csv
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine.checkpoint import atomic_save, load_checkpoint
from lung.timecourse import response
from test_timecourse_data_response import fixture_data, tiny_config


def assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert left.layout == right.layout and left.shape == right.shape
        if left.is_sparse:
            assert torch.equal(left.coalesce().indices(), right.coalesce().indices())
            assert torch.equal(left.coalesce().values(), right.coalesce().values())
        else:
            assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right


def config():
    cfg = tiny_config()
    cfg["response"]["conditions_per_update"] = 2
    return cfg


@pytest.mark.parametrize("interruption", ["after_update", "last_update", "epoch_commit", "partial_group", "telemetry"])
@pytest.mark.parametrize("backbone", ["gene_token", "compact_full_gene"])
def test_exact_update_recovery(tmp_path, interruption, backbone):
    torch.set_num_threads(1)
    data, cfg = fixture_data(), config()
    cfg["response"]["model"]["backbone"] = backbone
    keys = data.iter_response_epoch("T", cfg["response"].get("seed", 20260923))
    assert len(keys) % 2 == 1  # Exercise the actual final short optimizer group.
    updates = (len(keys) + 1) // 2
    seen = []
    actual_condition, actual_save = response.train_condition, response.save_training

    def track(*args, **kwargs):
        # Also exercise global Python/NumPy state; actual CFM uses global torch.
        random.random()
        np.random.random()
        seen.append(args[2])
        return actual_condition(*args, **kwargs)

    with patch.object(response, "train_condition", side_effect=track):
        direct = response.train_response(data, cfg, tmp_path / "direct")
    direct_seen = seen.copy()
    seen.clear()

    def stop_save(*args, **kwargs):
        if interruption == "epoch_commit" and kwargs["epoch"] == 1:
            raise KeyboardInterrupt("before epoch checkpoint commit")
        actual_save(*args, **kwargs)
        cursor = (kwargs.get("extra") or {}).get("update_resume")
        if cursor and ((interruption == "after_update" and kwargs["counters"]["optimizer_step"] == 1)
                       or (interruption == "last_update" and cursor["next_condition_index"] == len(keys))):
            raise KeyboardInterrupt("after complete update checkpoint")

    def partial(*args, **kwargs):
        result = track(*args, **kwargs)
        if interruption == "partial_group" and len(seen) == 3:
            raise KeyboardInterrupt("after partial group backward")
        return result

    actual_replace = response.os.replace

    def stop_metrics(src, dst):
        if interruption == "telemetry" and str(dst).endswith("metrics.csv"):
            raise KeyboardInterrupt("before metrics atomic replacement")
        return actual_replace(src, dst)

    with patch.object(response, "train_condition", side_effect=partial), \
         patch.object(response, "save_training", side_effect=stop_save), \
         patch.object(response.os, "replace", side_effect=stop_metrics):
        with pytest.raises(KeyboardInterrupt):
            response.train_response(data, cfg, tmp_path / "resumed")
    checkpoint = tmp_path / "resumed/member_1/last.pt"
    interrupted = load_checkpoint(checkpoint)
    cursor = interrupted["extra"]["update_resume"]
    expected_cursor = 2 if interruption in ("after_update", "partial_group") else len(keys)
    assert interrupted["epoch"] == 0
    assert cursor["next_condition_index"] == cursor["loss_count"] == expected_cursor
    assert interrupted["counters"]["optimizer_step"] == (expected_cursor + 1) // 2
    for stream in ("numpy_rng", "mmd_numpy_rng", "mmd_torch_rng"):
        assert stream in cursor
    atomic_save(tmp_path / "interrupted.pt", interrupted)
    seen.clear()
    with patch.object(response, "train_condition", side_effect=track):
        resumed = response.train_response(data, cfg, tmp_path / "resumed", resume=True)
    assert seen == direct_seen[expected_cursor:]
    expected, actual = load_checkpoint(direct["checkpoints"][0]), load_checkpoint(resumed["checkpoints"][0])
    for field in ("model", "optimizer", "rng", "counters", "extra"):
        assert_exact(expected[field], actual[field])
    assert actual["epoch"] == cfg["epochs"]
    assert actual["counters"]["optimizer_step"] == cfg["epochs"] * updates
    assert actual["metadata"]["selection"] == "fixed_final_epoch"
    assert actual["metadata"]["target_epochs"] == cfg["epochs"]
    before_rng = torch.get_rng_state().clone()
    frozen, _ = response.load_frozen_response(resumed["checkpoints"][0], data, "cpu")
    assert frozen.config.backbone == backbone and not frozen.training
    assert torch.equal(before_rng, torch.get_rng_state())
    assert (tmp_path / "direct/member_1/metrics.csv").read_bytes() == (tmp_path / "resumed/member_1/metrics.csv").read_bytes()
    with (tmp_path / "resumed/member_1/metrics.csv").open(newline="") as stream:
        assert [int(row["epoch"]) for row in csv.DictReader(stream)] == [1, 2]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
              for p in (tmp_path / "resumed").rglob("*") if p.is_file()}
    with patch.object(response, "train_condition", side_effect=AssertionError("completed run retrained")):
        assert response.train_response(data, cfg, tmp_path / "resumed", resume=True) == resumed
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
             for p in (tmp_path / "resumed").rglob("*") if p.is_file()}
    assert before == after
    (tmp_path / "verification.json").write_text(json.dumps({"scenario": interruption, "backbone": backbone,
        "resumed_cursor": expected_cursor, "condition_count": len(keys),
        "optimizer_step": actual["counters"]["optimizer_step"], "model_optimizer_rng_exact": True,
        "completed_read_only": True, "metrics_exact": True}), encoding="utf-8")


def test_legacy_epoch_checkpoint_and_nested_view_identity(tmp_path):
    torch.set_num_threads(1)
    data, cfg = fixture_data(), config()

    def interrupt(phase, current, total):
        if phase == "time_response_member_1" and current == 1:
            raise KeyboardInterrupt("legacy epoch boundary")

    with pytest.raises(KeyboardInterrupt):
        response.train_response(data, cfg, tmp_path / "legacy", progress=interrupt)
    saved = load_checkpoint(tmp_path / "legacy/member_1/last.pt")
    assert saved["epoch"] == 1 and "update_resume" not in saved["extra"]
    original = response.train_response(data, cfg, tmp_path / "direct")
    restored = response.train_response(data, cfg, tmp_path / "legacy", resume=True)
    assert_exact(load_checkpoint(original["checkpoints"][0]), load_checkpoint(restored["checkpoints"][0]))
    restricted = data.view([], [], "nested_holdout")
    with pytest.raises(ValueError, match="TIME_RESPONSE_RESUME_IDENTITY_MISMATCH"):
        response.train_response(restricted, cfg, tmp_path / "legacy", resume=True)
    bad = load_checkpoint(restored["checkpoints"][0])
    bad["epoch"] -= 1
    atomic_save(tmp_path / "not_final.pt", bad)
    with pytest.raises(ValueError, match="FINAL_TIME_RESPONSE_REQUIRED"):
        response.load_frozen_response(tmp_path / "not_final.pt", data, "cpu")
    (tmp_path / "verification.json").write_text(json.dumps({"legacy_epoch_resume_exact": True,
        "nested_view_rejected": True, "nonfinal_checkpoint_rejected": True}), encoding="utf-8")
