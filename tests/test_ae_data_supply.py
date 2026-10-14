"""AE input supply, caching and exact resume; constructed cells only, not biology."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from conftest import assert_same as assert_identical
from src.lung.data import TrainingData
from src.lung.data.ae_expression_cache import NormalizedExpressionCache
from src.lung.data import ae_expression_cache as cache_module
from src.lung.data.common import sha256
from src.lung.engine.ae_data import AEDataSupply, AEGroupPrefetch


class DrawSupply:
    def sample_group(self, keys, rng, role="T"):
        return [rng.integers(100000, size=(4, 6)) for _ in keys]


def test_prefetch_commits_only_consumed_rng_and_resume_replays_unused_group():
    supply = DrawSupply()
    expected_rng = np.random.default_rng(73)
    expected_first = supply.sample_group(["a", "b"], expected_rng)
    committed = deepcopy(expected_rng.bit_generator.state)
    expected_second = supply.sample_group(["c"], expected_rng)
    rng = np.random.default_rng(73)
    with AEGroupPrefetch(supply) as producer:
        producer.queue(["a", "b"], rng)
        assert rng.bit_generator.state == np.random.default_rng(73).bit_generator.state
        first = producer.consume(["a", "b"], rng)
        assert rng.bit_generator.state == committed
        producer.queue(["c"], rng)
    assert rng.bit_generator.state == committed
    for observed, expected in zip(first, expected_first):
        np.testing.assert_array_equal(observed, expected)
    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = committed
    with AEGroupPrefetch(supply) as resumed:
        second = resumed.consume(["c"], resumed_rng)
    np.testing.assert_array_equal(second[0], expected_second[0])
    assert resumed_rng.bit_generator.state == expected_rng.bit_generator.state


def test_prefetch_error_is_propagated_without_advancing_rng():
    class Broken(DrawSupply):
        def sample_group(self, keys, rng, role="T"):
            rng.integers(20)
            raise ValueError("corrupt input")
    rng = np.random.default_rng(18)
    original = deepcopy(rng.bit_generator.state)
    with AEGroupPrefetch(Broken()) as producer:
        with pytest.raises(ValueError, match="corrupt input"):
            producer.consume(["a"], rng)
    assert rng.bit_generator.state == original


def test_prefetch_rejects_changed_order():
    with AEGroupPrefetch(DrawSupply()) as producer:
        rng = np.random.default_rng(19)
        producer.queue(["a", "b"], rng)
        with pytest.raises(ValueError, match="ORDER_MISMATCH"):
            producer.consume(["b", "a"], rng)


def equal_batches(old, new):
    for expected, observed in zip(old, new):
        for field in ("control", "target", "control_cell_ids", "target_cell_ids"):
            np.testing.assert_array_equal(expected[field], observed[field])


def test_grouped_ae_reads_once_and_preserves_draws_and_duplicates(ae_reader, monkeypatch):
    keys = ae_reader.condition_ids("fit")[:2]
    keys = [*keys, keys[0]]
    old_rng, new_rng = np.random.default_rng(28), np.random.default_rng(28)
    old = [ae_reader.sample_response(key, 5, old_rng, role="fit") for key in keys]
    original_read = ae_reader.rows.read
    calls = []
    def counted(rows):
        calls.append(len(rows))
        return original_read(rows)
    monkeypatch.setattr(ae_reader.rows, "read", counted)
    new = ae_reader.sample_ae_group(keys, 5, new_rng, role="fit")
    equal_batches(old, new)
    assert old_rng.bit_generator.state == new_rng.bit_generator.state
    assert calls == [30]


def test_real_reader_supply_cache_matches_original_and_validation(ae_reader):
    class Facade:
        def __init__(self, value):
            self.reader = value
    keys = ae_reader.condition_ids("fit")[:2]
    old_rng, new_rng = np.random.default_rng(91), np.random.default_rng(91)
    expected = [ae_reader.sample_response(key, 4, old_rng) for key in keys]
    with AEDataSupply(Facade(ae_reader), 4) as supply:
        assert supply.expressions is not None
        calls = []
        original_read = supply.expressions.read
        def counted(rows, role):
            calls.append(len(rows))
            return original_read(rows, role)
        supply.expressions.read = counted
        with supply.groups() as producer:
            observed = producer.consume(keys, new_rng)
        assert calls == [16]
        equal_batches(expected, observed)
        assert old_rng.bit_generator.state == new_rng.bit_generator.state
        selection = ae_reader.condition_ids("selection")[0]
        rng = np.random.default_rng(92)
        expected_v = ae_reader.sample_response(selection, 4, rng, role="selection")
        observed_v = supply.sample_group([selection], np.random.default_rng(92), role="V")[0]
        equal_batches([expected_v], [observed_v])


def test_real_reader_prefetch_training_resumes_optimizer_rng_and_tail_group(ae_reader, tmp_path):
    import torch
    from types import SimpleNamespace
    from src.lung.engine.latent_pipeline import run_latent_ae
    from src.lung.engine.checkpoint import load_checkpoint

    class Facade:
        def __init__(self):
            self.reader = ae_reader
            self.n_genes, self.neighbors = ae_reader.n_genes, ae_reader.neighbors
            self.identity = ae_reader.manifest["signature"]
        def condition_ids(self, role):
            return ae_reader.condition_ids({"T": "fit", "V": "selection"}[role])
        def iter_response_epoch(self, role, seed):
            keys = self.condition_ids(role)
            return iter([keys[0], keys[1], keys[0], keys[1], keys[0]])

    def monitor(name):
        return SimpleNamespace(weights=tmp_path / name / "weights", experiment=tmp_path / name,
                               say=lambda *_: None, record=lambda *a, **k: None)

    class InterruptAfterOne:
        exhausted = False
        def check(self, step, epoch):
            self.exhausted = step >= 1
            return self.exhausted

    config = {"response": {"lr": 0.001, "weight_decay": 0.0001, "grad_clip": 1.,
                           "conditions_per_update": 2, "ot_samples": 4},
              "epochs": 1, "early_stopping": False, "device": "cpu", "seed": 38,
              "workflow": "modular_stage",
              "research": {"latent": {"d_model": 8, "latent_dim": 4, "ae": {"batch": 4}},
                           "precision": {"mode": "fp32"}}}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        continuous, resumed = monitor("continuous"), monitor("resumed")
        run_latent_ae(config, continuous, Facade())
        run_latent_ae(config, resumed, Facade(), InterruptAfterOne())
        path = resumed.weights / "latent_ae" / "last.pt"
        partial = load_checkpoint(path)
        assert partial["epoch"] == 0 and partial["extra"]["resume"]["next_group_start"] == 2
        assert partial["counters"]["optimizer_step"] == 1
        run_latent_ae(config, resumed, Facade())
        expected = load_checkpoint(continuous.weights / "latent_ae" / "last.pt")
        actual = load_checkpoint(path)
        for field in ("model", "optimizer", "rng", "epoch", "identity", "counters", "extra"):
            assert_identical(expected[field], actual[field])
        assert actual["counters"]["optimizer_step"] == 3
    finally:
        torch.set_num_threads(threads)


def sample_rows(ae_reader, role="fit"):
    condition = ae_reader.condition_ids(role)[0]
    target = ae_reader.role_cells(condition, role)
    controls = ae_reader._control_rows(condition, 5, np.random.default_rng(42), role)
    return pd.concat([target.iloc[[2, 0, 2]], controls], ignore_index=True)


@pytest.mark.parametrize("role", ["fit", "selection"])
def test_matches_original_transform_and_preserves_duplicates(ae_reader, role):
    rows = sample_rows(ae_reader, role)
    expected = ae_reader.normalize(ae_reader.rows.read(rows))
    with NormalizedExpressionCache(ae_reader) as cache:
        actual = cache.read(rows, role)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[0], actual[2])
        assert actual.dtype == np.float32
        assert cache.read(rows.iloc[:0], role).shape == (0, ae_reader.n_genes)


def test_warm_read_and_restart_do_not_read_raw_or_normalize_again(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        expected = cache.read(rows, "fit")
        directory = cache.directory
        before = {str(p): sha256(p) for p in directory.rglob("*.npy")}
        with patch.object(ae_reader.rows, "read", side_effect=AssertionError("raw read")), \
                patch("src.lung.data.ae_expression_cache.log_normalize",
                      side_effect=AssertionError("log normalization")), \
                patch("src.lung.data.ae_expression_cache.np.load", wraps=np.load) as load:
            np.testing.assert_array_equal(cache.read(rows, "fit"), expected)
            assert load.call_count == 0
    reopened_reader = TrainingData(ae_reader.project_root)
    with NormalizedExpressionCache(reopened_reader) as cache, \
            patch.object(reopened_reader.rows, "read", side_effect=AssertionError("raw read")), \
            patch("src.lung.data.ae_expression_cache.log_normalize",
                  side_effect=AssertionError("log normalization")):
        assert cache.directory == directory
        np.testing.assert_array_equal(cache.read(rows, "fit"), expected)
    assert before == {str(p): sha256(p) for p in directory.rglob("*.npy")}


def test_warm_read_does_not_resolve_cached_array_paths(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        expected = cache.read(rows, "fit")
        original, calls = Path.resolve, []

        def counted(path, *args, **kwargs):
            calls.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "resolve", counted):
            np.testing.assert_array_equal(cache.read(rows, "fit"), expected)
        # Namespace, condition, part and source boundaries remain checked;
        # individual cached index/CSR arrays need no repeated resolve.
        limit = (2 + 2 * rows.condition_id.nunique() + 2 * len(cache._mapped)
                 + 2 * rows.matrix_path.nunique())
        assert len(calls) <= limit


def test_mapped_parts_are_bounded_and_eviction_releases_them(ae_reader):
    with patch("src.lung.data.ae_expression_cache.MAX_MAPPED_PARTS", 1):
        with NormalizedExpressionCache(ae_reader) as cache:
            cache.prepare(("fit",))
            assert len(cache._mapped) == 1
            rows = sample_rows(ae_reader)
            expected = ae_reader.normalize(ae_reader.rows.read(rows))
            with patch("src.lung.data.ae_expression_cache.sha256",
                       side_effect=AssertionError("Unchanged arrays must not be rehashed on LRU eviction")):
                np.testing.assert_array_equal(cache.read(rows, "fit"), expected)
            assert len(cache._mapped) == 1


def test_group_read_audits_sources_once_and_bounds_dense_temporary_rows(ae_reader):
    rows = pd.concat([sample_rows(ae_reader)] * 300, ignore_index=True)
    expected = ae_reader.normalize(ae_reader.rows.read(rows))
    with NormalizedExpressionCache(ae_reader) as cache:
        cache.read(rows, "fit")
        original, shapes = sparse.csr_matrix.toarray, []

        def counted(matrix, *args, **kwargs):
            shapes.append(matrix.shape)
            return original(matrix, *args, **kwargs)

        with patch.object(sparse.csr_matrix, "toarray", counted), \
                patch.object(cache._source_audit, "source", wraps=cache._source_audit.source) as source:
            np.testing.assert_array_equal(cache.read(rows, "fit"), expected)
            assert source.call_count == rows.matrix_path.nunique()
        assert shapes and max(shape[0] for shape in shapes) <= 256


def test_missing_cell_source_identity_is_rejected(ae_reader):
    with NormalizedExpressionCache(ae_reader) as cache:
        with pytest.raises(ValueError, match="missing"):
            cache.read(sample_rows(ae_reader).assign(condition_id=None), "fit")


def test_role_and_original_cell_source_identity_are_enforced(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        with pytest.raises(PermissionError):
            cache.read(rows, "selection")
        with pytest.raises(PermissionError):
            cache.read(rows, "final_test")
        with pytest.raises(PermissionError):
            cache.read(rows.assign(matrix_row=rows.matrix_row + 1), "fit")
        with pytest.raises(PermissionError):
            cache.read(rows.assign(matrix_path="../outside.npz"), "fit")
        with pytest.raises(PermissionError):
            cache.read(rows.assign(role="selection"), "fit")


def test_prepare_covers_requested_roles_only_and_reports_progress(ae_reader):
    events = []
    with NormalizedExpressionCache(ae_reader) as cache:
        conditions = ae_reader.condition_ids("fit", include_controls=True)
        assert cache.prepare(("fit", "fit"), lambda *event: events.append(event)) == len(conditions)
        assert events[0] == ("ae_expression_cache", 0, len(conditions))
        assert events[-1] == ("ae_expression_cache", len(conditions), len(conditions))
        manifests = [json.loads(p.read_text()) for p in cache.directory.rglob("manifest.json")]
        assert {item["role"] for item in manifests} == {"fit"}
        assert len(manifests) == len(conditions)
        with pytest.raises(PermissionError):
            cache.prepare(("selection", "final_test"))
        assert not (cache.directory / "selection").exists()


def test_source_change_invalidates_cached_expression(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        cache.read(rows, "fit")
        source = ae_reader.project_root / "source.npz"
        values = sparse.load_npz(source)
        values.data[0] += 1
        sparse.save_npz(source, values)
        with pytest.raises(ValueError, match="[Ss]ource"):
            cache.read(rows, "fit")


def test_restart_source_digest_check_catches_preserved_mtime(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        cache.read(rows, "fit")
    source = ae_reader.project_root / "source.npz"
    stamp = source.stat()
    values = sparse.load_npz(source)
    values.data[0] += 1
    sparse.save_npz(source, values)
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    # A fresh ae_reader hashes its first encountered source. During one session,
    # deliberate size/mtime rollback retains the existing source-audit boundary.
    ae_reader = TrainingData(ae_reader.project_root)
    with NormalizedExpressionCache(ae_reader) as cache:
        with pytest.raises(ValueError, match="[Ss]ource"):
            cache.read(rows, "fit")


def test_cache_array_corruption_is_checked_on_change_and_restart(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        cache.read(rows, "fit")
        array = next(cache.directory.rglob("data.npy"))
        saved = np.load(array, allow_pickle=False).copy()
        saved[0] += 1
        # Windows prevents truncating an actively mapped file; same-sized writes
        # still exercise changed-file identity and checksum revalidation.
        with array.open("r+b") as stream:
            np.save(stream, saved, allow_pickle=False)
        with pytest.raises(ValueError, match="checksum"):
            cache.read(rows, "fit")
    with NormalizedExpressionCache(ae_reader) as cache:
        with pytest.raises(ValueError, match="checksum"):
            cache.read(rows, "fit")


def test_half_published_condition_is_not_reused(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        cache.read(rows, "fit")
        path = next(cache.directory.rglob("manifest.json"))
    path.unlink()
    with NormalizedExpressionCache(ae_reader) as cache:
        with pytest.raises(ValueError, match="incomplete"):
            cache.read(rows, "fit")


def test_failed_build_can_restart_without_overwriting_orphans(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        with patch("src.lung.data.ae_expression_cache.np.save", side_effect=OSError("write failed")):
            with pytest.raises(OSError, match="write failed"):
                cache.read(rows, "fit")
        assert not list(cache.directory.rglob("manifest.json"))
        orphan = cache.directory / "fit" / "unrelated.tmp-interrupted"
        orphan.mkdir(parents=True)
        (orphan / "partial.npy").write_bytes(b"interrupted")
        actual = cache.read(rows, "fit")
        np.testing.assert_array_equal(actual, ae_reader.normalize(ae_reader.rows.read(rows)))
        assert (orphan / "partial.npy").read_bytes() == b"interrupted"


def test_normalization_and_gene_axis_have_distinct_versions(ae_reader):
    with NormalizedExpressionCache(ae_reader) as first:
        original_directory = first.directory
    ae_reader.mean = ae_reader.mean + .001
    with NormalizedExpressionCache(ae_reader) as changed:
        assert changed.directory != original_directory
    ae_reader.gene_ids = ae_reader.gene_ids[::-1].copy()
    with NormalizedExpressionCache(ae_reader) as changed:
        assert changed.directory != original_directory


def test_cache_directory_must_stay_inside_project(ae_reader):
    with pytest.raises(ValueError, match="inside"):
        NormalizedExpressionCache(ae_reader, ae_reader.project_root.parent / "outside-cache")


def junction(link, target):
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
                   check=True, capture_output=True)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary regression")
def test_initial_cache_alias_retarget_is_detected_on_warm_read(ae_reader):
    root = ae_reader.project_root
    original, alias = root / "cache-real", root / "cache-alias"
    outside = root.parent / (root.name + "-outside")
    original.mkdir()
    outside.mkdir()
    junction(alias, original)
    try:
        with NormalizedExpressionCache(ae_reader, alias) as cache:
            rows = sample_rows(ae_reader)
            cache.read(rows, "fit")
            # Remove only this test-created junction, not its target directory.
            alias.rmdir()
            junction(alias, outside)
            with pytest.raises(ValueError, match="inside"):
                cache.read(rows, "fit")
    finally:
        alias.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary regression")
def test_warm_condition_directory_retarget_is_rejected(ae_reader):
    rows = sample_rows(ae_reader)
    with NormalizedExpressionCache(ae_reader) as cache:
        cache.read(rows, "fit")
        entry = next(iter(cache._conditions.values()))
        target = entry["target"]
        outside = ae_reader.project_root.parent / (ae_reader.project_root.name + "-condition-outside")
        # Drop only owned maps to allow Windows to move the directory while
        # retaining the warm metadata entry whose boundary must be checked.
        for _, _, arrays in cache._mapped.values():
            cache_module._close_arrays(arrays)
        cache._mapped.clear()
        target.rename(outside)
        junction(target, outside)
        try:
            with pytest.raises(ValueError, match="inside"):
                cache.read(rows, "fit")
        finally:
            target.rmdir()


def test_cached_zero_common_library_is_rejected_only_when_requested(ae_reader):
    rows = sample_rows(ae_reader)
    source = ae_reader.project_root / "source.npz"
    matrix = sparse.load_npz(source).tolil()
    matrix[int(rows.iloc[0].matrix_row)] = 0
    sparse.save_npz(source, matrix.tocsr())
    with NormalizedExpressionCache(ae_reader) as cache:
        valid_rows = rows.iloc[[1]]
        np.testing.assert_array_equal(cache.read(valid_rows, "fit"),
                                      ae_reader.normalize(ae_reader.rows.read(valid_rows)))
        with pytest.raises(ValueError, match="ZERO_COMMON_GENE_LIBRARY"):
            cache.read(rows, "fit")


def test_close_releases_mappings_and_prevents_further_use(ae_reader):
    cache = NormalizedExpressionCache(ae_reader)
    cache.read(sample_rows(ae_reader), "fit")
    directory = cache.directory
    cache.close()
    cache.close()
    moved = directory.with_name(directory.name + "-closed")
    directory.rename(moved)
    assert moved.exists()
    with pytest.raises(RuntimeError, match="closed"):
        cache.read(sample_rows(ae_reader), "fit")
