"""Role-aware sparse sampling for the frozen training assets."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.lung_data import PreparedLungData

from .common import OUTER, ROLES, check_shard, dataset_config, digest_json, inside, sha256, source_shard_registry
from .preprocess import standardized
from .sparse_cache import CSRCache, DEFAULT_CACHE, file_identity

METADATA_CACHE_ENTRIES = 128
METADATA_CACHE_BYTES = 64 * 1024 * 1024


class SparseRows:
    """Read memmapped CSR shards with a bounded immutable heap LRU."""

    def __init__(self, project_root, columns, cache_size=2, registry=None, *,
                 cache_dir=DEFAULT_CACHE, max_cache_bytes=0, use_disk_cache=True):
        self.root = Path(project_root).resolve()
        self.columns = np.asarray(columns, dtype=np.int64)
        if self.columns.ndim != 1 or np.any(self.columns < 0):
            raise ValueError("Gene columns must be a nonnegative vector")
        if cache_size < 1 or max_cache_bytes < 0:
            raise ValueError("Invalid sparse cache limits")
        self.cache_size = cache_size
        self.max_cache_bytes = max_cache_bytes
        self.cache_bytes = 0
        self.cache = OrderedDict()
        self.registry = registry or {}
        self.disk_cache = CSRCache(self.root, cache_dir) if use_disk_cache else None
        self._source_stamps = {}

    def _matrix(self, name):
        if self.registry and name not in self.registry:
            raise ValueError("Matrix is not registered in the source audit")
        if self.disk_cache and self.max_cache_bytes == 0:
            # No heap entry can bypass CSRCache.source's per-load source audit.
            matrix = self.disk_cache.load(name, self.registry.get(name))
        else:
            path = inside(self.root, name)
            check_shard(path, self.registry.get(name))
            stamp = file_identity(path)
            if name in self.cache and self._source_stamps[name] != stamp:
                self.cache_bytes -= self._nbytes(self.cache.pop(name))
            if name in self.cache:
                self.cache.move_to_end(name)
                return self.cache[name]
            matrix = (self.disk_cache.load(name, self.registry.get(name)) if self.disk_cache
                      else sparse.load_npz(path))
        if (not sparse.isspmatrix_csr(matrix)
                or (self.columns.size and matrix.shape[1] <= self.columns.max())):
            raise ValueError("Source count matrix is not the expected CSR gene space")
        size = self._nbytes(matrix)
        # Retain only bounded heap copies: persistent Windows mappings prevent
        # cache cleanup while callers/mocks retain references to this reader.
        if self.max_cache_bytes > 0 and size <= self.max_cache_bytes:
            while self.cache and (len(self.cache) >= self.cache_size
                                       or self.cache_bytes + size > self.max_cache_bytes):
                _, previous = self.cache.popitem(last=False)
                self.cache_bytes -= self._nbytes(previous)
            matrix = matrix.copy()
            for field in ("data", "indices", "indptr"):
                getattr(matrix, field).flags.writeable = False
            self.cache[name] = matrix
            self.cache_bytes += size
            self._source_stamps[name] = stamp
        return matrix

    @staticmethod
    def _nbytes(matrix):
        return sum(getattr(matrix, name).nbytes for name in ("data", "indices", "indptr"))

    def prepare_cache(self, names, progress=None):
        """Explicit, sequential preparation of only the caller's source list."""
        names = list(dict.fromkeys(names))
        for index, name in enumerate(names, 1):
            self._matrix(name)
            if progress is not None:
                progress(index, len(names), name)
        return len(names)

    def read(self, rows: pd.DataFrame):
        rows = rows.reset_index(drop=True)
        if rows.empty:
            return sparse.csr_matrix((0, len(self.columns)), dtype=np.float64)
        parts, order = [], []
        for name, part in rows.groupby("matrix_path", sort=False):
            matrix = self._matrix(name)
            indices = part["matrix_row"].to_numpy(dtype=np.int64)
            if (indices < 0).any() or (indices >= matrix.shape[0]).any():
                raise ValueError("Source matrix row out of range")
            parts.append(matrix[indices][:, self.columns])
            order.extend(part.index.to_list())
        return sparse.vstack(parts, format="csr")[np.argsort(order)]


class TrainingData:
    """Default permissions are T/fit and V/selection only.

    Auxiliary trainers explicitly request their documented roles. Test readers
    must use ``allow_test=True, allowed_roles=("final_test",)``; merely opening
    a test reader does not grant any weight-updating method extra privileges.
    """

    def __init__(self, project_root=None, config_path=None, *, allow_test=False,
                 allowed_roles=("fit", "selection"), require_preprocessing=True,
                 verify_hashes=True, row_cache_options=None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
        self.config = dataset_config(self.project_root, config_path)
        self.assets_path = inside(self.project_root, self.config["assets"])
        self.manifest = json.loads((self.assets_path / "manifest.json").read_text(encoding="utf-8"))
        source_hash = sha256(inside(self.project_root, self.config["bundle"]) / "manifest.json")
        signature = digest_json({"config": self.config, "source_manifest": source_hash, "training_schema": 1})
        if signature != self.manifest["signature"]:
            raise ValueError("Training configuration/source manifest differs from frozen assets")
        if self.manifest["status"] not in ("metadata_ready", "complete"):
            raise ValueError("Training metadata preparation is incomplete")
        if require_preprocessing and self.manifest["status"] != "complete":
            raise ValueError("PREPROCESSING_REQUIRED: fit-only normalization and graph are not ready")
        self.allowed_roles = frozenset(allowed_roles)
        if not self.allowed_roles.issubset(ROLES):
            raise ValueError("Unknown role")
        if "final_test" in self.allowed_roles and not allow_test:
            raise PermissionError("Final test requires explicit allow_test=True")
        self.allow_test = allow_test
        if verify_hashes:
            for name, item in self.manifest["files"].items():
                if sha256(inside(self.assets_path, name)) != item["sha256"]:
                    raise ValueError(f"Training asset hash mismatch: {name}")
        self.conditions = pd.read_parquet(self.assets_path / "conditions.parquet")
        self.condition_table = self.conditions.set_index("condition_id", drop=False)
        self.genes = pd.read_parquet(self.assets_path / "genes.parquet")
        self.gene_ids = self.genes["ensembl_id"].to_numpy(dtype=str)
        self.n_genes = len(self.genes)
        self.fingerprints = np.load(self.assets_path / "fingerprints.npy", allow_pickle=False)
        self.role_counts = pd.read_parquet(self.assets_path / "role_condition_counts.parquet")
        self.controls = pd.read_parquet(self.assets_path / "control_links.parquet")
        prepared = PreparedLungData(self.project_root, self.config["bundle"])
        self.rows = SparseRows(self.project_root, self.genes["tahoe_column"],
                               registry=source_shard_registry(prepared), **(row_cache_options or {}))
        self._metadata_cache = OrderedDict()
        self._metadata_cache_bytes = 0
        self._condition_cache = {}
        self._condition_sets = {}
        self._present_conditions = {role: frozenset(frame["condition_id"])
                                    for role, frame in self.role_counts.groupby("role", sort=False)}
        self._control_ids = {key: tuple(frame["control_condition_id"])
                             for key, frame in self.controls.groupby("condition_id", sort=False)}
        self.mean = self.scale = self.bandwidths = self.neighbors = None
        if (self.assets_path / "normalization.npz").exists():
            with np.load(self.assets_path / "normalization.npz", allow_pickle=False) as norm:
                self.mean, self.scale = norm["mean"], norm["scale"]
        if (self.assets_path / "neighbors.npz").exists():
            self.neighbors = np.load(self.assets_path / "neighbors.npz", allow_pickle=False)["neighbors"]
        if (self.assets_path / "bandwidths.npz").exists():
            self.bandwidths = np.load(self.assets_path / "bandwidths.npz", allow_pickle=False)["bandwidths"]
        self.assets = {"mu": self.mean, "scale": self.scale, "sd": self.scale,
                       "neighbors": self.neighbors, "bandwidths": self.bandwidths,
                       "gene_ids": self.gene_ids, "gene_count": self.n_genes,
                       "signature": self.manifest["signature"]}
        self.context_vocabulary = {name: sorted(self.conditions[name].dropna().unique().tolist())
                                   for name in ("source_id", "background_id", "protocol_id")}

    def _check_role(self, role):
        if role not in self.allowed_roles:
            raise PermissionError(f"This reader is not authorized for role {role}")

    def condition_ids(self, role="fit", *, include_controls=False):
        self._check_role(role)
        key = (role, include_controls)
        if key not in self._condition_cache:
            selected = self.role_counts.loc[self.role_counts["role"].eq(role), "condition_id"]
            rows = self.condition_table.loc[selected]
            if not include_controls:
                rows = rows.loc[rows["condition_type"].eq("treated")]
            self._condition_cache[key] = tuple(sorted(rows["condition_id"].tolist()))
            self._condition_sets[key] = frozenset(self._condition_cache[key])
        return list(self._condition_cache[key])

    def _is_response(self, condition_id, role):
        self._check_role(role)
        if (role, False) not in self._condition_sets:
            self.condition_ids(role)
        return condition_id in self._condition_sets[(role, False)]

    def cell_metadata(self, condition_id, role="fit"):
        return self._cell_metadata(condition_id, role).copy()

    def role_cells(self, condition_id, role="fit"):
        """All cell rows of one condition in a role, for representation caching.

        Returns the reader's canonical row frame (cell_id, matrix_path,
        matrix_row, condition_id); the caller owns normalization and encoding.
        """
        return self._cell_metadata(condition_id, role).copy()

    def _cell_metadata(self, condition_id, role):
        self._check_role(role)
        cache_key = (role, condition_id)
        if cache_key not in self._metadata_cache:
            frame = pd.read_parquet(self.assets_path / f"{OUTER[role]}.parquet",
                                    filters=[("role", "==", role), ("condition_id", "==", condition_id)])
            if frame.empty:
                raise ValueError(f"No cells for condition in allowed role: {condition_id}, {role}")
            # Measure once per loaded frame, including index/object columns.
            # Internal sampling only selects rows; public access returns a copy.
            frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
            if frame_bytes > METADATA_CACHE_BYTES:
                return frame
            while self._metadata_cache and (
                    len(self._metadata_cache) >= METADATA_CACHE_ENTRIES
                    or self._metadata_cache_bytes + frame_bytes > METADATA_CACHE_BYTES):
                _, (_, evicted_bytes) = self._metadata_cache.popitem(last=False)
                self._metadata_cache_bytes -= evicted_bytes
            self._metadata_cache[cache_key] = (frame, frame_bytes)
            self._metadata_cache_bytes += frame_bytes
        self._metadata_cache.move_to_end(cache_key)
        return self._metadata_cache[cache_key][0]

    def normalize(self, counts):
        if self.mean is None or self.scale is None:
            raise ValueError("PREPROCESSING_REQUIRED")
        return standardized(counts, self.mean, self.scale)

    def action_features(self, condition_id):
        row = self.condition_table.loc[condition_id]
        if not bool(row["eligible"]):
            raise ValueError("DRUG_IDENTITY_UNRESOLVED")
        vehicle = row["condition_type"] == "control"
        return {"fingerprint": self.fingerprints[int(row["fingerprint_row"])].astype(np.float32),
                "dose_um": float(row["dose_um"]) if not vehicle else 0.,
                "dose_missing": bool(vehicle), "is_vehicle": bool(vehicle),
                "duration_h": float(row["duration_h"]),
                **{name: self.context_vocabulary[name].index(row[name]) for name in self.context_vocabulary},
                "condition_id": condition_id, "drug_id": row["drug_id"],
                "chemical_id": row["chemical_or_experiment_group_id"]}

    def sample_response(self, condition_id, n=128, rng=None, role="fit", *,
                        control_n=None, target_n=None):
        control_rows, target_rows = self._response_rows(
            condition_id, n, rng, role, control_n=control_n, target_n=target_n)
        return {"control": self.normalize(self.rows.read(control_rows)),
                "target": self.normalize(self.rows.read(target_rows)),
                "control_cell_ids": control_rows["cell_id"].to_numpy(dtype=str),
                "target_cell_ids": target_rows["cell_id"].to_numpy(dtype=str),
                "condition": self.action_features(condition_id), "role": role,
                "coupling_type": "unpaired_independent_samples_for_explicit_OT_prior"}

    def _response_rows(self, condition_id, n, rng, role, *, control_n=None, target_n=None):
        self._check_role(role)
        rng = np.random.default_rng() if rng is None else rng
        n_control = n if control_n is None else control_n
        n_target = n if target_n is None else target_n
        if min(n_control, n_target) < 1:
            raise ValueError("Sample sizes must be positive")
        if not self._is_response(condition_id, role):
            raise PermissionError("Condition is not a treated response in the requested role")
        target_pool = self._cell_metadata(condition_id, role)
        target_rows = target_pool.iloc[rng.integers(len(target_pool), size=n_target)]
        control_rows = self._control_rows(condition_id, n_control, rng, role)
        return control_rows, target_rows

    def _response_result(self, condition_id, role, control_rows, target_rows,
                         control_counts, target_counts):
        return {"control": self.normalize(control_counts),
                "target": self.normalize(target_counts),
                "control_cell_ids": control_rows["cell_id"].to_numpy(dtype=str),
                "target_cell_ids": target_rows["cell_id"].to_numpy(dtype=str),
                "condition": self.action_features(condition_id), "role": role,
                "coupling_type": "unpaired_independent_samples_for_explicit_OT_prior"}

    def sample_response_group(self, keys, cfg, rng, mmd_rng, role="fit"):
        """Share shard reads within one group, preserving each stream's draw order."""
        self._check_role(role)
        requests, row_parts = [], []
        use_mmd = bool(cfg.get("mmd_weight", 1.))
        for key in keys:
            streams = [(cfg["ot_samples"], rng)]
            if use_mmd:
                streams.append((cfg["mmd_samples"], mmd_rng))
            for n, generator in streams:
                control, target = self._response_rows(key, n, generator, role)
                requests.append((key, control, target))
                row_parts.extend((control, target))
        if not requests:
            return []
        # SparseRows.read opens/audits each distinct shard once for this call.
        # Keep counts sparse across the group; normalize each original batch.
        counts = self.rows.read(pd.concat(row_parts, ignore_index=True))
        results, offset = [], 0
        for key, control, target in requests:
            middle, end = offset + len(control), offset + len(control) + len(target)
            results.append(self._response_result(
                key, role, control, target, counts[offset:middle], counts[middle:end]))
            offset = end
        if use_mmd:
            return list(zip(results[::2], results[1::2]))
        return [(result, None) for result in results]

    def _control_rows(self, condition_id, n, rng, role):
        self._check_role(role)
        control_ids = self._control_ids.get(condition_id, ())
        present = self._present_conditions.get(role, frozenset())
        pools = [self._cell_metadata(item, role) for item in control_ids if item in present]
        if not pools:
            raise ValueError("NO_SAME_ROLE_CONTROL")
        # Uniform technical control sample, then uniform cell within that sample.
        picks = rng.integers(len(pools), size=n)
        locations = np.asarray([rng.integers(len(pools[index])) for index in picks])
        if len(pools) == 1:
            return pools[0].iloc[locations].reset_index(drop=True)
        positions = [np.flatnonzero(picks == index) for index in range(len(pools))]
        selected = [pool.iloc[locations[position]] for pool, position in zip(pools, positions)]
        return pd.concat(selected, ignore_index=True).iloc[
            np.argsort(np.concatenate(positions))].reset_index(drop=True)

    def sample_controls(self, condition_id, n=128, rng=None, role="fit"):
        """Decision roots read reference controls only, never response outcomes."""
        self._check_role(role)
        if n < 1:
            raise ValueError("Sample size must be positive")
        if not self._is_response(condition_id, role):
            raise PermissionError("Condition is not in the requested decision-root role")
        rng = np.random.default_rng() if rng is None else rng
        rows = self._control_rows(condition_id, n, rng, role)
        return {"control": self.normalize(self.rows.read(rows)),
                "control_cell_ids": rows["cell_id"].to_numpy(dtype=str),
                "condition": self.action_features(condition_id), "role": role,
                "observation_role": "contemporaneous_control", "response_outcomes_read": False}

    def iter_response_epoch(self, role="fit", seed=0):
        """N_role uniform source -> condition draws, with replacement."""
        ids = self.condition_ids(role)
        selected = self.condition_table.loc[ids]
        groups = [frame["condition_id"].tolist() for _, frame in selected.groupby("source_id", sort=True)]
        rng = np.random.default_rng(seed)
        for _ in range(len(ids)):
            candidates = groups[rng.integers(len(groups))]
            yield candidates[rng.integers(len(candidates))]

    def state_arrays(self, fold="1_to_8", role="fit"):
        self._check_role(role)
        report = self.manifest["metadata"]["tidyi_folds"].get(fold)
        if report is None or report["status"] != "ready":
            raise ValueError("STATE_SPLIT_INSUFFICIENT")
        rows = pd.read_parquet(self.assets_path / "tidyi_splits.parquet",
                               filters=[("fold", "==", fold), ("role", "==", role)])
        if rows.empty:
            raise ValueError("No TIDYI cells for this fold/role")
        prepared = PreparedLungData(self.project_root, self.config["bundle"])
        source_name = prepared.manifest["matrices"]["tidyi_counts"]
        expected = {item["path"]: item for item in prepared.manifest.get("outputs", [])}
        if source_name in expected and sha256(inside(prepared.bundle_path, source_name)) != expected[source_name]["sha256"]:
            raise ValueError("TIDYI count matrix differs from the prepared source hash")
        counts, original, _ = prepared.load_tidyi()
        # Count order is the prepared metadata order, not necessarily matrix_row.
        locations = pd.Series(np.arange(len(original)), index=original["cell_id"])
        matrix = counts[locations.loc[rows["cell_id"]].to_numpy()]
        return self.normalize(matrix), rows["state_class"].to_numpy(dtype=np.int64)
