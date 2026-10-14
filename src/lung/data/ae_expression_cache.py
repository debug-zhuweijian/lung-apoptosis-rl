"""Role-aware, lossless log-expression caches for AE input sampling.

Only fixed common-gene library normalization is persisted. Frozen fit statistics
are applied to the requested rows at read time; no full dense atlas is stored.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat as stat_module
import tempfile
from threading import RLock

from filelock import FileLock, Timeout
import numpy as np
import pandas as pd
from scipy import sparse

from .common import OUTER, digest_json, inside, sha256
from .preprocess import log_normalize
from .reader import TrainingData
from .sparse_cache import CSRCache, file_identity

SCHEMA = 1
MAX_CONDITIONS = 128
MAX_MAPPED_PARTS = 16
MAX_DENSE_ROWS = 256
ROW_FIELDS = ("cell_id", "matrix_path", "matrix_row", "condition_id")


def _array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def _close_arrays(arrays):
    for value in arrays.values():
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _ordinary_stamps(paths):
    """Check immutable cache leaves without resolving every ancestor again."""
    stamps = []
    for path in paths:
        stat = path.lstat()
        if stat_module.S_ISLNK(stat.st_mode) or getattr(stat, "st_file_attributes", 0) & 0x400:
            raise ValueError("AE expression cache path uses a reparse point")
        stamps.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev))
    return tuple(stamps)


class NormalizedExpressionCache:
    """Persistent fixed inputs; ``read`` preserves cell order and multiplicity.

    Conditions are built lazily in bounded CSR parts. ``prepare`` can explicitly
    build fit/selection in advance and resume by reusing complete conditions.
    Source checks still run for every requested source, while content hashes and
    read-only mappings are reused until file identity changes.
    """

    def __init__(self, reader, directory=None):
        if not isinstance(reader, TrainingData):
            raise TypeError("AE expression cache requires real TrainingData")
        if reader.mean is None or reader.scale is None:
            raise ValueError("PREPROCESSING_REQUIRED")
        self.reader, self.root = reader, reader.project_root
        self.mean, self.scale = np.array(reader.mean, copy=True), np.array(reader.scale, copy=True)
        self.mean.flags.writeable = self.scale.flags.writeable = False
        self.identity = digest_json({
            "schema": SCHEMA, "training": reader.manifest["signature"],
            "gene_ids": reader.gene_ids.tolist(), "columns": reader.rows.columns.tolist(),
            "mean": _array_digest(self.mean), "scale": _array_digest(self.scale),
            "source_audit": digest_json(reader.rows.registry),
            "normalization": "float64_log1p_10000_common_library_v1",
        })
        requested_base = Path(directory or reader.assets_path / "ae-expression-cache-v1")
        if not requested_base.is_absolute():
            requested_base = self.root / requested_base
        # Keep Windows paths short; full 256-bit identities are checked in every
        # manifest, so a truncated directory-name collision is rejected.
        self._directory_path = requested_base / self.identity[:32]
        self.directory = inside(self.root, self._directory_path)
        self.block_size = int(reader.config["stream_batch_size"])
        self._source_audit = reader.rows.disk_cache or CSRCache(self.root)
        self._conditions, self._mapped = OrderedDict(), OrderedDict()
        self._validated = {}
        self._lock, self._closed = RLock(), False
        self._asset_stamps = {name: file_identity(reader.assets_path / name)
                              for name in ("train.parquet", "val.parquet", "genes.parquet",
                                           "normalization.npz")}

    def _check(self, role):
        if self._closed:
            raise RuntimeError("AE expression cache is closed")
        if role not in ("fit", "selection"):
            raise PermissionError("AE input cache permits fit/selection only")
        self.reader._check_role(role)
        # Re-resolve the namespace once, so replacement by a junction cannot
        # redirect cached absolute paths outside the project.
        if inside(self.root, self._directory_path) != self.directory:
            raise ValueError("AE expression cache directory alias changed")
        for name in (f"{OUTER[role]}.parquet", "genes.parquet", "normalization.npz"):
            if file_identity(self.reader.assets_path / name) != self._asset_stamps[name]:
                raise ValueError("Frozen AE expression input asset changed")

    def _sources(self, rows):
        records = {}
        for name in dict.fromkeys(rows["matrix_path"].tolist()):
            if self.reader.rows.registry and name not in self.reader.rows.registry:
                raise ValueError("Source matrix is not registered in the source audit")
            path, (stamp, digest, shape) = self._source_audit.source(
                name, self.reader.rows.registry.get(name))
            records[name] = (path, stamp, {"sha256": digest, "shape": list(shape)})
        return records

    @staticmethod
    def _save(target, name, value):
        path = target / (name + ".npy")
        with path.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        return {"path": path.name, "sha256": sha256(path),
                "shape": list(value.shape), "dtype": value.dtype.str}

    def _build(self, target, role, condition, rows, locator):
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = str(inside(self.root, target.parent / (target.name + ".lock")))
        # filelock's native Windows opener needs an explicit extended-length
        # path for nested content-addressed directories (Python file IO does not).
        if os.name == "nt" and not lock_path.startswith("\\\\?\\"):
            lock_path = ("\\\\?\\UNC\\" + lock_path[2:] if lock_path.startswith("\\\\")
                         else "\\\\?\\" + lock_path)
        lock = FileLock(lock_path, timeout=0)
        try:
            lock.acquire()
        except Timeout as error:
            raise RuntimeError("AE expression cache build locked; retry after builder exits") from error
        temporary = None
        try:
            if target.exists():
                return
            sources = self._sources(rows)
            temporary = Path(tempfile.mkdtemp(prefix=target.name + ".tmp-", dir=target.parent))
            index_path = temporary / "index"
            index_path.mkdir()
            index = {name: self._save(index_path, name, rows[name].to_numpy(
                dtype=np.int64 if name == "matrix_row" else str)) for name in ROW_FIELDS}
            parts = []
            for start in range(0, len(rows), self.block_size):
                stop = min(start + self.block_size, len(rows))
                values, valid = log_normalize(self.reader.rows.read(rows.iloc[start:stop]))
                part = temporary / f"part-{len(parts):06d}"
                part.mkdir()
                arrays = {name: self._save(part, name, value) for name, value in {
                    "data": values.data, "indices": values.indices,
                    "indptr": values.indptr, "valid": valid}.items()}
                parts.append({"start": start, "stop": stop, "path": part.name, "arrays": arrays})
            # No publication if a source changed during preparation.
            if {name: record[2] for name, record in self._sources(rows).items()} != {
                    name: record[2] for name, record in sources.items()}:
                raise ValueError("Source changed while building AE expression cache")
            manifest = {"schema": SCHEMA, "status": "complete", "identity": self.identity,
                        "role": role, "condition_id": condition, "locator": locator,
                        "rows": len(rows), "n_genes": self.reader.n_genes,
                        "sources": {name: record[2] for name, record in sources.items()},
                        "index": index, "parts": parts}
            with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(target)
            temporary = None
        finally:
            try:
                if temporary is not None:
                    shutil.rmtree(inside(self.root, temporary))
            finally:
                lock.release()

    def _entry(self, role, condition):
        key = (role, condition)
        entry = self._conditions.get(key)
        if entry is not None:
            if inside(self.root, entry["target"]) != entry["target"]:
                raise ValueError("AE expression cache condition path changed")
            _ordinary_stamps([entry["target"].parent, entry["target"], entry["target"] / "index"])
            current = _ordinary_stamps(entry["identity_paths"])
            if current == entry["stamp"]:
                self._conditions.move_to_end(key)
                return entry
            self._conditions.pop(key)
        rows = self.reader.role_cells(condition, role).loc[:, ROW_FIELDS].reset_index(drop=True)
        if rows.cell_id.duplicated().any():
            raise ValueError("Duplicate canonical AE cell ID")
        locator = digest_json(rows.assign(matrix_row=rows.matrix_row.astype(int)).values.tolist())
        target = inside(self.root, self.directory / role / digest_json({"condition": condition,
                                                                       "locator": locator})[:32])
        if not target.exists():
            self._build(target, role, condition, rows, locator)
        manifest_path = inside(self.root, target / "manifest.json")
        if not manifest_path.exists():
            raise ValueError("AE expression cache condition is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"schema": SCHEMA, "status": "complete", "identity": self.identity,
                    "role": role, "condition_id": condition, "locator": locator,
                    "rows": len(rows), "n_genes": self.reader.n_genes}
        if any(manifest.get(name) != value for name, value in expected.items()):
            raise ValueError("AE expression cache input identity mismatch or incomplete condition")
        paths = [manifest_path]
        for name in ROW_FIELDS:
            spec = manifest["index"][name]
            path = inside(target, Path("index") / spec["path"])
            if sha256(path) != spec["sha256"]:
                raise ValueError("AE expression cache index checksum mismatch")
            saved = np.load(path, allow_pickle=False)
            original = rows[name].to_numpy(dtype=np.int64 if name == "matrix_row" else str)
            if (not np.array_equal(saved, original) or saved.dtype.str != spec["dtype"]
                    or list(saved.shape) != spec["shape"]):
                raise ValueError("AE expression cache cell/source index differs")
            paths.append(path)
        stop = 0
        for part in manifest["parts"]:
            if part["start"] != stop or not stop < part["stop"] <= len(rows):
                raise ValueError("AE expression cache part coverage differs")
            stop = part["stop"]
        if stop != len(rows):
            raise ValueError("AE expression cache condition is incomplete")
        _ordinary_stamps([target.parent, target, target / "index"])
        part_paths = [{name: inside(target, Path(part["path"]) / item["path"])
                       for name, item in part["arrays"].items()} for part in manifest["parts"]]
        entry = {"rows": rows, "index": pd.Index(rows.cell_id), "target": target,
                 "manifest": manifest, "identity_paths": paths,
                 "part_paths": part_paths, "stamp": _ordinary_stamps(paths),
                 "part_identities": [digest_json(part) for part in manifest["parts"]]}
        if len(self._conditions) >= MAX_CONDITIONS:
            self._conditions.popitem(last=False)
        self._conditions[key] = entry
        return entry

    def _part(self, entry, number):
        spec = entry["manifest"]["parts"][number]
        if inside(self.root, entry["target"]) != entry["target"]:
            raise ValueError("AE expression cache condition path changed")
        key = (entry["target"], number)
        paths = entry["part_paths"][number]
        _ordinary_stamps([next(iter(paths.values())).parent])
        current = _ordinary_stamps(paths.values())
        identity = (current, entry["part_identities"][number])
        previous = self._mapped.get(key)
        if previous is not None and previous[0] == identity:
            self._mapped.move_to_end(key)
            return previous[1], previous[2]["valid"]
        if previous is not None:
            _close_arrays(self._mapped.pop(key)[2])
        validate = self._validated.get(key) != identity
        arrays = {}
        try:
            for name, path in paths.items():
                item = spec["arrays"][name]
                if validate and sha256(path) != item["sha256"]:
                    raise ValueError(f"AE expression cache checksum mismatch: {name}")
                arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
                if (list(arrays[name].shape) != item["shape"]
                        or arrays[name].dtype.str != item["dtype"]):
                    raise ValueError("AE expression cache array identity mismatch")
            shape = (spec["stop"] - spec["start"], self.reader.n_genes)
            matrix = sparse.csr_matrix(shape, dtype=np.float64)
            matrix.data, matrix.indices, matrix.indptr = (
                arrays[name] for name in ("data", "indices", "indptr"))
            if (matrix.data.dtype != np.float64 or arrays["valid"].dtype != bool
                    or arrays["valid"].shape != (shape[0],)):
                raise ValueError("AE expression cache values are invalid")
            if validate:
                if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
                    raise ValueError("AE expression cache values are invalid")
                matrix.check_format(full_check=True)
            if _ordinary_stamps(paths.values()) != current:
                raise ValueError("AE expression cache changed during validation")
        except BaseException:
            _close_arrays(arrays)
            raise
        if len(self._mapped) >= MAX_MAPPED_PARTS:
            _close_arrays(self._mapped.popitem(last=False)[1][2])
        self._validated[key] = identity
        self._mapped[key] = (identity, matrix, arrays)
        return matrix, arrays["valid"]

    def read(self, rows, role):
        """Return the original standardized float32 input, in requested order."""
        with self._lock:
            self._check(role)
            if "role" in rows and not rows["role"].eq(role).all():
                raise PermissionError("AE input rows differ from requested role")
            if not set(ROW_FIELDS).issubset(rows.columns) or rows[list(ROW_FIELDS)].isna().any().any():
                raise ValueError("AE input rows have missing cell/source identity")
            rows = rows.reset_index(drop=True)
            output = np.empty((len(rows), self.reader.n_genes), dtype=np.float32)
            requests = []
            for condition, requested in rows.groupby("condition_id", sort=False):
                try:
                    entry = self._entry(role, condition)
                except ValueError as error:
                    if "No cells for condition" in str(error):
                        raise PermissionError("AE condition differs from requested role") from error
                    raise
                positions = entry["index"].get_indexer(requested["cell_id"])
                if (positions < 0).any() or not np.array_equal(
                        requested.loc[:, ROW_FIELDS].to_numpy(),
                        entry["rows"].iloc[positions].loc[:, ROW_FIELDS].to_numpy()):
                    raise PermissionError("AE cell/source row identity differs from requested role")
                requests.append((entry, requested, positions))
            sources = self._sources(rows)
            for entry, requested, positions in requests:
                for name in dict.fromkeys(requested["matrix_path"].tolist()):
                    if entry["manifest"]["sources"].get(name) != sources[name][2]:
                        raise ValueError("Source differs from cached AE expression")
                for number, part in enumerate(entry["manifest"]["parts"]):
                    selected = np.flatnonzero((positions >= part["start"]) & (positions < part["stop"]))
                    if not len(selected):
                        continue
                    matrix, valid = self._part(entry, number)
                    for start in range(0, len(selected), MAX_DENSE_ROWS):
                        batch = selected[start:start + MAX_DENSE_ROWS]
                        offsets = positions[batch] - part["start"]
                        if not valid[offsets].all():
                            raise ValueError("ZERO_COMMON_GENE_LIBRARY: unusable expression row")
                        output[requested.index.to_numpy()[batch]] = (
                            (matrix[offsets].toarray() - self.mean) / self.scale).astype(np.float32)
            if any(file_identity(path) != stamp for path, stamp, _ in sources.values()):
                raise ValueError("Source changed during AE expression read")
            return output

    def prepare(self, roles=("fit", "selection"), progress=None):
        """Explicitly prepare complete conditions, reusing published input caches."""
        with self._lock:
            roles = tuple(dict.fromkeys(roles))
            for role in roles:
                self._check(role)
            plan = [(role, condition) for role in roles
                    for condition in self.reader.condition_ids(role, include_controls=True)]
            if progress is not None:
                progress("ae_expression_cache", 0, len(plan))
            for number, (role, condition) in enumerate(plan, 1):
                entry = self._entry(role, condition)
                sources = self._sources(entry["rows"])
                if {name: item[2] for name, item in sources.items()} != entry["manifest"]["sources"]:
                    raise ValueError("Source differs from cached AE expression")
                for part_number in range(len(entry["manifest"]["parts"])):
                    self._part(entry, part_number)
                if progress is not None:
                    progress("ae_expression_cache", number, len(plan))
            return len(plan)

    def close(self):
        with self._lock:
            for _, _, arrays in self._mapped.values():
                _close_arrays(arrays)
            self._mapped.clear()
            self._conditions.clear()
            self._validated.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
