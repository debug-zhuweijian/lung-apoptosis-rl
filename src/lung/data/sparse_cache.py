"""Lossless, content-addressed CSR storage; source NPZ files remain untouched."""

from __future__ import annotations

from collections import OrderedDict
import json
import logging
import os
from pathlib import Path
import shutil
import stat as stat_module
import tempfile
import time

import numpy as np
from filelock import FileLock, Timeout
from scipy import sparse

from .common import check_shard, digest_json, inside, sha256

SCHEMA = 1
DEFAULT_CACHE = "08-datasets/.csr-cache-v1"
LOG = logging.getLogger(__name__)


def file_identity(path):
    stat = path.stat()
    return _stat_identity(stat)


def _stat_identity(stat):
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev)


def _mapping_identity(target):
    """Cheap identity checks only for ordinary leaves of a freshly resolved target."""
    stamps = []
    for name in ("manifest.json", "data.npy", "indices.npy", "indptr.npy"):
        stat = (target / name).lstat()
        if (stat_module.S_ISLNK(stat.st_mode)
                or getattr(stat, "st_file_attributes", 0) & 0x400):
            return None  # Includes Windows junctions and other reparse points.
        stamps.append(_stat_identity(stat))
    return tuple(stamps)


def _mapped_matrix(manifest, arrays):
    # Give callers independent CSR wrappers sharing only read-only mapped arrays.
    matrix = sparse.csr_matrix(tuple(manifest["shape"]), dtype=arrays["data"].dtype)
    matrix.data, matrix.indices, matrix.indptr = (arrays[name]
                                                 for name in ("data", "indices", "indptr"))
    return matrix


def _inside_resolved(parent, name):
    """Resolve a child of a parent already resolved in this load only."""
    resolved = (parent / name).resolve()
    if not resolved.is_relative_to(parent):
        raise ValueError("Data path must remain inside the project")
    return resolved


class CSRCache:
    """Hash each source once per unchanged file identity in this reader.

    Cached arrays are hash-checked on first use and whenever their file identity
    changes. LRU eviction does not force another full scan. OS-backed read-only
    mappings avoid retaining decompressed shards in Python heap memory.
    max_open_mappings enables bounded reuse; zero keeps the original reopen path.
    Eviction drops references rather than closing mappings held by a caller.
    """

    def __init__(self, root, directory=DEFAULT_CACHE, *, max_open_mappings=0):
        if type(max_open_mappings) is not int or max_open_mappings < 0:
            raise ValueError("max_open_mappings must be a nonnegative integer")
        self.root = Path(root).resolve()
        self._directory_path = self.root / directory
        self.directory = inside(self.root, directory)
        self.max_open_mappings = max_open_mappings
        self._mappings = OrderedDict()
        self._sources = {}
        self._validated = {}
        self._last_progress = float("-inf")
        self._built = 0

    def clear(self):
        """Drop owned references; caller-held mmap views remain valid until released."""
        self._mappings.clear()
        self._sources.clear()
        self._validated.clear()

    def close(self):
        self.clear()

    def source(self, name, record=None):
        path = inside(self.root, name)
        check_shard(path, record)
        stamp = file_identity(path)
        previous = self._sources.get(path)
        if previous is None or previous[0] != stamp:
            digest = sha256(path)
            if file_identity(path) != stamp:
                raise ValueError("Source changed while hashing")
            audit = record or {}
            expected = audit.get("csr_sha256_from_checkpoint", audit.get("csr_sha256"))
            if expected is not None and digest != expected:
                raise ValueError("Source CSR hash differs from strict audit")
            with np.load(path, allow_pickle=False) as archive:
                shape = tuple(int(n) for n in archive["shape"])
            if "source_rows" in audit and shape[0] != audit["source_rows"]:
                raise ValueError("Source CSR dimensions differ from strict audit")
            if "source_columns" in audit and shape[1] != audit["source_columns"]:
                raise ValueError("Source CSR dimensions differ from strict audit")
            previous = (stamp, digest, shape)
            self._sources[path] = previous
        return path, previous

    def load(self, name, record=None):
        source, (stamp, digest, source_shape) = self.source(name, record)
        # Resolve again so a directory replaced with a symlink cannot escape.
        directory = inside(self.root, self._directory_path)
        key = digest_json({"schema": SCHEMA, "source_sha256": digest})
        target = _inside_resolved(directory, key)
        if not target.exists():
            self._build(source, stamp, digest, target)
            directory = inside(self.root, self._directory_path)
            target = _inside_resolved(directory, key)
        cached = self._mappings.pop(target, None)
        if cached is not None and _mapping_identity(target) == cached[0]:
            self._mappings[target] = cached
            return _mapped_matrix(cached[1], cached[2])
        manifest_path = _inside_resolved(target, "manifest.json")
        array_paths = {name: _inside_resolved(target, f"{name}.npy")
                       for name in ("data", "indices", "indptr")}
        stamp_paths = [manifest_path, *array_paths.values()]
        current = tuple(file_identity(path) for path in stamp_paths)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema"] != SCHEMA or manifest["source_sha256"] != digest:
            raise ValueError("CSR cache source/version identity mismatch")
        shape = tuple(manifest["shape"])
        if shape != source_shape:
            raise ValueError("CSR cache shape differs from source dimensions")
        if len(shape) != 2 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("CSR cache shape is invalid")
        arrays = {}
        validate = self._validated.get(target) != current
        for name, path in array_paths.items():
            spec = manifest["arrays"][name]
            if validate and sha256(path) != spec["sha256"]:
                raise ValueError(f"CSR cache checksum mismatch: {name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]:
                raise ValueError(f"CSR cache array identity mismatch: {name}")
            arrays[name] = array
        data, indices, indptr = (arrays[name] for name in ("data", "indices", "indptr"))
        if (data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1
                or len(data) != len(indices) or len(indptr) != shape[0] + 1
                or indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu"):
            raise ValueError("CSR cache dimensions are inconsistent")
        if validate:
            if (indptr[0] != 0 or indptr[-1] != len(data)
                    or np.any(indptr[1:] < indptr[:-1])
                    or np.any(indices < 0) or np.any(indices >= shape[1])):
                raise ValueError("CSR cache structure is invalid")
            if tuple(file_identity(path) for path in stamp_paths) != current:
                raise ValueError("CSR cache changed during validation")
            self._validated[target] = current
        if self.max_open_mappings:
            mapping_stamp = _mapping_identity(target)
            if mapping_stamp is not None:
                if mapping_stamp != current:
                    raise ValueError("CSR cache changed during validation")
                self._mappings[target] = (current, manifest, arrays)
                while len(self._mappings) > self.max_open_mappings:
                    self._mappings.popitem(last=False)
        # Direct assignment preserves even int64 index arrays without scipy's
        # constructor silently downcasting/copying them into heap allocations.
        return _mapped_matrix(manifest, arrays)

    def _build(self, source, stamp, digest, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = inside(target.parent, target.name + ".lock")
        lock = FileLock(lock_path, timeout=0)
        try:
            lock.acquire()
        except Timeout as error:
            raise RuntimeError(f"CSR cache build locked; retry after builder exits: {lock_path}") from error
        temporary = None
        try:
            if target.exists():
                return
            if time.monotonic() - self._last_progress >= 10:
                LOG.info("Preparing CSR cache (completed %d): %s", self._built, source.name)
                self._last_progress = time.monotonic()
            matrix = sparse.load_npz(source)
            if not sparse.isspmatrix_csr(matrix):
                raise ValueError("Source count matrix is not CSR")
            matrix.check_format(full_check=True)
            temporary = Path(tempfile.mkdtemp(prefix=target.name + ".tmp-", dir=target.parent))
            arrays = {}
            for name in ("data", "indices", "indptr"):
                value = getattr(matrix, name)
                path = temporary / f"{name}.npy"
                with path.open("wb") as stream:
                    np.save(stream, value, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                arrays[name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                                "sha256": sha256(path)}
            manifest = {"schema": SCHEMA, "source_sha256": digest,
                        "shape": list(matrix.shape), "arrays": arrays}
            with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            if file_identity(source) != stamp or sha256(source) != digest:
                raise ValueError("Source changed while building CSR cache")
            temporary.rename(target)
            self._built += 1
            temporary = None
        finally:
            try:
                if temporary is not None:
                    # Only the private directory created by this invocation.
                    shutil.rmtree(inside(target.parent, temporary))
            finally:
                # OS ownership vanishes on process exit; an old lock file does
                # not block restart. Never manually delete another writer's lock.
                lock.release()
