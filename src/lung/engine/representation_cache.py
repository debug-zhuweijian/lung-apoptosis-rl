"""Frozen h-representation cache (implementation task W3-B2).

Caches h = encode_cells(x)[1] (shared width, default 256, FP32) for every cell
of the requested roles, keyed by stable cell_id and the frozen condition axis.
z = latent_projection(h) is computed on the fly from the frozen projection, so
the cache stores exactly one array per cell and preserves the reference-token
semantics of the shared encoder.

Publishing contract: shard files land first; ``manifest.json`` lands last via
atomic replace and is the ONLY completion marker. Loading verifies every shard
hash plus the data/encoder identities; a mismatch raises instead of reusing.
T and V caches are built through the same frozen encoder; no V-fitted
normalization, encoder or latent scale ever enters this path.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import atomic_json

PROTOCOL = "latent-h-cache-v1"
SHARD_CELLS = 65536


def _tensor_digest(tensors):
    """Layout-aware digest: dense tensors by bytes; sparse COO by canonicalised
    indices/values/shape — the 32461x32461 graph is never densified (rework R2)."""
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(repr(value).encode("utf-8"))
            continue
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        if value.is_sparse:
            coalesced = value.detach().coalesce()
            digest.update(str(coalesced.layout).encode())
            digest.update(np.ascontiguousarray(coalesced.indices().cpu().numpy()).tobytes())
            digest.update(np.ascontiguousarray(
                coalesced.values().detach().cpu().numpy()).tobytes())
        else:
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_identity(shared, config):
    """Identity of the frozen shared module and its shape contract."""
    identity = {"weights": _tensor_digest(shared.state_dict()),
                "n_genes": int(config.n_genes), "d_model": int(config.d_model),
                "latent_dim": int(config.latent_dim)}
    checkpoint_hash = getattr(shared, "representation_hash", None)
    if checkpoint_hash is not None:
        identity["checkpoint_sha256"] = checkpoint_hash
    return identity


def _cache_roles(roles):
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or not set(roles) <= {"fit", "selection"}:
        raise ValueError("CACHE_ROLE_FORBIDDEN: only distinct fit/selection roles are allowed")
    return roles


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _registration(data, identity, roles, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("CACHE_BATCH_SIZE_INVALID")
    conditions = {role: list(data.reader.condition_ids(role, include_controls=True)) for role in roles}
    if any(not keys or len(set(keys)) != len(keys) for keys in conditions.values()):
        raise ValueError("CACHE_ROLE_EMPTY_OR_DUPLICATE_CONDITIONS")
    return {"data_identity": data.identity, "encoder": identity,
            "normalization_signature": data.manifest.get("signature"),
            "batch_size": min(batch_size, SHARD_CELLS), "conditions": conditions}


def _shard_path(directory, name):
    if Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("CACHE_SHARD_PATH_INVALID")
    return Path(directory) / name


def _verify_shard(directory, shard):
    path = _shard_path(directory, shard["file"])
    if not path.is_file() or _file_digest(path) != shard["sha256"]:
        raise ValueError(f"CACHE_SHARD_MISMATCH: {shard['file']}")
    return path


def build_representation_cache(data, shared, target, device, *, batch_size=512,
                               roles=("fit", "selection"), progress=None, identity=None):
    """Build every shard, then publish the complete cache manifest atomically."""
    roles = _cache_roles(roles)
    actual_identity = encoder_identity(shared, shared.config)
    if identity is not None and identity != actual_identity:
        raise ValueError("CACHE_ENCODER_IDENTITY_MISMATCH")
    identity = actual_identity
    registration = _registration(data, identity, roles, batch_size)
    target = Path(target)
    if (target / "manifest.json").exists():
        if not _manifest_matches(target, data.identity, identity, roles,
                                 data.manifest.get("signature")):
            raise ValueError("CACHE_TARGET_IDENTITY_MISMATCH: preserve the existing cache")
        return json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    shared.eval()
    shared.requires_grad_(False)
    reader = data.reader
    role_blocks = {}
    for role in roles:
        shards = []
        conditions = registration["conditions"][role]
        for position, condition in enumerate(conditions):
            rows = reader.role_cells(condition, role)
            if rows.empty:
                raise ValueError(f"CACHE_CONDITION_EMPTY: {role}/{condition}")
            for start in range(0, len(rows), registration["batch_size"]):
                chunk = rows.iloc[start:start + registration["batch_size"]]
                cell_ids = np.asarray(chunk["cell_id"].astype(str).tolist(), dtype=str)
                location = {"file": f"{role}_shard_{len(shards):04d}.npz", "rows": len(chunk),
                            "condition": condition, "start": start,
                            "cell_ids_sha256": _json_digest(cell_ids.tolist())}
                expression = reader.normalize(reader.rows.read(chunk))
                with torch.no_grad():
                    x = torch.as_tensor(expression, dtype=torch.float32, device=device)
                    _, encoded = shared.encode_h(x)
                    h = encoded.detach().to("cpu", torch.float32).numpy()
                if h.shape != (len(chunk), int(shared.config.d_model)) or not np.isfinite(h).all():
                    raise ValueError("CACHE_ENCODED_CHUNK_INVALID")
                path = _shard_path(target, location["file"])
                temporary = path.with_suffix(".tmp.npz")
                with temporary.open("wb") as stream:
                    np.savez(stream, h=h, cell_id=cell_ids,
                             condition_id=np.asarray([condition] * len(chunk), dtype=str))
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
                shards.append({**location, "sha256": _file_digest(path)})
            if progress is not None:
                progress(role, position + 1, len(conditions), condition)
        role_blocks[role] = {"cells": sum(shard["rows"] for shard in shards),
                             "conditions": len(conditions), "shards": shards}
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "status": "complete",
                "data_identity": data.identity,
                "normalization_signature": data.manifest.get("signature"),
                "encoder": identity,
                "h_dim": int(shared.config.d_model),
                "latent_dim": int(shared.config.latent_dim),
                "dtype": "float32", "roles": role_blocks,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%S")}
    atomic_json(target / "manifest.json", manifest)
    return manifest


def _manifest_matches(directory, data_identity, encoder, roles=None, normalization_signature=None):
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (manifest.get("status") == "complete"
            and manifest.get("protocol") == PROTOCOL
            and manifest.get("data_identity") == data_identity
            and manifest.get("encoder") == encoder
            and (roles is None or set(manifest.get("roles", {})) == set(roles))
            and (normalization_signature is None
                 or manifest.get("normalization_signature") == normalization_signature))


def ensure_representation_cache(data, shared, requested_dir, device, *, batch_size=512,
                                roles=("fit", "selection"), progress=None):
    """Reuse a complete matching cache; build a new version for other identities."""
    requested = Path(requested_dir)
    roles = _cache_roles(roles)
    identity = encoder_identity(shared, shared.config)
    registration = _registration(data, identity, roles, batch_size)
    signature = data.manifest.get("signature")
    if _manifest_matches(requested, data.identity, identity, roles, signature):
        return requested, "reused"
    versioned = requested.parent / f"{requested.name}-{_json_digest(registration)[:12]}"
    if _manifest_matches(versioned, data.identity, identity, roles, signature):
        return versioned, "reused"
    build_representation_cache(data, shared, versioned, device, batch_size=batch_size,
                               roles=roles, progress=progress, identity=identity)
    return versioned, "built"


class RepresentationCache:
    """Loaded, identity-verified h cache with per-role cell lookup on device."""

    def __init__(self, manifest, directory, device):
        self.manifest = manifest
        self.directory = Path(directory)
        self.device = torch.device(device)
        if manifest.get("protocol") != PROTOCOL or manifest.get("status") != "complete":
            raise ValueError("CACHE_INCOMPLETE: manifest is not a published completion marker")
        _cache_roles(manifest["roles"])
        self.h, self.cell_index, self.condition_ids = {}, {}, {}
        self.condition_pools = {}
        for role, block in manifest["roles"].items():
            parts, ids, conditions = [], [], []
            for shard in block["shards"]:
                path = _verify_shard(self.directory, shard)
                with np.load(path, allow_pickle=False) as saved:
                    parts.append(saved["h"])
                    ids.append(saved["cell_id"])
                    conditions.append(saved["condition_id"])
            self.h[role] = torch.as_tensor(np.concatenate(parts), dtype=torch.float32,
                                           device=self.device)
            cell_ids = np.concatenate(ids)
            self.condition_ids[role] = np.concatenate(conditions)
            self.cell_index[role] = {str(cell): row for row, cell in enumerate(cell_ids)}
            for condition in np.unique(self.condition_ids[role]):
                self.condition_pools[(role, str(condition))] = np.flatnonzero(
                    self.condition_ids[role] == condition)

    def lookup(self, role, cell_ids):
        rows = [self.cell_index[role][str(cell)] for cell in cell_ids]
        return self.h[role][rows]

    def pool(self, role, condition):
        key = (role, str(condition))
        if key not in self.condition_pools:
            raise ValueError(f"CACHE_CONDITION_MISSING: {role}/{condition}")
        return self.condition_pools[key]

    def draw(self, role, condition, n, rng):
        """Uniform cells with replacement from one condition's frozen pool."""
        pool = self.pool(role, condition)
        picks = rng.integers(len(pool), size=n)
        return self.h[role][pool[picks]], pool[picks]

    def summary(self):
        return {role: {"cells": int(self.h[role].shape[0]),
                       "conditions": len(np.unique(self.condition_ids[role]))}
                for role in self.h}


def load_representation_cache(directory, device, *, expected_data_identity=None,
                              expected_encoder=None):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("CACHE_MANIFEST_MISSING: an unfinished cache is never usable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (expected_data_identity is not None
            and manifest.get("data_identity") != expected_data_identity):
        raise ValueError("CACHE_DATA_IDENTITY_MISMATCH: identity mismatch is never silently reused")
    if (expected_encoder is not None
            and manifest.get("encoder") != expected_encoder):
        raise ValueError("CACHE_ENCODER_IDENTITY_MISMATCH: rebuild the cache for this encoder")
    return RepresentationCache(manifest, directory, device)
