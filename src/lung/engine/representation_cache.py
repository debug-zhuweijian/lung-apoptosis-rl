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
    return {"weights": _tensor_digest(shared.state_dict()),
            "n_genes": int(config.n_genes), "d_model": int(config.d_model),
            "latent_dim": int(config.latent_dim)}


def _encode_role(data, shared, role, device, batch_size, progress=None):
    """Return (h ndarray, cell_ids, condition_ids) for one reader role.

    Control conditions are included: the CFM/OT reference side draws same-plate
    control cells, so their representations must live in the cache too.
    """
    reader = data.reader
    conditions = reader.condition_ids(role, include_controls=True)
    hs, cell_ids, condition_ids = [], [], []
    for position, condition in enumerate(conditions):
        rows = reader.role_cells(condition, role)
        for start in range(0, len(rows), batch_size):
            chunk = rows.iloc[start:start + batch_size]
            counts = reader.rows.read(chunk)
            expression = reader.normalize(counts)
            with torch.no_grad():
                x = torch.as_tensor(expression, dtype=torch.float32, device=device)
                _, h = shared.encode_h(x)
            hs.append(h.detach().to("cpu", torch.float32).numpy())
            cell_ids.extend(chunk["cell_id"].astype(str).tolist())
            condition_ids.extend([condition] * len(chunk))
        if progress is not None:
            progress(role, position + 1, len(conditions), condition)
    if not hs:
        raise ValueError(f"CACHE_ROLE_EMPTY: {role}")
    return np.concatenate(hs), np.asarray(cell_ids, dtype=str), np.asarray(condition_ids, dtype=str)


def build_representation_cache(data, shared, target, device, *, batch_size=512,
                               roles=("fit", "selection"), progress=None, identity=None):
    """Write shards, then publish the manifest as the completion marker.

    The encoder identity is computed (or accepted) BEFORE the expensive encode
    loop so dependency mismatches fail fast (rework R2).
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    shared.eval()
    shared.requires_grad_(False)
    identity = identity or encoder_identity(shared, shared.config)
    role_blocks = {}
    for role in roles:
        h, cell_ids, condition_ids = _encode_role(data, shared, role, device, batch_size, progress)
        shards = []
        for start in range(0, len(cell_ids), SHARD_CELLS):
            stop = min(start + SHARD_CELLS, len(cell_ids))
            name = f"{role}_shard_{len(shards):04d}.npz"
            path = target / name
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, h=h[start:stop], cell_id=cell_ids[start:stop],
                     condition_id=condition_ids[start:stop])
            temporary.replace(path)
            shards.append({"file": name, "rows": int(stop - start),
                           "sha256": _file_digest(path)})
        role_blocks[role] = {"cells": int(len(cell_ids)),
                             "conditions": int(len(np.unique(condition_ids))),
                             "shards": shards}
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


def _manifest_matches(directory, data_identity, encoder):
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (manifest.get("status") == "complete"
            and manifest.get("data_identity") == data_identity
            and manifest.get("encoder", {}).get("weights") == encoder.get("weights"))


def ensure_representation_cache(data, shared, requested_dir, device, *, batch_size=512,
                                roles=("fit", "selection"), progress=None):
    """Reuse a matching cache, else build into a versioned directory (rework R2).

    A rebuild never reuses the requested directory: it lands in
    ``<requested>-<encoder hash prefix>`` so an old completion marker can never
    be mistaken for evidence about the new cache. Returns (directory, status).
    """
    requested = Path(requested_dir)
    identity = encoder_identity(shared, shared.config)
    if _manifest_matches(requested, data.identity, identity):
        return requested, "reused"
    versioned = requested.parent / f"{requested.name}-{identity['weights'][:12]}"
    if not _manifest_matches(versioned, data.identity, identity):
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
        self.h, self.cell_index, self.condition_ids = {}, {}, {}
        self.condition_pools = {}
        for role, block in manifest["roles"].items():
            parts, ids, conditions = [], [], []
            for shard in block["shards"]:
                path = self.directory / shard["file"]
                if not path.is_file() or _file_digest(path) != shard["sha256"]:
                    raise ValueError(f"CACHE_SHARD_MISMATCH: {shard['file']}")
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
            and manifest.get("encoder", {}).get("weights") != expected_encoder.get("weights")):
        raise ValueError("CACHE_ENCODER_IDENTITY_MISMATCH: rebuild the cache for this encoder")
    return RepresentationCache(manifest, directory, device)
