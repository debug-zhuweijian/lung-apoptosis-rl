"""A permission boundary over independent endpoint populations."""
import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from lung.data.preprocess import standardized
from lung.data.common import inside
from lung.data.reader import SparseRows
from lung.engine.config import config_hash, file_hash
from .protocol import GROUPS, VEHICLE
from .preparation import source_paths

KEYS = ["catalog_number", "time_point", "dose_uM", "replicate", "culture_plate"]


class TimeData:
    def __init__(self, root, config, fit_drugs, allowed_target_drugs, role):
        self.root, self.config = Path(root).resolve(), config
        _, self.assets_dir = source_paths(root, config)
        self.manifest = json.loads((self.assets_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["status"] != "complete":
            raise ValueError("TIME_DATA_INCOMPLETE")
        for name, digest in self.manifest["files"].items():
            if file_hash(self.assets_dir / name) != digest:
                raise ValueError("TIME_ASSET_HASH_MISMATCH")
        self.synthetic = False
        self.identity = self.manifest["identity"]
        self.coordinate_identity = self.manifest["coordinate_identity"]
        self.genes = pd.read_parquet(self.assets_dir / "genes.parquet").ensembl_id.to_numpy()
        self._cells = pd.read_parquet(self.assets_dir / "cells.parquet")
        self._fingerprints = json.loads((self.assets_dir / "fingerprints.json").read_text())
        with np.load(self.assets_dir / "preprocessing.npz") as saved:
            self.mu, self.scale, self.neighbors = saved["mu"], saved["scale"], saved["neighbors"]
            self.bandwidth = float(saved["bandwidth"])
        self.n_genes = len(self.genes)
        self._sparse_rows = None
        options = config.get("data_loading", {}).get("row_cache", {}) or {}
        if options.get("use_disk_cache", True):
            source = inside(self.root, self.manifest["source"])
            registry, self._matrix_paths = {}, {}
            for shard_id, entry in self.manifest["shards"].items():
                path = inside(source, entry["file"])
                name = path.relative_to(self.root).as_posix()
                stat = path.stat()
                registry[name] = {"csr_bytes": stat.st_size, "csr_mtime_ns": stat.st_mtime_ns,
                                  "csr_sha256": entry["sha256"], "source_rows": entry["rows"],
                                  "source_columns": self.n_genes}
                self._matrix_paths[int(shard_id)] = name
            self._sparse_rows = SparseRows(self.root, np.arange(self.n_genes),
                                          registry=registry, **options)
        self.drug_groups = copy.deepcopy(GROUPS)
        self._initialize(fit_drugs, allowed_target_drugs, role)

    def _initialize(self, fit_drugs, allowed_target_drugs, role):
        registered = set(sum(self.drug_groups.values(), []))
        self.fit_drugs = frozenset(fit_drugs)
        self.allowed_target_drugs = frozenset(allowed_target_drugs)
        self.role = role
        if not self.fit_drugs <= set(self.drug_groups["D"]) or not self.allowed_target_drugs <= registered:
            raise PermissionError("UNREGISTERED_OR_NONDEVELOPMENT_FIT_DRUG")
        if not self.fit_drugs <= self.allowed_target_drugs:
            raise PermissionError("FIT_DRUGS_REQUIRE_TARGET_PERMISSION")
        self.view_identity = config_hash({"data": self.identity, "fit": sorted(self.fit_drugs),
            "targets": sorted(self.allowed_target_drugs), "role": role})
        selected = self._cells.loc[self._cells.catalog_number.isin(registered | {VEHICLE}) & self._cells.time_point.isin([24, 72])]
        self._rows, records = {}, []
        for key, rows in selected.groupby(KEYS, sort=True, dropna=False):
            drug, time, dose, replicate, plate = key
            cid = config_hash(list(key))[:24]
            valid = rows.loc[rows.author_hash_qc.eq(True)]
            self._rows[cid] = valid
            fingerprint = np.asarray(self._fingerprints[drug], dtype=np.float32)
            if fingerprint.shape != (2048,) or not np.isfinite(fingerprint).all():
                raise ValueError("FINGERPRINT_MISMATCH")
            records.append(dict(condition_id=cid, catalog_number=drug, group_id=drug,
                time_point=float(time), duration=float(time), horizon=float(time), dose_uM=float(dose), dose=float(dose),
                replicate=str(replicate), culture_plate=str(plate), context=f"{time}:{replicate}:{plate}",
                source="sciPlex3", source_id=1, background_id=0, protocol_id=0,
                is_vehicle=drug == VEHICLE, fingerprint=fingerprint, n_cells=len(valid),
                status="empty" if len(valid) == 0 else "one_cell" if len(valid) == 1 else "available"))
        self.conditions = pd.DataFrame(records).set_index("condition_id", drop=False)
        reference = {row["context"]: row["condition_id"] for row in records if row["is_vehicle"]}
        self.conditions["reference_id"] = [reference.get(row["context"]) for row in records]
        self.conditions["reference_n"] = [len(self._rows.get(cid, [])) for cid in self.conditions.reference_id]
        self.conditions.loc[self.conditions.reference_n.eq(0), "status"] = "missing_matched_vehicle"

    def view(self, fit_drugs, allowed_target_drugs, role):
        result = copy.copy(self)
        result._initialize(fit_drugs, allowed_target_drugs, role)
        return result

    def training_view(self, *, known24_only=False, validation_fraction=0.):
        if not 0 <= validation_fraction < 1:
            raise ValueError("VALIDATION_FRACTION_RANGE")
        result = self.view(self.fit_drugs, self.allowed_target_drugs, self.role)
        result.known24_only = bool(known24_only)
        result.validation_fraction = float(validation_fraction)
        result.view_identity = config_hash({"base": self.view_identity, "known24_only": known24_only,
            "validation_fraction": validation_fraction, "split": "condition_row_tail_v1"})
        return result

    def condition_ids(self, role=None, drugs=None, time=72):
        frame = self.conditions
        if role in ("T", "V"):
            selected = (frame.time_point.eq(24) | frame.catalog_number.isin(self.fit_drugs))
            selected &= frame.n_cells.gt(0) & frame.reference_n.gt(0)
            if getattr(self, "known24_only", False):
                selected &= frame.time_point.eq(24)
            if role == "V":
                selected &= frame.n_cells.ge(2) & frame.reference_n.ge(2)
        else:
            selected = frame.time_point.eq(time) & frame.catalog_number.isin(self.allowed_target_drugs if drugs is None else drugs)
        return frame.index[selected].tolist()

    def iter_response_epoch(self, role, seed):
        keys = self.condition_ids(role)
        return np.random.default_rng(seed).permutation(keys).tolist()

    def action_features(self, condition_id):
        row = self.conditions.loc[condition_id].to_dict()
        row["fingerprint"] = np.asarray(row["fingerprint"], dtype=np.float32).copy()
        return row

    def _permit(self, condition_id, role):
        row = self.conditions.loc[condition_id]
        if getattr(self, "known24_only", False) and row.time_point != 24:
            raise PermissionError("PRETRAIN24_ONLY")
        if row.time_point == 72 and not row.is_vehicle:
            permitted = self.fit_drugs if role in ("T", "V") else self.allowed_target_drugs
            if row.catalog_number not in permitted:
                raise PermissionError("TARGET72_ACCESS_DENIED: " + row.catalog_number)

    @lru_cache(maxsize=3)
    def _shard(self, shard_id):
        entry = self.manifest["shards"][str(int(shard_id))]
        path = inside(inside(self.root, self.manifest["source"]), entry["file"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("SOURCE_SHARD_HASH_MISMATCH")
        matrix = sparse.load_npz(path)
        if matrix.shape != (entry["rows"], self.n_genes):
            raise ValueError("SOURCE_SHARD_SHAPE_MISMATCH")
        return matrix

    def _read(self, condition_id, n, rng, role=None):
        rows = self._rows[condition_id]
        fraction = getattr(self, "validation_fraction", 0.)
        if fraction and role in ("T", "V"):
            holdout = max(1, int(len(rows) * fraction)) if len(rows) >= 2 else 0
            rows = rows.iloc[len(rows)-holdout:] if role == "V" and holdout else (rows.iloc[:len(rows)-holdout] if role == "T" else rows.iloc[:0])
        if len(rows) == 0:
            raise ValueError("EMPTY_POPULATION")
        if n is not None:
            if n < 1:
                raise ValueError("SAMPLE_COUNT_POSITIVE")
            rows = rows.iloc[rng.choice(len(rows), n, replace=n > len(rows))]
        if self._sparse_rows is not None:
            matrix_rows = pd.DataFrame({
                "matrix_path": [self._matrix_paths[int(shard)] for shard in rows.shard_id],
                "matrix_row": rows.row_in_shard.to_numpy(int)})
            matrix = self._sparse_rows.read(matrix_rows)
        else:
            ordered = rows.reset_index(drop=True)
            pieces, positions = [], []
            for shard, part in ordered.groupby("shard_id", sort=False):
                pieces.append(self._shard(int(shard))[part.row_in_shard.to_numpy(int)])
                positions.extend(part.index.tolist())
            matrix = sparse.vstack(pieces).tocsr()[np.argsort(positions)]
        return standardized(matrix, self.mu, self.scale)

    def sample_target(self, condition_id, n=None, rng=None, role=None):
        self._permit(condition_id, role)
        return self._read(condition_id, n, rng if rng is not None else np.random.default_rng(0), role)

    def sample_controls(self, condition_id, n, rng, role=None):
        row = self.action_features(condition_id)
        if row["reference_n"] == 0:
            raise ValueError("MISSING_MATCHED_VEHICLE")
        return {**row, "control": self._read(row["reference_id"], n, rng, role)}

    def sample_response(self, condition_id, n, rng, role=None):
        self._permit(condition_id, role)
        return {**self.sample_controls(condition_id, n, rng, role),
                "target": self.sample_target(condition_id, n, rng, role)}

    def canonical_features(self, samples):
        values = np.asarray(samples)
        if values.ndim != 2 or values.shape[1] != self.n_genes or not len(values):
            raise ValueError("CANONICAL_FEATURE_AXIS_MISMATCH")
        return values.mean(axis=0)
