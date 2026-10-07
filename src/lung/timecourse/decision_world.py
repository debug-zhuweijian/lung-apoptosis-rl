"""Single-exposure 72 h matched-plate model environment; no measured reward access."""
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from lung.engine.config import config_hash, file_hash
from lung.engine.scoring import ScalarScoreCache
from .scoring import projected_seed_batch
from lung.rl import (Candidate, CandidateRegistry, DecisionState, NetworkSpec, ProtocolSpec,
                     RootContext, SimulationAdapter, VersionStamp)


@dataclass(frozen=True)
class FixedTimeState:
    condition_id: str
    coordinate_identity: str
    shape: tuple
    expression: bytes


class TimeDecisionWorld:
    def __init__(self, predictor, proxy, ids, board, role, identity, config):
        self.predictor, self.proxy, self.ids = predictor, proxy, tuple(ids)
        self.data, self.board, self.role, self.identity = predictor.data, board, role, identity
        self.device = torch.device(config.get("device", getattr(predictor, "cfg", {}).get("device", "cpu")))
        if not ids:
            raise ValueError("EMPTY_BOARD_CANDIDATES")
        self.n = int(config.get("population", 64))
        width = int(config.get("projection_width", 16))
        rng = np.random.default_rng(int(config.get("projection_seed", 917)))
        self.expression_projection = rng.normal(size=(self.data.n_genes, width)).astype(np.float32) / np.sqrt(self.data.n_genes)
        self._projection = torch.as_tensor(self.expression_projection, device=self.device)
        self._reference_cache = None
        self._score_cache = ScalarScoreCache(config.get("score_cache_capacity", 4096))
        self.score_batch_size = int(config.get("score_batch_size", 4))
        if self.score_batch_size < 1:
            raise ValueError("SCORE_BATCH_SIZE_INVALID")
        drug_projection = rng.normal(size=(2048, width)).astype(np.float32) / np.sqrt(2048)
        spec = NetworkSpec(state_dim=3*width+4, drug_dim=width, dose_dim=2, time_dim=2,
            history_dim=width, history_start=2*width, protocol_dim=2,
            encoder_hidden=(2*width,width), scorer_hidden=(width,), critic_hidden=(2*width,width))
        candidates = []
        for cid in ids:
            meta = self.data.action_features(cid)
            fp = np.asarray(meta["fingerprint"], dtype=np.float32)
            dose = float(meta["dose"])
            features = np.concatenate((fp @ drug_projection, [np.log1p(dose), dose/(1+dose)], [1., 0.]))
            candidates.append(Candidate(str(cid), str(meta["group_id"]), str(dose), "72", 72,
                                        torch.tensor(features, dtype=torch.float32)))
        candidates.append(Candidate("STOP", "STOP", "NONE", "NONE", 0, torch.zeros(spec.action_dim), True))
        self.registry = CandidateRegistry(candidates, spec)
        protocol_id = config_hash({"board": board, "ids": ids, "endpoint":72, "stop":"matched_vehicle"})
        self.protocol = ProtocolSpec(72, "vehicle_reference", frozenset({"endpoint"}),
            {("vehicle_reference", str(cid)): "endpoint" for cid in ids},
            {("vehicle_reference", 0): "matched_vehicle"}, protocol_id)
        versions = VersionStamp(identity, protocol_id, proxy.identity, self.registry.version)
        self.adapter = SimulationAdapter(self.registry, self.protocol, versions, self.observe, self.score,
            lambda state: 0., endpoint_available=True, population_resolution_passed=False, n_pop=self.n,
            evidence={"transfer_status":"unvalidated", "not_apoptosis_rate":True,
                "population_resolution":"D_only_population_audit_required",
                "potential":"explicit_zero_single_step", "root":"matched_72h_vehicle_population",
                "scope":"single_exposure_fixed_72h_proxy", "synthetic_test":config.get("synthetic",False)})
        self.adapter.score_many_fn = self.score_many
        controls = self.data.sample_controls(ids[0], self.n,
            np.random.default_rng(int(config.get("root_seed", 1907))))["control"]
        self.root = self.fixed_root(controls, str(board), role)

    def fixed_root(self, expression, root_id, role):
        x = np.asarray(expression, dtype=np.float32)
        if x.ndim != 2 or not len(x) or x.shape[1] != self.data.n_genes or not np.isfinite(x).all():
            raise ValueError("INVALID_FIXED_TIME_STATE")
        payload = FixedTimeState(self.ids[0], self.data.coordinate_identity, x.shape, x.tobytes())
        identity = config_hash({"input": hashlib.sha256(payload.expression).hexdigest(),
            "shape": payload.shape, "coordinates": payload.coordinate_identity, "role": role})
        return RootContext(f"{root_id}:{identity}", role, payload)

    def reference(self, root, population=None):
        fixed = root.payload
        if not isinstance(fixed, FixedTimeState) or fixed.coordinate_identity != self.data.coordinate_identity:
            raise ValueError("FIXED_TIME_STATE_REQUIRED")
        x = np.frombuffer(fixed.expression, dtype=np.float32).reshape(fixed.shape)
        if population is None or population == len(x):
            return x.copy()
        # Resolution audit repeats the same empirical starting population, not another biological sample.
        if population < 1 or population % len(x):
            raise ValueError("FIXED_POPULATION_REQUIRES_INTEGER_REPLICATION")
        return np.tile(x, (population // len(x), 1))

    def _reference_tensor(self, root, population=None):
        fixed = root.payload
        if not isinstance(fixed, FixedTimeState) or fixed.coordinate_identity != self.data.coordinate_identity:
            raise ValueError("FIXED_TIME_STATE_REQUIRED")
        n = fixed.shape[0] if population is None else population
        if n < 1 or n % fixed.shape[0]:
            raise ValueError("FIXED_POPULATION_REQUIRES_INTEGER_REPLICATION")
        if self._reference_cache is None or self._reference_cache[0] is not fixed:
            self._reference_cache = None
            initial = torch.tensor(self.reference(root), dtype=torch.float32, device=self.device)
            self._reference_cache = (fixed, initial)
        initial = self._reference_cache[1]
        return initial.clone() if n == len(initial) else initial.repeat(n // len(initial), 1)

    def enable_population_audit(self, registration_path, report_path):
        registration = json.loads(Path(registration_path).read_text())
        report = json.loads(Path(report_path).read_text())
        expected = {"environment":self.identity,"proxy":self.proxy.identity,
                    "board":list(self.board),"population":self.n}
        if self.role == "T":
            expected["root_id"] = self.root.root_id
        if (any(registration.get(k) != v for k,v in expected.items()) or
                report.get("registration_sha256") != file_hash(registration_path) or
                report.get("passed") is not True or
                (self.role == "T" and registration.get("candidates") != self.registry.version)):
            raise ValueError("POPULATION_AUDIT_IDENTITY_OR_RESULT_MISMATCH")
        rows = report.get("rows",[])
        if len(rows) != 2 or [r.get("plan") for r in rows] != registration.get("plans"):
            raise ValueError("POPULATION_AUDIT_ROWS_MISMATCH")
        for row in rows:
            means = np.asarray(row.get("means",[]),dtype=float)
            if (means.shape != (3,) or not np.isfinite(means).all() or
                    (means<0).any() or (means>1).any() or
                    row.get("sizes") != registration.get("sizes") or
                    float(means.max()-means.min()) > registration["tolerance"]):
                raise ValueError("POPULATION_AUDIT_NOT_PASSED")
        self.adapter.population_resolution_passed = True
        self.adapter.versions = VersionStamp(config_hash({"environment":self.identity,
            "registration":file_hash(registration_path),"audit":file_hash(report_path)}),
            self.protocol.version,self.proxy.identity,self.registry.version)
        self.adapter.evidence.update(population_resolution="D_only_same_frozen_environment",
            population_audit_sha256=file_hash(report_path),
            E_precision_guarantee=False,ode_convergence_audited=False)

    def observe(self, root, prefix, seed):
        x = self._reference_tensor(root)
        mean = x.mean(0).to(self._projection.dtype) @ self._projection
        sd = x.std(0, correction=0).to(self._projection.dtype) @ self._projection
        features = torch.cat((mean, sd, torch.zeros_like(mean), mean.new_tensor([0.,1.,1.,0.])))
        return DecisionState(features.float(),
            torch.empty(0,self.registry.spec.history_input_dim), (), 0, root.root_id, root.role)

    def score(self, root, sequence, seeds):
        return self.raw_score(root,sequence,seeds,self.n)

    def score_many(self, root, requests):
        return tuple(self.raw_score(root, sequence, seeds, self.n) for sequence, seeds in requests)

    def raw_score(self, root, sequence, seeds, population):
        if len(sequence) != 1:
            raise ValueError("SINGLE_EXPOSURE_ONLY")
        cid = sequence[0]
        if cid != "STOP" and cid not in self.ids:
            raise ValueError("CANDIDATE_ROLE_VIOLATION")
        reference = self._reference_tensor(root, population)
        fixed = root.payload
        steps = tuple(cfg.get("inference_ode_steps") for _, cfg in getattr(self.predictor, "models", ()))
        identity = config_hash({"environment": self.identity, "proxy": self.proxy.identity,
            "calibration": getattr(self.predictor, "calibration", None),
            "gate": getattr(self.predictor, "gate", None),
            "artifacts": getattr(self.predictor, "artifacts", None),
            "score_batch_size": self.score_batch_size,
            "inference_config": [cfg for _, cfg in getattr(self.predictor, "models", ())]})
        keys = [(root.root_id, root.role, hashlib.sha256(fixed.expression).hexdigest(),
                 fixed.coordinate_identity, fixed.shape, self.adapter.versions, identity,
                 self.protocol.version, tuple(sequence), population, steps, int(seed)) for seed in seeds]
        values = [self._score_cache.get(key) for key in keys]
        missing = [i for i, value in enumerate(values) if value is None]
        for start in range(0, len(missing), self.score_batch_size):
            indices = missing[start:start+self.score_batch_size]
            populations = ([reference for _ in indices] if cid == "STOP" else
                projected_seed_batch(self.predictor, cid, reference, [seeds[i] for i in indices]))
            for i, x in zip(indices, populations):
                values[i] = self._score_cache.put(keys[i], self.proxy(x))
        return torch.tensor(values, dtype=torch.float64)
