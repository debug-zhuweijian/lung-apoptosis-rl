"""Frozen biological world adapter: public planning populations and independent scores."""

import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lung.models import biological_features, physical_projection
from lung.rl import (Candidate, CandidateRegistry, DecisionState, NetworkSpec,
                     ProtocolSpec, RootContext, SimulationAdapter, VersionStamp, stream_seed)
from .auxiliary import load_auxiliary, encode_vectors, verify_auxiliary_dependencies
from .scoring import ScalarScoreCache
from .checkpoint import load_checkpoint
from .config import StageUnavailable, config_hash, file_hash, project_path, implementation_identity
from .response import condition_kwargs, tensor, projected
from .response_loader import load_response
from .representation import load_representation, verify_response_representation


INITIAL_STATE_VERSION = "fixed-observation-v2"


@dataclass(frozen=True)
class FixedInitialState:
    """Immutable, replayable initial population; expression uses normalized gene order."""

    condition_id: str
    role: str
    data_identity: str
    initial_seed: int
    population_size: int
    expression: bytes | None = None
    version: str = INITIAL_STATE_VERSION

    @property
    def identity(self):
        return config_hash({"condition_id": self.condition_id, "role": self.role,
            "data": self.data_identity, "seed": self.initial_seed,
            "population_size": self.population_size, "version": self.version,
            "expression": hashlib.sha256(self.expression).hexdigest() if self.expression is not None else None})


class BiologicalWorld:
    """Every returned state has only planned summaries; scores replay fixed actions."""

    def __init__(self, data, config, pipeline, *, require_resolution=True):
        self.data, self.config, self.pipeline = data, config, pipeline
        self.device = torch.device(config["device"])
        self._initial_cache = None
        self._score_cache = ScalarScoreCache(config.get("score_cache_capacity", 4096))
        self.score_batch_size = int(config.get("score_batch_size", 4))
        if self.score_batch_size < 1:
            raise ValueError("SCORE_BATCH_SIZE_INVALID")
        required = ("response", "pool", "state")
        missing = [key for key in required if not pipeline.get(key)]
        if missing:
            raise StageUnavailable(f"PIPELINE_INCOMPLETE: train these stages first: {missing}")
        for name in ("pool", "state", "backward", "correction", "uncertainty", "endpoint"):
            if pipeline.get(name):
                verify_auxiliary_dependencies(pipeline[name], pipeline)
        if config["task"] == "ANNEXIN-A549-V1" and not pipeline.get("endpoint"):
            raise StageUnavailable("ENDPOINT_UNAVAILABLE: matched expression/outcome chain is absent")
        self.inference_microbatch = config["response"].get("inference_microbatch", 1)
        if (isinstance(self.inference_microbatch, bool)
                or not isinstance(self.inference_microbatch, int) or self.inference_microbatch < 1):
            raise ValueError("inference_microbatch must be a positive integer.")
        self.members = [load_response(path, data, self.device) for path in pipeline["response"]]
        self.reference = self.members[0]
        self.shape = self.reference.config
        self.representation = (load_representation(pipeline["representation"], data, self.device)
                               if pipeline.get("representation") else self.reference)
        for member in self.members:
            if (member.config.n_genes != self.shape.n_genes
                    or member.config.d_model != self.shape.d_model
                    or member.config.context_dim != self.shape.context_dim):
                raise StageUnavailable("RESPONSE_MEMBER_SHAPE_MISMATCH")
            if hasattr(member, "representation_identity"):
                if not pipeline.get("representation"):
                    raise StageUnavailable("LATENT_REPRESENTATION_REQUIRED")
                verify_response_representation(member, self.representation)
                # A single frozen AE is shared after every member has been verified.
                member.shared = self.representation.shared
                object.__setattr__(member.member, "_shared_ref", self.representation.shared)
        if self.representation.config.d_model != self.shape.d_model:
            raise StageUnavailable("REPRESENTATION_WIDTH_MISMATCH")
        self.pool = load_auxiliary(pipeline["pool"], data, self.device)
        self.classifier = load_auxiliary(pipeline["state"], data, self.device)
        self.backward = load_auxiliary(pipeline["backward"], data, self.device) if pipeline.get("backward") else None
        self.correction = load_auxiliary(pipeline["correction"], data, self.device) if pipeline.get("correction") else None
        self.gate_table = json.loads(Path(pipeline["correction_gate"]).read_text(encoding="utf-8")) if pipeline.get("correction_gate") else None
        if self.gate_table is not None:
            expected_gate = {"data_identity": data.identity,
                "response_hashes": [file_hash(p) for p in pipeline["response"]],
                "pool_hash": file_hash(pipeline["pool"]), "backward_hash": file_hash(pipeline["backward"]),
                "correction_hash": file_hash(pipeline["correction"])}
            if any(self.gate_table.get(key) != value for key, value in expected_gate.items()):
                raise StageUnavailable("CORRECTION_GATE_VERSION_MISMATCH")
        self.n_pop = config["population"]
        self.steps = pipeline.get("inference_ode_steps", config["response"]["inference_ode_steps"])
        self.actions = self._action_metadata()
        self.registry, self.protocol = self._registry()
        biological_keys = ("representation", "response", "pool", "state", "backward", "correction", "correction_gate", "endpoint")
        weights = {key: [file_hash(path) for path in (pipeline[key] if isinstance(pipeline[key], list) else [pipeline[key]])]
                   for key in biological_keys if pipeline.get(key)}
        self.biological_identity = config_hash({"weights": weights, "data": data.identity,
            "implementation": implementation_identity(),
            "population": self.n_pop, "steps": self.steps, "protocol": self.protocol.version,
            "initial_state_interface": INITIAL_STATE_VERSION,
            "initial_state_seed": config.get("seed", 20260923),
            "inference_microbatch": self.inference_microbatch, "score_batch_size": self.score_batch_size,
            "task": config["task"], "history_support": config["history_support"]})
        self.potential = None
        if pipeline.get("potential"):
            from lung.rl import PotentialNetwork
            saved = load_checkpoint(pipeline["potential"])
            if saved["metadata"].get("biological_identity") != self.biological_identity:
                raise StageUnavailable("POTENTIAL_VERSION_MISMATCH")
            self.potential = PotentialNetwork(self.registry.spec).to(self.device)
            self.potential.load_state_dict(saved["model"])
            self.potential.eval().requires_grad_(False)
        self.endpoint = None
        if config["task"] == "ANNEXIN-A549-V1":
            self.endpoint = load_auxiliary(pipeline["endpoint"], data, self.device)
        resolution_passed = bool(config.get("synthetic", False))
        if pipeline.get("numerical_audit"):
            audit = json.loads(Path(pipeline["numerical_audit"]).read_text(encoding="utf-8"))
            resolution_passed = audit.get("passed") is True and audit.get("biological_identity") == self.biological_identity
            if not resolution_passed and pipeline.get("freeze_certificate"):
                certificate = json.loads(Path(pipeline["freeze_certificate"]).read_text(encoding="utf-8"))
                resolution_passed = (
                    audit.get("passed") is True
                    and certificate.get("audit_sha256") == file_hash(pipeline["numerical_audit"])
                    and certificate.get("precalibration_identity") == audit.get("biological_identity")
                    and certificate.get("frozen_identity") == self.biological_identity
                    and certificate.get("calibration_after_diagnostic") is True)
        if require_resolution and not resolution_passed:
            raise StageUnavailable("POPULATION_RESOLUTION_UNRESOLVED: run the potential stage to perform the V-only numerical audit first")
        version = VersionStamp(self.biological_identity, self.protocol.version,
                               config_hash({"task": config["task"], "weights": weights.get("state"),
                                            "endpoint": weights.get("endpoint"),
                                            "potential": file_hash(pipeline["potential"]) if pipeline.get("potential") else None}),
                               self.registry.version)
        self.adapter = SimulationAdapter(self.registry, self.protocol, version,
            self.observe, self.score, self.phi if self.potential is not None else None,
            task_id=config["task"], endpoint_available=True,
            population_resolution_passed=resolution_passed, n_pop=self.n_pop,
            evidence={"synthetic_test": config.get("synthetic", False),
                      "transfer_status": "unvalidated", "history_support": config["history_support"],
                      "coupling_assumption": "minibatch_entropic_ot_prior",
                      "initial_state_interface": INITIAL_STATE_VERSION,
                      "knowledge_assets": "missing_masked", "apoptosis_rate": None})

        self.adapter.score_many_fn = self.score_many

    def _action_metadata(self):
        if self.data.synthetic:
            ids = self.data.condition_ids("T")
        else:
            rows = self.data.reader.conditions
            ids = sorted(rows.loc[rows.eligible & rows.condition_type.eq("treated"), "condition_id"])
        actions = {}
        for condition_id in ids:
            value = self.data.action_features(condition_id)
            key = (value["group_id"], str(value["dose"]), str(value["duration"]))
            action_id = "a-" + config_hash(key)[:24]
            actions.setdefault(action_id, value)
        if not actions:
            raise StageUnavailable("ACTION_SUPPORT_EMPTY")
        return actions

    @torch.no_grad()
    def _registry(self):
        from lung.rl import exact_time_ticks
        unit, ticks = exact_time_ticks([str(item["duration"]) for item in self.actions.values()])
        s = self.shape
        spec = NetworkSpec(state_dim=s.state_dim, drug_dim=s.d_model, dose_dim=s.context_dim,
                           time_dim=s.context_dim, history_dim=s.d_model, history_start=s.d_model+27,
                           protocol_dim=s.context_dim,
                           encoder_hidden=(2*s.d_model, s.d_model),
                           scorer_hidden=(s.d_model, max(1, s.d_model//2)),
                           critic_hidden=(2*s.d_model, s.d_model))
        candidates = []
        for (action_id, batch), duration in zip(self.actions.items(), ticks):
            encoding = self.reference.condition_encoder
            drug = encoding.drug(tensor(batch["fingerprint"], self.device)[None])[0]
            dose = encoding.dose(tensor([batch["dose"]], self.device))[0]
            time = encoding.duration(tensor([batch["duration"]], self.device))[0]
            candidates.append(Candidate(action_id, batch["group_id"], str(batch["dose"]),
                                        str(batch["duration"]), duration, torch.cat((drug, dose, time))))
        candidates.append(Candidate("STOP", "STOP", "NONE", "NONE", 0,
                                    torch.zeros(spec.action_dim), is_stop=True))
        registry = CandidateRegistry(candidates, spec)
        if self.config.get("protocol"):
            manifest = json.loads(project_path(self.config["protocol"]).read_text(encoding="utf-8"))
            if self.config["history_support"] != "model_assumption":
                raise StageUnavailable("UNSUPPORTED_HISTORY_TRANSITION: current predictor lacks measured sequence history")
            if manifest.get("semantics") != "complete_replacement_no_carryover_assumption":
                raise StageUnavailable("UNSUPPORTED_PROTOCOL: register explicit complete-replacement assumption for V1")
            if manifest["ticks_per_hour"] != unit:
                raise StageUnavailable("PROTOCOL_TICK_MISMATCH")
            self.tail_rules = manifest.get("tail_rules", {})
            protocol = ProtocolSpec(int(manifest["horizon_ticks"]), manifest["initial_state"],
                frozenset(manifest["terminal_states"]),
                {(row["from"], row["action_id"]): row["to"] for row in manifest["transitions"]},
                {(row["state"], int(row["elapsed_ticks"])): row["rule_id"] for row in manifest.get("stop_tails", [])},
                config_hash(manifest), unit)
            for rule in protocol.stop_tails.values():
                if not self.tail_rules.get(rule) or any(a not in self.actions for a in self.tail_rules[rule]):
                    raise StageUnavailable("PROTOCOL_INCOMPLETE: tail must be an explicit supported action sequence")
            for (state, ticks), rule in protocol.stop_tails.items():
                # Tails follow the same switch machine as ordinary exposures.
                suffix = ProtocolSpec(protocol.horizon_ticks - ticks, state,
                    protocol.terminal_states, protocol.transitions, {}, protocol.version,
                    protocol.ticks_per_hour)
                suffix.validate_complete(registry, self.tail_rules[rule])
        else:
            if any(a.duration_ticks != 24*unit for a in candidates if not a.is_stop):
                raise StageUnavailable("PROTOCOL_INCOMPLETE: non-24h data need an explicit registered protocol")
            self.tail_rules = {}
            manifest = {"name": "single-exposure-24h", "horizon": 24, "stop_tail": None}
            protocol = ProtocolSpec(24*unit, "untreated", frozenset({"sampled"}),
                {("untreated", a.action_id): "sampled" for a in candidates if not a.is_stop},
                {}, config_hash(manifest), unit)
        return registry, protocol

    def roots(self, role):
        return [self.root(key, role) for key in self.data.condition_ids(role)]

    def root(self, condition_id, role, *, expression=None, gene_ids=None, data_identity=None):
        """Freeze a reference draw, or supplied normalized expression (no resampling).

        Explicit rows must use this data version's complete, ordered gene axis and
        train-fitted normalization. Too few rows for a requested population is an
        error; cells are never duplicated to manufacture additional observations.
        """
        if condition_id not in self.data.condition_ids(role):
            raise ValueError("ROOT_ROLE_MISMATCH")
        seed = stream_seed(self.config.get("seed", 20260923), INITIAL_STATE_VERSION,
                           self.data.identity, role, str(condition_id))
        count = max(self.n_pop, 16 if self.data.synthetic else 2048)
        frozen = None
        if expression is not None:
            expected_genes = (tuple(str(i) for i in range(self.shape.n_genes)) if self.data.synthetic
                              else tuple(self.data.reader.gene_ids))
            if data_identity != self.data.identity or gene_ids is None or tuple(gene_ids) != expected_genes:
                raise ValueError("ROOT_DATA_OR_GENE_AXIS_MISMATCH")
            values = np.asarray(expression)
            if (values.ndim != 2 or values.shape[1] != self.shape.n_genes
                    or values.shape[0] < self.n_pop or values.dtype.kind not in "fiu"
                    or not np.isfinite(values).all()):
                raise ValueError("ROOT_EXPRESSION_INVALID")
            with np.errstate(over="ignore"):
                values = values.astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("ROOT_EXPRESSION_INVALID")
            count, frozen = len(values), values.tobytes()
        elif gene_ids is not None or data_identity is not None:
            raise ValueError("ROOT_EXPRESSION_REQUIRED")
        payload = FixedInitialState(str(condition_id), role, self.data.identity, seed, count, frozen)
        return RootContext(f"{condition_id}:{payload.identity}", role, payload)

    def _expanded(self, sequence):
        prefix = ()
        for action in sequence:
            if action == "STOP":
                state, ticks, _ = self.protocol.position(self.registry, prefix)
                tail = tuple(self.tail_rules[self.protocol.stop_tails[state, ticks]])
                if sum(self.registry[a].duration_ticks for a in tail) != self.protocol.horizon_ticks - ticks:
                    raise StageUnavailable("PROTOCOL_INCOMPLETE: STOP tail does not reach the endpoint")
                self.protocol.validate_complete(self.registry, sequence)
                expanded = (*prefix, *tail)
                self.protocol.validate_complete(self.registry, expanded)
                return expanded
            prefix = (*prefix, action)
        return prefix

    @torch.no_grad()
    def _initial(self, root, n):
        payload = root.payload
        if (not isinstance(payload, FixedInitialState) or payload.version != INITIAL_STATE_VERSION
                or payload.data_identity != self.data.identity or payload.role != root.role
                or root.root_id != f"{payload.condition_id}:{payload.identity}"):
            raise ValueError("ROOT_FIXED_STATE_IDENTITY_MISMATCH")
        if isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= payload.population_size:
            raise ValueError("ROOT_POPULATION_SIZE_UNAVAILABLE")
        cached = self._initial_cache
        if cached is None or cached[0] != root.root_id:
            # Keep only the most recent root, including its full nested V population.
            self._initial_cache = None
            cached = None
            batch = self.data.sample_controls(payload.condition_id, payload.population_size,
                                              np.random.default_rng(payload.initial_seed), role=root.role)
            values = batch.pop("control")
            if payload.expression is not None:
                values = np.frombuffer(payload.expression, dtype=np.float32).reshape(
                    payload.population_size, self.shape.n_genes)
                batch = {**batch, "observation_role": "explicit_fixed_expression"}
                batch.pop("control_cell_ids", None)
            cached = (root.root_id, tensor(values.copy(), self.device), deepcopy(batch))
            self._initial_cache = cached
        # Neither model code nor callers may mutate the cached expression or metadata.
        return cached[1][:n].clone(), deepcopy(cached[2])

    @torch.no_grad()
    def _generate_member(self, model, x, batch, seed, step, member, elapsed, n, ode_steps):
        if hasattr(model, "condition_reference"):
            x = model.condition_reference(x)
        output = []
        for start in range(0, n, self.inference_microbatch):
            end = min(start + self.inference_microbatch, n)
            noises = []
            for index in range(start, end):
                generator = torch.Generator(device=self.device).manual_seed(stream_seed(seed, "noise", step, member, index))
                noises.append(model.sample_noise(x[index:index+1], generator=generator)
                              if hasattr(model, "sample_noise") else
                              torch.randn((1, self.shape.n_genes), generator=generator, device=self.device))
            noise = torch.cat(noises)
            ca = model.encode_condition(**condition_kwargs(batch, end - start, self.device, elapsed,
                        self.protocol.horizon_ticks / self.protocol.ticks_per_hour))
            output.append(model.sample(x[start:end], ca, noise=noise, steps=ode_steps, checkpoint=False))
        return torch.cat(output)

    @torch.no_grad()
    def propagate(self, root, prefix, seed, *, n=None, scoring_member=None, ode_steps=None):
        n, ode_steps = self.n_pop if n is None else n, ode_steps or self.steps
        initial, context = self._initial(root, n)
        populations = [initial for _ in self.members]
        elapsed = 0.
        for step, action_id in enumerate(self._expanded(prefix)):
            batch = {**self.actions[action_id], **{key: context[key] for key in ("source_id", "background_id", "protocol_id")}}
            prior = populations if scoring_member is None else [populations[scoring_member] for _ in self.members]
            raw = [self._generate_member(model, prior[i], batch, seed, step, i, elapsed, n, ode_steps)
                   for i, model in enumerate(self.members)]
            temporary = [projected(value, self.data) for value in raw]
            variance = torch.stack([value.mean(0) for value in temporary]).var(0, correction=0).mean()
            if self.backward is not None and self.correction is not None:
                vectors = encode_vectors(self.representation, torch.cat(temporary), self.device,
                                         batch_size=self.inference_microbatch, output_device=self.device)
                zb = self.pool(vectors)
                ca = self.reference.encode_condition(**condition_kwargs(batch, 1, self.device, elapsed,
                       self.protocol.horizon_ticks / self.protocol.ticks_per_hour))[0]
                reference_mean = torch.stack([value.mean(0) for value in prior]).mean(0)
                e_b = (self.backward(zb, ca) - reference_mean).square().mean()
                gate = self._gate(batch)
                raw = [self.correction(value, zb, ca, e_b, variance, gate=gate) for value in raw]
            populations = [projected(value, self.data) for value in raw]
            elapsed += batch["duration"]
        result = torch.cat(populations) if scoring_member is None else populations[scoring_member]
        return result, context

    def _gate(self, context):
        if self.gate_table is None:
            return False
        # Domain table interpretation is centralized in the residual stage.
        from .residuals import gate_for
        return gate_for(self.gate_table, context)

    @torch.no_grad()
    def _features(self, expression, context, prefix):
        vectors = encode_vectors(self.representation, expression, self.device,
                                 batch_size=self.inference_microbatch, output_device=self.device)
        pooled = self.pool(vectors)
        log_expression = physical_projection(expression, tensor(self.data.mu, self.device), tensor(self.data.scale, self.device)).y
        bio, _ = biological_features(log_expression, [None, None, None], [0, 0, 0])
        ce, d, c = self.reference.condition_encoder, self.shape.d_model, self.shape.context_dim
        backgrounds = [ce.background, ce.source, ce.protocol]
        embeddings = [module(torch.tensor(context[key], device=self.device))
                      for module, key in zip(backgrounds, ("background_id", "source_id", "protocol_id"))]
        elapsed, rows = 0, []
        for action_id in prefix:
            action = self.registry[action_id]
            if action.is_stop:
                raise ValueError("No policy state is constructed after STOP")
            rows.append(torch.cat((action.embedding.to(self.device),
                                   pooled.new_tensor([elapsed / self.protocol.horizon_ticks]), embeddings[2])))
            elapsed += action.duration_ticks
        features = torch.cat((pooled, bio, pooled.new_zeros(d), *embeddings,
                              pooled.new_tensor([elapsed/self.protocol.horizon_ticks, 1-elapsed/self.protocol.horizon_ticks])))
        history = torch.stack(rows) if rows else pooled.new_empty((0, self.registry.spec.history_input_dim))
        return features, history, elapsed

    @torch.no_grad()
    def observe(self, root, prefix, seed):
        expression, context = self.propagate(root, prefix, seed)
        features, history, elapsed = self._features(expression, context, prefix)
        return DecisionState(features.cpu(), history.cpu(), prefix, elapsed, root.root_id, root.role)

    @torch.no_grad()
    def score_one(self, root, sequence, seed, *, n=None, ode_steps=None):
        member = int(np.random.default_rng(stream_seed(seed, "member")).integers(len(self.members)))
        x, context = self.propagate(root, sequence, seed, n=n, scoring_member=member, ode_steps=ode_steps)
        z = encode_vectors(self.representation, x, self.device,
                           batch_size=self.inference_microbatch, output_device=self.device)
        if self.endpoint is None:
            return float(self.classifier.population_proxy(z))
        features, history, elapsed = self._features(x, context, self._expanded(sequence))
        # Endpoint module owns its history encoder; it never borrows Actor states.
        return float(self.endpoint(features[None], history[None], torch.tensor([len(history)], device=self.device))[0])

    @torch.no_grad()
    def _generate_requests(self, model, inputs, batch, seeds, step, member, elapsed, n, ode_steps):
        # Population-mean ablation is reapplied inside sample: combining populations
        # would change it. Preserve the exact scalar implementation for this mode.
        if getattr(model.config, "reference_mode", "cell") == "population_mean":
            return [self._generate_member(model, x, batch, seed, step, member, elapsed, n, ode_steps)
                    for x, seed in zip(inputs, seeds)]
        references = [model.condition_reference(x) if hasattr(model, "condition_reference") else x
                      for x in inputs]
        results = [[] for _ in seeds]
        for start in range(0, n, self.inference_microbatch):
            end = min(start + self.inference_microbatch, n)
            noises = []
            for x, seed in zip(references, seeds):
                for index in range(start, end):
                    generator = torch.Generator(device=self.device).manual_seed(
                        stream_seed(seed, "noise", step, member, index))
                    noises.append(model.sample_noise(x[index:index+1], generator=generator)
                                  if hasattr(model, "sample_noise") else torch.randn(
                                      (1, self.shape.n_genes), device=self.device, generator=generator))
            ca = model.encode_condition(**condition_kwargs(batch, (end-start)*len(seeds),
                self.device, elapsed, self.protocol.horizon_ticks / self.protocol.ticks_per_hour))
            raw = model.sample(torch.cat([x[start:end] for x in references]), ca,
                               noise=torch.cat(noises), steps=ode_steps, checkpoint=False)
            for pieces, value in zip(results, raw.split(end-start)):
                pieces.append(value)
        return [torch.cat(pieces) for pieces in results]

    @torch.no_grad()
    def _score_seed_batch(self, root, sequence, seeds, n, ode_steps):
        initial, context = self._initial(root, n)
        selected = [int(np.random.default_rng(stream_seed(seed, "member")).integers(len(self.members)))
                    for seed in seeds]
        populations = [[initial for _ in self.members] for _ in seeds]
        elapsed = 0.
        for step, action_id in enumerate(self._expanded(sequence)):
            batch = {**self.actions[action_id], **{key: context[key] for key in
                     ("source_id", "background_id", "protocol_id")}}
            prior = [[values[chosen] for _ in self.members] for values, chosen in zip(populations, selected)]
            by_member = [self._generate_requests(model, [p[i] for p in prior], batch,
                         seeds, step, i, elapsed, n, ode_steps) for i, model in enumerate(self.members)]
            next_populations = []
            for request, previous in enumerate(prior):
                raw = [rows[request] for rows in by_member]
                temporary = [projected(value, self.data) for value in raw]
                variance = torch.stack([value.mean(0) for value in temporary]).var(0, correction=0).mean()
                if self.backward is not None and self.correction is not None:
                    vectors = encode_vectors(self.representation, torch.cat(temporary), self.device,
                        batch_size=self.inference_microbatch, output_device=self.device)
                    zb = self.pool(vectors)
                    ca = self.reference.encode_condition(**condition_kwargs(batch, 1, self.device,
                        elapsed, self.protocol.horizon_ticks / self.protocol.ticks_per_hour))[0]
                    reference_mean = torch.stack([value.mean(0) for value in previous]).mean(0)
                    e_b = (self.backward(zb, ca) - reference_mean).square().mean()
                    raw = [self.correction(value, zb, ca, e_b, variance, gate=self._gate(batch)) for value in raw]
                next_populations.append([projected(value, self.data) for value in raw])
            populations = next_populations
            elapsed += batch["duration"]
        scores = []
        for values, chosen in zip(populations, selected):
            x = values[chosen]
            if self.endpoint is None:
                z = encode_vectors(self.representation, x, self.device,
                    batch_size=self.inference_microbatch, output_device=self.device)
                scores.append(float(self.classifier.population_proxy(z)))
            else:
                features, history, _ = self._features(x, context, self._expanded(sequence))
                scores.append(float(self.endpoint(features[None], history[None],
                    torch.tensor([len(history)], device=self.device))[0]))
        return scores

    def score(self, root, sequence, seeds):
        return self.score_many(root, ((sequence, seeds),))[0]

    def score_many(self, root, requests):
        # Validate the immutable root even when all requested scalars are cached.
        self._initial(root, self.n_pop)
        result = []
        for sequence, seeds in requests:
            keys = [(root.root_id, root.role, root.payload.identity, self.adapter.versions,
                     self.biological_identity, self.protocol.version, tuple(sequence),
                     self.n_pop, self.steps, int(seed)) for seed in seeds]
            values = [self._score_cache.get(key) for key in keys]
            missing = [i for i, value in enumerate(values) if value is None]
            for start in range(0, len(missing), self.score_batch_size):
                indices = missing[start:start+self.score_batch_size]
                scores = self._score_seed_batch(root, sequence, [seeds[i] for i in indices], self.n_pop, self.steps)
                for i, value in zip(indices, scores):
                    values[i] = self._score_cache.put(keys[i], value)
            result.append(torch.tensor(values, dtype=torch.float64))
        return tuple(result)

    def phi(self, state):
        if self.potential is None:
            raise StageUnavailable("POTENTIAL_UNAVAILABLE")
        with torch.no_grad():
            return float(self.potential([state])[0])
