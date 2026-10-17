"""Staged latent pipeline orchestration (implementation task W3; rework R3/R4/R5/R6).

B1 ``latent_ae``   pretrain the shared autoencoder on T only (dual-end MSE),
                   one optimizer step per update group (conditions_per_update).
B2 ``latent_cache`` freeze encoder+projection; ``ensure_representation_cache``
                   reuses a matching cache or builds a versioned directory.
B3 ``latent_cfm``  latent CFM on cached h; the main phase never solves ODEs.
                   ``coupling_space``: "latent" solves OT on frozen-projected z
                   of BOTH sides and pairs h/z by the coupling (rework R3);
                   "gene" solves OT on drawn full-gene cells and looks h up by
                   cell_id. Random pairing is NOT offered under either name.
B4 refinement      epoch-switched: member decoder (shared-initialised, frozen
                   during CFM) becomes trainable; decoded full-gene endpoint
                   MMD per condition via a differentiable latent Heun pass.

Every training call initializes a new model and optimizer. Best/final model
exports support downstream stages and inference, without training recovery state.
The entry-level numeric policy is re-applied at every seeding boundary.
"""

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lung.models import (ModelConfig, entropic_ot, rbf_mmd, sample_ot_pairs)
from ..models.flow import heun_integrate
from ..models.latent_staged import LatentMember, SharedAutoencoder
from .ae_data import AEDataSupply
from .checkpoint import atomic_save, load_checkpoint, finalize_training
from .config import config_hash
from .numerics import PrecisionPolicy, apply_seed, policy_autocast
from .representation_cache import (ensure_representation_cache, encoder_identity,
                                   load_representation_cache)
from .response import condition_kwargs, projected, tensor
from .response_training import batch_tensor

LATENT_RNG_POLICY = "latent_cfm_numpy_torch_separate_v1"


def latent_settings(config):
    settings = config.get("research", {}).get("latent", {})
    if not isinstance(settings, dict):
        raise ValueError("research.latent must be a mapping")
    coupling = settings.get("coupling_space", "latent")
    if coupling not in ("latent", "gene"):
        raise ValueError("research.latent.coupling_space must be 'latent' or 'gene'")
    refine = settings.get("refine", {})
    if not isinstance(refine, dict):
        raise ValueError("research.latent.refine must be a mapping")
    return {"d_model": int(settings.get("d_model", 256)),
            "latent_dim": int(settings.get("latent_dim", 256)),
            "coupling_space": coupling, "refine": refine,
            "ae": settings.get("ae", {}), "cache_dir": settings.get("cache_dir"),
            "ae_checkpoint": settings.get("ae_checkpoint")}


def _model_config(config, settings, n_genes):
    return ModelConfig(n_genes=n_genes, d_model=settings["d_model"],
                       latent_dim=settings["latent_dim"], backbone="compact_full_gene")


def _paths(monitor, config):
    root = Path(monitor.weights)
    settings = latent_settings(config)
    cache = Path(settings["cache_dir"]) if settings["cache_dir"] else root / "latent_cache"
    return {"ae": root / "latent_ae", "cache": cache, "cfm": root / "latent_cfm"}


def _resolve_ae_checkpoint(config, monitor):
    explicit = latent_settings(config)["ae_checkpoint"]
    path = Path(explicit) if explicit else _paths(monitor, config)["ae"] / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"LATENT_AE_REQUIRED: set research.latent.ae_checkpoint or train AE in this run: {path}")
    return path


def control_ids(data, condition):
    frame = data.reader.controls
    return tuple(frame.loc[frame["condition_id"].eq(condition), "control_condition_id"])


def draw_control_h(data, cache, condition, n, rng, role="fit"):
    """Mirror reader._control_rows exactly: uniform control condition draw, then
    uniform cell within the drawn pool, original batch order preserved (R3)."""
    pools = [cache.pool(role, control) for control in control_ids(data, condition)
             if (role, control) in cache.condition_pools]
    if not pools:
        raise ValueError(f"NO_SAME_ROLE_CONTROL_IN_CACHE: {condition}")
    picks = rng.integers(len(pools), size=n)
    locations = np.asarray([rng.integers(len(pools[index])) for index in picks])
    if len(pools) == 1:
        rows = pools[0][locations]
    else:
        positions = [np.flatnonzero(picks == index) for index in range(len(pools))]
        selected = [pool[locations[position]] for pool, position in zip(pools, positions)]
        rows = np.concatenate(selected)[np.argsort(np.concatenate(positions))]
    return cache.h[role][rows]


def _couple(condition, data, cache, shared, n, rng, device, coupling_space):
    """Return (paired reference h, paired target z, condition feature dict).

    Both spaces SOLVE entropic OT and sample pair indices from the coupling;
    neither is an independent random pairing (rework R3).
    """
    if coupling_space == "gene":
        drawn = data.sample_response(condition, n, rng, role="T")
        coupling = entropic_ot(tensor(drawn["control"], device),
                               tensor(drawn["target"], device))
        ii, jj = sample_ot_pairs(coupling, n_pairs=n)
        ids = ii.cpu().numpy()
        h_reference = cache.lookup("fit", np.asarray(drawn["control_cell_ids"])[ids])
        with torch.no_grad():
            z_target = shared.project(
                cache.lookup("fit", np.asarray(drawn["target_cell_ids"])[jj.cpu().numpy()]))
        return h_reference, z_target.detach(), drawn
    h_reference_pool = draw_control_h(data, cache, condition, n, rng, role="fit")
    h_target_pool, _ = cache.draw("fit", condition, n, rng)
    with torch.no_grad():
        z_reference_pool = shared.project(h_reference_pool)
        z_target_pool = shared.project(h_target_pool)
    coupling = entropic_ot(z_reference_pool, z_target_pool)
    ii, jj = sample_ot_pairs(coupling, n_pairs=n)
    return (h_reference_pool[ii], z_target_pool[jj].detach(),
            data.action_features(condition))


def _load_shared(config, data, device, ae_path=None):
    from .representation import load_representation
    settings = latent_settings(config)
    if ae_path is None:
        raise ValueError("LATENT_AE_REQUIRED: load a trained AE checkpoint")
    reference = load_representation(ae_path, data, device)
    if (reference.config.d_model != settings["d_model"]
            or reference.config.latent_dim != settings["latent_dim"]):
        raise ValueError("REPRESENTATION_MODEL_CONFIG_MISMATCH")
    return reference.shared


class _EarlyStop:
    """Early stopping with a minimum number of parameter updates."""

    def __init__(self, config):
        early = config.get("early_stopping", False)
        policy = early if isinstance(early, dict) else {"enabled": bool(early)}
        self.enabled = bool(policy.get("enabled", True))
        self.patience = int(policy.get("patience", 20))
        self.min_delta = float(policy.get("min_delta", 1e-4))
        self.min_updates = int(policy.get("min_updates", 1000))
        self.best, self.bad, self.updates, self.steps = math.inf, 0, 0, 0

    def observe(self, value):
        self.updates += 1
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return improved

    def step_done(self):
        self.steps += 1

    @property
    def stopped(self):
        return (self.enabled and self.steps >= self.min_updates
                and self.updates > 0 and self.bad >= self.patience)

    def reset(self):
        self.best, self.bad = math.inf, 0

    def state_dict(self):
        return {"best": self.best, "bad": self.bad, "updates": self.updates,
                "steps": self.steps}


def _payload(model_state, shared_state, *, epoch, identity, metadata, counters, extra):
    return {"schema_version": 1, "model": model_state, "shared": shared_state,
            "epoch": int(epoch), "identity": identity, "metadata": metadata,
            "counters": counters, "extra": extra}


def run_latent_ae(config, monitor, data):
    settings = latent_settings(config)
    ae = settings["ae"]
    device = torch.device(config["device"])
    policy = PrecisionPolicy.from_config(config)
    lr = float(ae.get("lr", config["response"]["lr"]))
    batch = int(ae.get("batch", config["response"]["ot_samples"]))
    cfg = config["response"]
    shape = _model_config(config, settings, data.n_genes)
    identity = config_hash({"data": data.identity, "stage": "latent_ae", "model": asdict(shape),
                            "lr": lr, "batch": batch, "epochs": config["epochs"],
                            "groups": cfg["conditions_per_update"],
                            "precision": policy.identity()})
    folder = _paths(monitor, config)["ae"]
    folder.mkdir(parents=True, exist_ok=True)
    apply_seed(config["seed"], policy if "research" in config else None)
    shared = SharedAutoencoder(shape, tensor(data.neighbors, device, torch.long)).to(device)
    if "research" in config:
        shared.precision_policy = policy
    optimizer = torch.optim.AdamW(shared.parameters(), lr=lr,
                                  weight_decay=cfg["weight_decay"])
    best, completed_epochs, selected = math.inf, 0, False
    stop = _EarlyStop(config)
    metadata = {"stage": "latent_ae", "model_config": asdict(shape),
                "data_identity": data.identity, "precision": policy.identity()}
    shared_state = {key: value.detach().to("cpu", copy=True)
                    for key, value in shared.state_dict().items()}
    with AEDataSupply(data, batch) as supply:
        for epoch in range(config["epochs"]):
            shared.train()
            rng = np.random.default_rng(config["seed"] + 100000 + epoch)
            keys = list(data.iter_response_epoch("T", config["seed"] + 100000 + epoch))
            group_size = cfg["conditions_per_update"]
            aggregate = 0.
            with supply.groups(enabled=supply.reader is not None) as prefetch:
                for start in range(0, len(keys), group_size):
                    group = keys[start:start + group_size]
                    batches = prefetch.consume(group, rng)
                    prefetch.queue(keys[start + len(group):start + len(group) + group_size], rng)
                    optimizer.zero_grad(set_to_none=True)
                    for drawn in batches:
                        x = batch_tensor(np.concatenate([drawn["control"], drawn["target"]]), device,
                                         pin_memory=supply.reader is not None)
                        with policy_autocast(shared, device):
                            loss = shared.reconstruction_loss(x).float()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("NONFINITE_AE_RECONSTRUCTION")
                        (loss / len(group)).backward()
                        aggregate += float(loss.detach()) / len(keys)
                    torch.nn.utils.clip_grad_norm_(shared.parameters(), cfg["grad_clip"],
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    stop.step_done()
            validation = _ae_validation(shared, data, batch, device, policy, supply=supply)
            improved = stop.observe(validation["loss"])
            if improved:
                best = validation["loss"]
            extra = {"best": best, "validation": validation,
                     "training_completion": {"status": "complete" if stop.stopped or epoch + 1 >= config["epochs"] else "running",
                         "reason": "validation_plateau" if stop.stopped else "max_epochs",
                         "target_epochs": config["epochs"], "actual_epochs": epoch + 1, "actual_updates": stop.steps}}
            payload = _payload(shared.state_dict(), shared_state,
                               epoch=epoch + 1, identity=identity,
                               metadata=metadata,
                               counters={"epoch": epoch + 1, "optimizer_step": stop.steps},
                               extra=extra)
            completed_epochs = epoch + 1
            if improved:
                atomic_save(folder / "best.pt", payload)
                selected = True
            monitor.record("latent_ae", epoch + 1,
                           {"reconstruction_loss": aggregate,
                            **{f"val/{key}": value for key, value in validation.items()}})
            if stop.stopped:
                break
    completion = {"status": "complete",
                  "reason": "validation_plateau" if stop.stopped else "max_epochs",
                  "target_epochs": config["epochs"], "actual_epochs": completed_epochs,
                  "actual_updates": stop.steps}
    atomic_save(folder / "final.pt", _payload(shared.state_dict(), shared_state,
        epoch=completed_epochs, identity=identity, metadata=metadata,
        counters={"epoch": completed_epochs, "optimizer_step": stop.steps},
        extra={"best": best, "training_completion": completion}))
    weights = folder / ("best.pt" if selected else "final.pt")
    if selected:
        finalize_training(weights, completion)
    return {"weights": str(weights), "best": best, "identity": identity}


@torch.no_grad()
def _ae_validation(shared, data, batch, device, policy=None, *, supply=None):
    apply_seed(20260923 + 998, policy)
    rng = np.random.default_rng(20260923 + 998)
    losses = []
    for condition in data.condition_ids("V"):
        drawn = (data.sample_response(condition, batch, rng, role="V") if supply is None
                 else supply.sample_group([condition], rng, role="V")[0])
        x = batch_tensor(np.concatenate([drawn["control"], drawn["target"]]), device,
                         pin_memory=supply is not None and supply.reader is not None)
        with policy_autocast(shared, device):
            losses.append(float(shared.reconstruction_loss(x).float()))
    return {"loss": float(np.mean(losses))}


def run_latent_cache(config, monitor, data):
    device = torch.device(config["device"])
    ae_path = _resolve_ae_checkpoint(config, monitor)
    shared = _load_shared(config, data, device, ae_path)

    def progress(role, done, total, condition=None):
        if done == total:
            monitor.say(f"表征缓存完成：{role} 共 {total} 个条件。")

    directory, status = ensure_representation_cache(
        data, shared, _paths(monitor, config)["cache"], device, progress=progress)
    cache = load_representation_cache(directory, "cpu",
                                      expected_data_identity=data.identity,
                                      expected_encoder=encoder_identity(shared, shared.config))
    manifest = cache.manifest  # the loader returns a RepresentationCache, not a dict (S04)
    monitor.record("latent_cache", 1, {**{f"cells/{role}": block["cells"]
                                          for role, block in manifest["roles"].items()},
                                       **{f"conditions/{role}": block["conditions"]
                                          for role, block in manifest["roles"].items()},
                                       "status_reused": float(status == "reused")})
    return {"cache": str(directory / "manifest.json"), "directory": str(directory),
            "status": status, "encoder": manifest["encoder"]}


def run_latent_cfm(config, monitor, data):
    settings = latent_settings(config)
    device = torch.device(config["device"])
    policy = PrecisionPolicy.from_config(config)
    shared = _load_shared(config, data, device,
                          ae_path=str(_resolve_ae_checkpoint(config, monitor)))
    directory, _ = ensure_representation_cache(
        data, shared, _paths(monitor, config)["cache"], device)
    cache = load_representation_cache(directory, device,
                                      expected_data_identity=data.identity,
                                      expected_encoder=encoder_identity(shared, shared.config))
    shape, cfg = shared.config, config["response"]
    refine_enabled = bool(settings["refine"].get("enabled", False))
    refine_epoch = int(settings["refine"].get("switch_epoch", 10 ** 9))
    refine_weight = float(settings["refine"].get("weight", 1.0))
    identity = config_hash({"data": data.identity, "stage": "latent_cfm",
                            "model": asdict(shape), "cache": cache.manifest,
                            "coupling_space": settings["coupling_space"],
                            "schedule": {"epochs": config["epochs"],
                                         "refine": settings["refine"]},
                            "response": {key: cfg[key] for key in
                                         ("lr", "weight_decay", "grad_clip", "ot_samples",
                                          "microbatch", "conditions_per_update")},
                            "refine_ode_steps": cfg["train_ode_steps"] if refine_enabled else None,
                            "precision": policy.identity(),
                            "rng_policy": LATENT_RNG_POLICY})
    folder = _paths(monitor, config)["cfm"] / "member_1"
    folder.mkdir(parents=True, exist_ok=True)
    apply_seed(config["seed"], policy)
    member = LatentMember(shape, shared).to(device)
    member.init_decoder_from_shared()
    member.decoder.requires_grad_(False)  # frozen during CFM (B4 switches it on)
    member.precision_policy = policy
    metadata = {"stage": "latent_cfm", "model_config": asdict(shape),
                "data_identity": data.identity,
                "representation_hash": shared.representation_hash,
                "representation_identity": shared.representation_identity,
                "coupling_space": settings["coupling_space"],
                "rng_policy": LATENT_RNG_POLICY, "cache": cache.manifest.get("encoder"),
                "precision": policy.identity()}
    stop = _EarlyStop(config)
    step, best, completed_epochs, selected = 0, math.inf, 0, False
    optimizer = torch.optim.AdamW(
        [{"params": list(member.velocity_model.parameters())
          + list(member.condition_encoder.parameters())}],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    shared_state = {key: value.detach().to("cpu", copy=True)
                    for key, value in shared.state_dict().items()}
    for epoch in range(config["epochs"]):
        refine_active = refine_enabled and epoch + 1 > refine_epoch
        if refine_active and not member.decoder[0].weight.requires_grad:
            member.decoder.requires_grad_(True)
            optimizer.add_param_group({"params": list(member.decoder.parameters()),
                                       "lr": optimizer.param_groups[0]["lr"],
                                       "weight_decay": cfg["weight_decay"]})
            best, selected = math.inf, False
            stop.reset()  # stage switch restarts selection (R4.3)
        member.train()
        rng = np.random.default_rng(config["seed"] + 200000 + epoch)
        generator = torch.Generator(device=device).manual_seed(config["seed"] + 200000 + epoch)
        keys = list(data.iter_response_epoch("T", config["seed"] + 200000 + epoch))
        aggregate = {"cfm_loss": 0., "mmd_loss": None, "loss": 0.}
        for start in range(0, len(keys), cfg["conditions_per_update"]):
            group = keys[start:start + cfg["conditions_per_update"]]
            optimizer.zero_grad(set_to_none=True)
            for condition in group:
                h_reference, z_target, features = _couple(
                    condition, data, cache, shared, cfg["ot_samples"], rng,
                    device, settings["coupling_space"])
                micro = cfg["microbatch"]
                for low in range(0, len(z_target), micro):
                    piece = slice(low, min(low + micro, len(z_target)))
                    rows = piece.stop - piece.start
                    with policy_autocast(member, device):
                        # One condition graph per microbatch; do not backward
                        # through an already freed shared ca graph a second time.
                        ca = member.condition_encoder(**condition_kwargs(features, rows, device))
                        loss = member.cfm_loss(z_target[piece], h_reference[piece],
                                               ca).float()
                    (loss * rows / len(z_target) / len(group)).backward()
                    aggregate["cfm_loss"] += float(loss.detach()) * rows / len(z_target) / len(keys)
                if refine_active:
                    mmd = _refine_endpoint_mmd(member, data, cache, condition, cfg,
                                               rng, generator, device,
                                               weight=refine_weight / len(group))
                    aggregate["mmd_loss"] = (aggregate["mmd_loss"] or 0.) + mmd / len(keys)
            aggregate["loss"] = aggregate["cfm_loss"] + (aggregate["mmd_loss"] or 0.)
            clip_parameters = [p for p in member.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(clip_parameters, cfg["grad_clip"],
                                           error_if_nonfinite=True)
            optimizer.step()
            stop.step_done()
            step += 1
        validation = _cfm_validation(member, data, cache, cfg, device, policy)
        improved = stop.observe(validation["loss"])
        if improved:
            best = validation["loss"]
        payload = _payload(member.state_dict(), shared_state,
                           epoch=epoch + 1, identity=identity, metadata=metadata,
                           counters={"optimizer_step": step},
                           extra={"best": best, "validation": validation,
                                  "training_completion": {"status": "complete" if stop.stopped or epoch + 1 >= config["epochs"] else "running",
                                      "reason": "validation_plateau" if stop.stopped else "max_epochs",
                                      "target_epochs": config["epochs"], "actual_epochs": epoch + 1, "actual_updates": step}})
        completed_epochs = epoch + 1
        if improved:
            atomic_save(folder / "best.pt", payload)
            selected = True
        monitor.record("latent_cfm", epoch + 1,
                       {**{f"train/{key}": value for key, value in aggregate.items()},
                        **{f"val/{key}": value for key, value in validation.items()}})
        if stop.stopped:
            break
    completion = {"status": "complete",
                  "reason": "validation_plateau" if stop.stopped else "max_epochs",
                  "target_epochs": config["epochs"], "actual_epochs": completed_epochs,
                  "actual_updates": step}
    atomic_save(folder / "final.pt", _payload(member.state_dict(), shared_state,
        epoch=completed_epochs, identity=identity, metadata=metadata,
        counters={"optimizer_step": step},
        extra={"best": best, "training_completion": completion}))
    weights = folder / ("best.pt" if selected else "final.pt")
    if selected:
        finalize_training(weights, completion)
    return {"weights": str(weights), "identity": identity, "best": best,
            "coupling_space": settings["coupling_space"]}


def _refine_endpoint_mmd(member, data, cache, condition, cfg, rng, generator, device, *, weight):
    """B4: decoded full-gene endpoint MMD; latent noise, no gene-space sigma on y."""
    drawn = data.sample_response(condition, cfg["mmd_samples"], rng, role="T")
    y = tensor(drawn["target"], device)
    h_reference = cache.lookup("fit", np.asarray(drawn["control_cell_ids"]))
    ca = member.condition_encoder(**condition_kwargs(drawn, len(y), device))
    noise = torch.randn((len(y), member.config.latent_dim), device=device,
                        generator=generator)
    latent = heun_integrate(
        lambda state, tau: member.velocity(state, tau, h_reference, ca),
        noise, cfg["train_ode_steps"], checkpoint=True)
    generated = member.decode(latent)
    loss = rbf_mmd(generated, y, bandwidth=data.bandwidth, unbiased=False)
    (loss * weight).backward()
    return float(loss.detach())


@torch.no_grad()
def _cfm_validation(member, data, cache, cfg, device, policy=None):
    """Free generation on V: latent ODE -> one decode -> projection -> MMD/MSE."""
    apply_seed(20260923 + 999, policy)
    rng = np.random.default_rng(20260923 + 999)
    generator = torch.Generator(device=device).manual_seed(20260923 + 999)
    rows = []
    for condition in data.condition_ids("V"):
        drawn = data.sample_response(condition, cfg["mmd_samples"], rng, role="V")
        y = tensor(drawn["target"], device)
        h_reference = cache.lookup("selection", np.asarray(drawn["control_cell_ids"]))
        ca = member.condition_encoder(**condition_kwargs(drawn, len(y), device))
        noise = torch.randn((len(y), member.config.latent_dim), device=device,
                            generator=generator)
        latent = member.sample_latent(h_reference, ca, steps=cfg["inference_ode_steps"],
                                      noise=noise)
        generated = projected(member.decode(latent), data)
        rows.append({"mmd_loss": float(rbf_mmd(generated, y, bandwidth=data.bandwidth)),
                     "mean_mse": float((generated.mean(0) - y.mean(0)).square().mean())})
    return {**{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            "loss": float(np.mean([row["mmd_loss"] + row["mean_mse"] for row in rows]))}


class LatentMemberAdapter:
    """Comparison interface over a staged latent member (rework R5.2).

    Exposes the evaluator's model contract: observed reference pass-through,
    member condition encoding, one-shot latent sampling + decode, latent noise.
    """

    def __init__(self, shared, member, config):
        self.shared, self.member, self.config = shared, member, config

    def encode_cells(self, expression):
        return self.shared.encode_h(expression)

    @property
    def condition_encoder(self):
        return self.member.condition_encoder

    def condition_reference(self, reference):
        return reference  # observed mode

    def encode_condition(self, **kwargs):
        return self.member.condition_encoder(**kwargs)

    def sample_noise(self, reference, generator=None):
        return torch.randn((len(reference), self.config.latent_dim),
                           device=reference.device, dtype=reference.dtype,
                           generator=generator)

    def sample(self, reference, condition, noise=None, steps=32, checkpoint=False,
               progress=None):
        _, h = self.shared.encode_h(reference)
        latent = self.member.sample_latent(h, condition, steps=steps, noise=noise)
        return self.member.decode(latent)


def load_latent_adapter(path, data, device, *, saved=None):
    """Load a latent_cfm/latent_members checkpoint with identity checks (R5.2)."""
    from .representation import latent_shape, require_finite_state, require_trained
    saved = load_checkpoint(path) if saved is None else saved
    metadata = saved.get("metadata", {})
    if metadata.get("stage") not in ("latent_cfm", "latent_members"):
        raise ValueError(f"LATENT_ADAPTER_STAGE_MISMATCH: {metadata.get('stage')}")
    require_trained(saved, "TRAINED_RESPONSE_REQUIRED")
    shape = latent_shape(metadata, data)
    require_finite_state(saved["shared"])
    require_finite_state(saved["model"])
    shared = SharedAutoencoder(shape, tensor(data.neighbors, "cpu", torch.long))
    shared.load_state_dict(saved["shared"])
    member = LatentMember(shape, shared.to(device))
    member.load_state_dict(saved["model"])
    shared.to(device).eval().requires_grad_(False)
    member.to(device).eval().requires_grad_(False)
    if metadata.get("representation_hash") is not None:
        shared.representation_hash = metadata["representation_hash"]
    adapter = LatentMemberAdapter(shared, member, shape)
    adapter.representation_identity = encoder_identity(shared, shape)
    adapter.representation_hash = metadata.get("representation_hash")
    for key in ("representation_identity", "cache"):
        expected = metadata.get(key)
        actual = adapter.representation_identity
        if expected is not None and "checkpoint_sha256" not in expected:
            actual = {name: value for name, value in actual.items() if name != "checkpoint_sha256"}
        if expected is not None and expected != actual:
            raise ValueError("LATENT_REPRESENTATION_IDENTITY_MISMATCH")
    return adapter


def run_latent_stage(mode, config, monitor, data):
    if mode == "latent_ae":
        result = run_latent_ae(config, monitor, data)
    elif mode == "latent_cache":
        result = run_latent_cache(config, monitor, data)
    elif mode == "latent_cfm":
        result = run_latent_cfm(config, monitor, data)
    else:
        raise ValueError(f"UNKNOWN_LATENT_STAGE: {mode}")
    return {"stage": mode, **result}
