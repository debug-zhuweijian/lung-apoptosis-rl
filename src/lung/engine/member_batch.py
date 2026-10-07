"""Five-member latent training batched in ONE GPU process (task W6; rework R1/R4.2/R7).

Structure: the frozen shared encoder/projection and the h cache are owned once;
each member owns an independent latent velocity network, condition encoder,
decoder copy, AdamW state, gradient clipping and random streams.

Batching contract (rework R1): the member forward runs through
``LatentMember.forward`` — condition encoding INSIDE the functional graph — via
``torch.func.stack_module_state`` + ``functional_call`` + ``vmap``. Gradients
land on the stacked member parameters and are scattered back ONLY to the
trainable whitelist (velocity network + condition encoder); frozen shared
weights and the frozen member decoder never receive gradients. The optimizer
steps ONCE per update group per member, with the same per-member loss scaling
as the single-member pipeline (no extra division by the member count).

Lockstep resume (rework R4.2): members advance together; checkpoints carry the
in-epoch group cursor plus every member's NumPy/Torch RNG states. A resume
whose member snapshots disagree on epoch/cursor raises instead of silently
skipping a lagging member's data.

Random streams are the registered protocol ``latent_member_batch_v1``:
per-member ``torch.Generator`` draws latent noise/tau and per-member NumPy
streams draw cells; this path does not replay any legacy trajectory and does
not claim to. ``member_workers`` must be 1 so a batched run never also spawns
processes.

v1 scope (recorded): batched CFM main phase; per-member refinement decoding and
per-member early stopping remain follow-ups — members stop together on budget
or epoch limits here.
"""

import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .async_checkpoint import BackgroundSaver, clone_snapshot
from .checkpoint import atomic_save, load_checkpoint, rng_state
from .config import config_hash
from .latent_pipeline import (_cfm_validation, _couple, _load_shared, _payload,
                              _resolve_ae_checkpoint, _resolve_cache_dir,
                              _write_stage_state, latent_settings)
from .numerics import PrecisionPolicy, apply_seed, policy_autocast
from .representation_cache import (encoder_identity, ensure_representation_cache,
                                   load_representation_cache)
from .response import condition_kwargs
from ..models.latent_staged import LatentMember

MEMBER_RNG_POLICY = "latent_member_batch_v1"
_TRAINABLE_PREFIXES = ("velocity_model.", "condition_encoder.")


def _validate_stacked_inputs(stacked, base):
    """Full input validation OUTSIDE vmap (rework S03).

    BatchedTensors cannot drive Python control flow, so every check the single
    member path performs in ``ConditionEncoder.forward`` is applied here on the
    stacked [members, rows, ...] tensors before the batched call.
    """
    encoder = base.condition_encoder
    fingerprint = stacked["fingerprint"]
    if fingerprint.ndim != 3 or fingerprint.shape[-1] != 2048 or not torch.isfinite(fingerprint).all():
        raise ValueError("INVALID_MEMBER_FINGERPRINT: finite [members, rows, 2048] required.")
    dose, duration, vehicle = stacked["dose"], stacked["duration"], stacked["vehicle"]
    if not torch.isfinite(dose).all() or ((dose < 0) & ~vehicle).any():
        raise ValueError("INVALID_DOSE: ordinary drugs require finite nonnegative dose.")
    if not torch.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("INVALID_DURATION: finite positive hours required.")
    elapsed, horizon = stacked["elapsed"], stacked["horizon"]
    if not torch.isfinite(elapsed).all() or not torch.isfinite(horizon).all():
        raise ValueError("Non-finite biological time.")
    if (horizon <= 0).any() or (elapsed < 0).any() or (elapsed > horizon).any():
        raise ValueError("Invalid biological horizon.")
    for key, embedding in (("background", encoder.background), ("source", encoder.source),
                           ("protocol", encoder.protocol)):
        index = stacked[key]
        if index.dtype != torch.long or (index < 0).any() or (index >= embedding.num_embeddings).any():
            raise ValueError(f"UNSUPPORTED_CONTEXT: {key} id not in the frozen registry.")


def _batched_forward(base, params, buffers):
    """vmap over members; condition encoding happens inside the graph (R1.2).

    Inputs must be validated by the caller (see _validate_stacked_inputs);
    the functional call routes through the unchecked dense encoder so no
    data-dependent Python branch executes inside vmap (rework S03).
    """
    def forward(z, h, fingerprint, dose, duration, background, source, protocol,
                elapsed, horizon, vehicle, noise, tau):
        def one(member_params, member_buffers, z1, h1, fp1, dose1, dur1, bg1, src1,
                proto1, el1, hz1, veh1, noise1, tau1):
            return torch.func.functional_call(
                base, (member_params, member_buffers),
                (z1, h1, fp1, dose1, dur1, bg1, src1, proto1, el1, hz1, veh1),
                kwargs={"noise": noise1, "tau": tau1, "validate": False})

        return torch.func.vmap(one)(params, buffers, z, h, fingerprint, dose,
                                    duration, background, source, protocol,
                                    elapsed, horizon, vehicle, noise, tau)
    return forward


def _condition_tensors(features, rows, device):
    return condition_kwargs(features, rows, device)


def _stacked_inputs(members, generators, member_draws, piece, rows, device):
    """Per-member stacked tensors for one micro-chunk (identical chunk grid)."""
    zs, hs, noises, taus = [], [], [], []
    feats = []
    for index, (h_reference, z_target, features) in enumerate(member_draws):
        noise = torch.randn((rows, members[index].config.latent_dim), device=device,
                            dtype=torch.float32, generator=generators[index])
        tau = torch.rand(rows, device=device, dtype=torch.float32,
                         generator=generators[index])
        zs.append(z_target[piece])
        hs.append(h_reference[piece])
        noises.append(noise)
        taus.append(tau)
        feats.append(_condition_tensors(features, rows, device))
    stacked = {"z": torch.stack(zs), "h": torch.stack(hs),
               "noise": torch.stack(noises), "tau": torch.stack(taus)}
    for key in ("fingerprint", "dose", "duration", "background", "source",
                "protocol", "elapsed", "horizon", "vehicle"):
        stacked[key] = torch.stack([entry[key] for entry in feats])
    return stacked


def train_latent_members(config, monitor, data, budget):
    members_count = int(config["response"]["members"])
    if members_count < 1:
        raise ValueError("response.members must be positive")
    if int(config["response"].get("member_workers", 1)) != 1:
        raise ValueError("LATENT_MEMBER_BATCH_REQUIRES_SINGLE_PROCESS: set member_workers: 1")
    settings = latent_settings(config)
    device = torch.device(config["device"])
    policy = PrecisionPolicy.from_config(config)
    shared = _load_shared(config, data, device,
                          ae_path=str(_resolve_ae_checkpoint(config, monitor)))
    directory, _ = ensure_representation_cache(
        data, shared, _resolve_cache_dir(config, monitor), device)
    cache = load_representation_cache(directory, device,
                                      expected_data_identity=data.identity,
                                      expected_encoder=encoder_identity(shared, shared.config))
    # Saved activation hooks would replay escaped vmap tensors during backward.
    # Keep the small member network's activations; the shared AE stays unchanged.
    shape = replace(shared.config, checkpoint_blocks=False)
    cfg = config["response"]
    base_identity = {"data": data.identity, "stage": "latent_members",
                     "model": asdict(shape), "cache": cache.manifest,
                     "coupling_space": settings["coupling_space"],
                     "members": members_count, "epochs": config["epochs"],
                     "response": {key: cfg[key] for key in
                                  ("lr", "weight_decay", "grad_clip", "ot_samples",
                                   "microbatch", "conditions_per_update")},
                     "precision": policy.identity(),
                     "rng_policy": MEMBER_RNG_POLICY}
    root = Path(monitor.weights) / "latent_members"
    apply_seed(config["seed"], policy)
    members, optimizers, bests, epochs0, cursors, steps = [], [], [], [], [], []
    streams = None
    for index in range(members_count):
        folder = root / f"member_{index + 1}"
        folder.mkdir(parents=True, exist_ok=True)
        member = LatentMember(shape, shared).to(device)
        member.init_decoder_from_shared()
        member.decoder.requires_grad_(False)
        member.precision_policy = policy
        member._folder = folder
        member._identity = config_hash({**base_identity, "member": index})
        member._metadata = {"stage": "latent_members", "member": index,
                            "model_config": asdict(shape), "data_identity": data.identity,
                            "coupling_space": settings["coupling_space"],
                            "rng_policy": MEMBER_RNG_POLICY}
        trainable = [parameter for name, parameter in member.named_parameters()
                     if name.startswith(_TRAINABLE_PREFIXES)]
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
        best, epoch0, step, cursor = math.inf, 0, 0, 0
        if (folder / "last.pt").exists():
            saved = load_checkpoint(folder / "last.pt", member._identity)
            member.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            epoch0 = saved["epoch"]
            best = saved["extra"]["best"]
            step = saved["counters"]["optimizer_step"]
            position = saved["extra"].get("resume") or {}
            if position.get("epoch") == epoch0:
                cursor = int(position.get("next_group_start", 0))
        members.append(member)
        optimizers.append(optimizer)
        bests.append(best)
        epochs0.append(epoch0)
        cursors.append(cursor)
        steps.append(step)
    if len(set(epochs0)) != 1 or len(set(cursors)) != 1 or len(set(steps)) != 1:
        raise ValueError("MEMBER_SNAPSHOT_DIVERGED: lockstep members must resume from "
                         "snapshots with identical epoch/cursor/step; rerun the stage "
                         "or delete the diverged member folders explicitly")
    epoch0, cursor, step = epochs0[0], cursors[0], steps[0]
    if (members[0]._folder / "last.pt").exists():
        saved = load_checkpoint(members[0]._folder / "last.pt", members[0]._identity)
        from .checkpoint import restore_rng
        restore_rng(saved["rng"])
        policy.apply()
        streams = saved["extra"].get("resume") if cursor else None
    shared_state = {key: value.detach().to("cpu", copy=True)
                    for key, value in shared.state_dict().items()}
    saver = BackgroundSaver()
    try:
        for epoch in range(epoch0, config["epochs"]):
            if budget.exhausted:
                break
            for member in members:
                member.train()
            rngs = [np.random.default_rng(config["seed"] + 300000 + m * 1000 + epoch)
                    for m in range(members_count)]
            generators = [torch.Generator(device=device).manual_seed(
                config["seed"] + 300000 + m * 1000 + epoch)
                for m in range(members_count)]
            keys = list(data.iter_response_epoch("T", config["seed"] + 300000 + epoch))
            if streams is not None:
                if streams["keys"] != keys:
                    raise ValueError("MEMBER_RESUME_CONDITION_ORDER_MISMATCH")
                for index in range(members_count):
                    rngs[index].bit_generator.state = streams["numpy_rngs"][index]
                    generators[index].set_state(streams["torch_rngs"][index])
                streams = None
            group_size = cfg["conditions_per_update"]
            if cursor > len(keys) or not (cursor == len(keys) or cursor % group_size == 0):
                raise ValueError("MEMBER_RESUME_CURSOR_MISMATCH")
            params, buffers = None, None
            for start in range(cursor, len(keys), group_size):
                group = keys[start:start + group_size]
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                for condition in group:
                    n = cfg["ot_samples"]
                    micro = min(cfg["microbatch"], n)
                    member_draws = [
                        _couple(condition, data, cache, shared, n, rngs[index],
                                device, settings["coupling_space"])
                        for index in range(members_count)]
                    for low in range(0, n, micro):
                        piece = slice(low, min(low + micro, n))
                        rows = piece.stop - piece.start
                        stacked = _stacked_inputs(members, generators, member_draws,
                                                  piece, rows, device)
                        if params is None:
                            params, buffers = torch.func.stack_module_state(members)
                            batched = _batched_forward(members[0], params, buffers)
                        _validate_stacked_inputs(stacked, members[0])  # outside vmap (S03)
                        with policy_autocast(members[0], device):
                            losses = batched(*[stacked[key] for key in
                                               ("z", "h", "fingerprint", "dose", "duration",
                                                "background", "source", "protocol",
                                                "elapsed", "horizon", "vehicle", "noise", "tau")]).float()
                        if not torch.isfinite(losses).all():
                            raise FloatingPointError("NONFINITE_MEMBER_CFM")
                        # Same per-member scaling as the single-member pipeline;
                        # summing over members does not rescale any member (R1.5).
                        (losses.sum() * rows / n / len(group)).backward()
                for index, member in enumerate(members):
                    for name, parameter in member.named_parameters():
                        if not name.startswith(_TRAINABLE_PREFIXES):
                            continue  # frozen shared/decoder: never touched (R1.3/R1.4)
                        gradient = params[name].grad
                        parameter.grad = (gradient[index].clone() if gradient is not None
                                          else None)
                    torch.nn.utils.clip_grad_norm_(
                        [p for name, p in member.named_parameters()
                         if name.startswith(_TRAINABLE_PREFIXES)],
                        cfg["grad_clip"], error_if_nonfinite=True)
                for optimizer in optimizers:
                    optimizer.step()  # exactly once per group per member (R1.5)
                step += 1
                for member in members:
                    member._step = step
                saver.observe_computed(step)
                exhausted = budget.check(step, epoch)
                position = {"epoch": epoch, "next_group_start": start + len(group),
                            "keys": keys,
                            "numpy_rngs": [rng.bit_generator.state for rng in rngs],
                            "torch_rngs": [generator.get_state() for generator in generators]}
                for member, optimizer, best in zip(members, optimizers, bests):
                    snapshot = clone_snapshot(member, optimizer)
                    payload = _payload(snapshot["model"], shared_state,
                                       snapshot["optimizer"], epoch=epoch,
                                       identity=member._identity,
                                       metadata=member._metadata,
                                       counters={"optimizer_step": step},
                                       extra={"best": best, "resume": position,
                                              "budget_exhausted": bool(exhausted)})
                    path = member._folder / "last.pt"
                    saver.submit(step, lambda p=payload, target=path:
                                 atomic_save(target, p), group=members_count)
                params, buffers = None, None  # stale after the optimizer step
                if exhausted:
                    saver.flush()
                    break
            cursor = 0
            if budget.exhausted:
                break
            saver.flush()
            for index, member in enumerate(members):
                validation = _cfm_validation(member, data, cache, cfg, device, policy)
                if validation["loss"] < bests[index]:
                    bests[index] = validation["loss"]
                    payload = _payload(member.state_dict(), shared_state,
                                       optimizers[index].state_dict(), epoch=epoch + 1,
                                       identity=member._identity,
                                       metadata=member._metadata,
                                       counters={"optimizer_step": step},
                                       extra={"best": bests[index],
                                              "validation": validation,
                                              "budget_exhausted": False})
                    atomic_save(member._folder / "best.pt", payload)
                monitor.record(f"latent_members/member_{index + 1}", epoch + 1,
                               {**{f"val/{key}": value for key, value in validation.items()},
                                "best": bests[index],
                                "saver/durable_step": float(saver.durable_step or 0)})
    finally:
        saver.close()
    weights = []
    for member in members:
        best, last = member._folder / "best.pt", member._folder / "last.pt"
        weights.append(str(best if best.exists() else last))
    _write_stage_state(monitor, latent_members={"weights": weights})
    return {"weights": weights, "members": members_count, "rng_policy": MEMBER_RNG_POLICY}
