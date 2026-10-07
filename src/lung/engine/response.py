"""The real CFM + full-ODE distribution objective, with exact gradient recomputation."""

import contextlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lung.models import ModelConfig, build_response_model, entropic_ot, sample_ot_pairs, rbf_mmd, physical_projection
from .checkpoint import seed_all, save_training, load_checkpoint, restore_rng, rng_state
from .config import config_hash
from .numerics import PrecisionPolicy, policy_autocast
from .timing import PhaseTimer
from .response_resume import validate_resume, update_state
from .response_training import generation_noise, mmd_backward, one_step_loss, train_conditions, condition_batch_mode, prefetch_enabled
from .response_prefetch import ResponsePrefetch
from .training_control import TrainingControl
from lung.monitoring.response_progress import ResponseTrainingProgress


RESPONSE_RNG_POLICY = "cfm_shared_mmd_separate_v1"


def response_rngs(seed, device):
    """Epoch-scoped streams; extra MMD draws never advance shared CFM draws."""
    mmd_seed = int(np.random.SeedSequence([seed, 0x4D4D44]).generate_state(1)[0])
    return (np.random.default_rng(seed), np.random.default_rng(mmd_seed),
            torch.Generator(device=device).manual_seed(mmd_seed))


def mmd_weight(cfg):
    value = float(cfg.get("mmd_weight", 1.))
    if not math.isfinite(value) or value < 0:
        raise ValueError("mmd_weight must be finite and nonnegative")
    return value


def response_checkpoint(path, identity):
    saved = load_checkpoint(path)
    if saved["metadata"].get("rng_policy") != RESPONSE_RNG_POLICY:
        raise ValueError("RESPONSE_RNG_POLICY_CHANGED: legacy checkpoint requires its original code; start a new run")
    if saved.get("identity") != identity:
        raise ValueError("CHECKPOINT_IDENTITY_MISMATCH: response objective/data changed")
    return saved


def tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(value).copy(), dtype=dtype, device=device)


def condition_kwargs(batch, n, device, elapsed=0., horizon=None):
    # A source-specific observed endpoint belongs to each condition. Explicit
    # world protocols still pass their common horizon when simulating actions.
    if horizon is None:
        horizon = batch.get("horizon", batch["duration"])
    def fill(value, dtype=torch.float32):
        return torch.full((n,), value, device=device, dtype=dtype)
    return {"fingerprint": tensor(batch["fingerprint"], device).reshape(1, -1).expand(n, -1),
            "dose": fill(batch["dose"]), "duration": fill(batch["duration"]),
            "background": fill(batch["background_id"], torch.long),
            "source": fill(batch["source_id"], torch.long),
            "protocol": fill(batch["protocol_id"], torch.long),
            "elapsed": fill(elapsed), "horizon": fill(horizon),
            "vehicle": fill(bool(batch.get("is_vehicle", False)), torch.bool)}


def projected(raw, data):
    result = physical_projection(raw, tensor(data.mu, raw.device), tensor(data.scale, raw.device))
    if not result.valid.all():
        raise FloatingPointError("INVALID_GENERATED_MASS: no fallback, dropping or fake zero")
    return result.x


def exact_mmd_backward(model, reference, target, encode_condition, *, bandwidth, steps,
                       scale=1., noise=None, progress=None, microbatch=1, mode="replay"):
    """Exact chain rule with a bounded ODE microbatch (dropout is zero).

    The MMD is coupled across the ENTIRE generated batch. First differentiate it
    w.r.t. generated endpoints; then replay identical full ODE microbatches and propagate
    that endpoint gradient. This does not truncate the ODE or detach its gradient.
    """
    if mode != "replay":
        return mmd_backward(model, reference, target, encode_condition, mode=mode,
            bandwidth=bandwidth, steps=steps, scale=scale, noise=noise,
            progress=progress, microbatch=microbatch)
    if not isinstance(microbatch, int) or microbatch < 1:
        raise ValueError('MMD microbatch must be a positive integer')
    noise = generation_noise(model, reference) if noise is None else noise
    if hasattr(model, "condition_reference"):
        reference = model.condition_reference(reference)
    outputs = []
    with torch.no_grad():
        for i in range(0, len(reference), microbatch):
            end = min(i + microbatch, len(reference))
            if progress is not None:
                progress("MMD generate", i, len(reference))
            outputs.append(model.sample(reference[i:end], encode_condition(end-i),
                noise=noise[i:end], steps=steps, checkpoint=False,
                progress=(lambda current, total: progress("ODE generate", current, total)) if progress else None))
        generated = torch.cat(outputs)
    leaf = generated.detach().requires_grad_(True)
    if progress is not None:
        progress("MMD endpoint gradient", 0, 1)
    loss = rbf_mmd(leaf, target, bandwidth=bandwidth, unbiased=False)
    endpoint_gradient, = torch.autograd.grad(loss * scale, leaf)
    for i in range(0, len(reference), microbatch):
        end = min(i + microbatch, len(reference))
        if progress is not None:
            progress("MMD replay", i, len(reference))
        output = model.sample(reference[i:end], encode_condition(end-i), noise=noise[i:end],
                              steps=steps, checkpoint=True,
                              progress=(lambda current, total: progress("ODE replay", current, total)) if progress else None)
        if progress is not None:
            progress("ODE backward start", i, len(reference))
        torch.autograd.backward(output, endpoint_gradient[i:end])
        if progress is not None:
            progress("ODE backward done", end, len(reference))
    return float(loss.detach())


def learning_rate(step, total, warmup, peak, minimum):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    fraction = (step - warmup) / max(total - warmup - 1, 1)
    return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * min(fraction, 1.)))


def train_condition(model, data, condition_id, rng, cfg, device, divisor, progress=None,
                    *, mmd_rng=None, mmd_generator=None):
    stage = cfg.get("_training_stage", "full")
    if stage not in {"full", "cfm", "one_step"}:
        raise ValueError("Unknown response training stage")
    if model.config.backbone == "latent_flow" and stage != "full":
        raise ValueError("LATENT_SCHEDULE_UNSUPPORTED: use full")
    weight = mmd_weight(cfg) if stage == "full" else 0.
    if (weight or stage == "one_step") and (mmd_rng is None or mmd_generator is None or mmd_rng is rng):
        raise ValueError("MMD requires separate NumPy and torch random streams")
    if progress is not None:
        progress("sampling", 0, 1)
    batch = data.sample_response(condition_id, cfg["ot_samples"], rng, role="T")
    reference, target = tensor(batch["control"], device), tensor(batch["target"], device)
    if progress is not None:
        progress("OT coupling", 0, 1)
    if cfg["coupling"] == "entropic_ot_prior":
        coupling = entropic_ot(reference, target)
    elif cfg["coupling"] == "random_product_control":
        coupling = reference.new_full((len(reference), len(target)), 1 / (len(reference) * len(target)))
    else:
        raise ValueError("Unsupported coupling; no automatic OT failure substitution")
    i, j = sample_ot_pairs(coupling, n_pairs=cfg["cfm_samples"])
    reference = model.condition_reference(reference)
    cfm, auxiliary, reconstruction = 0., 0., 0.
    micro = cfg["microbatch"]
    for start in range(0, len(i), micro):
        a, b = i[start:start+micro], j[start:start+micro]
        ca = model.encode_condition(**condition_kwargs(batch, len(a), device))
        with policy_autocast(model, device):
            if model.config.backbone == "latent_flow":
                components = model.loss_components(reference[a], target[b], ca)
                fit = components["latent_cfm_loss"].float()
                reconstruct = components["reconstruction_loss"].float()
                loss = fit + model.config.reconstruction_weight * reconstruct
            else:
                fit = model.cfm_loss(reference[a], target[b], ca).float()
                loss = fit
        if model.config.backbone == "latent_flow":
            reconstruction += float(reconstruct.detach()) * len(a) / len(i)
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_CFM")
        if progress is not None:
            progress("CFM backward start", start, len(i))
        (loss * len(a) / len(i) / divisor).backward()
        cfm += float(fit.detach()) * len(a) / len(i)
        if progress is not None:
            progress("CFM backward done", min(start+micro, len(i)), len(i))
    if stage == "one_step":
        # Independent population draws are NOT observed single-cell pairings.
        second = data.sample_response(condition_id, cfg["mmd_samples"], mmd_rng, role="T")
        x, y = tensor(second["control"], device), tensor(second["target"], device)
        noise = generation_noise(model, x, generator=mmd_generator)
        tau = torch.rand(len(x), device=device, dtype=x.dtype, generator=mmd_generator)
        ca = model.encode_condition(**condition_kwargs(second, len(x), device))
        extra = one_step_loss(model, x, y, ca, bandwidth=data.bandwidth,
            noise=noise, tau=tau, microbatch=cfg.get("mmd_microbatch", 1))
        if not torch.isfinite(extra):
            raise FloatingPointError("NONFINITE_ONE_STEP_MMD")
        (extra / divisor).backward()
        auxiliary = float(extra.detach())
    fit_total = cfm + auxiliary + model.config.reconstruction_weight * reconstruction
    diagnostic = {"one_step_auxiliary": auxiliary,
                  "reconstruction_loss": reconstruction if model.config.backbone == "latent_flow" else None,
                  "latent_cfm_loss": cfm if model.config.backbone == "latent_flow" else None}
    if weight == 0:
        return {"cfm_loss": cfm, "mmd_loss": None, "mmd_weight": weight,
                "weighted_mmd_loss": None, "mmd_computed": 0., "loss": fit_total, **diagnostic}
    # Independent cells and Gaussian noise; the shared CFM streams do not advance.
    second = data.sample_response(condition_id, cfg["mmd_samples"], mmd_rng, role="T")
    x, y = tensor(second["control"], device), tensor(second["target"], device)
    if model.config.backbone != "latent_flow":
        y = y + model.config.sigma_min * torch.randn(y.shape, device=device, dtype=y.dtype,
                                                    generator=mmd_generator)
    noise = generation_noise(model, x, generator=mmd_generator)
    mmd = exact_mmd_backward(model, x, y,
                            lambda n: model.encode_condition(**condition_kwargs(second, n, device)),
                            bandwidth=data.bandwidth, steps=cfg["train_ode_steps"], scale=weight/divisor,
                            noise=noise, progress=progress,
                            microbatch=cfg.get('mmd_microbatch', 1), mode=cfg.get('mmd_backward', 'auto'))
    return {"cfm_loss": cfm, "mmd_loss": mmd, "mmd_weight": weight,
            "weighted_mmd_loss": weight*mmd, "mmd_computed": 1., "loss": fit_total + weight*mmd,
            **diagnostic}


@torch.no_grad()
def evaluate_response(model, data, cfg, device, role="V", progress=lambda x, **kw: x):
    saved_rng = rng_state()
    was_training = model.training
    model.eval()
    rows = []
    try:
        seed_all(20260923 + 997)
        policy = getattr(model, "precision_policy", None)
        if policy is not None:
            policy.apply()  # the fixed validation seed must not undo the policy (R6)
        rng = np.random.default_rng(20260923 + 997)
        for condition_id in progress(data.condition_ids(role), desc=f"{role} response"):
            batch = data.sample_response(condition_id, cfg["mmd_samples"], rng, role=role)
            x, y = tensor(batch["control"], device), tensor(batch["target"], device)
            x = model.condition_reference(x)
            micro = cfg.get('mmd_microbatch', 1)
            raw = torch.cat([model.sample(x[i:i+micro], model.encode_condition(
                    **condition_kwargs(batch, len(x[i:i+micro]), device)),
                    steps=cfg["inference_ode_steps"], checkpoint=False)
                    for i in range(0, len(x), micro)])
            valid = projected(raw, data)
            mmd = rbf_mmd(valid, y, bandwidth=data.bandwidth)
            mse = (valid.mean(0) - y.mean(0)).square().mean()
            # Use the registered training coupling on an independent diagnostic batch.
            fit = data.sample_response(condition_id, cfg["ot_samples"], rng, role=role)
            a, b = tensor(fit["control"], device), tensor(fit["target"], device)
            if cfg["coupling"] == "entropic_ot_prior":
                plan = entropic_ot(a, b)
            elif cfg["coupling"] == "random_product_control":
                plan = a.new_full((len(a), len(b)), 1 / (len(a) * len(b)))
            else:
                raise ValueError("Unsupported coupling; no automatic substitution")
            ii, jj = sample_ot_pairs(plan, n_pairs=cfg["cfm_samples"])
            a = model.condition_reference(a)
            micro = cfg['microbatch']
            losses, reconstructions = [], []
            for k in range(0, len(ii), micro):
                left, right = a[ii[k:k+micro]], b[jj[k:k+micro]]
                condition = model.encode_condition(**condition_kwargs(fit, len(left), device))
                with policy_autocast(model, device):
                    if model.config.backbone == "latent_flow":
                        parts = model.loss_components(left, right, condition)
                        losses.append(parts["latent_cfm_loss"].float() * len(left))
                        reconstructions.append(parts["reconstruction_loss"].float() * len(left))
                    else:
                        losses.append(model.cfm_loss(left, right, condition).float() * len(left))
            cfm = torch.stack(losses).sum() / len(ii)
            rows.append({"condition_id": condition_id, "role": role,
                         "cfm_loss": float(cfm), "mmd_loss": float(mmd),
                         "mean_mse": float(mse), "delta_mse": float(mse),
                         "loss": float(mmd + mse)})
            if reconstructions:
                rows[-1].update(latent_cfm_loss=float(cfm),
                    reconstruction_loss=float(torch.stack(reconstructions).sum() / len(ii)))
        if not rows:
            raise ValueError(f"SPLIT_EMPTY: {role}")
        return {key: float(np.mean([row[key] for row in rows]))
                for key in rows[0] if key not in {"condition_id", "role"}}, rows
    finally:
        restore_rng(saved_rng)
        model.train(was_training)


class TrainingDetailProgress:
    """Keep live substeps without forcing a console repaint at every CUDA launch."""
    def __init__(self, bar, interval=.5):
        self.bar, self.interval = bar, interval
        self.phase, self.last = None, -float('inf')

    def __call__(self, phase, done, total):
        now = time.monotonic()
        if phase != self.phase or done == total or now - self.last >= self.interval:
            self.bar.set_postfix_str(f'{phase} {done}/{total}', refresh=True)
            self.phase, self.last = phase, now


def _phase(timer, name):
    return timer.phase(name) if timer is not None else contextlib.nullcontext()


def train_response(data, config, monitor, resume=False, stop_check=None):
    workers = config["response"].get("member_workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("response.member_workers must be a positive integer")
    if workers > 1 and config["response"]["members"] > 1:
        if stop_check is not None:
            raise ValueError("STOP_CHECK_REQUIRES_SINGLE_PROCESS_MEMBERS")
        from .response_parallel import train_parallel_members
        return train_parallel_members(data.identity, config, monitor, resume=resume)
    return _train_response_members(data, config, monitor, resume=resume, stop_check=stop_check)


def _train_response_members(data, config, monitor, resume=False, member_indices=None,
                            stop_check=None):
    """Train full independent members, preserving their global seed/index mapping.

    ``stop_check(step, epoch)`` is an optional caller-wired development budget:
    when it fires the loop force-saves a resumable checkpoint flagged
    ``budget_exhausted`` and stops without claiming convergence.
    """
    cfg = {**config["response"], "mmd_weight": mmd_weight(config["response"])}
    device = torch.device(config["device"])
    shape = ModelConfig(**cfg["model"])
    control_config = {**config, "training_schedule": cfg.get("training_schedule", config.get("training_schedule", {}))}
    if shape.backbone == "latent_flow" and control_config["training_schedule"].get("mode", "full") != "full":
        raise ValueError("LATENT_SCHEDULE_UNSUPPORTED: use full")
    # Registered numeric policy enters the resume identity (task W4). Legacy
    # suite configs carry None; they resume under the same rule as any source
    # change: new run, old checkpoints kept as the previous implementation.
    precision = PrecisionPolicy.from_config(config) if "research" in config else None
    schedule_mode = control_config["training_schedule"].get("mode", "full")
    initial_stage = "full" if schedule_mode == "full" else schedule_mode.split("_then")[0]
    timing_enabled = bool(config.get("research", {}).get("timing", {}).get("enabled"))
    identity = config_hash({"data": data.identity, "response": cfg, "task": "RESPONSE-A549-V1",
                            "one_step_objective": "full_population_rbf_mmd_v1",
                            "training_schedule": control_config["training_schedule"],
                            "training_control": config.get("training_control", {}),
                            "early_stopping": config.get("early_stopping", False),
                            "precision": precision.identity() if precision else None,
                            "rng_policy": RESPONSE_RNG_POLICY})
    model_paths = []
    for member in (range(cfg["members"]) if member_indices is None else member_indices):
        folder = monitor.weights / "response" / f"member_{member+1}"
        seed_all(cfg["seed"] + member)
        if precision is not None:
            precision.apply()  # seed_all resets numeric flags; re-apply policy (R6)
        model = build_response_model(shape, tensor(data.neighbors, device, torch.long)).to(device)
        if precision is not None:
            model.precision_policy = precision  # registered policy; legacy paths leave it unset
        timer = PhaseTimer(device) if timing_enabled else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        epoch0, step, best = 0, 0, float("inf")
        budget_exhausted = False
        saved = None
        metadata = {"stage": "response", "member": member, "model_config": asdict(shape),
                    "condition_batching": condition_batch_mode(model, {**cfg, "_training_stage": initial_stage})[1],
                    "precision": precision.identity() if precision else None,
                    "one_step_auxiliary_objective": "full_population_teacher_endpoint_rbf_mmd",
                    "fit_objective": "latent_cfm_plus_full_gene_reconstruction" if shape.backbone == "latent_flow" else "full_gene_cfm",
                    "response_config": cfg,
                    "rng_policy": RESPONSE_RNG_POLICY,
                    "mmd_training": "computed" if cfg["mmd_weight"] else "not_computed",
                    "data_identity": data.identity, "synthetic": config.get("synthetic", False),
                    "train_role": "T", "selection_role": "V", "coupling": cfg["coupling"]}
        if resume and (folder / "last.pt").exists():
            saved = response_checkpoint(folder / "last.pt", identity)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            epoch0, step = saved["epoch"], saved["counters"]["optimizer_step"]
            best = saved["extra"]["best"]
            restore_rng(saved["rng"])
        control = TrainingControl(control_config, config["epochs"],
                                  saved.get("extra", {}).get("training_control") if saved else None)
        if saved is not None and "training_control" not in saved.get("extra", {}):
            if control.enabled or control.schedule.get("mode", "full") != "full":
                raise ValueError("TRAINING_CONTROL_STATE_REQUIRED")
            control.advance(epoch0, step)
            control.state.update(best=best if math.isfinite(best) else None,
                last_validation_epoch=epoch0, last_validation_update=step,
                full_validations=epoch0, last_checkpoint_update=step)
        if saved is None:
            save_training(folder / "last.pt", model, optimizer, epoch=0, identity=identity,
                          metadata=metadata, counters={"optimizer_step": 0},
                          extra={"best": best, "training_control": control.state_dict()})
        draws = len(data.condition_ids("T"))
        updates = math.ceil(draws / cfg["conditions_per_update"])
        total = config["epochs"] * updates
        warmup = min(cfg["warmup_epochs"], config["epochs"]) * updates
        if saved is not None:
            validate_resume(saved, draws, cfg["conditions_per_update"], total, warmup)
        for epoch in range(epoch0, config["epochs"]):
            if control.stopped:
                break
            model.train()
            rng, mmd_rng, mmd_generator = response_rngs(cfg["seed"] + member * 100000 + epoch, device)
            keys = list(data.iter_response_epoch("T", cfg["seed"] + member * 100000 + epoch))
            computing_mmd = cfg["mmd_weight"] and control.stage == "full"
            aggregate = {"cfm_loss": 0., "mmd_loss": 0. if computing_mmd else None,
                         "weighted_mmd_loss": 0. if computing_mmd else None,
                         "mmd_weight": 0., "mmd_computed": 0., "loss": 0., "one_step_auxiliary": 0.,
                         "latent_cfm_loss": 0. if shape.backbone == "latent_flow" else None,
                         "reconstruction_loss": 0. if shape.backbone == "latent_flow" else None}
            if len(keys) != draws:
                raise ValueError("RESPONSE_RESUME_POSITION_MISMATCH: epoch draw count")
            position = 0
            state = saved.get("extra", {}).get("update_resume") if saved is not None else None
            if state is not None:
                if state["keys"] != keys:
                    raise ValueError("RESPONSE_RESUME_POSITION_MISMATCH: condition order")
                position, aggregate = state["next_condition_index"], state["aggregate"]
                rng.bit_generator.state = state["numpy_rng"]
                mmd_rng.bit_generator.state = state["mmd_numpy_rng"]
                mmd_generator.set_state(state["mmd_torch_rng"].cpu())
            saved = None
            lr = optimizer.param_groups[0]["lr"]
            bar = monitor.progress(total=len(keys), desc=f"P {member+1}/{cfg['members']} epoch {epoch+1}",
                                   position=0, unit="condition", disable=True)
            bar.update(position)
            detail = ResponseTrainingProgress(bar, bar_factory=monitor.progress,
                state_path=monitor.logdir / "response_progress.json",
                enabled=monitor.config.get("progress", True), mmd_microbatch=cfg.get("mmd_microbatch", 1))
            detail.state = {**detail.state, "member": member + 1, "members": cfg["members"], "epoch": epoch + 1}
            # Stage-aware prefetch config: a non-full stage draws no MMD samples
            # and never advances the separate MMD stream (task W2). A mid-epoch
            # cfm->full transition is handled by the lazy MMD draw in
            # train_conditions, which reloads seconds on demand.
            stage_cfg = {**cfg, "_training_stage": control.stage,
                         "mmd_weight": cfg["mmd_weight"] if control.stage == "full" else 0.}
            prefetch = ResponsePrefetch(data, stage_cfg, enabled=prefetch_enabled(model, stage_cfg))
            epoch_stage = control.stage
            try:
                for start in range(position, len(keys), cfg["conditions_per_update"]):
                    group = keys[start:start+cfg["conditions_per_update"]]
                    group_batches = prefetch.consume(group, rng, mmd_rng)
                    prefetch.queue(keys[start+len(group):start+len(group)+cfg["conditions_per_update"]], rng, mmd_rng)
                    optimizer.zero_grad(set_to_none=True)
                    lr = learning_rate(step, total, warmup, cfg["lr"], cfg["min_lr"])
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    active_cfg = {**cfg, "_training_stage": control.stage}
                    batch_width, batch_reason = condition_batch_mode(model, active_cfg)
                    detail.state = {**detail.state, "condition_batching": batch_reason}
                    for offset in range(0, len(group), batch_width):
                        block = group[offset:offset+batch_width]
                        detail.condition_start(start + offset + 1, len(keys), ",".join(map(str, block)))
                        with _phase(timer, "train_block"):
                            block_metrics = train_conditions(model, data, block, rng, active_cfg, device, len(group),
                                progress=detail, mmd_rng=mmd_rng, mmd_generator=mmd_generator,
                                batches=group_batches[offset:offset+batch_width] if group_batches is not None else None)
                        for metrics in block_metrics:
                            for metric, value in metrics.items():
                                if value is not None:
                                    aggregate[metric] = (aggregate.get(metric) or 0.) + value / len(keys)
                            bar.update()
                            detail.condition_done()
                    detail("optimizer", 0, 1)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                    with _phase(timer, "step"):
                        optimizer.step()
                    step += 1
                    control.advance(epoch, step)
                    if stop_check is not None and not budget_exhausted and stop_check(step, epoch):
                        budget_exhausted = True
                    # Checkpoint commits first. Logging can lag a committed update
                    # if interrupted here, but can never advertise unsaved progress.
                    state = update_state(epoch, start + len(group), keys, aggregate,
                                         rng, mmd_rng, mmd_generator, total, warmup)
                    stage_before = control.stage
                    improved_update = False
                    if control.validation_due():
                        prefetch.drain()
                        with _phase(timer, "validation"):
                            validation, _ = evaluate_response(model, data, cfg, device, progress=monitor.progress)
                        improved_update = control.observe(validation["loss"]) and stage_before == "full"
                        if improved_update:
                            best = validation["loss"]
                        if control.stage != stage_before:
                            best = float("inf")
                        monitor.record(f"response/member_{member+1}", step,
                            {f"val/{key}": value for key, value in validation.items()}, axis="optimizer_step")
                    if budget_exhausted or control.checkpoint_due(force=improved_update):
                        control.checkpoint_saved()
                        arguments = dict(epoch=epoch, identity=identity, metadata=metadata,
                            counters={"optimizer_step": step, "condition_draw": epoch*draws + start + len(group)},
                            extra={"best": best, "update_resume": state, "training_control": control.state_dict(),
                                   "budget_exhausted": budget_exhausted})
                        with _phase(timer, "checkpoint"):
                            if improved_update:
                                save_training(folder / "best.pt", model, optimizer, **arguments)
                            save_training(folder / "last.pt", model, optimizer, **arguments)
                    monitor.record(f"response/member_{member+1}", step, {**metrics, "lr": lr}, axis="optimizer_step")
                    if control.stage != epoch_stage:
                        # Mid-epoch stage switch: the queued group was loaded under
                        # the old stage config. Drop it without committing its
                        # captured streams, then re-arm under the new stage (C2).
                        prefetch.discard()
                        epoch_stage = control.stage
                        prefetch.cfg = {**cfg, "_training_stage": control.stage,
                                        "mmd_weight": cfg["mmd_weight"] if control.stage == "full" else 0.}
                    if budget_exhausted or control.stopped:
                        break
            except BaseException as error:
                detail.close("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
                raise
            else:
                detail.close()
            finally:
                prefetch.close()
                bar.close()
            if budget_exhausted:
                # Keep the mid-epoch snapshot (with update_resume) as last.pt;
                # epoch-completion logic would overwrite it without the cursor
                # and could mislabel a partial epoch as complete (rework R4.1).
                break
            if control.stopped:
                break
            stage_before = control.stage
            control.advance(epoch+1, step)
            if control.stage != stage_before:
                best = float("inf")
            validation = {}
            improved = False
            if not budget_exhausted and (control.validation_due() or control.stopped):
                with _phase(timer, "validation"):
                    validation, _ = evaluate_response(model, data, cfg, device, progress=monitor.progress)
                # An epoch-boundary switch must not seed full-stage selection with
                # a checkpoint that has received no full-objective update yet.
                improved = (control.observe(validation["loss"]) if control.stage == stage_before else False)
                improved = improved and stage_before == "full"
                if improved:
                    best = validation["loss"]
                if control.stage != stage_before:
                    best = float("inf")
            counters = {"optimizer_step": step, "condition_draw": (epoch+1)*draws}
            control.checkpoint_saved()
            arguments = dict(epoch=epoch+1, identity=identity, metadata=metadata, counters=counters, extra={"best": best,
                "training_control": control.state_dict(),
                "schedule": {"total_updates": total, "warmup_updates": warmup},
                "train_aggregate": aggregate, "budget_exhausted": budget_exhausted})
            # Write the selected model before advancing last.pt's selection state.
            with _phase(timer, "checkpoint"):
                if improved:
                    save_training(folder / "best.pt", model, optimizer, **arguments)
                save_training(folder / "last.pt", model, optimizer, **arguments)
                if (epoch+1) % config["save_period"] == 0:
                    save_training(folder / f"epoch_{epoch+1}.pt", model, optimizer, **arguments)
            monitor.record(f"response/member_{member+1}", epoch+1,
                           {**{f"train/{key}": value for key, value in aggregate.items()},
                            **{f"val/{key}": value for key, value in validation.items()}, "lr": lr,
                            "gpu_gb": torch.cuda.memory_allocated(device)/1e9 if device.type == "cuda" else 0,
                            **({f"timing/{key}_s": value for key, value in timer.summary()["host_seconds"].items()}
                               if timer is not None else {})})
            if budget_exhausted:
                break
        model_paths.append(str(folder / "best.pt"))
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return model_paths


def load_response(path, data, device):
    saved = load_checkpoint(path)
    if saved["metadata"].get("data_identity") != data.identity:
        raise ValueError("MODEL_DATA_IDENTITY_MISMATCH")
    if saved["metadata"]["stage"] != "response" or saved.get("counters", {}).get("optimizer_step", 0) < 1:
        raise ValueError("TRAINED_RESPONSE_REQUIRED")
    cfg = ModelConfig(**saved["metadata"]["model_config"])
    model = build_response_model(cfg, tensor(data.neighbors, "cpu", torch.long))
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False)
