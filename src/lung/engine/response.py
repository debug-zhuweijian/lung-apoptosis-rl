"""The real CFM + full-ODE distribution objective, with exact gradient recomputation."""

import math

import numpy as np
import torch

from lung.models import ModelConfig, build_response_model, entropic_ot, sample_ot_pairs, rbf_mmd, physical_projection
from .checkpoint import load_checkpoint
from .numerics import policy_autocast
from .response_training import generation_noise, mmd_backward, one_step_loss


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


def load_response(path, data, device, *, saved=None):
    saved = load_checkpoint(path) if saved is None else saved
    if saved["metadata"].get("data_identity") != data.identity:
        raise ValueError("MODEL_DATA_IDENTITY_MISMATCH")
    if saved["metadata"]["stage"] != "response" or saved.get("counters", {}).get("optimizer_step", 0) < 1:
        raise ValueError("TRAINED_RESPONSE_REQUIRED")
    cfg = ModelConfig(**saved["metadata"]["model_config"])
    model = build_response_model(cfg, tensor(data.neighbors, "cpu", torch.long))
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False)
