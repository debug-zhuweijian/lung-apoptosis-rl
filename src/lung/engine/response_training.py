"""Shared response objectives; direct and replay retain the entire ODE objective."""

import torch

from .numerics import policy_autocast
from lung.models import rbf_mmd
from .inference_noise import response_noise


def generation_noise(model, reference, generator=None):
    return response_noise(model, reference, generator)


def resolve_mmd_mode(model, mode="auto"):
    if mode not in {"auto", "direct", "replay"}:
        raise ValueError("MMD backward mode must be auto, direct or replay")
    if mode == "auto":
        return "replay" if model.config.backbone == "gene_token" else "direct"
    return mode


def mmd_backward(model, reference, target, encode_condition, *, mode="auto", **kwargs):
    if resolve_mmd_mode(model, mode) == "replay":
        from .response import exact_mmd_backward
        return exact_mmd_backward(model, reference, target, encode_condition, **kwargs)
    return direct_mmd_backward(model, reference, target, encode_condition, **kwargs)


def direct_mmd_backward(model, reference, target, encode_condition, *, bandwidth, steps,
                        scale=1., noise=None, progress=None, microbatch=1):
    """One differentiable graph, full population MMD, all Heun steps retained."""
    if not isinstance(microbatch, int) or microbatch < 1:
        raise ValueError("MMD microbatch must be a positive integer")
    noise = generation_noise(model, reference) if noise is None else noise
    reference = model.condition_reference(reference)
    outputs = []
    for start in range(0, len(reference), microbatch):
        end = min(start + microbatch, len(reference))
        if progress is not None:
            progress("MMD direct", start, len(reference))
        outputs.append(model.sample(reference[start:end], encode_condition(end-start),
                                    noise=noise[start:end], steps=steps, checkpoint=False))
    loss = rbf_mmd(torch.cat(outputs), target, bandwidth=bandwidth, unbiased=False)
    if not torch.isfinite(loss):
        raise FloatingPointError("NONFINITE_MMD")
    (loss * scale).backward()
    if progress is not None:
        progress("MMD direct", len(reference), len(reference))
    return float(loss.detach())


def one_step_loss(model, reference, target, condition, *, bandwidth, noise=None, tau=None,
                  microbatch=None):
    """One population MMD, including cross-microbatch kernels; never average MMDs."""
    if model.config.backbone == "latent_flow":
        raise ValueError("LATENT_ONE_STEP_UNSUPPORTED")
    microbatch = len(reference) if microbatch is None else microbatch
    if not isinstance(microbatch, int) or microbatch < 1:
        raise ValueError("One-step microbatch must be positive")
    reference = model.condition_reference(reference)
    noise = generation_noise(model, reference) if noise is None else noise
    tau = torch.rand(len(reference), device=reference.device, dtype=reference.dtype) if tau is None else tau
    predictions, endpoints = [], []
    for start in range(0, len(reference), microbatch):
        stop = min(start + microbatch, len(reference))
        predicted, endpoint = model.one_step_endpoints(reference[start:stop], target[start:stop],
            condition[start:stop], noise=noise[start:stop], tau=tau[start:stop])
        predictions.append(predicted)
        endpoints.append(endpoint)
    return rbf_mmd(torch.cat(predictions), torch.cat(endpoints), bandwidth=bandwidth, unbiased=False)


def condition_batch_mode(model, cfg):
    requested = cfg.get("condition_microbatch", 1)
    if type(requested) is not int or requested < 1:
        raise ValueError("condition_microbatch must be a positive integer")
    if requested == 1:
        return 1, "serial_requested"
    if model.config.backbone != "compact_full_gene":
        # latent_flow stays serial: the grouped MMD path adds gene-space sigma
        # noise to targets, which the latent objective deliberately avoids.
        return 1, "serial_backbone"
    stage = cfg.get("_training_stage", "full")
    if stage not in {"full", "cfm"}:
        # one_step keeps the serial path: its auxiliary draws and one-step MMD
        # have no grouped implementation; grouping would silently drop them.
        return 1, "serial_training_stage"
    if model.config.reference_mode == "population_mean":
        # Per-condition control means must not blend across a grouped batch;
        # the per-condition pre-application variant is a separate follow-up.
        return 1, "serial_reference_mode"
    if resolve_mmd_mode(model, cfg.get("mmd_backward", "auto")) != "direct":
        return 1, "serial_backward_mode"
    if model.config.reference_mode == "observed":
        return requested, f"grouped_compact_observed_{stage}_direct"
    return requested, f"grouped_compact_condition_only_{stage}_direct"


def prefetch_enabled(model, cfg):
    for key in ("prefetch_group", "pin_memory"):
        if type(cfg.get(key, False)) is not bool:
            raise ValueError(f"{key} must be boolean")
    return cfg.get("prefetch_group", False) and condition_batch_mode(model, cfg)[0] > 1


def batch_tensor(value, device, pin_memory=False):
    from .response import tensor
    if pin_memory and torch.device(device).type == "cuda":
        import numpy as np
        host = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).pin_memory()
        return host.to(device=device, non_blocking=True)
    return tensor(value, device)


def train_conditions(model, data, keys, rng, cfg, device, divisor, progress=None,
                     *, mmd_rng=None, mmd_generator=None, batches=None):
    """Independent objectives share forwards, never population kernels.

    Prepare random inputs in serial order, including ORIGINAL per-microbatch draw
    shapes. Combining random draws can change CUDA generator offsets and is forbidden.
    """
    from .response import train_condition, condition_kwargs, mmd_weight
    from lung.models import entropic_ot, sample_ot_pairs
    width, _ = condition_batch_mode(model, cfg)
    if not keys:
        raise ValueError("train_conditions requires at least one condition key")
    if width > 1 and len(keys) > width:
        raise ValueError("Condition group exceeds condition_microbatch")
    if batches is not None and (width == 1 or len(batches) != len(keys)):
        raise ValueError("PREFETCH_REQUIRES_MATCHING_GROUPED_CONDITIONS")
    if width == 1 or (len(keys) == 1 and batches is None):
        return [train_condition(model, data, key, rng, cfg, device, divisor, progress,
                mmd_rng=mmd_rng, mmd_generator=mmd_generator) for key in keys]
    weight = mmd_weight(cfg) if cfg.get("_training_stage", "full") == "full" else 0.
    if weight and (mmd_rng is None or mmd_generator is None or mmd_rng is rng):
        raise ValueError("MMD requires separate NumPy and torch random streams")
    prepared = []
    for index, key in enumerate(keys):
        batch = batches[index][0] if batches is not None else data.sample_response(key, cfg["ot_samples"], rng, role="T")
        x, y = (batch_tensor(batch[name], device, cfg.get("pin_memory", False)) for name in ("control", "target"))
        if cfg["coupling"] == "entropic_ot_prior":
            coupling = entropic_ot(x, y)
        elif cfg["coupling"] == "random_product_control":
            coupling = x.new_full((len(x), len(y)), 1 / (len(x) * len(y)))
        else:
            raise ValueError("Unsupported coupling; no automatic substitution")
        ii, jj = sample_ot_pairs(coupling, n_pairs=cfg["cfm_samples"])
        chunks = []
        for start in range(0, len(ii), cfg["microbatch"]):
            left, right = x[ii[start:start+cfg["microbatch"]]], y[jj[start:start+cfg["microbatch"]]]
            noise = torch.randn_like(right)
            tau = torch.rand(len(right), device=right.device, dtype=right.dtype)
            chunks.append((left, right, noise, tau))
        item = {"batch": batch, "chunks": chunks, "cfm": 0.}
        if weight:
            # A group prefetched under a pre-transition stage carries no MMD draw;
            # draw it here so the separate MMD stream advances exactly once, as
            # in the serial path.
            second = (batches[index][1] if batches is not None and batches[index][1] is not None
                      else data.sample_response(key, cfg["mmd_samples"], mmd_rng, role="T"))
            mx, my = (batch_tensor(second[name], device, cfg.get("pin_memory", False)) for name in ("control", "target"))
            my = my + model.config.sigma_min * torch.randn(my.shape, device=device,
                dtype=my.dtype, generator=mmd_generator)
            item.update(second=second, x=mx, y=my,
                        noise=generation_noise(model, mx, generator=mmd_generator))
        prepared.append(item)

    def encode(batches, sizes):
        conditions = [condition_kwargs(batch, size, device) for batch, size in zip(batches, sizes)]
        return model.encode_condition(**{key: torch.cat([entry[key] for entry in conditions])
                                         for key in conditions[0]})

    for chunk_index in range(len(prepared[0]["chunks"])):
        chunks = [item["chunks"][chunk_index] for item in prepared]
        sizes = [len(chunk[0]) for chunk in chunks]
        left, right, noise, tau = [torch.cat([chunk[index] for chunk in chunks]) for index in range(4)]
        condition = encode([item["batch"] for item in prepared], sizes)
        with policy_autocast(model, device):
            losses = model.cfm_loss(left, right, condition, noise=noise, tau=tau,
                                    reduction="none").float()
        if not torch.isfinite(losses).all():
            raise FloatingPointError("NONFINITE_CFM")
        (losses.sum() / cfg["cfm_samples"] / divisor).backward()
        for item, values in zip(prepared, losses.detach().split(sizes)):
            item["cfm"] += float(values.sum()) / cfg["cfm_samples"]
        if progress:
            progress("CFM grouped backward", chunk_index+1, len(prepared[0]["chunks"]))

    mmds = [None] * len(prepared)
    if weight:
        endpoints = [[] for _ in prepared]
        micro = cfg.get("mmd_microbatch", 1)
        for start in range(0, cfg["mmd_samples"], micro):
            references = [item["x"][start:start+micro] for item in prepared]
            sizes = [len(x) for x in references]
            condition = encode([item["second"] for item in prepared], sizes)
            output = model.sample(torch.cat(references), condition,
                noise=torch.cat([item["noise"][start:start+micro] for item in prepared]),
                steps=cfg["train_ode_steps"], checkpoint=False)
            for destination, part in zip(endpoints, output.split(sizes)):
                destination.append(part)
        losses = [rbf_mmd(torch.cat(parts), item["y"], bandwidth=data.bandwidth, unbiased=False)
                  for parts, item in zip(endpoints, prepared)]
        total = torch.stack(losses).sum() * weight / divisor
        if not torch.isfinite(total):
            raise FloatingPointError("NONFINITE_MMD")
        total.backward()
        mmds = [float(loss.detach()) for loss in losses]
        if progress:
            progress("MMD grouped backward", len(keys), len(keys))
    return [{"cfm_loss": item["cfm"], "mmd_loss": mmd, "mmd_weight": weight,
             "weighted_mmd_loss": weight*mmd if mmd is not None else None,
             "mmd_computed": float(mmd is not None), "loss": item["cfm"] + weight*(mmd or 0.),
             "one_step_auxiliary": 0., "latent_cfm_loss": None, "reconstruction_loss": None}
            for item, mmd in zip(prepared, mmds)]
