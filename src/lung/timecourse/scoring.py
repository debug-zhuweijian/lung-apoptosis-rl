"""Bounded response batching; ensemble mixtures and chain statistics stay per seed."""
import torch
from lung.engine.response import condition_kwargs, projected
from .nested import FrozenTimePredictor
from .nested_features import project, chain_features
from .auxiliary import infer


@torch.no_grad()
def projected_seed_batch(predictor, cid, reference, seeds):
    # Foreign adapters retain their documented scalar callback.
    if not isinstance(predictor, FrozenTimePredictor) or any(
            getattr(model.config, "reference_mode", "cell") == "population_mean"
            for model, _ in getattr(predictor, "models", ())):
        return [predictor.predict(cid, sample_count=len(reference), seed=seed,
                                  reference=reference)["projected"] for seed in seeds]
    n, device = len(reference), predictor.cfg["device"]
    populations = [[] for _ in seeds]
    batch = predictor.data.action_features(cid)
    offset = 0
    for member, (model, cfg) in enumerate(predictor.models):
        count = n // len(predictor.models) + int(member < n % len(predictor.models))
        if not count:
            continue
        initial = reference[offset:offset+count]
        offset += count
        noises = []
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed+member)
            noises.append(model.sample_noise(initial, generator=generator) if hasattr(model, "sample_noise")
                          else torch.randn(initial.shape, device=device, generator=generator))
        pieces = [[] for _ in seeds]
        micro = cfg.get("inference_microbatch", 2)
        for start in range(0, count, micro):
            end = min(start+micro, count)
            ca = model.encode_condition(**condition_kwargs(batch, (end-start)*len(seeds), device))
            result = model.sample(torch.cat([initial[start:end] for _ in seeds]), ca,
                noise=torch.cat([noise[start:end] for noise in noises]),
                steps=cfg["inference_ode_steps"], checkpoint=False)
            for rows, values in zip(pieces, result.split(end-start)):
                rows.append(values)
        for population, rows in zip(populations, pieces):
            population.append(projected(torch.cat(rows), predictor.data))
    outputs = []
    for population in populations:
        off = project(torch.cat(population), predictor.data)
        features = chain_features(predictor.data, cid, off, reference, predictor.regressors['B'], device)
        corrected = project(off + infer(predictor.regressors['K'], features, device), predictor.data)
        # Preserve the frozen U chain execution; it is not an Actor feature or reward penalty.
        infer(predictor.regressors['U'], features, device).clamp(0, 50).expm1().clamp_min(0)
        outputs.append(corrected if predictor.gate['enabled'] else off)
    return outputs
