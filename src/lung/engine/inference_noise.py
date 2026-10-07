"""Draw in the model's declared generative space without changing legacy RNG."""
import torch


def response_noise(model, reference, generator):
    sampler = getattr(model, "sample_noise", None)
    if sampler is not None:
        return sampler(reference, generator=generator)
    return torch.randn(reference.shape, generator=generator,
                       device=reference.device, dtype=reference.dtype)
