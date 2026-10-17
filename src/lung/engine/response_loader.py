"""Load a response checkpoint once and dispatch by its saved stage."""

from .checkpoint import load_checkpoint


def load_response(path, data, device, *, saved=None):
    saved = load_checkpoint(path) if saved is None else saved
    stage = saved.get("metadata", {}).get("stage")
    if stage == "response":
        from .response import load_response as load_legacy_response
        return load_legacy_response(path, data, device, saved=saved)
    if stage in ("latent_cfm", "latent_members"):
        from .latent_pipeline import load_latent_adapter
        return load_latent_adapter(path, data, device, saved=saved)
    raise ValueError(f"RESPONSE_UNSUPPORTED_STAGE: {stage} ({path})")
