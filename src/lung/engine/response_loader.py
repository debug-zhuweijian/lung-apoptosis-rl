"""One stage dispatcher for frozen response inference and numerical audits."""

from .checkpoint import load_checkpoint


def load_response(path, data, device):
    stage = load_checkpoint(path).get("metadata", {}).get("stage")
    if stage == "response":
        from .response import load_response as load_legacy_response
        return load_legacy_response(path, data, device)
    if stage in ("latent_cfm", "latent_members"):
        from .latent_pipeline import load_latent_adapter
        return load_latent_adapter(path, data, device)
    raise ValueError(f"RESPONSE_UNSUPPORTED_STAGE: {stage} ({path})")
