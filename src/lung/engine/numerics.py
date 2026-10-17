"""Registered numeric policy for response research runs (implementation task W4).

The policy is stored in run configs and model metadata, and is applied
consistently at training, validation and member initialisation. ``seed_all``
defaults to IEEE FP32 with TF32 off; research entries pass their policy explicitly.

Registered meaning per mode:

- ``fp32``: every op on IEEE FP32, TF32 off. The strict comparison control.
- ``bf16``: dense network forwards run under CUDA BF16 autocast; parameters,
  optimizer state, AdamW moments, ODE state updates, CFM residual/reduction,
  Sinkhorn OT (FP64) and generative-time features keep their required dtype.
  Network outputs are cast back before any accumulation outside the context.
  Non-CUDA devices do not get autocast; the fallback is reported, not hidden.
- ``tf32``: TF32 allowed for matmul/cudnn. Listed separately so it can never be
  relabelled as "strict FP32"; not a default of any shipped config.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch

MODES = ("fp32", "bf16", "tf32")


@dataclass(frozen=True)
class PrecisionPolicy:
    mode: str = "fp32"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"PRECISION_MODE_UNKNOWN: {self.mode!r} not in {MODES}")

    @classmethod
    def from_config(cls, config):
        section = config.get("research", {}) if isinstance(config, dict) else {}
        section = section.get("precision", {}) if isinstance(section, dict) else {}
        mode = section.get("mode", "fp32")
        if not isinstance(mode, str):
            raise ValueError("research.precision.mode must be a string")
        return cls(mode=mode)

    def identity(self):
        """Serializable description entering config and checkpoint identities."""
        return {"mode": self.mode,
                "dense_forwards": "cuda_bf16_autocast" if self.mode == "bf16" else self.mode,
                "parameters": "fp32", "optimizer_state": "fp32",
                "loss_reduction": "fp32", "ode_state": "fp32",
                "ot_solver": "fp64_sinkhorn", "generative_time_features": "fp64",
                "tf32_enabled": self.mode == "tf32"}

    def apply(self):
        """Set the global matmul/cudnn flags registered by this policy.

        Called after ``seed_all`` by the research entry.
        """
        if self.mode == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # bf16 mode keeps every non-autocast op on IEEE FP32 (see autocast()).

    def bf16_active(self, device):
        """Whether BF16 autocast actually applies on this device."""
        return self.mode == "bf16" and torch.device(device).type == "cuda"

    def autocast(self, device):
        """Context for dense network forwards; FP32 modes are a no-op."""
        if not self.bf16_active(device):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def as_dict(self):
        return asdict(self)


def policy_autocast(model, device):
    """Wrap one dense forward in the model's registered policy (no-op if unset).

    Research runs set ``model.precision_policy`` after construction.
    Loss reductions stay on the caller's registered
    precision because call sites cast results with ``.float()`` before use.
    """
    policy = getattr(model, "precision_policy", None)
    return policy.autocast(device) if policy is not None else nullcontext()


def apply_seed(seed, policy=None):
    """Seed all streams, then re-apply the registered numeric policy.

    ``seed_all`` resets matmul/cudnn precision to its legacy defaults; research
    paths call this helper at every seeding and validation boundary so
    a registered TF32/BF16 policy is never silently undone (rework R6).
    """
    from .checkpoint import seed_all

    seed_all(seed)
    if policy is not None:
        policy.apply()
