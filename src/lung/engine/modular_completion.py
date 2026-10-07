"""Bind selected weights to evidence that their independent training finished."""

from pathlib import Path

from .checkpoint import atomic_save, load_checkpoint
from .config import StageUnavailable, config_hash, file_hash
from .training_control import checkpoint_completion


def completion_record(path, config):
    saved = load_checkpoint(path)
    extra = saved.get("extra", {})
    stage = saved["metadata"]["stage"]
    sealed = extra.get("module_completion")
    if sealed is not None:
        if (sealed.get("status") != "complete" or sealed.get("identity") != saved["identity"]
                or sealed.get("stage") != stage or sealed.get("data_identity") != saved["metadata"]["data_identity"]
                or sealed.get("selected_updates") != saved["counters"].get("optimizer_step")
                or sealed.get("actual_updates", 0) < sealed.get("selected_updates", 0)):
            raise StageUnavailable("MODULE_COMPLETION_MISMATCH")
        return sealed
    if extra.get("resume") or extra.get("update_resume") or extra.get("pause_reason") or extra.get("budget_exhausted"):
        raise StageUnavailable("MODULE_TRAINING_INCOMPLETE")
    tail_path = Path(path).parent / "last.pt"
    tail = load_checkpoint(tail_path) if tail_path.is_file() else saved
    if tail["identity"] != saved["identity"]:
        raise StageUnavailable("MODULE_COMPLETION_IDENTITY_MISMATCH")
    progress = tail.get("extra", {})
    if progress.get("resume") or progress.get("update_resume") or progress.get("pause_reason") or progress.get("budget_exhausted"):
        raise StageUnavailable("MODULE_TRAINING_INCOMPLETE")
    if stage in {"latent_ae", "latent_cfm", "latent_members"}:
        completion = progress.get("training_completion")
        if not completion or completion.get("status") != "complete":
            raise StageUnavailable("MODULE_COMPLETION_EVIDENCE_REQUIRED")
        if (completion.get("actual_epochs") != tail["epoch"]
                or completion.get("actual_updates") != tail["counters"].get("optimizer_step")):
            raise StageUnavailable("MODULE_COMPLETION_COUNTER_MISMATCH")
    else:
        target = progress.get("training_control", {}).get("target_epochs", config["epochs"])
        try:
            completion = checkpoint_completion(tail, target)
        except ValueError as error:
            raise StageUnavailable("MODULE_TRAINING_INCOMPLETE") from error
    return {**completion, "identity": saved["identity"], "stage": stage,
            "data_identity": saved["metadata"]["data_identity"],
            "selected_updates": saved["counters"].get("optimizer_step", 0),
            "selected_sha256": file_hash(path), "last_sha256": file_hash(tail_path) if tail_path.is_file() else file_hash(path)}


def seal_completed(path, config):
    """The immutable selected file survives later changes to best.pt or last.pt."""
    saved = load_checkpoint(path)
    if saved.get("extra", {}).get("module_completion"):
        completion_record(path, config)
        return str(path)
    completion = completion_record(path, config)
    target = Path(path).with_name(f"completed-{file_hash(path)[:16]}-{config_hash(completion)[:8]}.pt")
    if not target.is_file():
        atomic_save(target, {**saved, "extra": {**saved.get("extra", {}), "module_completion": completion}})
    completion_record(target, config)
    return str(target)
