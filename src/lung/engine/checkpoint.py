"""Atomic model artifacts and random-stream isolation helpers."""

import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _replace_checkpoint(temporary, path):
    """Keep the old destination intact during transient Windows file sharing."""
    delays = (.05, .1, .2, .4, .8)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if (os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33)
                    or attempt == len(delays)):
                raise
            time.sleep(delays[attempt])


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_checkpoint(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, expected=None):
    # Checkpoints are generated locally by this project. Never load untrusted pickles.
    result = torch.load(Path(path), map_location="cpu", weights_only=False)
    if result.get("schema_version") != 1:
        raise ValueError("CHECKPOINT_SCHEMA_MISMATCH")
    if expected is not None and result.get("identity") != expected:
        raise ValueError("CHECKPOINT_IDENTITY_MISMATCH: data/model/reward/protocol changed")
    return result


def save_training(path, model, *, epoch, identity, metadata, counters=None,
                  extra=None):
    atomic_save(path, {"schema_version": 1, "model": model.state_dict(),
                      "epoch": int(epoch), "identity": identity,
                      "metadata": metadata, "counters": counters or {},
                      "extra": extra or {}})


def finalize_training(path, completion):
    """Attach run completion without changing selected weights or their counters."""
    from .training_control import checkpoint_completion
    saved = load_checkpoint(path)
    completed = {**saved, "extra": {**saved.get("extra", {}),
                                  "training_completion": dict(completion)}}
    checkpoint_completion(completed, completion["target_epochs"])
    atomic_save(path, completed)
