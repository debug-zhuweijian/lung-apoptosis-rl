"""Project-relative configuration; no dependence on the IDE working directory."""

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
STAGES = ("response", "pool", "state", "backward", "correction", "uncertainty",
          "endpoint", "potential", "daac", "dlpa", "all")


class StageUnavailable(RuntimeError):
    """A required artifact or a scientifically valid supervision chain is missing."""


def read_yaml(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def project_path(value, root=ROOT):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if (
            isinstance(value, dict) and isinstance(result.get(key), dict)
        ) else copy.deepcopy(value)
    return result


def config_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_identity():
    """Bind runs and simulated environments to executable project source content."""
    paths = [ROOT / "train.py", ROOT / "test.py", *sorted((ROOT / "src/lung").rglob("*.py"))]
    return config_hash({path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths})


def load_config(path=None):
    config = read_yaml(project_path(path or "configs/train.yaml"))
    for name in ("response", "daac", "dlpa"):
        config[name] = merge(read_yaml(ROOT / "configs" / f"{name}.yaml"),
                             config.get(name, {}))
    validate_config(config)
    return config


def validate_config(config):
    if config.get("stage") not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if not isinstance(config.get("epochs"), int) or config["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    if config.get("task") not in {"RESPONSE-A549-V1", "PROXY-A549-V1", "ANNEXIN-A549-V1"}:
        raise ValueError("Unknown task identity; no implicit reward substitution")
    from .training_control import TrainingControl
    TrainingControl(config, config["epochs"])
    if config.get("num_workers", 0) != 0:
        raise ValueError("This Windows V1 loader uses num_workers=0")
    for name in ('microbatch', 'mmd_microbatch', 'inference_microbatch'):
        value = config['response'].get(name, 1)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'response.{name} must be a positive integer')
    for name in ('member_workers', 'member_cpu_threads', 'condition_microbatch'):
        value = config['response'].get(name, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f'response.{name} must be a positive integer')
    if not config.get("synthetic", False) and config["response"]["model"]["n_genes"] != 32461:
        raise ValueError("The main V1 protocol requires the full 32461-gene axis")
    forbidden = {"cost", "risk_penalty", "dose_penalty", "time_penalty", "step_penalty"}
    if forbidden.intersection(config.get("reward", {})):
        raise ValueError("V1 has no cost/risk reward penalties")
    state_objective(config)


def state_objective(config):
    """Canonical state objective, also stored in run/checkpoint identities."""
    objective = config.get("state_loss", "weighted_ce")
    weight = config.get("state_aggregate_lambda", 0.)
    if objective not in ("weighted_ce", "ce", "ce_aggregate"):
        raise ValueError("state_loss must be weighted_ce, ce or ce_aggregate")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("state_aggregate_lambda must be finite and nonnegative")
    if objective != "ce_aggregate" and weight != 0:
        raise ValueError("state_aggregate_lambda must be zero unless state_loss is ce_aggregate")
    return {"state_loss": objective, "state_aggregate_lambda": float(weight)}


def verify_interpreter():
    expected = json.loads((ROOT / "configs/runtime.json").read_text(encoding="utf-8-sig"))
    selected = project_path(expected["python_executable"])
    if Path(sys.executable).resolve() != selected:
        raise StageUnavailable(
            f"INTERPRETER_MISMATCH: PyCharm 请选择 {selected}; 当前为 {sys.executable}")
    return expected
