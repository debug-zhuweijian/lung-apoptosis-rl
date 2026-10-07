"""Atomic suite state and content identities. Never trust a directory as success."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8")).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def suite_path(base, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", run_id):
        raise ValueError("Suite ID must be a plain identifier, not a path")
    target = (Path(base) / run_id).resolve()
    if target.parent != Path(base).resolve():
        raise ValueError("Suite escaped research-suites directory")
    return target


def data_identity(value, root):
    """Refresh declared file paths; retain scalar versions/hashes supplied by caller."""
    if isinstance(value, dict):
        return {str(key): (file_digest(Path(key)) if Path(str(key)).is_absolute()
                          else data_identity(item, root)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [data_identity(item, root) for item in value]
    if isinstance(value, (str, Path)):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            if candidate.is_file():
                return {"path": str(candidate.resolve()), "sha256": file_digest(candidate)}
        except OSError:
            pass
    return value


def identity(config, root):
    paths = [*root.glob("*.py"), *(root / "src").rglob("*.py"),
             *(root / "scripts").rglob("*.py")]
    source = {str(p.relative_to(root)): file_digest(p) for p in sorted(set(paths))}
    effective = {key: value for key, value in config.items()
                 if key not in {"dashboard", "open_browser", "progress"}}
    return {"config": digest(effective), "source": digest(source),
            "data": digest(data_identity(config.get("suite_data_identity", {}), root))}


def artifact_hashes(result):
    paths = result.get("artifacts") if isinstance(result, dict) else None
    if not isinstance(paths, list) or not paths:
        raise ValueError("Task result must contain a nonempty artifacts list")
    hashes = {}
    for value in paths:
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Missing or non-absolute task artifact: {path}")
        hashes[str(path.resolve())] = file_digest(path)
    return hashes


def valid_completed(task, binding):
    if task.get("status") != "complete" or task.get("binding") != binding:
        return False
    try:
        return artifact_hashes(task["result"]) == task["artifact_hashes"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
