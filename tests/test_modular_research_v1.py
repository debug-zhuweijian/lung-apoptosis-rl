"""CPU synthetic full modular path; no biological efficacy or audit claim."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine import data as data_module
from lung.engine import modular_config, modular_research, train, world
from lung.engine.checkpoint import atomic_json, atomic_save, load_checkpoint, seed_all
from lung.engine.config import ROOT as PROJECT_ROOT, StageUnavailable, file_hash, merge
from lung.engine.representation_cache import load_representation_cache
from lung.engine.synthetic import SyntheticData, smoke_config
from lung.monitoring import RunMonitor
from lung.monitoring import run as monitor_module
from test_response_research_runtime import TinyData, _equal


class ModularSyntheticData(TinyData):
    """Three conditions: two doses of drug A and one dose of drug B."""

    synthetic = True
    identity = "SYNTHETIC-MODULAR-INTEGRATION-H8-Z4-v1"
    manifest = {"synthetic": True, "signature": identity}

    def __init__(self):
        self.read_roles = []
        super().__init__()
        self.assets_dir = Path("SYNTHETIC-NO-REAL-DATA")
        self.controls = pd.DataFrame([
            {"condition_id": condition, "control_condition_id": f"{role}-control"}
            for role in ("T", "V") for condition in self.condition_ids(role)])

    def condition_ids(self, role, include_controls=False):
        prefix = {"T": "T", "V": "V", "fit": "T", "selection": "V"}[role]
        return [f"{prefix}-{i}" for i in range(3)] + ([f"{prefix}-control"] if include_controls else [])

    def role_cells(self, condition, role):
        self.read_roles.append(role)
        return super().role_cells(condition, role)

    def action_features(self, condition):
        index = int(condition[-1])
        group = "A" if index < 2 else "B"
        fingerprint = np.zeros(2048, np.float32)
        fingerprint[0 if group == "A" else 1] = 1
        dose = float(index + 1 if index < 2 else 1)
        return {"fingerprint": fingerprint, "dose": dose, "duration": 24.,
            "background_id": 0, "source_id": 0, "protocol_id": 0,
            "condition_id": condition, "group_id": group,
            "condition": {"dose_value": dose, "treatment_hours": 24}}

    def sample_response(self, condition, n, rng, role):
        self.read_roles.append(role)
        return super().sample_response(condition, n, rng, role)

    def sample_controls(self, condition, n, rng, role):
        return {key: value for key, value in self.sample_response(condition, n, rng, role).items()
                if key not in {"target", "target_cell_ids"}}

    def state_arrays(self, fold, role):
        self.read_roles.append(role)
        return SyntheticData.state_arrays(self, fold, role)


def tiny_config():
    config = smoke_config(modular_config.load_modular_config(PROJECT_ROOT / "configs/modular_research.yaml"))
    return merge(config, {
        "epochs": 1, "progress": False, "early_stopping": False,
        "calibration_steps": 1, "population": 2, "scoring_repeats": 1,
        "validation_candidates": 2, "decision_validation_repeats": 1, "diagnostic_repeats": 1,
        "potential_train_prefixes": 2, "potential_val_prefixes": 1,
        "potential_continuations": 1, "potential_batch": 2,
        "response": {"members": 1, "conditions_per_update": 1, "ot_samples": 2,
                     "mmd_samples": 2, "inference_microbatch": 2, "inference_ode_steps": 2},
        "research": {"precision": {"mode": "fp32"},
                     "latent": {"d_model": 8, "latent_dim": 4, "ae": {"batch": 2}}},
        "daac": {"actor_delay": 1, "warmup_transitions": 2, "batch_size": 2,
                 "actor_schedule": {"drug": 1, "dose": 1, "joint": 1}},
        "modular": {
            "stages": {name: {"epochs": 1} for name in modular_config.DEFAULT_STAGES}}})


@pytest.fixture
def integration_environment(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    data = ModularSyntheticData()
    for module in (modular_research, train, monitor_module):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for module in (modular_research, train, monitor_module, world):
        monkeypatch.setattr(module, "implementation_identity", lambda: "synthetic-integration-fixed-source")
    monkeypatch.setattr(data_module, "open_data", lambda *args, **kwargs: data)
    seed_all(712)
    yield tmp_path, data, tiny_config()
    torch.set_num_threads(old_threads)


def run(config, run_id, pipeline=None, state=None, stage=None, resume=False):
    monitor = RunMonitor(config, run_id=run_id, resume=resume)
    try:
        return modular_research.run_pipeline(config, monitor, pipeline or {}, state=state, stage=stage)
    finally:
        monitor.close("synthetic_test_finished")


def test_synthetic_full_chain_trains_all_modules_and_reuses_completed_artifacts(integration_environment, monkeypatch):
    root, data, config = integration_environment
    # Retired limits in older configurations must not stop any pipeline stage.
    config = merge(config, {"daac": {"max_seconds": 1e-9},
        "modular": {"response_budget_seconds": 1e-9,
            "stages": {name: {"budget_seconds": 1e-9} for name in modular_config.DEFAULT_STAGES}}})
    result = run(config, "modular-full")
    assert result["status"] == "complete"
    pipeline, state = result["pipeline"], result["state"]
    assert set(state["stages"]) == set(modular_config.DEFAULT_STAGES)
    ids = [row["run_id"] for row in state["stages"].values()]
    assert len(set(ids)) == len(ids) == 9
    assert all(row["status"] == "complete" for row in state["stages"].values())
    assert all("budget" not in row for row in state["stages"].values())
    assert not list((root / "experiments").rglob("*budget*.json"))
    for key in ("representation", "pool", "state", "potential"):
        saved = load_checkpoint(pipeline[key])
        assert saved["counters"]["optimizer_step"] > 0
        assert saved["metadata"]["data_identity"] == data.identity
        if key != "representation":
            assert saved["metadata"]["synthetic"]
        assert Path(pipeline[key]).name.startswith("selected-")
    representation = load_checkpoint(pipeline["representation"])
    shape = representation["metadata"]["model_config"]
    assert shape["d_model"] == 8 and shape["latent_dim"] == 4
    cache = load_representation_cache(pipeline["representation_cache"], "cpu", expected_data_identity=data.identity)
    assert set(cache.manifest["roles"]) == {"fit", "selection"}
    assert set(cache.cell_index["fit"]).isdisjoint(cache.cell_index["selection"])
    response = load_checkpoint(pipeline["response"][0])
    assert response["counters"]["optimizer_step"] > 0 and response["metadata"]["stage"] == "latent_cfm"
    policy = load_checkpoint(pipeline["daac"][0])
    assert policy["counters"]["actor_step"] == 3
    assert policy["counters"]["critic_step"] >= 3
    assert policy["learner"]["actor_schedule"]["completed"] == {"drug": 1, "dose": 1, "joint": 1}
    assert policy["extra"]["decision_runtime"]["status"] == "complete"
    assert {event["stage"] for event in policy["learner"]["actor_stage_events"]} == {"drug", "dose", "joint"}
    audit = json.loads(Path(pipeline["ode_audit"]).read_text(encoding="utf-8"))
    assert audit["synthetic_test_only"] and not audit["biological_numerical_resolution_verified"]
    assert set(data.read_roles) <= {"T", "V", "fit", "selection", "calibration"}
    metric_rows = pd.read_csv(root / "experiments" / "modular-full-daac" / "metrics.csv")
    assert np.isfinite(metric_rows["value"]).all()
    manifests = list((root / "experiments").glob("*/run_manifest.json"))
    assert all(json.loads(path.read_text(encoding="utf-8"))["formal_training"] is False for path in manifests)
    before = {path: path.stat().st_mtime_ns for path in (root / "checkpoints").rglob("*.pt")}
    second = run(config, "modular-full", pipeline, state, resume=True)
    assert second["status"] == "complete"
    assert all(row["reused"] for row in second["state"]["stages"].values())
    assert before == {path: path.stat().st_mtime_ns for path in (root / "checkpoints").rglob("*.pt")}

    # Verify reuse checks the policy itself against the currently assembled world.
    incompatible = deepcopy(policy)
    incompatible["metadata"]["versions"] = replace(policy["metadata"]["versions"], candidates="wrong-order")
    bad_policy = root / "wrong-candidates.pt"
    atomic_save(bad_policy, incompatible)
    with pytest.raises((ValueError, StageUnavailable), match="MISMATCH"):
        modular_research.verify_artifact("daac", {**pipeline, "daac": [str(bad_policy)]}, data, config)

    bad_cache = root / "forbidden-role-cache"
    shutil.copytree(pipeline["representation_cache"], bad_cache)
    manifest_path = bad_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["roles"]["final_test"] = manifest["roles"].pop("selection")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="CACHE_ROLE_FORBIDDEN"):
        modular_research.verify_artifact("cache", {**pipeline, "representation_cache": str(bad_cache)}, data, config)

    # A child directory alone is not a resumable learner: retain explicit Actor initialization.
    upstream = {key: value for key, value in pipeline.items() if key not in {"daac", "daac_best"}}
    initialize = merge(config, {"daac": {"init_from": pipeline["daac"][0]}})
    execute = modular_research.execute_stage

    def interrupt_before_training(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(modular_research, "execute_stage", interrupt_before_training)
    with pytest.raises(KeyboardInterrupt):
        run(initialize, "initialize-paused", upstream, stage="daac")
    paused = json.loads((root / "experiments/initialize-paused/modular_state.json").read_text(encoding="utf-8"))
    assert paused["status"] == "interrupted"
    assert (root / "experiments/initialize-paused-daac/active_config.yaml").is_file()
    assert not (root / "checkpoints/initialize-paused-daac/daac/seed_0/last.pt").exists()
    monkeypatch.setattr(modular_research, "execute_stage", execute)
    resumed = run(initialize, "initialize-paused", upstream, paused, stage="daac", resume=True)
    assert resumed["status"] == "stage_complete"
    initialized = load_checkpoint(resumed["pipeline"]["daac"][0])
    assert initialized["metadata"]["initialization"]["source_sha256"] == file_hash(pipeline["daac"][0])
    assert initialized["metadata"]["initialization"]["components"] == ["actor"]
    assert initialized["counters"]["actor_step"] == 3


def test_default_config_is_single_actor_modular_route():
    config = modular_config.load_modular_config(PROJECT_ROOT / "configs/modular_research.yaml")
    assert config["policy_seeds"] == [0]
    assert config["modular"]["run_ablations"] is False
    assert config["modular"]["mode"] == "modular_pipeline"
    assert config["daac"]["actor_schedule"] == {"drug": 20, "dose": 20, "joint": 160}
    assert config["response"]["members"] == 1 and "suite" not in config
    assert "dlpa" not in modular_config.DEFAULT_STAGES
    assert "max_seconds" not in config["daac"]
    assert "response_budget_seconds" not in config["modular"]
    assert all("budget_seconds" not in stage for stage in config["modular"]["stages"].values())


def test_legacy_budget_paused_ae_resumes_without_time_limit_and_preserves_updates(integration_environment, monkeypatch):
    from lung.engine import latent_pipeline
    from lung.engine.modular_budget import StagePaused

    root, _, config = integration_environment
    complete = run(config, "ae-continuous", stage="ae")
    expected = load_checkpoint(complete["pipeline"]["representation"])
    original = latent_pipeline.atomic_save

    def save_legacy_pause(path, payload):
        if Path(path).name == "last.pt" and payload["counters"]["optimizer_step"] == 1:
            original(path, {**payload, "extra": {**payload["extra"], "budget_exhausted": True}})
            raise StagePaused("budget_exhausted", path, stage="ae")
        original(path, payload)

    monkeypatch.setattr(latent_pipeline, "atomic_save", save_legacy_pause)
    first = run(config, "ae-paused", stage="ae")
    assert first["status"] == "paused" and first["stage"] == "ae"
    assert "representation" not in first["pipeline"]
    folder = root / "checkpoints" / "ae-paused-ae" / "latent_ae"
    interrupted = load_checkpoint(folder / "last.pt")
    assert interrupted["counters"]["optimizer_step"] == 1
    assert interrupted["extra"]["budget_exhausted"]
    assert not list(folder.glob("selected-*.pt"))
    assert not (root / "experiments" / "ae-paused" / "pipeline.json").exists()
    monkeypatch.setattr(latent_pipeline, "atomic_save", original)
    legacy_config = merge(config, {"modular": {"response_budget_seconds": 1.,
        "stages": {"ae": {"budget_seconds": 1.}}}})
    assert modular_config.science_identity(legacy_config) == modular_config.science_identity(config)
    old_limits = {"schema_version": 1, "budget_total_seconds": 1.,
                  "budget_consumed_seconds": 2707., "branches": {}}
    paths = [root / "experiments/ae-paused/response_budget.json",
             root / "experiments/ae-paused-ae/module_budget.json"]
    for path in paths:
        atomic_json(path, old_limits)
    before = {path: path.read_bytes() for path in paths}
    final = run(config, "ae-paused", first["pipeline"], first["state"], stage="ae", resume=True)
    assert final["status"] == "stage_complete"
    actual = load_checkpoint(final["pipeline"]["representation"])
    assert actual["counters"]["optimizer_step"] == expected["counters"]["optimizer_step"] == 3
    assert all(_equal(value, expected["model"][key]) for key, value in actual["model"].items())
    assert final["state"]["stages"]["ae"]["run_id"] == "ae-paused-ae"
    assert before == {path: path.read_bytes() for path in paths}
    assert not actual["extra"]["budget_exhausted"]


def test_dry_run_only_reads_and_prints_the_dependency_plan(integration_environment, monkeypatch, capsys):
    root, _, config = integration_environment
    path = root / "synthetic.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run opened data, a monitor, or interpreter runtime")

    monkeypatch.setattr(data_module, "open_data", forbidden)
    monkeypatch.setattr(modular_research, "verify_interpreter", forbidden)
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    assert modular_research.main(["--config", str(path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["workflow"] == "modular_pipeline" and report["run_ablations"] is False
    assert [row["stage"] for row in report["stages"]] == list(modular_config.DEFAULT_STAGES)
    assert all(row["action"] == "train" for row in report["stages"])
    assert before == {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    assert not (root / "experiments").exists() and not (root / "checkpoints").exists()


def test_resume_rejects_changed_scientific_configuration_before_opening_data(integration_environment, monkeypatch):
    _, _, config = integration_environment
    state = {"schema_version": 1, "identity": modular_config.science_identity(config), "stages": {}}
    changed = merge(config, {"daac": {"actor_lr": config["daac"]["actor_lr"] * 2}})

    def forbidden(*args, **kwargs):
        raise AssertionError("data opened before rejecting changed scientific configuration")

    monkeypatch.setattr(data_module, "open_data", forbidden)
    with pytest.raises(ValueError, match="MODULAR_RESUME_CONFIG_MISMATCH"):
        run(changed, "bad-science", state=state)
