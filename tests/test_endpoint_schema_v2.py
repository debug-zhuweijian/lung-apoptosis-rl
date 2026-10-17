"""Synthetic contract/integration checks, never evidence of Annexin efficacy."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models import ModelConfig
from lung.engine.auxiliary import verify_auxiliary_dependencies
from lung.engine.checkpoint import load_checkpoint, save_training
from lung.engine.config import StageUnavailable, file_hash
from lung.engine.endpoint_contract import (
    REQUIRED_ASSETS, dependency_files_v2, endpoint_unit_index, feature_dependencies_v2,
    frozen_feature_identity_v2, make_manifest_v2,
)
from lung.engine.endpoints import (
    read_manifest, load_units, train_endpoint, load_endpoint, finalize_calibration,
)


class Monitor:
    def __init__(self, root):
        self.weights = root / "weights"
        self.weights.mkdir()

    def progress(self, values, **kwargs):
        return values

    def record(self, *args):
        pass


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    shape = ModelConfig(n_genes=8, d_model=8, n_heads=2, encoder_layers=1,
                        pad_layers=1, query_chunk=4, context_dim=2)
    data = SimpleNamespace(identity="schema2-synthetic-engineering-only")
    pipeline = {}
    for key in ("representation", "response", "pool", "state", "backward", "correction"):
        checkpoint = tmp_path / f"{key}.pt"
        save_training(checkpoint, torch.nn.Linear(2, 2), epoch=1,
                      identity="synthetic-" + key,
                      metadata={"data_identity": data.identity, "model_config": vars(shape),
                                "synthetic": True}, counters={"optimizer_step": 1})
        pipeline[key] = [str(checkpoint)] if key == "response" else str(checkpoint)
    gate = tmp_path / "gate.json"
    write_json(gate, {"synthetic_test_only": True, "enabled": False})
    pipeline["correction_gate"] = str(gate)
    assets = {}
    for key in sorted(REQUIRED_ASSETS):
        asset = tmp_path / f"{key}.json"
        write_json(asset, {"synthetic_asset": key})
        assets[key] = [{"path": str(asset), "sha256": file_hash(asset)}]
    measured = tmp_path / "synthetic-measurements.csv"
    measured.write_text("role,unit,y\n" + "".join(
        f"{role},{i},{.2 + .2 * i}\n" for role in ("T", "V", "C", "E") for i in range(2)),
        encoding="utf-8")
    dependencies = feature_dependencies_v2(pipeline, assets, base=tmp_path)
    generation = {"algorithm": "synthetic-test-only", "seed": 5, "population": 2, "ode_steps": 1}
    identity = frozen_feature_identity_v2(data.identity, dependencies, generation)
    protocols = {"fixture": {"label_type": "Annexin_positive_fraction", "denominator": "all events",
                              "sampling_gate": "synthetic gate", "protocol_identity": "synthetic-24h"}}
    rows_by_role, partitions = {}, {}
    for role in ("T", "V", "C", "E"):
        rows = []
        for i in range(2):
            observation = f"{role}-observation-{i}"
            rows.append({"role": role, "experimental_condition_id": f"{role}-condition-{i}",
                         "observation_ids": [observation], "experiment_id": f"{role}-experiment",
                         "source_id": "same-publication", "group_id": f"{role}-group-{i}",
                         "chemical_group_id": f"{role}-chemical-{i}", "protocol_id": "fixture",
                         "label_type": "Annexin_positive_fraction", "label_kind": "measured",
                         "denominator": "all events", "sampling_gate": "synthetic gate",
                         "initial_state": {"kind": "reference", "reference": "separate-control-population",
                                           "evidence": "synthetic unpaired reference only"},
                         "measurement_provenance": [{"observation_id": observation, "path": str(measured),
                            "sha256": file_hash(measured), "locator": f"{role}-line-{i}"}],
                         "feature_kind": "predicted", "feature_status": "complete",
                         "frozen_feature_identity": identity,
                         "population": [.1 + i / 10] * shape.d_model, "bio": [.2] * 27,
                         "context": [.1] * (3 * shape.context_dim),
                         "history": [[.2] * shape.history_input_dim],
                         "elapsed": 24., "horizon": 24., "label_fraction": .2 + i * .2})
        partition = tmp_path / f"endpoint-{role}.json"
        write_json(partition, rows)
        rows_by_role[role] = rows
        partitions[role] = {"path": partition.name, "sha256": file_hash(partition),
                            "unit_index": [endpoint_unit_index(row) for row in rows]}
    manifest = make_manifest_v2(data.identity, pipeline, assets, protocols, partitions,
        {"chemical_holdout": "identity_group", "source_holdout": False}, generation, base=tmp_path)
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    config = {"endpoint_manifest": str(path), "device": "cpu", "epochs": 1,
              "aux_batch": 2, "aux_lr": .001, "early_stopping": False,
              "calibration_steps": 2, "population": 2, "seed": 17, "synthetic": True,
              "response": {"inference_ode_steps": 1}, "history_support": "single_exposure_only"}
    yield SimpleNamespace(root=tmp_path, shape=shape, data=data, pipeline=pipeline,
                          assets=assets, manifest=manifest, path=path, config=config,
                          rows=rows_by_role, monitor=Monitor(tmp_path))
    torch.set_num_threads(previous_threads)


def rewrite_partition(case, role, rows, *, index=True):
    partition = case.root / case.manifest["partitions"][role]["path"]
    write_json(partition, rows)
    entry = case.manifest["partitions"][role]
    entry["sha256"] = file_hash(partition)
    if index:
        entry["unit_index"] = [endpoint_unit_index(row) for row in rows]
    write_json(case.path, case.manifest)


def test_schema2_fits_predicted_measured_without_observed_ids_or_rg(case, monkeypatch):
    original = Path.read_text
    opened = []

    def tracked(path, *args, **kwargs):
        opened.append(path.name)
        assert path.name not in {"endpoint-C.json", "endpoint-E.json"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    pipeline = train_endpoint(case.data, case.config, case.monitor, case.pipeline)
    saved = load_checkpoint(pipeline["endpoint"])
    assert {"endpoint-T.json", "endpoint-V.json"} <= set(opened)
    assert saved["metadata"]["selected_by"] == "V"
    assert saved["metadata"]["supervision_kind"] == "predicted_features_measured_endpoint"
    assert saved["metadata"]["development_groups"] == ["T-group-0", "T-group-1", "V-group-0", "V-group-1"]
    assert saved["counters"]["optimizer_step"] == 1
    assert not (case.monitor.weights / "endpoint" / "observed").exists()
    assert verify_auxiliary_dependencies(pipeline["endpoint"], pipeline)
    assert torch.isfinite(load_endpoint(pipeline["endpoint"], case.data, "cpu")(case.rows["T"][0]))


def test_schema2_calibration_preserves_roles_and_e_is_unopened(case, monkeypatch):
    pipeline = train_endpoint(case.data, case.config, case.monitor, case.pipeline)
    original = Path.read_text
    opened = []

    def tracked(path, *args, **kwargs):
        opened.append(path.name)
        assert path.name != "endpoint-E.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    pipeline = finalize_calibration(case.data, case.config, case.monitor, pipeline)
    assert "endpoint-C.json" in opened
    model = load_endpoint(pipeline["endpoint"], case.data, "cpu")
    assert model.calibrated
    assert model.endpoint_schema_version == 2 and model.endpoint_protocol_ids == ("fixture",)
    assert model.feature_generation == case.manifest["feature_generation"]
    assert finalize_calibration(case.data, case.config, case.monitor, pipeline) == pipeline
    assert model(torch.zeros(2, case.shape.state_dim), torch.zeros(2, 1, case.shape.history_input_dim),
                 torch.tensor([1, 1]), endpoint_protocol_id="fixture").shape == (2,)
    for protocol in (None, "unvalidated-protocol"):
        with pytest.raises(StageUnavailable, match="PROTOCOL_OUTSIDE_TRAINED_SCOPE"):
            model(torch.zeros(2, case.shape.state_dim), torch.zeros(2, 1, case.shape.history_input_dim),
                  torch.tensor([1, 1]), endpoint_protocol_id=protocol)


@pytest.mark.parametrize("stage", ["representation", "response", "pool", "state", "backward", "correction", "correction_gate"])
def test_every_actual_frozen_feature_dependency_is_bound(case, stage):
    path = case.pipeline[stage][0] if stage == "response" else case.pipeline[stage]
    with Path(path).open("ab") as stream:
        stream.write(b"changed-content")
    with pytest.raises(StageUnavailable, match="FROZEN_FEATURE_IDENTITY_MISMATCH"):
        read_manifest(case.data, case.config, case.pipeline)


@pytest.mark.parametrize("asset", sorted(REQUIRED_ASSETS))
def test_feature_assets_are_content_checked(case, asset):
    Path(case.assets[asset][0]["path"]).write_text("changed-content", encoding="utf-8")
    with pytest.raises(StageUnavailable, match="ASSET_HASH_MISMATCH"):
        read_manifest(case.data, case.config, case.pipeline)


def test_old_generator_source_records_do_not_constrain_feature_loading(case):
    stale = {"path": str(case.root / "old-generator.py"), "sha256": "0" * 64}
    dependencies = case.manifest["feature_dependencies"]
    dependencies["assets"]["feature_generator"] = [stale]
    dependencies["assets"]["condition_inputs"].append(stale)
    identity = frozen_feature_identity_v2(case.data.identity, dependencies, case.manifest["feature_generation"])
    case.manifest["frozen_feature_identity"] = identity
    for role, rows in case.rows.items():
        rewrite_partition(case, role, [{**row, "frozen_feature_identity": identity} for row in rows])
    path, manifest = read_manifest(case.data, case.config, case.pipeline)
    assert len(load_units(path, manifest, "T", case.shape)) == 2
    assert all(p.suffix != ".py" for p in dependency_files_v2(path, manifest))
    assert "feature_generator" not in feature_dependencies_v2(case.pipeline, dependencies["assets"], base=case.root)["assets"]


def test_missing_or_untrained_checkpoints_do_not_become_ready(case):
    with pytest.raises(StageUnavailable, match="CHECKPOINTS_REQUIRED"):
        feature_dependencies_v2({k: v for k, v in case.pipeline.items() if k != "response"}, case.assets, base=case.root)
    saved = load_checkpoint(case.pipeline["state"])
    saved["epoch"] = 0
    torch.save(saved, case.pipeline["state"])
    case.manifest["feature_dependencies"] = feature_dependencies_v2(case.pipeline, case.assets, base=case.root)
    case.manifest["frozen_feature_identity"] = frozen_feature_identity_v2(case.data.identity,
        case.manifest["feature_dependencies"], case.manifest["feature_generation"])
    write_json(case.path, case.manifest)
    with pytest.raises(StageUnavailable, match="TRAINED_FEATURE_CHECKPOINT_REQUIRED"):
        read_manifest(case.data, case.config, case.pipeline)


@pytest.mark.parametrize("role", ["C", "E"])
@pytest.mark.parametrize("field", ["observation_ids", "chemical_group_id", "experiment_id", "group_id"])
def test_all_role_index_leakage_is_rejected_before_fitting(case, role, field):
    case.manifest["partitions"][role]["unit_index"][0][field] = copy.deepcopy(case.rows["T"][0][field])
    if field == "observation_ids":
        case.manifest["partitions"][role]["unit_index"][0]["measurement_provenance"][0]["observation_id"] = case.rows["T"][0]["observation_ids"][0]
    write_json(case.path, case.manifest)
    with pytest.raises(ValueError, match="DUPLICATE_ENDPOINT_UNIT|GROUP_LEAKAGE"):
        read_manifest(case.data, case.config, case.pipeline)


def test_relabelled_observation_and_group_cannot_duplicate_physical_measurement(case):
    index = case.manifest["partitions"]["E"]["unit_index"][0]
    index["measurement_provenance"][0]["locator"] = case.rows["T"][0]["measurement_provenance"][0]["locator"]
    write_json(case.path, case.manifest)
    with pytest.raises(ValueError, match="DUPLICATE_ENDPOINT_UNIT: location"):
        read_manifest(case.data, case.config, case.pipeline)


def test_source_and_chemical_holdout_modes_are_explicit(case):
    case.manifest["split_policy"]["source_holdout"] = True
    write_json(case.path, case.manifest)
    with pytest.raises(ValueError, match="GROUP_LEAKAGE: source_id"):
        read_manifest(case.data, case.config, case.pipeline)
    case.manifest["split_policy"] = {"source_holdout": False, "chemical_holdout": "condition"}
    case.manifest["partitions"]["E"]["unit_index"][0]["chemical_group_id"] = case.rows["T"][0]["chemical_group_id"]
    write_json(case.path, case.manifest)
    read_manifest(case.data, case.config, case.pipeline)


@pytest.mark.parametrize("field,value", [("label_type", "SYTOX_lethal_fraction"),
    ("denominator", "surviving cells only"), ("sampling_gate", "different gating"),
    ("label_kind", "predicted")])
def test_label_protocol_and_measurement_kind_cannot_be_substituted(case, field, value):
    case.manifest["partitions"]["T"]["unit_index"][0][field] = value
    write_json(case.path, case.manifest)
    with pytest.raises(StageUnavailable, match="UNIT_INDEX_INVALID|PROTOCOL_MISMATCH"):
        read_manifest(case.data, case.config, case.pipeline)


def test_observed_initial_state_requires_actual_sample_and_pairing_evidence(case):
    case.manifest["partitions"]["T"]["unit_index"][0]["initial_state"]["kind"] = "observed_population"
    write_json(case.path, case.manifest)
    with pytest.raises(StageUnavailable, match="OBSERVED_INITIAL_STATE_UNVERIFIED"):
        read_manifest(case.data, case.config, case.pipeline)


@pytest.mark.parametrize("field,value", [("feature_kind", "observed"), ("feature_status", "pending"),
    ("frozen_feature_identity", "stale"), ("expression_sample_id", "invented")])
def test_schema2_rejects_observed_pairing_bypass_and_incomplete_features(case, field, value):
    rows = copy.deepcopy(case.rows["T"])
    rows[0][field] = value
    rewrite_partition(case, "T", rows)
    path, manifest = read_manifest(case.data, case.config, case.pipeline)
    with pytest.raises(StageUnavailable, match="PREDICTED_FEATURES_REQUIRED"):
        load_units(path, manifest, "T", case.shape)


@pytest.mark.parametrize("field,value", [("population", []), ("bio", [1e309] * 27),
    ("history", []), ("label_fraction", 1.1), ("horizon", 0)])
def test_schema2_rejects_missing_nonfinite_features_and_invalid_targets(case, field, value):
    rows = copy.deepcopy(case.rows["T"])
    rows[0][field] = value
    partition = case.root / "endpoint-T.json"
    partition.write_text(json.dumps(rows), encoding="utf-8")
    case.manifest["partitions"]["T"]["sha256"] = file_hash(partition)
    write_json(case.path, case.manifest)
    with pytest.raises(ValueError, match="ENDPOINT_"):
        load_units(case.path, case.manifest, "T", case.shape)


def test_partition_rows_must_match_metadata_index(case):
    rows = copy.deepcopy(case.rows["T"])
    rows[0]["observation_ids"] = ["unindexed-observation"]
    rewrite_partition(case, "T", rows, index=False)
    with pytest.raises(StageUnavailable, match="PARTITION_INDEX_MISMATCH"):
        load_units(case.path, case.manifest, "T", case.shape)


def test_schema1_still_requires_original_observed_pairing_fields(case):
    row = copy.deepcopy(case.rows["T"][0])
    row.update(feature_kind="observed", pairing_verified=True, pairing_evidence="fixture")
    legacy = case.root / "legacy.json"
    write_json(legacy, [row])
    manifest = {"schema_version": 1, "partitions": {"T": {"path": legacy.name, "sha256": file_hash(legacy)}}}
    with pytest.raises(StageUnavailable, match="PAIRING_UNVERIFIED"):
        load_units(case.path, manifest, "T", case.shape)


def test_complete_partition_contract_and_non_numeric_index_are_required(case):
    case.manifest["partitions"]["C"]["unit_index"][0]["label_fraction"] = .2
    write_json(case.path, case.manifest)
    with pytest.raises(StageUnavailable, match="UNIT_INDEX_INVALID"):
        read_manifest(case.data, case.config, case.pipeline)
    del case.manifest["partitions"]["E"]
    write_json(case.path, case.manifest)
    with pytest.raises(StageUnavailable, match="PARTITIONS_REQUIRED"):
        read_manifest(case.data, case.config, case.pipeline)


def test_ready_builder_rejects_missing_heldout_features(case):
    rows = copy.deepcopy(case.rows["E"])
    rows[0]["population"] = []
    rewrite_partition(case, "E", rows)
    with pytest.raises(ValueError, match="FEATURE_SHAPE_OR_VALUE_MISMATCH"):
        make_manifest_v2(case.data.identity, case.pipeline, case.assets, case.manifest["protocols"],
                         case.manifest["partitions"], case.manifest["split_policy"],
                         case.manifest["feature_generation"], base=case.root)


def test_same_model_cannot_mix_incompatible_observation_definitions(case):
    case.manifest["protocols"]["different"] = {**case.manifest["protocols"]["fixture"],
        "denominator": "adherent survivors", "protocol_identity": "different-denominator"}
    write_json(case.path, case.manifest)
    with pytest.raises(StageUnavailable, match="INCOMPATIBLE_OBSERVATION_DEFINITIONS"):
        read_manifest(case.data, case.config, case.pipeline)


def test_multiple_reviewed_protocols_with_exact_same_observation_definition(case):
    case.manifest["protocols"]["second-source"] = {**case.manifest["protocols"]["fixture"],
        "protocol_identity": "same-detection-definition-other-source"}
    rows = copy.deepcopy(case.rows["T"])
    rows[0]["protocol_id"] = "second-source"
    rewrite_partition(case, "T", rows)
    pipeline = train_endpoint(case.data, case.config, case.monitor, case.pipeline)
    model = load_endpoint(pipeline["endpoint"], case.data, "cpu")
    assert model.endpoint_protocol_ids == ("fixture", "second-source")
    assert torch.isfinite(model(rows[0]))


def test_metadata_index_rejects_nested_measurement_values(case):
    case.manifest["partitions"]["E"]["unit_index"][0]["measurement_provenance"][0]["value"] = .2
    write_json(case.path, case.manifest)
    with pytest.raises(StageUnavailable, match="MEASUREMENT_PROVENANCE_REQUIRED"):
        read_manifest(case.data, case.config, case.pipeline)
