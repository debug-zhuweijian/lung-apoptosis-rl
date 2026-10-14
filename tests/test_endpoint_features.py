"""Executable frozen-feature plumbing on synthetic data, not biology evidence."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.models import ModelConfig, ResponseModel
from lung.engine.auxiliary import train_auxiliary
from lung.engine.checkpoint import atomic_json, save_training
from lung.engine.config import StageUnavailable
from lung.engine.endpoint_contract import feature_dependencies_v2, frozen_feature_identity_v2
from lung.engine.endpoint_features import (load_condition_inputs, condition_review, preflight,
    mapped_conditions, generate_to_directory, record, checked, ExternalConditionWorld,
    propose_reference_inputs, validate_reference)
from lung.engine.endpoints import MatchedEndpointModel
from lung.engine.response import condition_kwargs, tensor
from lung.engine.synthetic import SyntheticData


class Monitor:
    def __init__(self, root):
        self.weights, self.experiment = root / "weights", root / "experiment"
        self.weights.mkdir()
        self.experiment.mkdir()

    def progress(self, values, **kwargs):
        return values

    def record(self, *args):
        pass

    def say(self, *args):
        pass


class ReferenceData(SyntheticData):
    def sample_controls(self, condition_id, n, rng, role):
        values = super().sample_controls(condition_id, n, rng, role)
        return {**values, "control_cell_ids": np.asarray([f"synthetic-cell-{i}" for i in range(n)])}


@pytest.fixture
def fixture(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    shape = ModelConfig(n_genes=8, d_model=8, n_heads=2, encoder_layers=1,
                        pad_layers=1, query_chunk=4, context_dim=2)
    config = {"device": "cpu", "epochs": 1, "seed": 19, "population": 2,
        "aux_batch": 2, "state_batch": 6, "aux_lr": .001, "save_period": 1,
        "calibration_steps": 2, "state_fold": "1_to_8", "synthetic": True,
        "task": "PROXY-A549-V1", "history_support": "single_exposure_only",
        "response": {"model": vars(shape), "inference_ode_steps": 1, "inference_microbatch": 2}}
    data = ReferenceData(config)
    model = ResponseModel(shape, torch.as_tensor(data.neighbors))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    sample = data.sample_response("T-0", 2, np.random.default_rng(1), "T")
    ca = model.encode_condition(**condition_kwargs(sample, 2, "cpu"))
    model.cfm_loss(tensor(sample["control"], "cpu"), tensor(sample["target"], "cpu"), ca).backward()
    optimizer.step()
    path = tmp_path / "response.pt"
    save_training(path, model, optimizer, epoch=1, identity="synthetic-fixture",
        metadata={"stage": "response", "data_identity": data.identity, "model_config": vars(shape), "synthetic": True},
        counters={"optimizer_step": 1})
    pipeline = {"response": [str(path)]}
    monitor = Monitor(tmp_path)
    for stage in ("pool", "state"):
        pipeline = train_auxiliary(stage, data, config, monitor, pipeline)
    data.assets_dir = tmp_path / "assets"
    data.assets_dir.mkdir()
    pd.DataFrame({"ensembl_id": [str(i) for i in range(8)]}).to_parquet(data.assets_dir / "genes.parquet")
    np.savez(data.assets_dir / "normalization.npz", mean=data.mu, scale=data.scale)
    np.savez(data.assets_dir / "neighbors.npz", neighbors=data.neighbors)
    np.savez(data.assets_dir / "bandwidths.npz", bandwidths=np.ones(5))
    pd.DataFrame({"condition_id": ["T-0"], "control_condition_id": ["synthetic-control"]}).to_parquet(data.assets_dir / "control_links.parquet")
    atomic_json(data.assets_dir / "manifest.json", {"signature": "synthetic", "status": "complete",
        "files": {p.name: {"sha256": record(p)["sha256"]} for p in data.assets_dir.iterdir()}})
    identity = tmp_path / "chemical-identity.json"
    atomic_json(identity, {"synthetic_only": True})
    data_config = tmp_path / "data.json"
    atomic_json(data_config, {"assets": "assets", "drug_identity_manifest": identity.name,
        "drug_identity_manifest_sha256": record(identity)["sha256"]})
    data.reader = SimpleNamespace(conditions=pd.DataFrame([{"condition_id": k, "role": "fit",
        "condition_type": "treated", "chemical_or_experiment_group_id": "chemical-A", "dose_um": i+1.,
        "source_id": "Tahoe-fixture", "background_id": "A549-fixture", "protocol_id": "24h-fixture", "plate": "synthetic-plate"}
        for i, k in enumerate(("T-0", "T-1"))]))
    reference = {"kind": "reference", "reference_id": "synthetic-unpaired-fit-reference", "role": "T",
        "condition_id": "T-0", "data_identity": data.identity, "source_id": "Tahoe-fixture",
        "background_id": "A549-fixture", "protocol_id": "24h-fixture", "reference_plate": "synthetic-plate", "evidence": "synthetic fixture only",
        "source_files": [record(data.assets_dir / "manifest.json"), record(data.assets_dir / "control_links.parquet")]}
    mappings = [{"source_id": "external-source", "cell_line": "A549", "protocol_id": "external-flow",
        "target_source_id": "Tahoe-fixture", "target_background_id": "A549-fixture", "target_protocol_id": "24h-fixture",
        "reference_id": reference["reference_id"], "target_reference_plate": reference["reference_plate"],
        "status": "explicit_assumption_unvalidated", "evidence": "synthetic transfer assumption only",
        "source_files": [record(data.assets_dir / "manifest.json")]}]
    reference_path, mapping_path = tmp_path / "reference.json", tmp_path / "mapping.json"
    atomic_json(reference_path, reference)
    atomic_json(mapping_path, mappings)
    inputs, curated = tmp_path / "inputs", tmp_path / "curated"
    inputs.mkdir(); curated.mkdir()
    protocols = {"external-flow": {"endpoint": "Annexin_positive_fraction"},
                 "lethal": {"endpoint": "SYTOX_lethal_fraction"}}
    atomic_json(curated / "protocols.json", protocols)
    atomic_json(curated / "manifest.json", {"files": [{"file": "protocols.json", "sha256": record(curated / "protocols.json")["sha256"]}]})
    fp = np.zeros((2, 2048), dtype=np.uint8); fp[0, 1] = 1
    np.save(inputs / "fingerprints.npy", fp)
    compounds = pd.DataFrame([{"compound_id": "A", "fingerprint_row": 0, "structure_eligible": True,
        "identity_id": "chemical-A", "canonical_full_smiles": "CCO", "warning": "synthetic chemical only"},
        {"compound_id": "B", "fingerprint_row": 1, "structure_eligible": False,
        "identity_id": None, "canonical_full_smiles": None, "warning": "unresolved"}])
    requests, components = [], []
    for name, duration, design, chosen, protocol, control in [
        ("ready",24,"single_exposure",[0],"external-flow",False),
        ("late",48,"single_exposure",[0],"external-flow",False),
        ("combination",24,"simultaneous_combination",[0,1],"external-flow",False),
        ("unreviewed",24,"single_exposure",[1],"external-flow",False),
        ("death",24,"single_exposure",[0],"lethal",False),
        ("control",24,"single_exposure",[],"external-flow",True)]:
        requests.append({"condition_id": name, "source_id": "external-source", "protocol_id": protocol,
            "cell_line": "A549", "input_contract_blockers": "", "initial_state_kind": "reference",
            "same_sample_expression_endpoint": False, "is_control": control, "treatment_hours": duration,
            "exposure_design": design})
        components.extend({"condition_id": name, **compounds.iloc[index].to_dict(),
            "dose_value": 8., "dose_unit": "uM"} for index in chosen)
    for name, frame in (("compounds",compounds),("condition_requests",pd.DataFrame(requests)),("components",pd.DataFrame(components))):
        frame.to_parquet(inputs / f"{name}.parquet")
    atomic_json(inputs / "manifest.json", {"schema_version": 1,
        "files": {p.name: record(p)["sha256"] for p in inputs.iterdir()},
        "inputs_sha256": {"curated/manifest.json": record(curated / "manifest.json")["sha256"]}})
    bundle = load_condition_inputs(inputs, tmp_path)
    yield SimpleNamespace(root=tmp_path, data=data, config=config, pipeline=pipeline,
        reference=reference, mappings=mappings, reference_path=reference_path, mapping_path=mapping_path,
        data_config=data_config, bundle=bundle, inputs=inputs)
    torch.set_num_threads(previous_threads)


def test_actual_frozen_generation_matches_world_g_forward_and_records_extrapolation(fixture):
    f = fixture
    report = generate_to_directory(f.root, f.data, f.config, f.pipeline, f.bundle, f.reference, f.mappings,
        f.root / "generated", reference_path=f.reference_path, mapping_path=f.mapping_path,
        data_config_path=f.data_config, script_path=Path(__file__))
    assert report["generated_conditions"] == 1 and report["formal_training_ready"] is False
    assert report["measured_endpoint_values_read"] is False
    row = json.loads(Path(report["features"]["path"]).read_text())[0]
    assert row["source_transfer"]["outside_global_fit_dose_range"]
    assert row["source_transfer"]["background_transfer_validated"] is False
    assert row["initial_state"]["kind"] == "reference" and "expression_sample_id" not in row
    assert np.isfinite(row["population"]).all() and np.asarray(row["population"]).any()
    assert row["elapsed"] == row["horizon"] == 24
    assert row["state_probabilities_used_by_endpoint_head"] is False
    actions, provenance, _ = mapped_conditions(f.bundle, f.data, f.reference, f.mappings, base=f.root)
    world = ExternalConditionWorld(f.data, f.config, f.pipeline, actions)
    root = world.root("T-0", "T")
    x, context = world.propagate(root, ("ready",), row["generation_seed"], scoring_member=row["scoring_member"])
    features, history, _ = world._features(x, context, ("ready",))
    endpoint = MatchedEndpointModel(world.shape)
    assert torch.allclose(endpoint(row), endpoint(features[None], history[None], torch.tensor([1]))[0])
    current = feature_dependencies_v2(f.pipeline, report["feature_assets"], base=f.root)
    assert frozen_feature_identity_v2(f.data.identity, current, report["feature_generation"]) == row["frozen_feature_identity"]


def test_unreviewed_time_combination_and_other_endpoint_are_not_generated(fixture):
    rows = {r["experimental_condition_id"]: r for r in condition_review(fixture.bundle)}
    assert rows["ready"]["eligible_condition_input"]
    assert not any(rows[k]["eligible_condition_input"] for k in ("late","combination","unreviewed","death","control"))
    assert "unsupported_time_not_24h" in rows["late"]["reasons"]
    assert "not_Annexin_schema2_endpoint" in rows["death"]["reasons"]


def test_unknown_source_has_no_numeric_context_fallback(fixture):
    f = fixture
    actions, _, review = mapped_conditions(f.bundle, f.data, f.reference, [], base=f.root)
    assert not actions
    assert "explicit_source_context_mapping_missing" in next(r for r in review if r["experimental_condition_id"] == "ready")["reasons"]
    changed = copy.deepcopy(f.mappings)
    changed[0]["target_source_id"] = "unknown-plate"
    with pytest.raises(StageUnavailable, match="TARGET_NOT_REFERENCE_DOMAIN"):
        mapped_conditions(f.bundle, f.data, f.reference, changed, base=f.root)


@pytest.mark.parametrize("field", ["reference_id", "target_reference_plate"])
@pytest.mark.parametrize("value", ["WRONG", None])
def test_mapping_reference_identity_and_plate_must_match(fixture, field, value):
    f = fixture
    changed = copy.deepcopy(f.mappings)
    if value is None:
        changed[0].pop(field)
    else:
        changed[0][field] = value
    with pytest.raises(StageUnavailable, match="SOURCE_CONTEXT_REFERENCE_(ID|PLATE)_MISMATCH"):
        mapped_conditions(f.bundle, f.data, f.reference, changed, base=f.root)


def test_declared_reference_plate_must_match_actual_training_metadata(fixture):
    f = fixture
    reference = {**f.reference, "reference_plate": "WRONG"}
    mappings = copy.deepcopy(f.mappings)
    mappings[0]["target_reference_plate"] = "WRONG"
    with pytest.raises(StageUnavailable, match="REFERENCE_PLATE_MISMATCH"):
        mapped_conditions(f.bundle, f.data, reference, mappings, base=f.root)


@pytest.mark.parametrize("field", ["reference_id", "reference_plate"])
def test_missing_reference_identity_or_plate_is_rejected(fixture, field):
    f = fixture
    reference = {k: v for k, v in f.reference.items() if k != field}
    with pytest.raises(StageUnavailable, match="REFERENCE_ID_AND_PLATE_REQUIRED"):
        mapped_conditions(f.bundle, f.data, reference, f.mappings, base=f.root)


@pytest.mark.parametrize("descriptor", ["reference", "mapping"])
def test_generation_objects_cannot_diverge_from_descriptor_evidence_files(fixture, descriptor):
    f = fixture
    reference, mappings = copy.deepcopy(f.reference), copy.deepcopy(f.mappings)
    if descriptor == "reference":
        reference["evidence"] = "different object than the recorded reference JSON"
    else:
        mappings[0]["evidence"] = "different object than the recorded mapping JSON"
    with pytest.raises(StageUnavailable, match="FEATURE_DESCRIPTOR_CONTENT_MISMATCH"):
        generate_to_directory(f.root, f.data, f.config, f.pipeline, f.bundle, reference, mappings,
            f.root / "divergent", reference_path=f.reference_path, mapping_path=f.mapping_path,
            data_config_path=f.data_config, script_path=Path(__file__))
    assert not (f.root / "divergent").exists()


def test_descriptor_changed_during_generation_cannot_be_published(fixture, monkeypatch):
    from lung.engine import endpoint_features as module
    f = fixture
    generate = module.generate_feature_rows

    def changed_after_generation(*args, **kwargs):
        rows = generate(*args, **kwargs)
        changed = copy.deepcopy(f.mappings)
        changed[0]["evidence"] = "descriptor replaced during inference"
        atomic_json(f.mapping_path, changed)
        return rows

    monkeypatch.setattr(module, "generate_feature_rows", changed_after_generation)
    with pytest.raises(StageUnavailable, match="FEATURE_SOURCE_MISSING_OR_CHANGED"):
        generate_to_directory(f.root, f.data, f.config, f.pipeline, f.bundle, f.reference, f.mappings,
            f.root / "changed-during-generation", reference_path=f.reference_path, mapping_path=f.mapping_path,
            data_config_path=f.data_config, script_path=Path(__file__))
    assert not (f.root / "changed-during-generation" / "manifest.json").exists()
    assert not (f.root / "changed-during-generation" / "features.json").exists()


def test_preflight_collects_missing_assets_and_weights_without_placeholder_features(fixture):
    f = fixture
    manifest_path = f.data.assets_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "metadata_ready"
    manifest["files"].pop("normalization.npz")
    atomic_json(manifest_path, manifest)
    report = preflight(f.root, f.data_config, {}, f.bundle)
    assert report["status"] == "pending"
    assert {"MISSING_FROZEN_CHECKPOINT: response", "MISSING_FROZEN_CHECKPOINT: pool",
        "MISSING_FROZEN_CHECKPOINT: state", "MISSING_FROZEN_DATA_ASSET: normalization.npz"} <= set(report["missing_or_invalid"])
    assert not (f.root / "features.json").exists()


def test_source_hash_change_invalidates_input_and_reference(fixture):
    f = fixture
    with (f.inputs / "fingerprints.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(StageUnavailable, match="MISSING_OR_CHANGED"):
        load_condition_inputs(f.inputs, f.root)
    with pytest.raises(StageUnavailable, match="MISSING_OR_CHANGED"):
        checked({"path": str(f.reference_path), "sha256": "not-the-content-hash"}, f.root)


def test_reference_observed_claim_or_wrong_data_identity_is_rejected(fixture):
    f = fixture
    reference = {**f.reference, "kind": "observed_population"}
    with pytest.raises(StageUnavailable, match="EXPLICIT_FIT_REFERENCE_IDENTITY_REQUIRED"):
        generate_to_directory(f.root, f.data, f.config, f.pipeline, f.bundle, reference, f.mappings,
            f.root / "bad", reference_path=f.reference_path, mapping_path=f.mapping_path,
            data_config_path=f.data_config, script_path=Path(__file__))
    assert not (f.root / "bad").exists()


def test_generator_never_constructs_missing_models_or_zero_feature_fallback(fixture):
    f = fixture
    with pytest.raises(StageUnavailable, match="PIPELINE_INCOMPLETE"):
        generate_to_directory(f.root, f.data, f.config, {}, f.bundle, f.reference, f.mappings,
            f.root / "no-models", reference_path=f.reference_path, mapping_path=f.mapping_path,
            data_config_path=f.data_config, script_path=Path(__file__))
    assert not (f.root / "no-models").exists()


def test_reference_proposal_uses_real_index_hashes_and_excludes_nonfit_cells(fixture):
    f = fixture
    common = {"source_id": "Tahoe-100M", "background_id": "CVCL_0023",
              "protocol_id": "actual-training-protocol", "plate": "actual-plate"}
    conditions = pd.DataFrame([{**common, "condition_id": "tahoe:T-0", "role": "fit", "eligible": True,
        "condition_type": "treated", "duration_h": 24.},
        {**common, "condition_id": "tahoe:C-0", "role": "fit", "eligible": True,
         "condition_type": "control", "duration_h": 24.}])
    links = pd.DataFrame([{"condition_id": "tahoe:T-0", "control_condition_id": "tahoe:C-0", "plate": "actual-plate"}])
    counts = pd.DataFrame([{"condition_id": k, "role": "fit", "n_cells": 2} for k in conditions.condition_id])
    np.savez(f.root / "raw-fixture.npz", counts=np.arange(24).reshape(3, 8))
    cells = pd.DataFrame([{**common, "condition_id": "tahoe:C-0", "cell_id": f"synthetic-{i}",
        "matrix_path": "raw-fixture.npz", "matrix_row": i, "role": "fit" if i < 2 else "final_test"}
        for i in range(3)])
    for name, frame in (("conditions", conditions), ("control_links", links), ("role_condition_counts", counts), ("train", cells)):
        frame.to_parquet(f.data.assets_dir / f"{name}.parquet")
    manifest = json.loads((f.data.assets_dir / "manifest.json").read_text())
    for name in ("conditions", "control_links", "role_condition_counts", "train"):
        manifest["files"][f"{name}.parquet"] = {"sha256": record(f.data.assets_dir / f"{name}.parquet")["sha256"]}
    atomic_json(f.data.assets_dir / "manifest.json", manifest)
    reference_path, mappings_path = propose_reference_inputs(f.root, f.data_config, f.bundle, f.root / "proposal")
    reference = json.loads(reference_path.read_text())
    chosen = pd.read_parquet(reference["cell_index"]["path"])
    assert reference["unique_fit_cell_indices"] == 2 and chosen.role.eq("fit").all()
    assert reference["matched_initial_state_of_endpoint_experiment"] is False
    assert reference["external_endpoint_values_used_for_selection"] is False
    assert reference["kind"] == "reference" and reference["status"] == "cross_source_transfer_unvalidated"
    assert reference["control_condition_ids"] == ["tahoe:C-0"]
    assert all(m["target_source_id"] == "Tahoe-100M" and m["background_transfer_validated"] is False
               for m in json.loads(mappings_path.read_text()))
    validate_reference(reference, f.data.assets_dir / "manifest.json", f.data.assets_dir,
                       reference["data_identity"], f.root)
    with (f.root / "raw-fixture.npz").open("ab") as stream:
        stream.write(b"changed source")
    with pytest.raises(StageUnavailable, match="SOURCE_MISSING_OR_CHANGED"):
        validate_reference(reference, f.data.assets_dir / "manifest.json", f.data.assets_dir,
                           reference["data_identity"], f.root)
