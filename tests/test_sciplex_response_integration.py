"""Small real-call-path fixtures for source routing and endpoint-only inclusion."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from src.lung.data import TrainingData, prepare_training
from src.lung.data.common import sha256
from src.lung.data.sciplex_source import SPLIT_POLICY, SOURCE, _catalog
from src.lung.engine.ae_data import AEDataSupply
from src.lung.engine.world import BiologicalWorld
from tests.test_identity_prepare_v2 import identity_fixture, write_json


@pytest.fixture
def mixed_fixture(identity_fixture):
    root, base_config, chemical, fp = identity_fixture
    gene_path = root / "prepared/gene_map.parquet"
    genes = pd.read_parquet(gene_path)
    genes["tahoe_column"] = [5, 4, 3, 2, 1, 0]
    genes.to_parquet(gene_path, index=False)
    full_axis = pd.read_parquet(root / "prepared/tahoe_genes.parquet")
    full_axis["ensembl_id"] = list(reversed(full_axis.ensembl_id.tolist()))
    full_axis.to_parquet(root / "prepared/tahoe_genes.parquet", index=False)
    manifest_path = root / "prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["outputs"]:
        item["sha256"] = sha256(root / "prepared" / item["path"])
    write_json(manifest_path, manifest)
    base = prepare_training(root, base_config, build_preprocessing=False)
    source, sidecar, protocol = root / "sciplex", root / "sciplex-chemicals", root / "sciplex-protocol"
    for path in (source, sidecar, protocol):
        path.mkdir()
    chosen = chemical[chemical.eligible_for_structural_features].iloc[:12].copy().reset_index(drop=True)
    source_fp = fp[chosen.fingerprint_row].copy()
    chosen["catalog_number"] = [f"S{1000+i}" for i in range(len(chosen))]
    chosen["drug_id"] = "sciplex:" + chosen.catalog_number
    chosen["source_name"] = chosen.drug_name
    chosen["chemical_or_experiment_group_id"] = chosen.identity_id
    chosen["fingerprint_row"] = np.arange(len(chosen))
    chosen["identity_status"] = "source_reported_structure_not_independently_verified"
    chosen["identity_version"] = "sciplex-source-structures-v2"
    chosen.to_parquet(sidecar / "chemical_identity.parquet", index=False)
    np.save(sidecar / "fingerprints.npy", source_fp, allow_pickle=False)
    write_json(sidecar / "manifest.json", {"input_path_base": "project_root", "input_base": ".",
        "inputs_sha256": {}, "outputs_sha256": {name: sha256(sidecar / name)
            for name in ("chemical_identity.parquet", "fingerprints.npy")}})
    records, planned = [], []
    for time in (24, 72):
        for drug in ["S0000", *chosen.catalog_number]:
            for well in (["w0", "w1"] if drug == "S0000" else [drug]):
                n = 20 if drug == "S0000" else 8
                if time == 72 and drug == chosen.catalog_number.iloc[-1]:
                    n = 0
                planned.append({"Combo": well + f"p{time}", "well_oligo": well, "plate_oligo": f"p{time}",
                    "time_point": time, "replicate": "rep1", "catalog_number": drug,
                    "dose": 0 if drug == "S0000" else 1000, "vehicle": drug == "S0000", "hash_pass_cells": n})
                for _ in range(n):
                    index = len(records)
                    records.append({"cell": f"sci{index}", "sample": "synthetic", "cell_type": "A549", "source_cell_index": index,
                        "catalog_number": drug, "vehicle": drug == "S0000", "dose_uM": 0. if drug == "S0000" else 1.,
                        "time_point": float(time), "replicate": "rep1", "culture_plate": f"p{time}", "well_oligo": well,
                        "plate_oligo": f"p{time}", "author_hash_qc": True, "shard_id": 0, "row_in_shard": index})
    pd.DataFrame(planned).to_csv(protocol / "planned.csv", index=False)
    write_json(protocol / "contract.json", {"withdrawal": None, "decision_eligible": False,
                                             "time_semantics": "author_reported_endpoint_after_treatment"})
    pd.DataFrame(records).to_parquet(source / "a549_cells.parquet", index=False)
    genes.to_parquet(source / "common_genes.parquet", index=False)
    counts = np.random.default_rng(7).poisson(3, (len(records), 6)).astype(np.int32) + 1
    sparse.save_npz(source / "common.npz", sparse.csr_matrix(counts))
    write_json(source / "extraction_status.json", {"status": "complete_expression_extraction_not_training_ready",
        "shards": [{"shard_id": 0, "common": {"file": "common.npz", "bytes": (source / "common.npz").stat().st_size,
            "sha256": sha256(source / "common.npz"), "columns": 6, "rows": len(records)}}]})
    cfg = json.loads(base_config.read_text())
    cfg["assets"] = "training-joint"
    cfg["sciplex_source"] = {"split_policy": SPLIT_POLICY, "planned_wells": "sciplex-protocol/planned.csv",
        "time_contract": "sciplex-protocol/contract.json",
        "planned_dose_unit": "nM", "protocol_files": {p.relative_to(root).as_posix(): sha256(p) for p in protocol.iterdir()}}
    for key, path in (("extraction_manifest", source / "extraction_status.json"),
                      ("chemical_manifest", sidecar / "manifest.json"), ("base_training_manifest", base / "manifest.json")):
        cfg["sciplex_source"][key] = path.relative_to(root).as_posix()
        cfg["sciplex_source"][key + "_sha256"] = sha256(path)
    config = root / "configs/joint.json"
    write_json(config, cfg)
    return root, config, counts, base


def test_metadata_routes_both_axes_and_preserves_missing_planned_well(mixed_fixture):
    root, config, expected, base = mixed_fixture
    old_hash = sha256(base / "manifest.json")
    assets = prepare_training(root, config, build_preprocessing=False)
    reader = TrainingData(root, config, require_preprocessing=False, row_cache_options={"use_disk_cache": False})
    new = reader.conditions[reader.conditions.source_id.eq(SOURCE)]
    assert set(new.duration_h) == {24, 72} and not new.decision_eligible.any()
    assert new.exposure_duration_h.isna().all() and new.withdrawal_protocol.isna().all()
    assert new.missing_reason.eq("planned_well_no_qc_cells").sum() == 1
    assert reader.manifest["schema_version"] == 3
    rows = pd.concat(pd.read_parquet(assets / f"{split}.parquet") for split in ("train", "val", "test"))
    assert not rows.cell_id.duplicated().any()
    assert rows.groupby("split_group_id").role.nunique().max() == 1
    assert not rows.condition_id.isin(new.loc[~new.eligible, "condition_id"]).any()
    sci = rows[rows.source_id.eq(SOURCE)].iloc[:3]
    np.testing.assert_array_equal(reader.rows.read(sci).toarray(), expected[sci.matrix_row])
    old = rows[~rows.source_id.eq(SOURCE)].iloc[:3]
    original = sparse.load_npz(root / "source.npz")
    np.testing.assert_array_equal(reader.rows.read(old).toarray(), original[old.matrix_row][:, [5, 4, 3, 2, 1, 0]].toarray())
    combined = pd.concat([sci.iloc[[2]], old.iloc[[0]], sci.iloc[[0]], old.iloc[[0]]])
    expected_rows = np.vstack([expected[sci.iloc[2].matrix_row], original[old.iloc[0].matrix_row].toarray()[0][::-1],
                              expected[sci.iloc[0].matrix_row], original[old.iloc[0].matrix_row].toarray()[0][::-1]])
    np.testing.assert_array_equal(reader.rows.read(combined).toarray(), expected_rows)
    assert sha256(base / "manifest.json") == old_hash
    with pytest.raises(ValueError, match="PREPROCESSING_REQUIRED"):
        TrainingData(root, config)


def test_small_fit_preprocessing_bandwidth_and_ae_clone_use_same_router(mixed_fixture):
    root, config, _, _ = mixed_fixture
    prepare_training(root, config, device="cpu")
    reader = TrainingData(root, config)
    assert reader.manifest["preprocessing"]["sources"] == 2
    assert reader.bandwidths.shape == (5,)
    keys = [cid for cid in reader.condition_ids() if cid.startswith("sciplex:")][:1]
    assert keys
    data = SimpleNamespace(reader=reader)
    with AEDataSupply(data, 3) as supply:
        assert hasattr(supply.reader.rows, "clone_for_ae")
        actual = supply.sample_group(keys, np.random.default_rng(9))[0]
        expected = reader.sample_response(keys[0], n=3, rng=np.random.default_rng(9))
        np.testing.assert_array_equal(actual["control_cell_ids"], expected["control_cell_ids"])
        np.testing.assert_array_equal(actual["control"], expected["control"])
        np.testing.assert_array_equal(actual["target"], expected["target"])


def test_source_dummy_atoms_are_not_usable_structures(mixed_fixture):
    root, _, _, base = mixed_fixture
    path = root / "sciplex-chemicals/chemical_identity.parquet"
    source = pd.read_parquet(path)
    source.loc[0, "canonical_full_smiles"] = "*.Cl.CC"
    source.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="dummy atoms"):
        _catalog(path.parent, pd.read_parquet(base / "chemicals.parquet"), np.load(base / "fingerprints.npy"))


def test_endpoint_response_conditions_cannot_enter_existing_world_actions_or_roots():
    rows = pd.DataFrame([{"condition_id": "old", "eligible": True, "condition_type": "treated", "decision_eligible": True},
                         {"condition_id": "sci72", "eligible": True, "condition_type": "treated", "decision_eligible": False}])
    world = BiologicalWorld.__new__(BiologicalWorld)
    world.data = SimpleNamespace(synthetic=False, reader=SimpleNamespace(conditions=rows),
        condition_ids=lambda role: ["old", "sci72"], action_features=lambda cid:
            {"condition_id": cid, "group_id": cid, "dose": 1., "duration": 24. if cid == "old" else 72.})
    assert [row["condition_id"] for row in world._action_metadata().values()] == ["old"]
    with pytest.raises(ValueError, match="RESPONSE_ONLY_CONDITION"):
        world.root("sci72", "T")
    world.root = lambda key, role: key
    assert world.roots("T") == ["old"]


def test_unresolved_relative_stereochemistry_has_no_main_fingerprint(mixed_fixture):
    root, config, _, _ = mixed_fixture
    directory = root / "sciplex-chemicals"
    chemical = pd.read_parquet(directory / "chemical_identity.parquet")
    chemical.loc[chemical.index[-1], "eligible_for_structural_features"] = False
    chemical.loc[chemical.index[-1], "fingerprint_row"] = -1
    chemical.loc[chemical.index[-1], "identity_status"] = "source_relative_stereochemistry_not_unique_absolute_structure"
    chemical.to_parquet(directory / "chemical_identity.parquet", index=False)
    fp = np.load(directory / "fingerprints.npy")[:-1]
    np.save(directory / "fingerprints.npy", fp, allow_pickle=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["outputs_sha256"] = {name: sha256(directory / name) for name in manifest["outputs_sha256"]}
    write_json(directory / "manifest.json", manifest)
    cfg = json.loads(config.read_text())
    cfg["sciplex_source"]["chemical_manifest_sha256"] = sha256(directory / "manifest.json")
    write_json(config, cfg)
    prepare_training(root, config, build_preprocessing=False)
    reader = TrainingData(root, config, require_preprocessing=False)
    held = reader.conditions[reader.conditions.drug_id.eq(chemical.drug_id.iloc[-1])]
    assert held.fingerprint_row.eq(-1).all() and not held.eligible.any()
    assert held.missing_reason.eq("source_structure_not_encodable").sum() == 1
    for cid in held.condition_id:
        with pytest.raises(ValueError, match="DRUG_IDENTITY_UNRESOLVED"):
            reader.action_features(cid)


def test_engine_facade_reaches_time_and_source_condition_encoder(mixed_fixture, monkeypatch):
    root, config, _, _ = mixed_fixture
    prepare_training(root, config, build_preprocessing=False)
    from src.lung.engine import data as engine_data
    from src.lung.engine.response import condition_kwargs
    from src.lung.models import ModelConfig
    from src.lung.models.encoders import ConditionEncoder
    monkeypatch.setattr(engine_data, "ROOT", root)
    monkeypatch.setattr(engine_data, "project_path", lambda value: root / value)
    cfg = {"data": config.relative_to(root).as_posix(), "response": {"model": {"n_genes": 6, "n_sources": 2, "n_protocols": 2}}}
    data = engine_data.EngineData(cfg, require_preprocessing=False)
    rows = data.reader.conditions
    cid = rows.loc[rows.source_id.eq(SOURCE) & rows.duration_h.eq(72) & rows.eligible & rows.condition_type.eq("treated"), "condition_id"].iloc[0]
    action = data.action_features(cid)
    assert action["duration"] == 72 and action["source_id"] != data.reader.context_vocabulary["source_id"].index("Tahoe-100M")
    encoder = ConditionEncoder(ModelConfig(**cfg["response"]["model"]))
    with torch.no_grad():
        encoded = encoder(**condition_kwargs(action, 3, "cpu"))
    assert encoded.shape == (3, 256) and torch.isfinite(encoded).all()
    cfg["response"]["model"]["n_protocols"] = 1
    with pytest.raises(ValueError, match="MODEL_CONTEXT_CARDINALITY"):
        engine_data.EngineData(cfg, require_preprocessing=False)


@pytest.mark.parametrize("name", ["sciplex-chemicals/fingerprints.npy", "sciplex-protocol/planned.csv"])
def test_new_source_dependencies_are_required_even_without_asset_hash_checks(mixed_fixture, name):
    root, config, _, _ = mixed_fixture
    prepare_training(root, config, build_preprocessing=False)
    with (root / name).open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        TrainingData(root, config, require_preprocessing=False, verify_hashes=False)
