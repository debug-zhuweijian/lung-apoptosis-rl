"""CPU interface tests with constructed fixtures, not real biological evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_modular_real.py"
SPEC = importlib.util.spec_from_file_location("verify_modular_real", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class SourceRows:
    def __init__(self):
        self.calls = []

    def read(self, rows):
        self.calls.append(rows.cell_id.tolist())
        n = len(rows)
        return sparse.csr_matrix((rows.matrix_row.to_numpy(dtype=np.float32) + 1,
                                  (np.arange(n), np.zeros(n, dtype=int))), shape=(n, 32461))


class SourceReader:
    def __init__(self, assets_path):
        self.assets_path = assets_path
        self.n_genes = 32461
        self.gene_ids = np.asarray([f"gene-{index}" for index in range(self.n_genes)])
        self.manifest = {"signature": "frozen-source-normalization", "status": "complete"}
        self.rows, self.state_calls = SourceRows(), []
        conditions, links, self.frames = [], [], {}
        for role in ("fit", "selection"):
            for index, condition in enumerate(self.condition_ids(role, include_controls=True)):
                is_control = "control" in condition
                conditions.append({"condition_id": condition, "source_id": "tahoe",
                                   "condition_type": "control" if is_control else "treated", "eligible": True})
                self.frames[role, condition] = pd.DataFrame({
                    "cell_id": [f"{condition}-cell-{j}" for j in range(6)],
                    "matrix_path": ["test-shard.npz"] * 6, "matrix_row": np.arange(6) + index * 6,
                    "condition_id": [condition] * 6, "role": [role] * 6})
                if not is_control:
                    links.extend({"condition_id": condition, "control_condition_id": f"{role}-control-{j}"}
                                 for j in range(2))
        self.conditions, self.controls = pd.DataFrame(conditions), pd.DataFrame(links)
        state_rows = [{"cell_id": f"state-{role}-{j}", "fold": "1_to_8", "role": role, "state_class": j}
                      for role in ("fit", "selection", "calibration", "final_test") for j in range(3)]
        self.state_rows = pd.DataFrame(state_rows)
        self.state_rows.to_parquet(assets_path / "tidyi_splits.parquet", index=False)

    def condition_ids(self, role, include_controls=False):
        if role not in ("fit", "selection"):
            raise PermissionError(role)
        return [f"{role}-treated-{j}" for j in range(3)] + (
            [f"{role}-control-{j}" for j in range(2)] if include_controls else [])

    def role_cells(self, condition, role):
        return self.frames[role, condition].copy()

    def normalize(self, counts):
        return counts.toarray().astype(np.float32)

    def action_features(self, condition):
        return {"fingerprint": np.zeros(2048, np.float32), "dose_um": 1., "duration_h": 24.,
                "source_id": 0, "background_id": 0, "protocol_id": 0, "condition_id": condition,
                "chemical_id": condition, "is_vehicle": "control" in condition}

    def state_arrays(self, fold, role):
        assert fold == "1_to_8" and role in ("fit", "selection", "calibration")
        self.state_calls.append((fold, role))
        selected = self.state_rows.loc[self.state_rows.role.eq(role)]
        return np.zeros((len(selected), self.n_genes), np.float32), selected.state_class.to_numpy()


@pytest.fixture
def source(tmp_path):
    reader = SourceReader(tmp_path)
    return SimpleNamespace(synthetic=False, identity="original-frozen-source", reader=reader,
        manifest=reader.manifest, n_genes=32461, mu=np.zeros(32461, np.float32),
        scale=np.ones(32461, np.float32), neighbors=np.arange(32461).reshape(-1, 1),
        assets_dir=tmp_path, bandwidth=1.)


def bounded(source):
    return verify.BoundedRealData(source, train_conditions=2, val_conditions=1, cells_per_condition=3)


def test_registration_is_deterministic_full_gene_and_separate_from_source(source):
    data = bounded(source)
    assert data.identity == bounded(source).identity and data.identity != source.identity
    assert data.synthetic is False and data.n_genes == 32461
    assert data.evidence["is_bounded_real_view"] and data.manifest["is_bounded_real_view"]
    assert data.evidence["source_identity"] == source.identity
    assert data.evidence["view_identity"] == data.identity and data.evidence["formal_training"] is False
    assert data.condition_ids("T") == ["fit-treated-0", "fit-treated-1"]
    assert data.condition_ids("V") == ["selection-treated-0"]
    assert data.reader.rows.source_cells_read == 0 and not source.reader.rows.calls
    for role, conditions in data.selection["cells"].items():
        assert any("control" in key for key in conditions)
        assert all(len(cells) <= 3 for cells in conditions.values())
        assert all(cell in source.reader.frames[role, key].cell_id.tolist()
                   for key, cells in conditions.items() for cell in cells)
    assert np.array_equal(data.reader.gene_ids, source.reader.gene_ids)


def test_sampling_uses_only_registered_rows_and_reuses_actual_source_reads(source):
    data = bounded(source)
    first = data.sample_response("fit-treated-0", 128, np.random.default_rng(17), role="T")
    calls = len(source.reader.rows.calls)
    second = data.sample_response("fit-treated-0", 128, np.random.default_rng(17), role="T")
    assert first["control"].shape == first["target"].shape == (128, 32461)
    assert np.array_equal(first["target"], second["target"])
    assert np.array_equal(first["control_cell_ids"], second["control_cell_ids"])
    assert len(source.reader.rows.calls) == calls
    registered = set().union(*(set(value) for value in data.selection["cells"]["fit"].values()))
    assert set(first["control_cell_ids"]) <= registered and set(first["target_cell_ids"]) <= registered
    assert data.reader.rows.source_cells_read <= 9
    assert first["coupling_type"] == "unpaired_independent_samples_for_explicit_OT_prior"


def test_epoch_sampling_keeps_original_source_condition_draw_count(source):
    data = bounded(source)
    drawn = list(data.iter_response_epoch("T", 53))
    assert len(drawn) == 2 and set(drawn) <= set(data.condition_ids("T"))
    assert drawn == list(data.iter_response_epoch("T", 53))


@pytest.mark.parametrize("role", ["fit", "selection", "calibration"])
def test_state_keeps_every_original_cell_and_label_in_each_allowed_role(source, role):
    data = bounded(source)
    arrays = data.state_arrays("1_to_8", role)
    assert arrays["x"].shape == (3, 32461)
    assert arrays["labels"].tolist() == [0, 1, 2]
    assert source.reader.state_calls == [("1_to_8", role)]
    registered = [row for row in data.selection["state_rows"] if row["role"] == role]
    assert len(registered) == 3


@pytest.mark.parametrize("role", ["calibration", "final_test", "E", "C", "residual_fit"])
def test_response_roles_cannot_escape_registered_T_V(source, role):
    data = bounded(source)
    with pytest.raises(PermissionError):
        data.reader.condition_ids(role)
    assert not source.reader.rows.calls


def test_wrong_role_condition_and_final_state_never_read_source(source):
    data = bounded(source)
    with pytest.raises(PermissionError):
        data.sample_response("fit-treated-0", 3, np.random.default_rng(2), role="V")
    with pytest.raises(PermissionError):
        data.state_arrays("1_to_8", "final_test")
    with pytest.raises(PermissionError):
        data.state_arrays("8_to_1", "fit")
    assert not source.reader.rows.calls and not source.reader.state_calls


def test_forged_row_pointer_is_rejected_before_source_read(source):
    data = bounded(source)
    row = data.reader.role_cells("fit-treated-0", "fit").iloc[:1].copy()
    row["matrix_row"] = 999
    with pytest.raises(PermissionError, match="ROW_LOCATION_MISMATCH"):
        data.reader.rows.read(row)
    assert not source.reader.rows.calls


@pytest.mark.parametrize("changed", ["source", "normalization", "gene_order", "cell_ids"])
def test_view_hash_binds_source_gene_normalization_and_exact_cell_ids(source, changed):
    before = bounded(source).identity
    if changed == "source":
        source.identity = "source-B"
    elif changed == "normalization":
        source.manifest = {**source.manifest, "signature": "normalization-B"}
    elif changed == "gene_order":
        source.reader.gene_ids = source.reader.gene_ids[::-1].copy()
    else:
        frame = source.reader.frames["fit", "fit-treated-0"].copy()
        frame.loc[0, "cell_id"] = "replacement-cell"
        source.reader.frames["fit", "fit-treated-0"] = frame
    assert before != bounded(source).identity


@pytest.mark.parametrize("field,value", [("synthetic", True), ("n_genes", 8)])
def test_source_must_be_real_and_full_gene_axis(source, field, value):
    setattr(source, field, value)
    with pytest.raises(ValueError, match="REQUIRES_FULL_32461_GENES"):
        bounded(source)


def test_cli_defaults_to_non_executing_plan(monkeypatch, capsys):
    monkeypatch.setattr(verify, "execute_verification", lambda _: pytest.fail("Default must not execute"))
    assert verify.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run" and result["requires"] == "--run"
    assert result["T_conditions"] == 16 and result["V_conditions"] == 1
    assert result["max_cells_per_condition"] == result["samples_per_update_condition"] == 128
    assert result["formal_training"] is False and result["full_audit_or_rl"] is False


def test_verification_config_keeps_calibration_semantics_and_resource_shape():
    base = {"synthetic": False, "calibration_steps": 2000, "calibration_lr": .01,
            "state_fold": "1_to_8", "state_loss": "weighted_ce",
            "research": {"precision": {"mode": "bf16"}}, "response": {"mmd_samples": 32}}
    original = deepcopy(base)
    config = verify.verification_config(base, verify.parser().parse_args([]))
    assert base == original and config["calibration_steps"] == 2000
    assert config["state_loss"] == "weighted_ce" and config["epochs"] == 1
    assert config["response"]["conditions_per_update"] == 16
    assert config["response"]["ot_samples"] == config["response"]["microbatch"] == 128
    assert config["research"]["latent"]["d_model"] == config["research"]["latent"]["latent_dim"] == 256


def test_checkpoint_evidence_marks_bounded_provenance_and_checks_real_updates(tmp_path, source):
    data = bounded(source)
    path = tmp_path / "model" / "best.pt"
    payload = {"schema_version": 1, "metadata": {"stage": "latent_ae", "data_identity": data.identity},
               "model": {"weight": torch.ones(2)}, "optimizer": {}, "shared": {},
               "counters": {"optimizer_step": 1}, "epoch": 1, "extra": {}}
    verify.atomic_save(path, payload)
    evidence = verify.checkpoint_evidence(path, data, expected_updates=1)
    saved = verify.load_checkpoint(path)
    assert evidence["finite"] and evidence["optimizer_steps"] == 1
    assert saved["metadata"]["source_identity"] == source.identity
    assert saved["metadata"]["view_identity"] == data.identity
    assert saved["metadata"]["is_bounded_real_view"] and saved["metadata"]["synthetic"] is False
    verify.atomic_save(path, {**saved, "counters": {"optimizer_step": 0}})
    with pytest.raises(ValueError, match="UPDATE_OR_FINITE"):
        verify.checkpoint_evidence(path, data)
