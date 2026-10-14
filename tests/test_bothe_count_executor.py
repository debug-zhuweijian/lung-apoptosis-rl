"""Tiny/audit tests. No actual registered RNA or native human run launched."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lung.data import bothe_count_executor as executor
from lung.data import bothe_count_production as production
from lung.data.bothe_count_matrix import sample_axis, sha


@pytest.mark.parametrize("version", ["../escape", "..", "x/y", "x\\y", "", "/root", "C:abc", "x:y", ".hidden", "v" * 81])
def test_path_escape(version):
    with pytest.raises(ValueError):
        executor.version_path(version)


def test_version_safe():
    assert executor.version_path("rna-v1").parent == executor.PROCESSED.resolve()


@pytest.mark.parametrize("runs", [[], ["invented"], ["ERR5000305", "ERR5000305"], [True], "ERR5000305"])
def test_registered_runs_only(runs):
    with pytest.raises(ValueError):
        executor.registered_rows(runs)


def test_controlled_public_signatures():
    assert list(inspect.signature(executor.execute_registered_runs).parameters) == ["runs", "version"]
    assert list(inspect.signature(executor.execute_registered_run).parameters) == ["run", "version"]
    assert list(inspect.signature(executor.execute_registered_when_ready).parameters) == ["runs", "version"]
    assert list(inspect.signature(production.load_production).parameters) == ["version"]
    with pytest.raises(ValueError, match="CONTROLLED"):
        executor._CompletedRun({"status": "complete"})
    with pytest.raises(ValueError, match="CONTROLLED"):
        production._publish(Path("none"), [{"status": "complete", "actual_registered": True}], {})


def test_fixed_native_argv(tmp_path):
    row = executor.registered_rows(["ERR5000305"])[0]
    commands = executor.commands(executor.PROCESSED / "audit-argv" / row["run"], row)
    subjunc = commands["subjunc"]
    assert subjunc[:5] == [str(executor.TOOLS / "subjunc.exe"), "-T", "1", "-P", "3"]
    assert subjunc[-3:] == ["--SAMoutput", "-o", "aligned.sam"]
    assert subjunc[subjunc.index("-r") + 1].endswith("ERR5000305_1.fastq.gz")
    assert subjunc[subjunc.index("-R") + 1].endswith("ERR5000305_2.fastq.gz")
    counts = commands["featureCounts"]
    assert counts[-11:] == ["-f", "-p", "--countReadPairs", "-B", "-C", "--primary", "-Q", "10", "-s", "0", "strict_pairs/filtered.sam"]
    assert "-O" not in counts and "--fraction" not in counts


def test_synthetic_FASTQ_never_registered():
    row = sample_axis()[0]
    with pytest.raises(ValueError, match="REGISTERED"):
        production._validation_bindings(row, {"status": "complete", "mode": "audit_mock", "actual_registered": False})


def test_pending_preflight_does_not_scan_raw(monkeypatch):
    monkeypatch.setattr(executor, "pinned_bindings", lambda **kwargs: {"audit": "metadata"})
    monkeypatch.setattr(executor, "download_receipt", lambda run: Path("definitely-missing-receipt"))
    result = executor.preflight("ERR5000305")
    assert result["status"] == "pending_raw_download"
    assert result["RNA_counts_recovered"] is False
    assert result["index_hashes_actually_recomputed"] is False


def test_estimate_uses_ENA_denominator():
    value = executor.estimate_storage("ERR5000305")
    assert value["declared_ENA_read_count"] == 70453460
    assert value["estimated_primary_only_SAM_bytes_range"] == [70453460 * 250, 70453460 * 600]
    assert "unmeasured" in value["RAM"]
    assert "unconfirmed" in value["denominator_assumption"]


@pytest.mark.parametrize("kind", ["nonzero", "spawn"])
def test_native_failure_records_status(tmp_path, monkeypatch, kind):
    def fail(*args, **kwargs):
        if kind == "spawn":
            raise OSError("audit missing executable")
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(executor.subprocess, "run", fail)
    with pytest.raises((RuntimeError, OSError)):
        executor._stage("audit", ["audit-fixture"], tmp_path)
    value = json.loads((tmp_path / "audit.command.json").read_text())
    assert value["status"] == "failed"
    assert (tmp_path / "audit.log").exists()


def test_existing_version_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "version_path", lambda version: tmp_path)
    with pytest.raises(FileExistsError, match="NO_OVERWRITE"):
        executor.execute_registered_run("ERR5000305", "audit-version")


def test_failure_preserved_no_publication(tmp_path, monkeypatch):
    directory = tmp_path / "audit-version"
    monkeypatch.setattr(executor, "PROCESSED", tmp_path)
    monkeypatch.setattr(executor, "version_path", lambda version: directory)
    monkeypatch.setattr(executor, "pinned_bindings", lambda **kwargs: {})
    def reject(row, output):
        output.mkdir()
        (output / "unqualified.part").write_bytes(b"audit-only")
        raise ValueError("audit fixture qualification rejected")
    monkeypatch.setattr(executor, "_execute_one", reject)
    with pytest.raises(ValueError, match="audit fixture"):
        executor.execute_registered_run("ERR5000305", "audit-version")
    assert json.loads((directory / "driver.json").read_text())["status"] == "failed"
    assert (directory / "ERR5000305/unqualified.part").exists()
    assert not (directory / "manifest.json").exists()


@pytest.mark.parametrize("field,value", [("status", "running"), ("model_eligible", True), ("evidence", "audit_mock"), ("domain", "sc_UMI")])
def test_loader_rejects_unqualified_early(tmp_path, monkeypatch, field, value):
    version = "audit-fixture"
    run = "ERR5000305"
    manifest = {"schema": "bothe-source-bound-counts-v1", "version": version, "evidence": "actual_registered_RNA_native_execution",
                "domain": executor.DOMAIN, "runs": [run], "readyRNA": False, "model_eligible": False, "function_eligible": False}
    lifecycle = {"status": "complete", "runs": [run], "completed_runs": [run], "model_forward_training_or_PT_load": False}
    if field == "status": lifecycle[field] = value
    else: manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "driver.json").write_text(json.dumps(lifecycle))
    monkeypatch.setattr(executor, "version_path", lambda version: tmp_path)
    with pytest.raises(ValueError, match="LINEAGE"):
        production.load_production(version)


def test_real_pins_metadata_only():
    value = executor.pinned_bindings(verify_index_files=False)
    assert len([p for p in value if Path(p).parent == executor.INDEX and Path(p).name.startswith("hg19")]) == 6
    assert value[str(executor.ROOT / "src/lung/data/bothe_count_matrix.py")] == "e59cae1286c551e86e074225adac6f1105612bb0b83ae11c3f45151f44c55dd1"


def test_readback_object_keeps_NA_and_no_model_qualification():
    value = production.BulkCountReadback("audit", ("s",), ("1", "2"), ("retained", "no_source_intron"),
                                          ((0,), (None,)), ("s",), False)
    assert value.values == ((0,), (None,))
    assert value.model_eligible is False and value.function_eligible is False
    with pytest.raises(TypeError):
        production.BulkCountReadback("audit", (), (), (), (), (), False, model_eligible=True)


def test_matrix_fragment_zero_and_missing_mask(tmp_path, monkeypatch):
    row = executor.registered_rows(["ERR5000305"])[0]
    run_dir = tmp_path / row["run"]
    run_dir.mkdir()
    (run_dir / "counts.tsv").write_text(
        "Geneid\tChr\tStart\tEnd\tStrand\tLength\tstrict_pairs/filtered.sam\n"
        "f1\tchr1\t1\t10\t+\t10\t0\n"
        "f2\tchr1\t21\t30\t+\t10\t5\n", encoding="utf-8")
    genes = [{"ncbi_gene_id": "1", "status": "retained"},
             {"ncbi_gene_id": "2", "status": "retained"},
             {"ncbi_gene_id": "3", "status": "no_source_intron"}]
    features = {"f1": ("1", ("chr1", "1", "10", "+", "10")),
                "f2": ("2", ("chr1", "21", "30", "+", "10"))}
    monkeypatch.setattr(production, "annotation_axis", lambda: (genes, features))
    samples, matrix = production._matrix([row], tmp_path)
    position = [s["sample"] for s in samples].index(row["sample"]) + 2
    assert matrix[0][position] == "0" and matrix[1][position] == "5"
    assert matrix[2][position] == ""
    assert matrix[0].count("") == 11
    assert matrix[2][2:] == [""] * 12


def test_stage_complete_is_not_RNA_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(executor.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    value = executor._stage("audit_fixture", ["audit_mock"], tmp_path)
    assert value["status"] == "complete"
    assert "RNA_counts_recovered" not in value
    assert not (tmp_path / "manifest.json").exists()


def test_required_bindings_cannot_be_removed(tmp_path, monkeypatch):
    run, version = "ERR5000305", "audit-forged-binding"
    manifest = {"schema": "bothe-source-bound-counts-v1", "version": version,
                "evidence": "actual_registered_RNA_native_execution", "domain": executor.DOMAIN,
                "runs": [run], "readyRNA": False, "model_eligible": False, "function_eligible": False,
                "bindings": {}}
    lifecycle = {"status": "complete", "runs": [run], "completed_runs": [run], "model_forward_training_or_PT_load": False}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "driver.json").write_text(json.dumps(lifecycle))
    directory = tmp_path / run
    directory.mkdir()
    (directory / "execution.json").write_text("{}")
    # Isolate the mandatory-binding set check; fabricated fixture is rejected.
    monkeypatch.setattr(executor, "version_path", lambda version: tmp_path)
    monkeypatch.setattr(executor, "pinned_bindings", lambda **kwargs: {"audit_required": "audit_hash"})
    monkeypatch.setattr(production, "_run_bindings", lambda *args: {})
    with pytest.raises(ValueError, match="BINDINGS_MISMATCH"):
        production.load_production(version)


def test_wait_requires_commit_and_both_complete_files(tmp_path, monkeypatch):
    row = executor.registered_rows(["ERR5000305"])[0]
    monkeypatch.setattr(executor, "DATA", tmp_path / "data")
    receipt = tmp_path / "download.json"
    monkeypatch.setattr(executor, "download_receipt", lambda run: receipt)
    raw = executor.DATA / "raw" / row["run"]
    raw.mkdir(parents=True)
    part = raw / f"{row['run']}_2.fastq.gz.part"
    part.write_text("audit partial")
    steps = []

    def advance(seconds):
        steps.append(seconds)
        if len(steps) == 1:
            receipt.write_text("audit commit marker; not qualification")
            (raw / f"{row['run']}_1.fastq.gz").write_text("audit marker")
        elif len(steps) == 2:
            (raw / f"{row['run']}_2.fastq.gz").write_text("audit marker")
    monkeypatch.setattr(executor.time, "sleep", advance)
    lifecycle = {"completed_runs": [], "model_forward_training_or_PT_load": False}
    executor._wait_for_registered_download(row, tmp_path, lifecycle)
    assert steps == [30, 30] and lifecycle["completed_runs"] == []
    assert "waiting_run" not in lifecycle
    assert part.exists() and not (tmp_path / "manifest.json").exists()


def test_wait_complete_markers_do_not_grant_counts(tmp_path, monkeypatch):
    row = executor.registered_rows(["ERR5000305"])[0]
    monkeypatch.setattr(executor, "DATA", tmp_path / "data")
    receipt = tmp_path / "download.json"
    receipt.write_text("audit marker")
    monkeypatch.setattr(executor, "download_receipt", lambda run: receipt)
    raw = executor.DATA / "raw" / row["run"]
    raw.mkdir(parents=True)
    for end in (1, 2):
        (raw / f"{row['run']}_{end}.fastq.gz").write_text("audit marker")
    def unexpected_sleep(seconds):
        raise AssertionError("Complete markers should proceed to actual validator")
    monkeypatch.setattr(executor.time, "sleep", unexpected_sleep)
    lifecycle = {"completed_runs": []}
    executor._wait_for_registered_download(row, tmp_path, lifecycle)
    assert lifecycle == {"completed_runs": []}
    assert not (tmp_path / "manifest.json").exists()


def test_wait_interruption_preserves_waiting_evidence(tmp_path, monkeypatch):
    row = executor.registered_rows(["ERR5000305"])[0]
    monkeypatch.setattr(executor, "DATA", tmp_path / "data")
    monkeypatch.setattr(executor, "download_receipt", lambda run: tmp_path / "missing.json")
    def interrupted(seconds):
        raise KeyboardInterrupt("audit interruption")
    monkeypatch.setattr(executor.time, "sleep", interrupted)
    lifecycle = {"completed_runs": []}
    with pytest.raises(KeyboardInterrupt):
        executor._wait_for_registered_download(row, tmp_path, lifecycle)
    assert json.loads((tmp_path / "driver.json").read_text())["waiting_run"] == row["run"]
    assert lifecycle["completed_runs"] == []
