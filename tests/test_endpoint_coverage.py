"""Coverage counts must respect connected experiment and chemical blocks."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("coverage_audit", Path(__file__).resolve().parents[1]
    / "scripts/audit_endpoint_coverage.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(name, source, experiment, chemical):
    return {"condition_id": name, "source_id": source, "experiment_block": experiment,
            "chemical_group_ids": [chemical]}


def test_shared_chemical_connects_experiments_transitively():
    rows = [row("a", "s1", "e1", "drug1"), row("b", "s1", "e1", "drug2"),
            row("c", "s2", "e2", "drug2"), row("d", "s3", "e3", "drug3")]
    assert audit.connected_groups(rows, source_holdout=False) == [["a", "b", "c"], ["d"]]


def test_source_holdout_does_not_count_same_study_as_independent_sources():
    rows = [row("a", "s1", "e1", "d1"), row("b", "s1", "e2", "d2")]
    assert len(audit.connected_groups(rows, source_holdout=False)) == 2
    assert len(audit.connected_groups(rows, source_holdout=True)) == 1


def test_duplicate_condition_rejected():
    item = row("a", "s1", "e1", "d1")
    with pytest.raises(ValueError, match="duplicate condition"):
        audit.connected_groups([item, item], source_holdout=True)
