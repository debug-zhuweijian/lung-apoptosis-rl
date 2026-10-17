"""The entry selects one stage; omitted arguments cannot launch training."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung.engine import data as data_module
from lung.engine import modular_research
from lung.engine.config import StageUnavailable


def test_no_stage_does_not_load_config_or_start_training(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("No training stage was selected")

    monkeypatch.setattr(modular_research, "load_modular_config", forbidden)
    assert modular_research.main([]) == 0
    assert "--stage" in capsys.readouterr().out


def test_dry_run_only_reads_and_prints_selected_dependencies(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run opened data")

    monkeypatch.setattr(data_module, "open_data", forbidden)
    assert modular_research.main(["--stage", "response", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {"stage": "response", "depends_on": ["representation", "representation_cache"],
                      "missing": ["representation", "representation_cache"]}


def test_missing_upstream_model_stops_before_opening_data(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Upstream model was not supplied")

    monkeypatch.setattr(data_module, "open_data", forbidden)
    with pytest.raises(StageUnavailable, match="MODULAR_DEPENDENCIES_REQUIRED"):
        modular_research.run_pipeline({}, None, {}, stage="response")
