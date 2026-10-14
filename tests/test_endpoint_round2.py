"""Regression against original printed flow quadrants, including source limits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extend_endpoint_alignment_round2 import STUDIES, quadrant_total


def test_early_fraction_is_not_total_annexin():
    assert quadrant_total(["75.9", "0.58", "19.4", "4.2"]) == pytest.approx(.236)


def test_published_rounding_is_allowed_but_missing_quadrant_mass_is_not():
    assert quadrant_total(["96.9", "1.56", "1.57", "0.010"]) == pytest.approx(.0158)
    with pytest.raises(ValueError, match="rounding"):
        quadrant_total(["90.00", "1.00", "3.27", "3.31"])


def test_real_exposure_and_missing_method_are_preserved():
    eight_h = next(s for s in STUDIES if s["compound"] == "8h")
    assert eight_h["hours"] == 48
    assert eight_h["hold"] == "supplement_assay_method_not_verified"
    for study in STUDIES:
        for _, values in study["rows"]:
            quadrant_total(values)
