"""Measured-unit/scale regressions; no model fitting or feature generation."""

import copy

import pytest

from lung.data.endpoint_protocol_supervision import EndpointProtocolSupervision, fraction_observation


@pytest.fixture(scope="module")
def supervision():
    return EndpointProtocolSupervision.load()


def test_alternative_nav_doses_share_exact_observations_and_total_weight(supervision):
    caption = supervision.view("nav_figure_caption_10uM")
    methods = supervision.view("nav_methods_11uM")
    assert {r["observation_id"] for r in caption} == {r["observation_id"] for r in methods}
    assert {r["candidate_id"] for r in caption}.isdisjoint(r["candidate_id"] for r in methods)
    assert sum(r["weight"] for r in caption) == sum(r["weight"] for r in methods) == 4
    for a, b in zip(caption, methods):
        assert a["label_fraction"] == b["label_fraction"]


def test_percent_conversion_does_not_change_raw_value_or_assay(supervision):
    nav = supervision.view("nav_figure_caption_10uM")
    row = next(r for r in nav if r["state_id"] == "nonsenescent" and ":navitoclax:" in r["observation_id"])
    assert row["raw_value"] == pytest.approx(70.66)
    assert row["label_fraction"] == pytest.approx(0.7066)
    assert row["raw_unit"] == "percent" and row["unit_conversion_multiplier"] == .01
    assert row["endpoint_definition"]["denominator"] == "total_image_counted_cells"
    assert row["reading_bounds_are_biological_confidence_interval"] is False


def test_real_total_annexin_keeps_harvested_population_and_unique_units(supervision):
    units = supervision.view("vielhauer_source_nominal")
    assert len(units) == len({r["unit_id"] for r in units}) == 20
    assert all(r["raw_unit"] == "fraction" and r["unit_conversion_multiplier"] == 1 for r in units)
    assert all(r["endpoint_definition"]["denominator"] == "post_protocol_harvested_parent_population" for r in units)
    assert all(r["terminal_time_h"] == 24 and not r["raw_replicates_recovered"] for r in units)


@pytest.mark.parametrize("view_ids", [
    ["nav_figure_caption_10uM", "nav_methods_11uM"],
    ["vielhauer_source_nominal", "vielhauer_source_nominal"],
])
def test_duplicate_Y_cannot_become_two_training_units(supervision, view_ids):
    with pytest.raises(ValueError, match="OVERLAPPING"):
        supervision.combine_views(view_ids)


def test_two_roles_allowed_without_inventing_calibration_or_selection(supervision):
    units = supervision.view("vielhauer_source_nominal")
    roles = {r["observation_id"]: ("fit" if i < 12 else "final_test") for i, r in enumerate(units)}
    assigned = supervision.view("vielhauer_source_nominal", roles=roles)
    assert {r["role"] for r in assigned} == {"fit", "final_test"}
    assert all(r["role"] == "unassigned" for r in supervision.view("vielhauer_source_nominal"))


def test_roles_cannot_silently_omit_or_add_an_observation(supervision):
    with pytest.raises(ValueError, match="ROLE_ASSIGNMENT"):
        supervision.view("nav_figure_caption_10uM", roles={})


def test_combined_sources_retain_definition_instead_of_numeric_pooling(supervision):
    rows = supervision.combine_views(["vielhauer_source_nominal", "nav_methods_11uM"])
    assert len(rows) == 24
    assert {r["endpoint_definition"]["assay"] for r in rows} == {"Annexin_V_PI", "image_Annexin_V"}


def test_returned_source_records_cannot_mutate_frozen_label_views(supervision):
    row = supervision.view("nav_methods_11uM")[0]
    row["source_record"]["digitized_mean_percent"] = -1
    assert supervision.view("nav_methods_11uM")[0]["source_record"]["digitized_mean_percent"] >= 0


@pytest.mark.parametrize("mutation", ["wrong_unit", "inverted_bounds", "nan"])
def test_invalid_labels_are_not_zero_filled(supervision, mutation):
    observation = copy.deepcopy(next(iter(supervision._labels.values())))
    if mutation == "wrong_unit":
        observation["unit"] = "relative_viability"
    elif mutation == "inverted_bounds":
        observation["source_record"]["measured_endpoint"]["pixel_lower_fraction"] = 1.1
    else:
        observation["value"] = float("nan")
    with pytest.raises(ValueError):
        fraction_observation(observation)
