"""Scientific data-boundary regressions, independent of formal training."""

import pandas as pd
import pytest

from lung.data.endpoint_alignment import (aggregate_units, assert_split_isolation,
                                          validate_alignment, validate_source_links)


def tables():
    conditions = pd.DataFrame([{
        "condition_id": "a", "source_id": "study", "protocol_id": "flow",
        "cell_line": "A549", "initial_state_id": "reference",
        "initial_state_kind": "reference", "treatment_hours": 24,
        "exposure_design": "single_exposure", "split_group": "study-series",
    }])
    interventions = pd.DataFrame([{
        "condition_id": "a", "compound_id": "compound", "source_compound_name": "drug",
        "dose_value": 0.5, "dose_unit": "uM", "identity_status": "reviewed",
    }])
    observations = pd.DataFrame([{
        "observation_id": f"y{i}", "condition_id": "a", "endpoint": "Annexin_positive_fraction",
        "value": y, "unit": "fraction", "label_kind": "measured", "replicate_id": str(i),
        "source_file": "source.xlsx", "source_sha256": "a" * 64,
        "source_locator": f"Sheet1!C{i}", "measurement_status": "observed",
        "exclusion_reason": "", "replicate_kind": "independent_experiment",
    } for i, y in enumerate((0.1, 0.2, 0.3), 1)])
    protocols = {"flow": {"denominator": "gated events", "sampling_gate": "singlets",
                          "source": "doi:source", "endpoint": "Annexin_positive_fraction"}}
    return conditions, interventions, observations, protocols


def test_measured_replicates_make_one_condition_unit_not_generated_cell_labels():
    c, i, o, p = tables()
    assert validate_alignment(c, i, o, p)["observed"] == 3
    unit = aggregate_units(c, o)
    assert len(unit) == 1
    assert unit.iloc[0].mean_fraction == pytest.approx(0.2)
    assert unit.iloc[0].measured_replicates == 3


def test_one_source_measurement_cannot_be_copied_into_extra_supervision():
    c, i, o, p = tables()
    duplicate = o.iloc[[0]].assign(observation_id="generated-extra")
    with pytest.raises(ValueError, match="duplicate"):
        validate_alignment(c, i, pd.concat([o, duplicate]), p)


@pytest.mark.parametrize("field,value,reason", [
    ("label_kind", "predicted", "generated"),
    ("value", 75.0, "fraction"),
    ("endpoint", "MTT_viability", "interchangeable"),
    ("source_sha256", "", "provenance"),
])
def test_invalid_supervision_is_rejected(field, value, reason):
    c, i, o, p = tables()
    changed = o.assign(**{field: value})
    with pytest.raises(ValueError, match=reason):
        validate_alignment(c, i, changed, p)


def test_missing_values_and_single_panels_do_not_acquire_fake_precision():
    c, i, o, p = tables()
    missing = o.iloc[[1]].assign(value=float("nan"), measurement_status="missing",
                                exclusion_reason="missing in source")
    chosen = pd.concat([o.iloc[[0]], missing])
    validate_alignment(c, i, chosen, p)
    unit = aggregate_units(c, chosen).iloc[0]
    assert unit.measured_replicates == 1
    assert pd.isna(unit.sd_fraction)


def test_source_series_cannot_cross_training_and_test():
    c, _, _, _ = tables()
    c = pd.concat([c, c.assign(condition_id="b")])
    assignment = pd.DataFrame({"condition_id": ["a", "b"], "partition": ["fit", "test"]})
    with pytest.raises(ValueError, match="crosses"):
        assert_split_isolation(c, assignment)


def test_distinct_endpoint_definitions_do_not_aggregate_together():
    c, _, o, _ = tables()
    alternate = o.assign(endpoint="SYTOX_lethal_fraction", observation_id=["s1", "s2", "s3"])
    assert len(aggregate_units(c, pd.concat([o, alternate]))) == 2


def test_death_fraction_cannot_be_relabelled_as_flow_protocol():
    c, i, o, p = tables()
    with pytest.raises(ValueError, match="disagrees"):
        validate_alignment(c, i, o.assign(endpoint="SYTOX_lethal_fraction"), p)


def test_reference_state_cannot_be_promoted_to_observed_without_evidence():
    c, i, o, p = tables()
    with pytest.raises(ValueError, match="requires sample"):
        validate_alignment(c.assign(initial_state_kind="observed_cell"), i, o, p)


def test_source_hashes_are_bound_per_observation():
    _, _, o, _ = tables()
    sources = [{"file": "source.xlsx", "sha256": "a" * 64}]
    validate_source_links(o, sources)
    with pytest.raises(ValueError, match="not bound"):
        validate_source_links(o.assign(source_sha256="b" * 64), sources)


def test_duplicate_replicate_at_a_different_location_is_not_an_extra_experiment():
    c, i, o, p = tables()
    duplicate = o.iloc[[0]].assign(observation_id="y-other", source_locator="Fig2!B5")
    with pytest.raises(ValueError, match="duplicate experimental replicate"):
        validate_alignment(c, i, pd.concat([o, duplicate]), p)


def test_same_sample_claim_requires_separate_evidence():
    c, i, o, p = tables()
    with pytest.raises(ValueError, match="pairing requires"):
        validate_alignment(c.assign(same_sample_expression_endpoint=True), i, o, p)


def test_reported_panel_does_not_invent_independent_replicate_count():
    c, _, o, _ = tables()
    panel = o.iloc[[0]].assign(replicate_kind="reported_panel_not_raw_replicate")
    row = aggregate_units(c, panel).iloc[0]
    assert row.available_numeric_records == 1
    assert row.measured_replicates == 0
    assert pd.isna(row.sd_fraction)
