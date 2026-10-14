"""Synthetic source-contract counterexamples; no model or biological inference."""

import csv
import json

import pytest

from scripts import build_vielhauer_duration_challenge as challenge


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "synthetic-source"
    source.mkdir()
    conditions, components, totals, comparisons = [], [], [], []
    common = {"protocol_id": challenge.PROTOCOL, "denominator": challenge.DENOMINATOR,
              "sampling_gate": challenge.GATE}
    for figure, (arm, duration, wash, recovery) in challenge.ARMS.items():
        specs = [("vehicle", 0)] + [(drug, dose) for drug in ("bocnho", "cbi_indole2")
                                    for dose in (.05, .5, 10, 40, 1000)]
        for drug, dose in specs:
            cid = f"synthetic:{drug}:{dose}:{duration}"
            n, note = "3", "three_independent_experiments"
            if dose == 1000:
                n, note = (("2", "general_Annexin_1uM_each_drug_one_outlier_excluded") if duration == 24
                           else ("", "washout_section_n3_but_general_1uM_outlier_n2_scope_ambiguous"))
            conditions.append({**common, "condition_id": cid, "cell_line": "A549",
                "state_id": "a549_parental_standard_culture", "drug_id": drug, "dose_nM": dose,
                "drug_name": "synthetic source material", "protocol_arm": arm, "source_figure": figure,
                "exposure_duration_h": duration, "washout_at_h": "" if wash is None else wash,
                "drug_free_recovery_h": recovery, "terminal_time_h": 24,
                "is_vehicle": "true" if drug == "vehicle" else "false",
                "initial_RNA_reference": "not_available_in_this_source",
                "cross_figure_replicate_pairing": "not_established", "published_n": n, "n_status": note})
            measurement = {**common, "condition_id": cid, "source_image": "synthetic-source.txt",
                "source_figure": figure, "value_status": "digitized_published_group_mean",
                "uncertainty_kind": "pixel_reading_bounds_not_SEM"}
            for component, value in (("necrosis", 7), ("early_apoptosis", 10), ("late_apoptosis", 20)):
                components.append({**measurement, "component": component, "estimate_percent": value,
                    "reading_lower_percent": value - 1, "reading_upper_percent": value + 1})
            totals.append({**measurement, "label_type": "Annexin_positive_fraction",
                "estimate_percent": 30, "reading_lower_percent": 29, "reading_upper_percent": 31,
                "estimate_fraction": .30, "reading_lower_fraction": .29,
                "reading_upper_fraction": .31, "sem": ""})
    for drug in ("bocnho", "cbi_indole2"):
        for dose in (.05, .5, 10, 40, 1000):
            comparisons.append({"drug_id": drug, "dose_nM": dose, "terminal_time_h": 24,
                "continuous_condition_id": f"synthetic:{drug}:{dose}:24",
                "pulse_condition_id": f"synthetic:{drug}:{dose}:6", "duration_p_value": ""})
    for filename, records in zip(challenge.SOURCE_TABLES, (conditions, components, totals, comparisons)):
        write_csv(source / filename, records)
    for name in ("pixel_landmarks.json", "Vielhauer2013_PMC_fulltext.md", "synthetic-source.txt"):
        (source / name).write_text("SYNTHETIC engineering fixture, not scientific evidence", encoding="utf-8")
    return source


def change(source, name, edit):
    records = challenge.rows(source / name)
    edit(records)
    write_csv(source / name, records)


def test_real_clocks_and_units_survive_round_trip(source, tmp_path):
    target = tmp_path / "challenge"
    built = challenge.build(target, source)
    assert challenge.load_challenge(target) == built
    assert len(built["active_conditions"]) == 20
    assert len(built["reference_conditions"]) == 2
    pulse = [r for r in built["active_conditions"] if r["exposure_duration_h"] == 6]
    assert len(pulse) == 10 and all(r["terminal_time_h"] == 24 and r["drug_free_recovery_h"] == 18 for r in pulse)
    assert sorted({r["dose_uM"] for r in pulse}) == [.00005, .0005, .01, .04, 1]
    assert all(r["predicted_features"] is None and r["gene_expression_sample_id"] is None for r in pulse)
    assert built["contract"]["formal_training_ready"] is False
    with pytest.raises(ValueError, match="new output"):
        challenge.build(target, source)


@pytest.mark.parametrize("field,value,message", [
    ("terminal_time_h", "30", "common 24h"),
    ("exposure_duration_h", "24", "clocks mismatch"),
    ("drug_free_recovery_h", "24", "clocks mismatch"),
    ("washout_at_h", "", "clocks mismatch"),
    ("dose_nM", "50000", "drug or dose"),
    ("initial_RNA_reference", "observed_t0", "Unverified RNA"),
    ("denominator", "original_seeded_cells", "Condition protocol"),
])
def test_rejects_false_exposure_pairing_and_units(source, field, value, message):
    def edit(rows):
        row = next(r for r in rows if r["is_vehicle"] == "false" and r["source_figure"] == "Fig5C_left")
        row[field] = value
    change(source, "bocnho_conditions.csv", edit)
    with pytest.raises(ValueError, match=message):
        challenge.prepare(source)


def test_cannot_call_whole_stacked_bar_annexin(source):
    def edit(rows):
        rows[1].update(estimate_percent="37", reading_upper_percent="38", estimate_fraction=".37", reading_upper_fraction=".38")
    change(source, "bocnho_total_annexin.csv", edit)
    with pytest.raises(ValueError, match="excluding necrosis"):
        challenge.prepare(source)


def test_high_dose_n_is_not_filled_from_generic_n3(source):
    def edit(rows):
        row = next(r for r in rows if r["dose_nM"] == "1000" and r["source_figure"] == "Fig5C_left")
        row["published_n"], row["n_status"] = "3", "three_independent_experiments"
    change(source, "bocnho_conditions.csv", edit)
    with pytest.raises(ValueError, match="replicate-count exception"):
        challenge.prepare(source)


@pytest.mark.parametrize("field,value,message", [
    ("sem", ".02", "manufacture total SEM"),
    ("uncertainty_kind", "confidence_interval", "statistical intervals"),
    ("estimate_fraction", "30", "conversion mismatch"),
    ("estimate_percent", "nan", "Non-finite"),
])
def test_preserves_measurement_kind_and_scale(source, field, value, message):
    change(source, "bocnho_total_annexin.csv", lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError, match=message):
        challenge.prepare(source)


def test_duration_pairs_cannot_mix_doses(source):
    def edit(rows):
        rows[0]["pulse_condition_id"] = rows[1]["pulse_condition_id"]
    change(source, "bocnho_duration_comparisons.csv", edit)
    with pytest.raises(ValueError, match="changes drug, dose"):
        challenge.prepare(source)


def test_frozen_source_and_output_edits_are_detected(source, tmp_path):
    target = tmp_path / "challenge"
    challenge.build(target, source)
    path = target / "contract.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["matched_initial_RNA"] = True
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from reviewed source"):
        challenge.load_challenge(target)
    (source / "synthetic-source.txt").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source"):
        challenge.load_challenge(target)


def test_source_figures_cannot_be_mixed(source):
    change(source, "bocnho_component_means.csv", lambda rows: rows[1].update(source_figure="Fig5C_left"))
    with pytest.raises(ValueError, match="source figure mismatch"):
        challenge.prepare(source)


def test_input_ready_manifest_cannot_claim_training_ready(source, tmp_path):
    target = tmp_path / "challenge"
    challenge.build(target, source)
    path = target / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["formal_training_ready"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="readiness or schema"):
        challenge.load_challenge(target)
