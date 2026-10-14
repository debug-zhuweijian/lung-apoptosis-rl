"""Register a group-state Annexin challenge without inventing exact dose or RNA."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1"
SOURCE = BASE / "source_reviews/state_function_round4/nav_gal"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError("Challenge version already exists")
    digitized = SOURCE / "digitized-v1"
    recorded = json.loads((digitized / "manifest.json").read_text(encoding="utf8"))
    for base, values in [(ROOT, recorded["sources_sha256"]), (digitized, recorded["files_sha256"])]:
        for name, expected in values.items():
            if sha(base / name) != expected:
                raise ValueError("Digitization provenance mismatch: " + name)
    values = pd.read_csv(digitized / "published_means.csv")
    if len(values) != 60 or values.observation_id.duplicated().any():
        raise ValueError("Published plot units are missing or duplicated")
    curves = []
    for curve, rows in values.groupby("curve_id", sort=True):
        row = rows.iloc[0]
        if sorted(rows.hours_after_challenge) != list(range(0, 37, 4)):
            raise ValueError("Curve time coordinates do not match the inspected plot")
        senescent = row.state_id == "cisplatin_induced_senescent"
        vehicle = row.challenge_treatment == "vehicle"
        curves.append({
            "condition_group_id": curve, "state_id": row.state_id,
            "state_source_label": row.state_figure_label, "cell_line": "A549",
            "state_evidence": "author_reported_group_induction_and_markers_not_paired_RNA",
            "induction_compound": "cisplatin" if senescent else None,
            "induction_nominal_uM": 15 if senescent else None,
            "induction_days": 10 if senescent else None,
            "cisplatin_washout_before_challenge": "not_explicitly_reported" if senescent else "not_applicable",
            "nominal_seeded_cells_per_well": 4000 if senescent else 6000,
            "hours_after_replating_before_challenge": 24,
            "challenge_treatment": row.challenge_treatment, "is_vehicle": vehicle,
            "reported_dose_candidates_uM": [] if vehicle else [10, 11],
            "dose_evidence": "vehicle_amount_not_recovered" if vehicle else "Fig4_caption_10_vs_Methods_4.6_11",
            "actual_dose_verified": False,
            "protocol_id": "GonzalezGualda2020:Fig4b:imaged_Annexin_V",
            "reference_expression_sample_ids": [], "same_sample_expression_endpoint": False,
            "baseline_observation_id": rows.loc[rows.hours_after_challenge.eq(0), "observation_id"].item(),
            "baseline_interpretation": "plotted_time_zero_group_aggregate_first_scan_precedes_challenge_in_methods",
            "biological_repeats_reported": 3, "technical_per_biological_reported": 3,
            "raw_replicate_identifiers_available": False,
            "source_sections": ["acel13142-sec-0014", "acel13142-sec-0016", "acel13142-fig-0004"],
        })
    if len(curves) != 6:
        raise ValueError("Expected two group states and three treatment arms")
    contract = {
        "status": "conditional_published_group_state_challenge",
        "source_doi": "10.1111/acel.13142", "cell_line": "A549",
        "endpoint_definition": "Annexin_V_positive_cells_divided_by_total_image_counted_cells",
        "observation_kind": "digitized_published_mean_not_raw_replicate",
        "primary_readout_hours": 24,
        "primary_readout_reason": "matches_the_existing_24h_pilot_definition_not_selected_by_model_results",
        "evaluation_is_retrospective": True,
        "secondary_readouts": "remaining_plotted_times_are_correlated_descriptive_followup",
        "dose_scenarios": {"figure_caption": 10, "methods": 11},
        "dose_scenario_policy": "evaluate_both_reported_values_without_counting_the_same_Y_twice",
        "dose_scenario_limit": "two_reported_values_do_not_prove_the_actual_dose_or_cover_all_possible_errors",
        "uncertainty_policy": "reading_bounds_are_pixel_assumptions_not_author_SD_CI_or_biological_variance",
        "state_covariates": "group_state_and_nominal_seeding_density_vary_together",
        "state_claim_boundary": "predictive_group_context_heterogeneity_not_isolated_causal_state_effect",
        "duration_boundary": "serial_continuous_exposure_readouts_do_not_observe_early_withdrawal_at_common_Tstar",
        "gene_expression_pairing": "none_verified",
        "chemistry_input_status": "source_compound_names_defined_machine_structures_under_separate_review",
        "prediction_requirements": ["frozen_model_and_preprocessing", "explicit_state_input_provenance",
                                    "unvalidated_transfer_flags", "same_endpoint_definition",
                                    "both_dose_scenarios_or_explicit_missing_prediction"],
        "permitted_metrics": ["descriptive_point_error_with_reading_bounds",
                              "within_state_pair_order_if_intervals_separate",
                              "state_blind_vs_state_aware_predictions_under_same_information_budget"],
        "unsupported_claims": ["60_independent_experiments", "recovered_original_replicates",
                               "calibrated_single_cell_fate", "RNA_state_increment_proved",
                               "drug_rank_reversal_proved", "common_endpoint_duration_selection_proved"],
        "formal_training_ready": False,
    }
    output.mkdir(parents=True)
    (output / "conditions.json").write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding="utf8")
    (output / "challenge_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf8")
    values.to_csv(output / "observations.csv", index=False, encoding="utf-8-sig")
    primary = values.loc[values.hours_after_challenge.eq(24)].copy()
    primary.to_csv(output / "primary_24h.csv", index=False, encoding="utf-8-sig")
    assert len(primary) == 6 and (~primary.challenge_treatment.eq("vehicle")).sum() == 4
    parent = digitized / "manifest.json"
    manifest = {"schema_version": 1, "status": contract["status"],
                "condition_groups": 6, "state_groups": 2, "active_drug_names": 2,
                "published_timepoint_means": 60, "primary_24h_drug_state_means": 4,
                "original_biological_repeat_values_recovered": 0, "formal_training_ready": False,
                "inputs_sha256": {str(parent.relative_to(ROOT)): sha(parent),
                                  str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
                "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "challenge-v1")
    main(parser.parse_args().output)
