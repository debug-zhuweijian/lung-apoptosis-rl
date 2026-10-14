"""Freeze Kotani Fig4 as a separate ND-gated challenge, without RNA pairing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round8/kotani"
PROTOCOL = "Kotani2024:Fig4C:ND_gated_flow"
CONTROL = "no_axitinib_control"
PRETREATED = "axitinib_2.5uM_72h_pretreated"
IMAGE = "cancers-16-02782_p09_image1_Im0.png"
SOURCE_FILES = ("Fig4C_digitized_means.csv", "Fig4B_author_printed_values.csv",
                "Fig4_extraction_audit.json", "protocol_evidence.json", "validation.json",
                "PMC11352446.xml", "cancers-16-02782.pdf",
                "cancers-3131450-supplementary.pdf", "extract_fig4.py", IMAGE)
OUTPUT_FILES = ("conditions.json", "reference_conditions.json", "observations.csv",
                "representative_panels.csv", "challenge_contract.json")
CONDITION_FIELDS = ("condition_id", "cell_line", "state", "pretreatment_compound",
                    "pretreatment_dose_uM", "pretreatment_duration_h", "transition_handling",
                    "challenge_compound", "challenge_dose_uM", "challenge_duration_h",
                    "no_ABT_reference_clock", "vehicle_identity", "source_column_1based", "pairing_level")
DEFINITIONS = {
    "Annexin_positive": {"endpoint": "Annexin_given_ND", "denominator": "ND_gate_events",
        "sampling_gate": "FSC-defined non-destroyed (ND) gate; Annexin V-positive events within ND, including PI-negative and PI-positive quadrants"},
    "ND": {"endpoint": "ND_fraction_of_acquired_events",
        "denominator": "all_acquired_events_including_cells_and_debris",
        "sampling_gate": "FSC-defined ND population within all acquired cells and debris"},
    "PI_positive": {"endpoint": "PI_given_ND", "denominator": "ND_gate_events",
        "sampling_gate": "FSC-defined ND gate; PI-positive events within ND"},
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def physical_id(image_hash, locator):
    return "kotani_observation:" + hashlib.sha256(f"{image_hash}:{locator}".encode()).hexdigest()[:24]


def prepare_tables(source):
    """Validate the reviewed extraction and retain its exact state/time identities."""
    source = Path(source)
    inventory = {row["file"]: row for row in read_json(source / "local_file_inventory.json")}
    for name in SOURCE_FILES:
        if name not in inventory or sha(source / name) != inventory[name]["sha256"]:
            raise ValueError("Kotani source inventory hash mismatch: " + name)
    audit = read_json(source / "Fig4_extraction_audit.json")
    if (audit["source_image"] != IMAGE or audit["source_image_sha256"] != sha(source / IMAGE)
            or audit["B_and_C_kept_separate"] is not True
            or any(audit[key] is not False for key in ("replicate_values_recovered",
                   "standard_deviations_recovered", "natural_population_apoptosis_inferred",
                   "no_ABT_rows_assumed_time_matched_vehicle"))):
        raise ValueError("Kotani extraction evidence boundary changed")
    means = pd.read_csv(source / "Fig4C_digitized_means.csv")
    panels = pd.read_csv(source / "Fig4B_author_printed_values.csv")
    if (len(means) != 18 or len(panels) != 24
            or means.duplicated(["condition_id", "endpoint"]).any()
            or panels.duplicated(["condition_id", "endpoint"]).any()
            or set(means.endpoint) != set(DEFINITIONS)
            or set(panels.endpoint) != {"ND", "Annexin_negative_PI_positive",
                                        "Annexin_positive_PI_positive", "Annexin_positive_PI_negative"}):
        raise ValueError("Expected 18 unique means and 24 separate printed-panel values")
    condition_rows = means[list(CONDITION_FIELDS)].drop_duplicates().sort_values("source_column_1based")
    panel_conditions = panels[list(CONDITION_FIELDS)].drop_duplicates().sort_values("source_column_1based")
    pd.testing.assert_frame_equal(condition_rows.reset_index(drop=True), panel_conditions.reset_index(drop=True), check_dtype=False)
    if (len(condition_rows) != 6 or not condition_rows.condition_id.is_unique
            or condition_rows.source_column_1based.tolist() != [1, 2, 3, 4, 5, 6]):
        raise ValueError("State/time condition IDs are missing or merged")
    if not (means.groupby("condition_id").size().eq(3).all() and panels.groupby("condition_id").size().eq(4).all()):
        raise ValueError("Each condition requires three means and four separate panel readings")
    if not condition_rows.cell_line.eq("A549").all() or set(condition_rows.state) != {CONTROL, PRETREATED}:
        raise ValueError("Unexpected Kotani cell line or state")
    active = condition_rows.loc[condition_rows.challenge_compound.eq("ABT-263")].copy()
    references = condition_rows.loc[condition_rows.challenge_compound.eq("none")].copy()
    expected = {(CONTROL, 24.), (PRETREATED, 6.), (PRETREATED, 12.), (PRETREATED, 24.)}
    if (len(active) != 4 or set(zip(active.state, active.challenge_duration_h)) != expected
            or not active.challenge_dose_uM.eq(1).all() or not active.no_ABT_reference_clock.eq("not_applicable").all()):
        raise ValueError("Active state/dose/time grid changed")
    if (len(references) != 2 or set(references.state) != {CONTROL, PRETREATED}
            or not references.challenge_duration_h.isna().all()
            or not references.challenge_dose_uM.eq(0).all()
            or not references.no_ABT_reference_clock.eq("not_stated").all()):
        raise ValueError("Unknown reference clock must remain unknown")
    for row in records(condition_rows):
        control = row["state"] == CONTROL
        expected_id = "Kotani2024_A549_" + ("control" if control else "axitinib") + "_"
        expected_id += "no_ABT" if row["challenge_compound"] == "none" else f"ABT_{int(row['challenge_duration_h'])}h"
        if (row["condition_id"] != expected_id or row["pretreatment_duration_h"] != 72
                or row["pretreatment_compound"] != ("none" if control else "axitinib")
                or row["pretreatment_dose_uM"] != (0 if control else 2.5)
                or row["transition_handling"] != "harvest_then_reculture;washout_and_carryover_unspecified"
                or row["vehicle_identity"] != "not_stated"
                or row["pairing_level"] != "same_study_defined_population_state_not_same_cell"):
            raise ValueError("Condition ID or pretreatment history changed")
    bounds = means[["readout_lower", "value", "readout_upper"]].to_numpy(float)
    if (not np.isfinite(bounds).all() or (bounds < 0).any() or (bounds > 100).any()
            or (np.diff(bounds, axis=1) < 0).any()):
        raise ValueError("Invalid digitization value or readout range")
    if (not means.unit.eq("percent").all() or not means.figure.eq("Fig4C").all()
            or not means.author_reported_n.eq(3).all() or not means.replicate_kind.eq("not_specified").all()
            or not means.value_origin.eq("digitized_published_bar_mean").all()
            or not means.summary_statistic.eq("mean_of_3_replicates_per_caption").all()
            or not means.author_error_bar.eq("SD_not_digitized").all()
            or not means.interval_kind.eq("pixel_readout_range_not_statistical").all()):
        raise ValueError("Published means and reading ranges must not become raw repeats or SD")
    if (not panels.unit.eq("percent").all() or not panels.figure.eq("Fig4B").all()
            or not panels.value_origin.eq("author_printed_flow_plot_label").all()
            or not panels.summary_statistic.eq("single_displayed_flow_plot_not_Fig4C_mean").all()
            or panels.replicate_id.notna().any() or panels.raw_events_or_counts_available.any()
            or not np.isfinite(panels.value).all() or not panels.value.between(0, 100).all()):
        raise ValueError("Representative panels must remain separate from mean/repeat evidence")
    for frame in (means, panels):
        if not frame.source_image.eq(IMAGE).all() or not frame.source_pdf_page.eq(9).all():
            raise ValueError("Source image locator mismatch")
        expected_denominator = np.where(frame.endpoint.eq("ND"), DEFINITIONS["ND"]["denominator"], "ND_gate_events")
        if not np.array_equal(frame.denominator.to_numpy(), expected_denominator):
            raise ValueError("ND and within-ND denominators cannot be interchanged")

    def condition_records(frame, role):
        return [{**row, "source_id": "Kotani2024", "experiment_block": "Fig4",
                 "split_group": "Kotani2024:Fig4", "protocol_id": PROTOCOL,
                 "condition_role": role, "initial_state_kind": "reference",
                 "initial_expression_sample_id": None, "same_sample_expression_endpoint": False,
                 "state_evidence": "author_defined_population_state_and_parallel_markers_not_paired_RNA"}
                for row in records(frame)]
    observations = means.copy()
    observations["source_endpoint"] = observations.endpoint
    observations["source_value_percent"] = observations.value
    observations["source_readout_lower_percent"] = observations.readout_lower
    observations["source_readout_upper_percent"] = observations.readout_upper
    for key in ("value", "readout_lower", "readout_upper"):
        observations[key] = observations[key] / 100
    observations["unit"], observations["label_kind"] = "fraction", "measured"
    observations["protocol_id"] = PROTOCOL
    observations["endpoint"] = observations.source_endpoint.map(lambda key: DEFINITIONS[key]["endpoint"])
    observations["sampling_gate"] = observations.source_endpoint.map(lambda key: DEFINITIONS[key]["sampling_gate"])
    observations["source_locator"] = [f"Fig4C:{r.source_endpoint}:column={r.source_column_1based}" for r in observations.itertuples()]
    observations["observation_id"] = observations.source_locator.map(lambda s: physical_id(audit["source_image_sha256"], s))
    observations["source_image_sha256"] = audit["source_image_sha256"]
    primary = active.loc[active.challenge_duration_h.eq(24)].set_index("state").condition_id.to_dict()
    contract = {
        "schema_version": 1, "challenge_id": "Kotani2024_Fig4C_ND", "protocol_id": PROTOCOL,
        "endpoint_definitions": {row["endpoint"]: row for row in DEFINITIONS.values()},
        "primary_endpoint": "Annexin_given_ND", "primary_readout_hours": 24,
        "primary_condition_ids": primary, "evaluation_is_retrospective": True,
        "primary_comparison": "axitinib_pretreated_minus_no_axitinib_under_ABT263_1uM_24h",
        "secondary_scope": "6h_and_12h_point_errors_only_no_duration_selection_or_pooling",
        "reference_scope": "unknown_clock_reference_display_only_no_time_matched_subtraction",
        "uncertainty_policy": "pixel_readout_ranges_not_SD_SE_CI_or_biological_variance",
        "replicate_policy": "one_published_mean_per_condition_endpoint_n3_reported_repeat_kind_unknown_no_raw_replicates",
        "gene_expression_pairing": "none_verified_no_initial_expression_sample_id",
        "state_claim_boundary": "defined_group_history_comparison_not_isolated_causal_state_effect_or_RNA_increment",
        "sampling_boundary": "Annexin_and_PI_given_ND_not_original_culture_or_seeded_population_rates",
        "core_G_compatible": False, "formal_training_ready": False,
        "prediction_provenance_policy": "model_and_state_input_declarations_recorded_not_independently_validated",
    }
    return {"conditions": condition_records(active, "active"),
            "references": condition_records(references, "unknown_clock_reference"),
            "observations": observations, "panels": panels, "contract": contract}


def build(output, source=SOURCE, root=ROOT):
    root, source, output = Path(root).resolve(), Path(source).resolve(), Path(output)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if output.exists():
        raise FileExistsError("Kotani challenge is immutable; choose a new output version")
    inputs = [source / name for name in (*SOURCE_FILES, "local_file_inventory.json")]
    source_hashes = {p.relative_to(root).as_posix(): sha(p) for p in inputs}
    tables = prepare_tables(source)
    output.mkdir(parents=True)
    for name, key in (("conditions.json", "conditions"), ("reference_conditions.json", "references"),
                      ("challenge_contract.json", "contract")):
        (output / name).write_text(json.dumps(tables[key], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tables["observations"].to_csv(output / "observations.csv", index=False, encoding="utf-8-sig")
    tables["panels"].to_csv(output / "representative_panels.csv", index=False, encoding="utf-8-sig")
    if any(sha(root / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Kotani source changed during challenge preparation")
    manifest = {"schema_version": 1, "status": "input-ready/prediction-missing",
                "source_directory": source.relative_to(root).as_posix(),
                "active_conditions": 4, "unknown_clock_references": 2,
                "published_mean_observations": 18, "representative_panel_values": 24,
                "formal_training_ready": False, "core_G_compatible": False,
                "implementation_sha256": {Path(__file__).name: sha(__file__)},
                "inputs_sha256": source_hashes,
                "files_sha256": {name: sha(output / name) for name in OUTPUT_FILES}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=SOURCE / "challenge-v1")
    args = parser.parse_args()
    result = build(args.output, args.source)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}, ensure_ascii=False))
