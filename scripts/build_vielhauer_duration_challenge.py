"""Build/read the reviewed 24h harvested-Annexin exposure challenge; no models."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration"
PROTOCOL = "vielhauer2013_a549_annexin_pi_harvested_parent"
DENOMINATOR = "post_protocol_harvested_parent_population"
GATE = "parent_population_gate_not_further_documented"
ARMS = {"Fig3C": ("continuous_24h", 24, None, 0),
        "Fig5C_left": ("pulse_6h_washout_recovery18h", 6, 6, 18)}
SOURCE_TABLES = ("bocnho_conditions.csv", "bocnho_component_means.csv",
                 "bocnho_total_annexin.csv", "bocnho_duration_comparisons.csv")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite source number")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(base, name):
    base = Path(base).resolve()
    path = (base / name).resolve()
    require(path.is_relative_to(base), "Source path escapes declared directory")
    return path


def prepare(source=SOURCE):
    source = Path(source).resolve()
    conditions, components, totals, comparisons = [rows(source / n) for n in SOURCE_TABLES]
    require((len(conditions), len(components), len(totals), len(comparisons)) == (22, 66, 22, 10),
            "The source challenge must contain 22 conditions, 66 components and 10 comparisons")
    ids = {r["condition_id"] for r in conditions}
    require(len(ids) == 22 and {r["condition_id"] for r in totals} == ids,
            "Missing, duplicated or mismatched condition identities")
    by_component = {}
    for row in components:
        key = (row["condition_id"], row["component"])
        require(key not in by_component and row["condition_id"] in ids, "Duplicated/unknown component")
        by_component[key] = row
    for row in (*components, *totals):
        require(row["protocol_id"] == PROTOCOL and row["denominator"] == DENOMINATOR
                and row["sampling_gate"] == GATE, "Endpoint protocol, denominator or gate changed")
        require(row["value_status"] == "digitized_published_group_mean"
                and row["uncertainty_kind"] == "pixel_reading_bounds_not_SEM",
                "Published means/pixel intervals cannot become raw replicates or statistical intervals")
        low, mid, high = [number(row[k]) for k in (
            "reading_lower_percent", "estimate_percent", "reading_upper_percent")]
        require(0 <= low <= mid <= high <= 100, "Invalid reading interval")
    measurements = {}
    for row in totals:
        cid = row["condition_id"]
        require({c for (i, c) in by_component if i == cid}
                == {"necrosis", "early_apoptosis", "late_apoptosis"}, "Missing or renamed component")
        require(all(by_component[(cid, c)][k] == row[k]
                    for c in ("necrosis", "early_apoptosis", "late_apoptosis")
                    for k in ("source_image", "source_figure")), "Component/total source figure mismatch")
        annexin = sum(number(by_component[(cid, c)]["estimate_percent"])
                      for c in ("early_apoptosis", "late_apoptosis"))
        # Three independently rounded values, each with four decimal places.
        require(abs(annexin - number(row["estimate_percent"])) <= .0001500001,
                "Total Annexin must be early + late, excluding necrosis")
        require(row["label_type"] == "Annexin_positive_fraction" and row["sem"] == "",
                "Do not substitute another endpoint or manufacture total SEM")
        for percent, fraction in (("estimate_percent", "estimate_fraction"),
                                  ("reading_lower_percent", "reading_lower_fraction"),
                                  ("reading_upper_percent", "reading_upper_fraction")):
            require(abs(number(row[percent]) / 100 - number(row[fraction])) < 1e-10,
                    "Percent/fraction conversion mismatch")
        measurements[cid] = {"mean_fraction": number(row["estimate_fraction"]),
            "pixel_lower_fraction": number(row["reading_lower_fraction"]),
            "pixel_upper_fraction": number(row["reading_upper_fraction"]),
            "sem": None, "value_origin": "digitized_published_component_means_sum",
            "source_image": row["source_image"], "source_figure": row["source_figure"]}

    active, references = [], []
    grid = set()
    for row in conditions:
        require(row["cell_line"] == "A549" and row["state_id"] == "a549_parental_standard_culture",
                "This challenge has one A549 background, not multiple initial states")
        require(row["protocol_id"] == PROTOCOL and row["denominator"] == DENOMINATOR
                and row["sampling_gate"] == GATE, "Condition protocol mismatch")
        require(row["is_vehicle"] in ("true", "false"), "Unknown vehicle status")
        require(measurements[row["condition_id"]]["source_figure"] == row["source_figure"],
                "Condition/measurement source figure mismatch")
        require(number(row["terminal_time_h"]) == 24, "Only the common 24h endpoint is supported")
        require(row["initial_RNA_reference"] == "not_available_in_this_source"
                and row["cross_figure_replicate_pairing"] == "not_established",
                "Unverified RNA or replicate pairing cannot be introduced")
        vehicle = row["is_vehicle"] == "true"
        entry = {**row, "is_vehicle": vehicle, "measured_endpoint": measurements[row["condition_id"]],
                 "gene_expression_sample_id": None, "predicted_features": None}
        if vehicle:
            # Preserve source control metadata; do not assume a verified sham-wash protocol.
            references.append(entry)
            continue
        exposure = number(row["exposure_duration_h"])
        wash = None if row["washout_at_h"] == "" else number(row["washout_at_h"])
        recovery = number(row["drug_free_recovery_h"])
        require((row["protocol_arm"], exposure, wash, recovery) == ARMS.get(row["source_figure"]),
                "Exposure/washout/recovery clocks mismatch")
        dose = number(row["dose_nM"])
        require(row["drug_id"] in ("bocnho", "cbi_indole2")
                and dose in (.05, .5, 10, 40, 1000), "Unexpected source drug or dose")
        expected_n = ("3", "three_independent_experiments")
        if dose == 1000:
            expected_n = (("2", "general_Annexin_1uM_each_drug_one_outlier_excluded")
                          if exposure == 24 else ("", "washout_section_n3_but_general_1uM_outlier_n2_scope_ambiguous"))
        require((row["published_n"], row["n_status"]) == expected_n,
                "High-dose replicate-count exception or uncertainty was lost")
        key = (row["drug_id"], dose, exposure)
        require(key not in grid, "Duplicated active candidate")
        grid.add(key)
        active.append({**entry, "dose_nM": dose, "dose_uM": dose / 1000,
                       "exposure_duration_h": exposure, "washout_at_h": wash,
                       "drug_free_recovery_h": recovery, "terminal_time_h": 24,
                       "source_material_identity": "Vielhauer2013:" + row["drug_id"]})
    require(len(active) == 20 and len(references) == 2, "Incorrect active/control partition")
    keyed = {r["condition_id"]: r for r in active}
    seen_pairs = set()
    for comparison in comparisons:
        a, b = (keyed.get(comparison[k]) for k in ("continuous_condition_id", "pulse_condition_id"))
        require(a is not None and b is not None, "Comparison points to non-active conditions")
        require(a["drug_id"] == b["drug_id"] == comparison["drug_id"]
                and a["dose_nM"] == b["dose_nM"] == number(comparison["dose_nM"])
                and a["exposure_duration_h"] == 24 and b["exposure_duration_h"] == 6
                and number(comparison["terminal_time_h"]) == 24,
                "Duration comparison changes drug, dose, clock or arm")
        pair = (a["condition_id"], b["condition_id"])
        require(pair not in seen_pairs and comparison["duration_p_value"] == "",
                "Duplicate comparison or unsupported cross-arm p value")
        seen_pairs.add(pair)
    contract = {"protocol_id": PROTOCOL, "label_type": "Annexin_positive_fraction",
        "denominator": DENOMINATOR, "sampling_gate": GATE, "common_terminal_time_h": 24,
        "state_count": 1, "active_conditions": 20, "separate_references": 2,
        "duration_comparisons": 10, "clinical_or_natural_population_rate": False,
        "same_initial_cohort_denominator_verified": False, "verified_sham_wash_control": False,
        "matched_initial_RNA": False, "complete_generated_features": 0,
        "formal_training_ready": False, "current_G_training_manifest": False,
        "data_exposure": "source outcomes inspected during design; retrospective challenge",
        "interval_kind": "pixel_reading_bounds_not_SEM_or_confidence_interval",
        "policy_or_prediction_evaluation_performed": False,
        "source_statistical_notes": "Preserve per-condition n_status; no raw replicates or total SEM recovered"}
    names = set(SOURCE_TABLES) | {"pixel_landmarks.json", "Vielhauer2013_PMC_fulltext.md"}
    names.update(r["source_image"] for r in totals)
    dependencies = {name: sha(inside(source, name)) for name in sorted(names)}
    return {"active_conditions": active, "reference_conditions": references,
            "duration_comparisons": comparisons, "contract": contract}, dependencies


def load_challenge(path):
    path = Path(path).resolve()
    manifest = read(path / "manifest.json")
    require(manifest.get("schema_version") == 1 and manifest.get("status") == "input-ready/prediction-missing"
            and manifest.get("formal_training_ready") is False, "Challenge readiness or schema changed")
    require(manifest["builder_sha256"] == sha(__file__), "Builder version changed")
    prepared, dependencies = prepare(manifest["source_directory"])
    require(dependencies == manifest["source_files_sha256"], "Frozen source content changed")
    require(set(manifest["files_sha256"]) == {k + ".json" for k in prepared}, "Incomplete output bindings")
    for key, expected in prepared.items():
        file = path / (key + ".json")
        require(sha(file) == manifest["files_sha256"][file.name] and read(file) == expected,
                "Derived challenge differs from reviewed source")
    return prepared


def build(output, source=SOURCE):
    output = Path(output).resolve()
    require(not output.exists(), "Use a new output directory; frozen packages are not overwritten")
    prepared, dependencies = prepare(source)
    output.mkdir(parents=True)
    for name, value in prepared.items():
        (output / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "status": "input-ready/prediction-missing",
        "source_directory": str(Path(source).resolve()), "source_files_sha256": dependencies,
        "builder_sha256": sha(__file__), "formal_training_ready": False,
        "files_sha256": {name + ".json": sha(output / (name + ".json")) for name in prepared}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return load_challenge(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = load_challenge(args.output) if args.verify else build(args.output, args.source)
    print(json.dumps({"status": "input-ready/prediction-missing", **result["contract"]}, ensure_ascii=False))
