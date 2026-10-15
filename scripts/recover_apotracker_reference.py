"""Recover source-native PS/PI trajectories from Ortmayr & Zampieri EV6.

These imaging observations are neither Annexin measurements nor matched RNA.
Medium context, literal times, source repeat labels, and channel definitions
stay explicit. No model fit, candidate expansion, or frozen-role change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
import pandas as pd

KEYS = ["cell_line", "medium_context", "source_treatment", "source_repeat", "time_h"]
CHANNELS = {
    "nuclei_count": ("nuclei_count", "Nuclei counts"),
    "fraction_Green": ("green_only_percent", "ApoTracker Green only"),
    "fraction_GreenRed": ("green_red_percent", "ApoTracker Green & propidium iodide"),
    "fraction_Red": ("red_only_percent", "Propidium iodide only"),
}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def treatment_events(name):
    """Exact source vocabulary; no implicit dose conversion or inferred salts."""
    recipes = {
        "Untreated_serumfree": [], "Untreated_fullmedium": [],
        "5fluorouracil_500uM": [("5-fluorouracil", 500)],
        "Etomoxir_10uM": [("etomoxir", 10)],
        "Trimetazidine_1000uM": [("trimetazidine", 1000)],
        "Trimetazidine_400uM": [("trimetazidine", 400)],
        "TrimetazidineEtomoxir_1000uM10uM": [("trimetazidine", 1000), ("etomoxir", 10)],
        "TrimetazidineEtomoxir_400uM10uM": [("trimetazidine", 400), ("etomoxir", 10)],
    }
    if name not in recipes:
        raise ValueError(f"Unrecognized source treatment: {name}")
    return [{"drug_name": drug, "dose_uM": dose, "source_unit": "uM", "start_h": 0,
             "mode": "continuous_concurrent", "chemical_structure_qualified": False,
             "material_form": "dihydrochloride" if drug == "trimetazidine" else "not_verified"}
            for drug, dose in recipes[name]]


def read_trajectories(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    result, expected_keys = None, None
    for suffix, (value_name, marker) in CHANNELS.items():
        sheet = "Apoptosis_" + suffix
        rows = list(workbook[sheet].values)
        if marker not in str(rows[0][0]) or list(rows[3][:4]) != ["Cell line", "Condition", "Treatment", "Replicate"]:
            raise ValueError(f"Changed source schema or endpoint definition: {sheet}")
        times = np.asarray(rows[3][4:], dtype=float)
        if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("Invalid source time grid")
        records = []
        for source_row, row in enumerate(rows[4:], 5):
            if row[0] not in {"A549", "HCT116"} or row[1] not in {"S-", "Pro"}:
                raise ValueError("Unrecognized source background or medium context")
            treatment_events(row[2])
            if row[3] not in {1, 2, 3}:
                raise ValueError("Unrecognized source replicate label")
            if row[2].startswith("Untreated_") and row[2] != {
                "S-": "Untreated_serumfree", "Pro": "Untreated_fullmedium"}[row[1]]:
                raise ValueError("Control medium identity conflict")
            values = np.asarray(row[4:], dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Missing, nonfinite or negative source observation")
            if suffix == "nuclei_count":
                if np.any(values != np.floor(values)) or np.any(values <= 0):
                    raise ValueError("Nuclei denominator must be positive integer counts")
            elif np.any(values > 100):
                raise ValueError("Source percent outside [0,100]")
            for col, (tm, value) in enumerate(zip(times, values), 5):
                record = dict(zip(KEYS, [*row[:4], float(tm)]))
                record[value_name] = float(value)
                record[value_name + "_source_cell"] = f"{sheet}!R{source_row}C{col}"
                records.append(record)
        frame = pd.DataFrame(records)
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate trajectory measurement identity")
        keys = set(frame[KEYS].itertuples(index=False, name=None))
        if expected_keys is not None and keys != expected_keys:
            raise ValueError("Channel trajectory keys differ; cannot infer pairing")
        expected_keys = keys
        result = frame if result is None else result.merge(frame, on=KEYS, validate="one_to_one")
    workbook.close()
    total = result.green_only_percent + result.green_red_percent + result.red_only_percent
    if (total > 100).any():
        raise ValueError("Disjoint source channel sum exceeds 100; do not clip")
    # Source workbook explicitly says Green only / Green & Red / Red only.
    result["ps_positive_percent"] = result.green_only_percent + result.green_red_percent
    result["pi_positive_percent"] = result.red_only_percent + result.green_red_percent
    result["double_negative_percent"] = 100 - total
    result["nuclei_count"] = result.nuclei_count.astype(np.int64)
    result["assay"] = "ApoTracker_Green_PI_imaging"
    result["denominator"] = "image_detected_Hoechst_positive_nuclei_at_same_time"
    result["has_matched_RNA"] = False
    result["annexin_equivalence_validated"] = False
    result["initial_expression_observed"] = False
    result["culture_context"] = result.medium_context.map({"S-": "serum_free", "Pro": "5_percent_dialyzed_FBS"})
    result["trajectory_id"] = result[KEYS[:-1]].astype(str).agg("|".join, axis=1)
    result["same_cell_identity_tracking"] = False
    result["repeat_scope"] = "source_culture_repeat_not_proven_independent_experiment_day"
    return result.sort_values(KEYS).reset_index(drop=True)


def summarize(frame):
    """Unweighted source-repeat means, not cell-pooled pseudo-replication."""
    groups = ["cell_line", "medium_context", "source_treatment", "time_h"]
    endpoints = ["green_only_percent", "green_red_percent", "red_only_percent",
                 "ps_positive_percent", "pi_positive_percent", "nuclei_count"]
    summary = frame.groupby(groups)[endpoints].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    # A descriptive context interaction for matched nominal single drugs only.
    contrasts = []
    for (cell, tm), group in summary.groupby(["cell_line", "time_h"]):
        table = group.set_index(["medium_context", "source_treatment"])
        for drug in ["5fluorouracil_500uM", "Trimetazidine_1000uM"]:
            keys = [("S-", drug), ("S-", "Untreated_serumfree"),
                    ("Pro", drug), ("Pro", "Untreated_fullmedium")]
            if not all(k in table.index for k in keys):
                continue
            for endpoint in ["ps_positive_percent", "pi_positive_percent"]:
                v = [float(table.loc[k, endpoint + "_mean"]) for k in keys]
                contrasts.append({"cell_line": cell, "time_h": tm, "source_treatment": drug,
                    "endpoint": endpoint, "serumfree_drug_minus_control_pp": v[0] - v[1],
                    "fullmedium_drug_minus_control_pp": v[2] - v[3],
                    "context_interaction_pp": v[0] - v[1] - v[2] + v[3],
                    "interpretation": "descriptive_medium_context_interaction_not_RNA_state_increment",
                    "independent_timepoints": False})
    return summary, pd.DataFrame(contrasts)


def recover(workbook, output):
    out = Path(output)
    if out.exists():
        raise ValueError("Use a new output directory; no historical results overwritten")
    frame = read_trajectories(workbook)
    summary, contrasts = summarize(frame)
    out.mkdir(parents=True)
    frame.to_csv(out / "source_trajectories.csv", index=False)
    summary.to_csv(out / "repeat_group_summary.csv", index=False)
    contrasts.to_csv(out / "context_contrasts.csv", index=False)
    (out / "source_treatment_events.json").write_text(json.dumps(
        {name: treatment_events(name) for name in sorted(frame.source_treatment.unique())}, indent=2) + "\n")
    manifest = {"source_doi": "10.15252/msb.202110716", "source_dataset": "EV6",
        "source_sha256": digest(workbook), "script_sha256": digest(__file__),
        "synthetic": False, "rows": len(frame), "A549_rows": int(frame.cell_line.eq("A549").sum()),
        "trajectories": frame.trajectory_id.nunique(),
        "source_conditions": len(frame[KEYS[:3]].drop_duplicates()),
        "times_h": sorted(frame.time_h.unique().tolist()),
        "input_signals": list(CHANNELS), "fraction_scale": "source_percentage_points",
        "fraction_precision": "published_values_are_integer_percentages_no_exact_event_counts_reconstructed",
        "matched_RNA": False, "Annexin_V_measurement": False,
        "state_expression_increment_validated": False, "true_sequential_intervention": False,
        "frozen_training_roles_changed": False, "biological_independence_across_times": False,
        "source_medium_is_context_not_measured_initial_state": True,
        "outputs": {p.name: digest(p) for p in sorted(out.iterdir())}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workbook", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(recover(args.workbook, args.output), indent=2))
