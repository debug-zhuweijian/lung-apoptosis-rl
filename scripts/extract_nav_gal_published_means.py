"""Read published Nav-Gal Annexin bar means, not raw biological replicates.

Calibration and color identities were read from Fig4b. Pixel intervals describe
digitization uncertainty only; neither the paper's SD nor biological confidence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import find_objects, label

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1"
SOURCE = BASE / "source_reviews/state_function_round4/nav_gal"
# Original image pixels. y=0 and y=100 are plot coordinates, not cell counts.
PANELS = {
    "vehicle": {"roi": [1265, 1690, 300, 347], "y_zero": 347.5, "y_100": 68.5},
    "navitoclax": {"roi": [1265, 1690, 459, 745], "y_zero": 745.5, "y_100": 457.5},
    "nav_gal": {"roi": [1265, 1690, 852, 1132], "y_zero": 1132.5, "y_100": 850.5},
}
TIMES = list(range(0, 37, 4))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(y, zero, top, edge_error=3.0, axis_error=1.0):
    """Conservative envelope under the declared pixel calibration assumptions."""
    point = 100 * (zero - y) / (zero - top)
    values = [100 * (z - v) / (z - t)
              for z in (zero - axis_error, zero + axis_error)
              for t in (top - axis_error, top + axis_error)
              for v in (y - edge_error, y + edge_error)]
    return tuple(float(np.clip(x, 0, 100)) for x in (point, min(values), max(values)))


def bars(rgb, spec):
    x0, x1, y0, y1 = spec["roi"]
    roi = rgb[y0:y1, x0:x1]
    red = (roi[..., 0] > 140) & (roi[..., 1] < 110) & (roi[..., 2] < 110)
    objects, _ = label(red)
    rectangles = [s for s in find_objects(objects) if s is not None
                  and s[1].stop - s[1].start >= 3 and s[0].stop - s[0].start >= 2]
    rectangles.sort(key=lambda s: s[1].start)
    if len(rectangles) != len(TIMES):
        raise ValueError(f"Expected ten plotted red bars, found {len(rectangles)}")
    black = np.max(rgb, axis=2) < 95
    for time, rectangle in zip(TIMES, rectangles):
        red_x = x0 + (rectangle[1].start + rectangle[1].stop - 1) / 2
        # The colored fill starts inside the black mean-outline stroke.
        red_y = y0 + rectangle[0].start - 1.5
        black_x = int(round(red_x - 16))
        bottom = int(spec["y_zero"]) - 2
        top = int(spec["y_100"])
        wide = black[top:bottom + 1, black_x - 4:black_x + 5].mean(1) >= .65
        # Keep the filled body connected to the bottom, not the narrow SD stem/cap.
        rows = np.flatnonzero(wide)
        runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
        valid = [run for run in runs if len(run) and run[-1] >= bottom - top - 2]
        if not valid:
            raise ValueError(f"Black bar at {time} h cannot be resolved from axis/error bar")
        black_y = top + min(valid, key=lambda run: run[0])[0]
        yield time, [("nonsenescent", "Control", black_x, float(black_y)),
                     ("cisplatin_induced_senescent", "Senescent", red_x, red_y)]


def main(output):
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError("Choose a new immutable digitization output")
    source = SOURCE / "Fig4_embedded_original.jpg"
    rgb = np.asarray(Image.open(source).convert("RGB"))
    if rgb.shape != (2792, 1880, 3):
        raise ValueError("Figure pixels differ from the inspected original")
    rows = []
    for treatment, spec in PANELS.items():
        for time, states in bars(rgb, spec):
            for state, original_label, x, y in states:
                mean, lower, upper = interval(y, spec["y_zero"], spec["y_100"])
                curve = f"GonzalezGualda2020:Fig4b:{treatment}:{state}"
                rows.append({
                    "curve_id": curve, "observation_id": f"{curve}:h{time}",
                    "state_id": state, "state_figure_label": original_label,
                    "challenge_treatment": treatment, "hours_after_challenge": time,
                    "digitized_mean_percent": round(mean, 2),
                    "reading_lower_percent": round(lower, 2), "reading_upper_percent": round(upper, 2),
                    "bar_center_x_px": x, "mean_edge_y_px": y,
                    "measurement_kind": "published_aggregate_mean_digitized",
                    "endpoint": "image_Annexin_V_positive_among_image_counted_cells",
                    "n_biological_reported": 3, "technical_per_biological_reported": 3,
                    "raw_biological_replicates_available": False, "author_sd_digitized": False,
                    "dose_figure_uM": None if treatment == "vehicle" else 10,
                    "dose_methods_uM": None if treatment == "vehicle" else 11,
                    "dose_conflict": treatment != "vehicle", "actual_dose_verified": False,
                    "same_sample_RNA_pairing": False, "formal_training_ready": False,
                    "source_locator": f"Fig4b {treatment}; {original_label}; plotted {time} h bar",
                })
    frame = pd.DataFrame(rows)
    assert len(frame) == 60 and frame.observation_id.is_unique
    assert frame.curve_id.nunique() == 6
    # Independent readback by a reviewer, before this pixel calculation.
    visual = {("vehicle", 24, "nonsenescent"): (0, 3),
              ("vehicle", 36, "nonsenescent"): (0, 3),
              ("vehicle", 24, "cisplatin_induced_senescent"): (3, 7),
              ("vehicle", 36, "cisplatin_induced_senescent"): (4, 8),
              ("navitoclax", 24, "nonsenescent"): (68, 72),
              ("navitoclax", 36, "nonsenescent"): (72, 76),
              ("nav_gal", 24, "nonsenescent"): (27, 32),
              ("nav_gal", 36, "nonsenescent"): (27, 32)}
    for treatment in ("navitoclax", "nav_gal"):
        for time in (24, 36):
            visual[treatment, time, "cisplatin_induced_senescent"] = (98, 100)
    for (treatment, time, state), (low, high) in visual.items():
        value = frame.loc[frame.challenge_treatment.eq(treatment) & frame.hours_after_challenge.eq(time)
                          & frame.state_id.eq(state), "digitized_mean_percent"].item()
        if not low <= value <= high:
            raise ValueError(f"Digitized mean disagrees with independent reading: {(treatment,time,state,value)}")
    output.mkdir(parents=True)
    frame.to_csv(output / "published_means.csv", index=False, encoding="utf-8-sig")
    calibration = {"panels": PANELS, "plotted_times_h": TIMES, "mean_edge_error_px": 3,
                   "axis_error_px": 1, "red_fill_outline_offset_px": 1.5,
                   "uncertainty_scope": "analyst_declared_image_reading_envelope_not_author_SD_or_CI",
                   "image_changes": "none", "independent_visual_crosschecks": 12}
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf8")
    sources = [source, SOURCE / "GonzalezGualda2020_NavGal_published.pdf",
               BASE / "source_reviews/state_function_round3/PMC7189993.xml", Path(__file__)]
    manifest = {"status": "digitized_published_aggregates_not_raw_replicates", "curve_groups": 6,
                "plotted_means": 60, "biological_repeats_reported": 3,
                "independent_experiments_cannot_be_counted_from_60_bars": True,
                "core_training_alignment_unchanged": True, "formal_training_ready": False,
                "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "files_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(frame.loc[frame.hours_after_challenge.isin([24, 36]),
          ["challenge_treatment", "hours_after_challenge", "state_id", "digitized_mean_percent",
           "reading_lower_percent", "reading_upper_percent"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "digitized-v1")
    main(parser.parse_args().output)
