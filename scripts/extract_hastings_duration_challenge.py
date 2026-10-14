"""Read two published 72 h protein means; do not infer apoptosis fractions."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/duration_round5"
SOURCE = BASE / "pulse"
OUTPUT = BASE / "cleaved_caspase_challenge_v1"
IMAGE = SOURCE / "elife-53367-fig1-figsupp2-v1.jpg"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_value(y: float, zero: float, eight: float) -> float:
    return 8.0 * (zero - y) / (zero - eight)


def extract() -> tuple[list[dict], dict]:
    pixels = np.asarray(Image.open(IMAGE).convert("RGB")).astype(np.int16)
    if pixels.shape != (5264, 3981, 3):
        raise ValueError("Unexpected original figure dimensions")
    # Last panel: source marker at 72 h. Centre of the printed 0 and 8 ticks.
    zero_y, eight_y = 5071.0, 4352.5
    x0, x1, y0, y1 = 3819, 3861, 4750, 5010
    region = pixels[y0:y1, x0:x1]
    red = ((region[:, :, 0] > 140) & (region[:, :, 1] < 130)
           & (region[:, :, 2] < 130)
           & ((region[:, :, 0] - region[:, :, 1]) > 60))
    grey = ((region.max(axis=2) - region.min(axis=2) < 14)
            & (region.mean(axis=2) > 80) & (region.mean(axis=2) < 205))
    rows = []
    details = {}
    for schedule, mask, exposure, visual_range in [
        ("pulse_2h_then_drug_removal", red, 2.0, (1.10, 1.40)),
        ("continuous_72h", grey, 72.0, (2.75, 3.10)),
    ]:
        distance = distance_transform_edt(mask)
        if distance.max() < 10:
            raise ValueError(f"Filled marker could not be distinguished: {schedule}")
        yy, xx = np.where(distance == distance.max())
        marker_y, marker_x = float(np.median(yy) + y0), float(np.median(xx) + x0)
        mean = relative_value(marker_y, zero_y, eight_y)
        envelope = [relative_value(marker_y + dy, zero_y + dz, eight_y + de)
                    for dy, dz, de in itertools.product((-4, 4), (-2, 2), (-2, 2))]
        if not visual_range[0] <= mean <= visual_range[1]:
            raise ValueError(f"Outside independent visual reading: {schedule}")
        condition = f"Hastings2020:Fig1S2:A549:{schedule}"
        rows.append({
            "observation_id": condition + ":h72:relative_cleaved_caspase3",
            "condition_id": condition,
            "cell_line": "A549",
            "compound": "cisplatin",
            "nominal_concentration": 5.0,
            "concentration_unit": "ug/mL",
            "exposure_start_hours": 0.0,
            "exposure_end_hours": exposure,
            "exposure_duration_hours": exposure,
            "elapsed_observation_hours": 72.0,
            "post_removal_followup_hours": 72.0 - exposure,
            "protocol": schedule,
            "readout": "relative_cleaved_caspase3_MAGPIX_signal",
            "unit": "published_relative_signal",
            "digitized_mean": round(mean, 4),
            "digitization_lower": round(min(envelope), 4),
            "digitization_upper": round(max(envelope), 4),
            "marker_x": marker_x,
            "marker_y": marker_y,
            "reported_n": 3,
            "raw_replicate_values_available_for_this_comparison": False,
            "author_sd_digitized": False,
            "same_sample_RNA_pair": False,
            "formal_training_ready": False,
        })
        details[schedule] = {"marker_x": marker_x, "marker_y": marker_y,
                             "mask_interior_radius": float(distance.max()),
                             "independent_visual_bounds": visual_range}
    return rows, {
        "source_shape": list(pixels.shape), "panel": "Fig1-figure supplement2 last/right panel",
        "axis_zero_y": zero_y, "axis_eight_y": eight_y,
        "axis_tick_pixel_runs": {"zero": [5067, 5075], "eight": [4349, 4356]},
        "marker_roi_xyxy": [x0, y0, x1, y1],
        "marker_y_tolerance_pixels": 4, "axis_y_tolerance_pixels": 2,
        "uncertainty_kind": "analyst_digitization_envelope_not_author_SD_CI_or_biological_variance",
        "method": "colour-specific filled-marker centre from distance-transform maximum; linear printed y axis",
        "markers": details,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite the frozen challenge; use a new version")
    rows, calibration = extract()
    sources = [IMAGE, SOURCE / "elife-53367-fig1-figsupp1-v1.jpg",
               SOURCE / "PMC7282820.xml", SOURCE / "elife-53367-v1.pdf", Path(__file__)]
    before = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
    contract = {
        "status": "published_relative_protein_duration_challenge",
        "source_doi": "10.7554/eLife.53367",
        "source_sections": ["s2-1", "s4-2", "s4-5", "fig1s1", "fig1s2"],
        "common_observation_time_hours": 72,
        "common_drug_start_hours": 0,
        "finite_exposure_durations_hours": [2, 72],
        "drug_removal_in_pulse": "explicit_in_fig1s1A",
        "wash_count_and_residual_drug_clearance": "not_explicitly_quantified_in_checked_source",
        "background": "Advanced RPMI; 1% FCS; 1% GlutaMAX; 5% CO2; 20% O2",
        "normalization": "published linear relative signal; exact denominator for this panel not independently recovered",
        "PCA_log_transformation_applied": False,
        "allowed_evaluation": ["error_on_published_relative_protein_means",
                               "within_source_order_of_two_exposure_schedules_at_common_72h"],
        "not_established": ["Annexin_positive_fraction", "single_cell_future_fate",
                            "RNA_increment", "dose_optimization", "broad_drug_generalization",
                            "whole_model_data_feasibility", "independent_blind_test"],
        "evaluation_is_retrospective": True,
        "independent_observations_not_multiplied_by_digitization": True,
        "biological_replicates_reported": 3,
        "original_replicate_values_recovered": 0,
        "RNA_samples_for_these_endpoints": [],
        "formal_training_ready": False,
    }
    if any(sha256(ROOT / name) != expected for name, expected in before.items()):
        raise ValueError("Source changed during extraction")
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, value in [("calibration.json", calibration), ("contract.json", contract)]:
        (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "status": contract["status"], "observations": 2,
                "source_base": "project_root", "inputs_sha256": before,
                "files_sha256": {p.name: sha256(p) for p in sorted(OUTPUT.iterdir())},
                "formal_training_ready": False}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "observations": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
