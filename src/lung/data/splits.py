"""Deterministic, role-isolated splits; no response-driven rebalancing."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .common import OUTER, ROLES, STATE_LABELS


def grouped_roles(groups, prefix="lung-v1|") -> dict[str, str]:
    groups = sorted(set(str(value) for value in groups))
    ordered = sorted(groups, key=lambda value: (hashlib.sha256((prefix + value).encode()).hexdigest(), value))
    boundaries = [0] + [math.floor(len(ordered) * p) for p in (.5, .6, .7, .8, .9)] + [len(ordered)]
    return {group: role for role, left, right in zip(ROLES, boundaries, boundaries[1:])
            for group in ordered[left:right]}


def condition_roles(conditions: pd.DataFrame, chemical: pd.DataFrame):
    columns = ["drug_id", "chemical_or_experiment_group_id", "identity_status", "fingerprint_row"]
    reviewed = "eligible_for_structural_features" in chemical.columns
    if reviewed:
        columns += ["eligible_for_structural_features", "is_vehicle", "identity_version"]
    result = conditions.merge(chemical[columns], on="drug_id", how="left", validate="many_to_one")
    treated = result["condition_type"].eq("treated")
    if reviewed:
        accepted_status = (result["identity_status"].eq("original_cid_structure_verified")
                           | (result["identity_status"].eq("manufacturer_nominal_form_verified")
                              & result["identity_version"].eq("identity-v3")))
        valid = (accepted_status
                 & result["eligible_for_structural_features"].eq(True) & treated
                 & result["chemical_or_experiment_group_id"].notna())
        controls = result["condition_type"].eq("control")
        if not (result.loc[controls, "is_vehicle"].eq(True)
                & result.loc[controls, "identity_status"].eq("vehicle_control")).all():
            raise ValueError("Identity-v2 control conditions require a verified vehicle")
    else:
        valid = result["identity_status"].eq("parsed_full_form") & treated
    resolved = result.loc[valid].copy()
    role_map = grouped_roles(resolved["chemical_or_experiment_group_id"])
    result["role"] = result["chemical_or_experiment_group_id"].map(role_map)
    if reviewed:
        result["role"] = result["role"].where(valid)
    result["outer_split"] = result["role"].map(OUTER)
    result["eligible"] = valid | result["condition_type"].eq("control")
    result["source_id"] = result["source"]
    result["background_id"] = result["cell_line_id"]
    result["protocol_id"] = "tahoe_continuous_exposure_24h_v1"
    result["duration_h"] = result["treatment_hours"].astype(float)
    result["dose_um"] = result["dose_value"].astype(float)
    if not result.loc[valid, "dose_unit"].isin(["uM", "µM", "μM"]).all():
        raise ValueError("Unexpected source dose units; conversion must be explicit")
    if (result.loc[valid, "dose_um"].isna().any()
            or (result.loc[valid, "dose_um"] < 0).any()
            or not result["duration_h"].eq(24).all()
            or result["is_sequential"].any()):
        raise ValueError("Prepared V1 is only supported for recorded single 24 h conditions")
    # The source has no biological replicate identifier; sample IDs are retained.
    result["replicate_id"] = result["source_sample_id"]
    result["replicate_scope"] = "source_sample_unit_not_verified_biological_replicate"
    return result


def control_roles(controls: pd.DataFrame) -> dict[str, str]:
    result = {}
    for (source, plate), rows in controls.groupby(["source_id", "plate"], sort=True):
        # Physical control cells are never repeated between role containers.
        mapping = grouped_roles(rows["cell_id"], prefix=f"lung-v1|control|{source}|{plate}|")
        result.update(mapping)
    return result


def tidyi_splits(cells: pd.DataFrame):
    cells = cells.copy()
    columns = cells["well"].str.extract(r"^[A-Za-z]+(\d+)$", expand=False)
    if columns.isna().any() or cells["cell_id"].duplicated().any():
        raise ValueError("Invalid or duplicate TIDYI cell / well metadata")
    cells["well_column"] = columns.astype(int)
    if not cells["state_label_original"].isin(STATE_LABELS).all():
        raise ValueError("Unexpected TIDYI original state label")
    cells["state_class"] = cells["state_label_original"].map(dict(zip(STATE_LABELS, range(3))))
    plates = sorted(cells["plate"].unique())
    if plates != [1, 8]:
        raise ValueError("TIDYI V1 requires exactly the registered plates 1 and 8")
    frames, reports = [], {}
    for train_plate, test_plate in ((1, 8), (8, 1)):
        fold = f"{train_plate}_to_{test_plate}"
        ids = cells.loc[cells["plate"].eq(train_plate), "well_column"].unique()
        ordered = sorted(ids, key=lambda col: hashlib.sha256(f"tidyi-v1|{train_plate}|{col}".encode()).hexdigest())
        b1, b2 = math.floor(len(ordered) * .70), math.floor(len(ordered) * .85)
        assignment = {int(col): role for role, values in (
            ("fit", ordered[:b1]), ("selection", ordered[b1:b2]),
            ("calibration", ordered[b2:])) for col in values}
        frame = cells.copy()
        frame["role"] = frame["well_column"].map(assignment)
        frame.loc[frame["plate"].eq(test_plate), "role"] = "final_test"
        frame["outer_split"] = frame["role"].map(OUTER)
        frame["fold"] = fold
        counts = {role: {label: int((part["state_label_original"] == label).sum())
                         for label in STATE_LABELS} for role, part in frame.groupby("role")}
        missing = {role: [label for label in STATE_LABELS if counts.get(role, {}).get(label, 0) == 0]
                   for role in ("fit", "selection", "calibration", "final_test")}
        reports[fold] = {"status": "ready" if not any(missing.values()) else "STATE_SPLIT_INSUFFICIENT",
                         "class_counts": counts, "missing_classes": missing,
                         "train_plate": train_plate, "test_plate": test_plate,
                         "column_roles": {str(key): value for key, value in assignment.items()}}
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), reports
