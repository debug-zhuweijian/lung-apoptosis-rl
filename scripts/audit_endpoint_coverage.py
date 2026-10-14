"""Audit real endpoint coverage before assigning splits or generating features.

This writes a preparation inventory, never a ready G manifest. No label is
duplicated over expression cells, and no synthetic feature vector is generated.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.endpoint_alignment import aggregate_units, file_hash, load_alignment, stable_id

BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1"


def connected_groups(rows, *, source_holdout):
    """Count independent split blocks, joining shared experiments and chemicals."""
    parents = {row["condition_id"]: row["condition_id"] for row in rows}
    if len(parents) != len(rows):
        raise ValueError("duplicate condition in coverage inventory")

    def root(value):
        while parents[value] != value:
            value = parents[value]
        return value

    owners = {}
    for row in rows:
        keys = [("experiment", row["source_id"], row["experiment_block"])]
        keys += [("chemical", x) for x in row["chemical_group_ids"]]
        if source_holdout:
            keys.append(("source", row["source_id"]))
        for key in keys:
            owner = owners.setdefault(key, row["condition_id"])
            left, right = root(row["condition_id"]), root(owner)
            parents[max(left, right)] = min(left, right)
    groups = {}
    for condition in sorted(parents):
        groups.setdefault(root(condition), []).append(condition)
    return list(groups.values())


def summarize(records):
    candidates = [row for row in records if row["annexin_input_candidate"]]
    definitions = {}
    for row in candidates:
        definitions.setdefault(row["observation_definition_id"], []).append(row)
    coverage = []
    for definition, rows in sorted(definitions.items()):
        blocks = connected_groups(rows, source_holdout=True)
        coverage.append({"observation_definition_id": definition,
            "protocol_ids": sorted({x["protocol_id"] for x in rows}),
            "condition_count": len(rows),
            "source_count": len({x["source_id"] for x in rows}),
            "chemical_count": len({g for x in rows for g in x["chemical_group_ids"]}),
            "source_and_chemical_holdout_blocks": len(blocks),
            "experiment_and_chemical_holdout_blocks": len(connected_groups(rows, source_holdout=False)),
            "four_nonempty_partitions_possible_under_source_holdout": len(blocks) >= 4})
    return {"conditions": len(records),
        "measured_annexin_condition_units": sum(x["has_measured_annexin"] for x in records),
        "annexin_single_drug_24h_chemical_input_candidates": len(candidates),
        "candidate_sources": sorted({x["source_id"] for x in candidates}),
        "candidate_chemical_groups": len({g for x in candidates for g in x["chemical_group_ids"]}),
        "observed_initial_expression_conditions": sum(x["initial_state_kind"] != "reference" for x in records),
        "complete_expression_feature_conditions": sum(x["expression_feature_status"] == "complete" for x in records),
        "observation_definition_coverage": coverage,
        "formal_endpoint_training_ready": False,
        "reason": "No generated response features; input candidates are not qualified supervised training units.",
        "split_scope": "Counts are combinatorial coverage, not statistical power or generalization evidence.",
        "controls_scope": "Controls remain separate; no control label is copied into drug conditions."}


def main(alignment, inputs, output):
    if (output / "manifest.json").exists():
        raise FileExistsError("Coverage snapshot exists; choose a new version")
    data = load_alignment(alignment, ROOT)
    contract = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in contract["files"].items():
        if file_hash(inputs / name) != digest:
            raise ValueError("condition input changed: " + name)
    for name, digest in contract["inputs_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("condition input dependency changed: " + name)
    aligned_key = (alignment / "manifest.json").relative_to(ROOT).as_posix()
    if contract["inputs_sha256"].get(aligned_key) != file_hash(alignment / "manifest.json"):
        raise ValueError("condition inputs use a different endpoint alignment")
    requests = pd.read_parquet(inputs / "condition_requests.parquet")
    components = pd.read_parquet(inputs / "components.parquet")
    if requests.condition_id.duplicated().any() or set(requests.condition_id) != set(data["conditions"].condition_id):
        raise ValueError("condition request coverage mismatch")
    original = data["conditions"].set_index("condition_id")
    for row in requests.to_dict("records"):
        for field in ("source_id", "protocol_id", "experiment_block", "treatment_hours", "initial_state_kind"):
            if row[field] != original.loc[row["condition_id"], field]:
                raise ValueError("request condition identity mismatch: " + field)
    if not set(components.condition_id) <= set(requests.condition_id):
        raise ValueError("unknown component condition")
    units = aggregate_units(data["conditions"], data["observations"])
    by_condition = {row["condition_id"]: row for row in units.to_dict("records")}
    records = []
    for row in requests.to_dict("records"):
        parts = components[components.condition_id == row["condition_id"]]
        unit = by_condition.get(row["condition_id"], {})
        protocol = data["protocols"][row["protocol_id"]]
        chemical_ids = sorted(set(parts.loc[parts.structure_eligible, "identity_id"].dropna()))
        blockers = [x for x in row["input_contract_blockers"].split(";") if x]
        annexin = unit.get("endpoint") == "Annexin_positive_fraction"
        input_candidate = bool(annexin and not blockers and not row["is_control"]
            and len(parts) == 1 and len(chemical_ids) == 1 and row["treatment_hours"] == 24)
        usable = data["observations"][(data["observations"].condition_id == row["condition_id"])
            & data["observations"].measurement_status.eq("observed")
            & data["observations"].exclusion_reason.fillna("").eq("")]
        records.append({"condition_id": row["condition_id"], "source_id": row["source_id"],
            "protocol_id": row["protocol_id"], "experiment_block": row["experiment_block"],
            "cell_line": row["cell_line"], "initial_state_kind": row["initial_state_kind"],
            "initial_state_id": row["initial_state_id"], "treatment_hours": row["treatment_hours"],
            "is_control": row["is_control"], "chemical_group_ids": chemical_ids,
            "source_compounds": parts.compound_id.tolist(),
            "has_measured_annexin": annexin, "annexin_input_candidate": input_candidate,
            "endpoint": protocol["endpoint"], "denominator": protocol["denominator"],
            "sampling_gate": protocol["sampling_gate"],
            "observation_definition_id": stable_id("observation_definition", protocol["endpoint"],
                protocol["denominator"], protocol["sampling_gate"]),
            "observation_ids": usable.observation_id.tolist(),
            "available_numeric_records": int(unit.get("available_numeric_records", 0)),
            "measured_replicates": int(unit.get("measured_replicates", 0)),
            "input_contract_blockers": blockers,
            "expression_feature_status": row["expression_feature_status"],
            "assigned_partition": None, "same_sample_expression_endpoint": False,
            "feature_generation_ready": False})
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [("condition_inventory.json", records), ("coverage.json", summarize(records))]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"status": "coverage_audited_features_pending", "formal_training_ready": False,
        "inputs": {str(p.resolve()): file_hash(p) for p in
            (alignment / "manifest.json", inputs / "manifest.json", Path(__file__))},
        "files": {name: file_hash(output / name) for name in ("condition_inventory.json", "coverage.json")}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summarize(records), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=BASE / "curated-20261010-r3")
    parser.add_argument("--inputs", type=Path, default=BASE / "condition-inputs-v2")
    parser.add_argument("--output", type=Path, default=BASE / "coverage-v1")
    args = parser.parse_args()
    main(*(p if p.is_absolute() else ROOT / p for p in (args.alignment, args.inputs, args.output)))
