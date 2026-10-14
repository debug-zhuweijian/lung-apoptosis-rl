"""Measured endpoint views; this explicit label reader is never a P producer.

The two Nav-Gal dose interpretations are alternative views of four observations,
not eight independently measured labels. Roles are supplied by a study protocol.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

from .endpoint_protocol_inputs import BASE, EndpointProtocolInputs


VIEW_SPECS = {
    "vielhauer_source_nominal": ("Vielhauer2013", "source_nominal"),
    "nav_figure_caption_10uM": ("GonzalezGualda2020", "figure_caption"),
    "nav_methods_11uM": ("GonzalezGualda2020", "methods"),
}
ROLES = {"fit", "selection", "calibration", "final_test", "external_test", "unassigned"}


def fraction_observation(observation):
    """Convert only unit scale, never the assay definition or population."""
    source = observation["source_record"]
    raw_value, unit = observation["value"], observation["unit"]
    if unit == "fraction":
        lower = source["measured_endpoint"]["pixel_lower_fraction"]
        upper = source["measured_endpoint"]["pixel_upper_fraction"]
        scale = 1.0
    elif unit == "percent":
        lower, upper = source["reading_lower_percent"], source["reading_upper_percent"]
        scale = 0.01
    else:
        raise ValueError("ENDPOINT_UNIT_NOT_FRACTION_OR_PERCENT")
    values = (float(lower) * scale, float(raw_value) * scale, float(upper) * scale)
    if not all(math.isfinite(v) for v in values) or not 0 <= values[0] <= values[1] <= values[2] <= 1:
        raise ValueError("ENDPOINT_READING_BOUNDS_OR_VALUE_INVALID")
    return {"label_fraction": values[1], "raw_value": raw_value, "raw_unit": unit,
            "unit_conversion_multiplier": scale, "reading_lower_fraction": values[0],
            "reading_upper_fraction": values[2],
            "uncertainty_kind": observation["uncertainty_kind"],
            "reading_bounds_are_biological_confidence_interval": False}


class EndpointProtocolSupervision:
    """Read labels deliberately; serving/generating features uses another API."""

    def __init__(self, bundle):
        self._bundle = bundle
        labels = bundle.observations()
        if len({r["observation_id"] for r in labels}) != len(labels):
            raise ValueError("DUPLICATE_SOURCE_OBSERVATION")
        self._labels = {r["observation_id"]: r for r in labels}
        self._mapping = {r["candidate_id"]: r["observation_id"] for r in bundle.observation_mapping()}

    @classmethod
    def load(cls, directory=BASE / "endpoint-protocol-inputs-v1"):
        return cls(EndpointProtocolInputs.load(directory))

    def view(self, view_id, *, roles=None):
        if view_id not in VIEW_SPECS:
            raise ValueError("UNKNOWN_SUPERVISION_VIEW")
        source_id, scenario = VIEW_SPECS[view_id]
        units, seen = [], set()
        for cid in self._bundle.candidate_ids:
            candidate = self._bundle.candidate(cid)
            if candidate["source_id"] != source_id or candidate["dose_scenario"] != scenario:
                continue
            oid = self._mapping[cid]
            if oid in seen:
                raise ValueError("DOSE_SCENARIOS_MUST_NOT_EXPAND_SOURCE_Y")
            seen.add(oid)
            observation = self._labels[oid]
            units.append({"unit_id": oid, "observation_id": oid, "candidate_id": cid,
                "condition_id": candidate["condition_id"], "source_id": source_id,
                "state_id": candidate["state_id"], "dose_scenario": scenario,
                "p_request_id": candidate["p_request_id"], "role": "unassigned",
                "label_kind": "measured_published_group_mean", "weight": 1.0,
                "weight_meaning": "one_source_observation_not_number_of_generated_cells",
                "endpoint_definition": copy.deepcopy(candidate["endpoint_definition"]),
                "shared_control_id": candidate["shared_control_id"],
                "terminal_time_h": candidate["protocol"]["terminal_time_h"],
                "raw_replicates_recovered": False,
                "independent_biological_replicate_ids_verified": False,
                "data_exposure": "public_outcomes_inspected_retrospective_design",
                "source_record": copy.deepcopy(observation["source_record"]),
                **fraction_observation(observation)})
        expected = 20 if source_id == "Vielhauer2013" else 4
        if len(units) != expected:
            raise ValueError("SUPERVISION_VIEW_INCOMPLETE")
        if roles is not None:
            if set(roles) != seen or any(role not in ROLES for role in roles.values()):
                raise ValueError("EXPLICIT_OBSERVATION_ROLE_ASSIGNMENT_REQUIRED")
            units = [{**unit, "role": roles[unit["observation_id"]]} for unit in units]
        return units

    def combine_views(self, view_ids, *, roles=None):
        """Combine explicit nonoverlapping units, retaining different definitions.

        This is an indexed label collection, not permission to fit one assay head
        or to rank across observation definitions. A model declares that scope.
        """
        rows = [row for view_id in view_ids for row in self.view(view_id)]
        ids = [r["observation_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("OVERLAPPING_SUPERVISION_VIEWS: alternative dose interpretations share Y")
        if roles is not None:
            if set(roles) != set(ids) or any(role not in ROLES for role in roles.values()):
                raise ValueError("EXPLICIT_OBSERVATION_ROLE_ASSIGNMENT_REQUIRED")
            rows = [{**row, "role": roles[row["observation_id"]]} for row in rows]
        return rows


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_views(output, directory=BASE / "endpoint-protocol-inputs-v1"):
    """Write measured views into a new directory, with no model features."""
    output, directory = Path(output).resolve(), Path(directory).resolve()
    if output.exists():
        raise FileExistsError("Use a new supervision version; refusing overwrite")
    supervision = EndpointProtocolSupervision.load(directory)
    views = {name: supervision.view(name) for name in VIEW_SPECS}
    unique_observations = {row["observation_id"] for rows in views.values() for row in rows}
    if len(unique_observations) != 24:
        raise ValueError("SUPERVISION_SOURCE_UNIT_COUNT_CHANGED")
    output.mkdir(parents=True)
    for name, rows in views.items():
        (output / (name + ".json")).write_text(json.dumps(rows, ensure_ascii=False,
            allow_nan=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "status": "measured_label_views_ready_features_and_roles_pending",
        "input_directory": str(directory), "input_manifest_sha256": file_sha(directory / "manifest.json"),
        "label_reader_sha256": file_sha(__file__), "unique_active_observation_ids": 24,
        "view_unit_counts": {name: len(rows) for name, rows in views.items()},
        "roles_frozen": False, "formal_training_ready": False,
        "source_observation_definitions_must_remain_distinct": True,
        "alternative_views_must_not_be_concatenated_as_independent_Y": True,
        "generated_P_descriptors": 0, "model_execution_performed": False,
        "files_sha256": {name + ".json": file_sha(output / (name + ".json")) for name in views}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
        allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return load_views(output)


def load_views(output):
    """Reproduce each unit from the frozen source, not just a self-reported hash."""
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("formal_training_ready") is not False:
        raise ValueError("SUPERVISION_MANIFEST_INVALID")
    directory = Path(manifest["input_directory"])
    if (file_sha(directory / "manifest.json") != manifest["input_manifest_sha256"]
            or file_sha(__file__) != manifest["label_reader_sha256"]):
        raise ValueError("SUPERVISION_SOURCE_OR_READER_CHANGED")
    source = EndpointProtocolSupervision.load(directory)
    if set(manifest["files_sha256"]) != {name + ".json" for name in VIEW_SPECS}:
        raise ValueError("SUPERVISION_VIEW_FILES_MISSING")
    for name in VIEW_SPECS:
        path = output / (name + ".json")
        expected = source.view(name)
        if (file_sha(path) != manifest["files_sha256"][path.name]
                or json.loads(path.read_text(encoding="utf-8")) != expected
                or manifest["view_unit_counts"][name] != len(expected)):
            raise ValueError("SUPERVISION_VIEW_CHANGED")
    if (manifest["unique_active_observation_ids"] != 24 or manifest["roles_frozen"] is not False
            or manifest["generated_P_descriptors"] != 0 or manifest["model_execution_performed"] is not False
            or manifest["source_observation_definitions_must_remain_distinct"] is not True
            or manifest["alternative_views_must_not_be_concatenated_as_independent_Y"] is not True):
        raise ValueError("SUPERVISION_READINESS_OR_UNIT_CLAIM_CHANGED")
    return manifest
