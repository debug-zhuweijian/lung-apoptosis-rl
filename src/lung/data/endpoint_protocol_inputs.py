"""Reviewed endpoint protocol inputs, separate from observations and model outputs.

The standard continuous-24h P descriptor is a proposed reference feature,
not observed or predicted washout-terminal RNA. No model is executed here.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import data_file_bindings

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration"
VIEL = BASE / "vielhauer-duration-challenge-v1"
CHEMICAL = ROOT / "experiments/20261010-data-alignment-round11/chemical_condition_inputs.json"
NAV = BASE / "endpoint-alignment-v1/source_reviews/state_function_round4/nav_gal"
FEATURE_NAMES = ("dose_uM", "exposure_duration_h", "washout_at_h",
                 "drug_free_recovery_h", "terminal_time_h")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_bindings(bindings):
    for name, digest in data_file_bindings(bindings).items():
        require(sha(name) == digest, "Source hash mismatch: " + name)


def bind(bindings, path, expected=None):
    path = Path(path).resolve()
    if path.suffix.lower() == ".py":
        return
    actual = sha(path)
    require(expected is None or actual == expected, "Source hash mismatch: " + str(path))
    require(str(path) not in bindings or bindings[str(path)] == actual, "Conflicting source binding")
    bindings[str(path)] = actual


def validate_protocol(row):
    require(set(row["protocol"]) == {
        "time_origin", "exposure_duration_h", "washout_at_h", "washout_status",
        "drug_free_recovery_h", "terminal_time_h", "exposure_intervals",
        "media_events", "prior_drug_retention", "floating_at_washout_recovered"},
        "Protocol schema mismatch")
    p = row["protocol"]
    require(p["time_origin"] == "challenge_drug_start", "Physical time origin changed")
    for key in ("exposure_duration_h", "drug_free_recovery_h", "terminal_time_h"):
        require(isinstance(p[key], (int, float)) and math.isfinite(p[key]) and p[key] >= 0,
                "Invalid physical clock")
    require(p["terminal_time_h"] == 24 and
            p["exposure_duration_h"] + p["drug_free_recovery_h"] == 24,
            "Exposure plus recovery must equal common 24h endpoint")
    require(p["exposure_intervals"] == [[0, p["exposure_duration_h"]]], "Exposure path mismatch")
    if row["protocol_arm"] == "continuous_24h":
        require(p["exposure_duration_h"] == 24 and p["washout_at_h"] is None
                and p["washout_status"] == "not_applicable_no_challenge_withdrawal"
                and p["media_events"] == [], "Continuous protocol became withdrawal")
    else:
        require(row["protocol_arm"] == "pulse_6h_washout_recovery18h"
                and p["exposure_duration_h"] == p["washout_at_h"] == 6
                and p["washout_status"] == "reported_media_decant_rinse_fresh_media"
                and p["media_events"] == [{"at_h": 6, "event": "media_decant_rinse_fresh_media"}],
                "Pulse/withdrawal physical clock mismatch")
    require(row["dose_uM"] > 0 and math.isfinite(row["dose_uM"]), "Invalid actual candidate dose")


def protocol(arm, prior="not_applicable"):
    pulse = arm == "pulse_6h_washout_recovery18h"
    return {"time_origin": "challenge_drug_start", "exposure_duration_h": 6 if pulse else 24,
            "washout_at_h": 6 if pulse else None,
            "washout_status": "reported_media_decant_rinse_fresh_media" if pulse else
                              "not_applicable_no_challenge_withdrawal",
            "drug_free_recovery_h": 18 if pulse else 0, "terminal_time_h": 24,
            "exposure_intervals": [[0, 6 if pulse else 24]],
            "media_events": [{"at_h": 6, "event": "media_decant_rinse_fresh_media"}] if pulse else [],
            "prior_drug_retention": prior,
            "floating_at_washout_recovered": "not_reported" if pulse else "not_applicable"}


def request_for(row):
    # State-specific support is intentionally not claimed. The reference must
    # be supplied and versioned by a later actual descriptor producer.
    key = [row["source_id"], row["chemical"]["compound_id"], row["dose_uM"], "continuous_24h"]
    request_id = "standardP:" + hashlib.sha256(json.dumps(key).encode()).hexdigest()
    return {"request_id": request_id, "descriptor_role": "standard_reference",
            "compound_id": row["chemical"]["compound_id"], "dose_uM": row["dose_uM"],
            "reference_protocol_arm": "continuous_24h", "reference_terminal_time_h": 24,
            "reference_cell_line": "A549", "reference_source_id": None,
            "reference_state_id": None, "reference_measurement_time_h": None,
            "support_status": "pending_reference_and_response_features",
            "interpretation": "standard_continuous24h_response_descriptor_not_washout_terminal_RNA"}


def prepare(viel=VIEL, chemical=CHEMICAL, nav=NAV):
    from scripts.build_vielhauer_duration_challenge import load_challenge, prepare as prepare_viel
    from scripts.evaluate_nav_gal_challenge import load_challenge as load_nav

    bindings = {}
    viel, chemical = Path(viel).resolve(), Path(chemical).resolve()
    challenge = load_challenge(viel)
    vm = read(viel / "manifest.json")
    bind(bindings, viel / "manifest.json")
    for name, digest in vm["files_sha256"].items():
        bind(bindings, viel / name, digest)
    _, upstream = prepare_viel(vm["source_directory"])
    for name, digest in upstream.items():
        bind(bindings, Path(vm["source_directory"]) / name, digest)
    source_directory = Path(vm["source_directory"])
    source_manifest = source_directory / "manifest.json"
    bind(bindings, source_manifest)
    for name, entry in read(source_manifest)["files"].items():
        bind(bindings, source_directory / name, entry["sha256"])
    assembly_path = source_directory / "figure5_highres_assembly.json"
    bind(bindings, assembly_path)
    for tile in read(assembly_path)["tiles"]:
        bind(bindings, source_directory / tile["file"], tile["sha256"])
    sidecar = read(chemical)
    bind(bindings, chemical)
    for name, digest in sidecar["input_files_sha256"].items():
        bind(bindings, name, digest)
    identities_path = Path(sidecar["fingerprints_path"]).parent / "nominal_structures.json"
    identity_manifest = read(identities_path.parent / "manifest.json")
    for entry in identity_manifest["files"] + identity_manifest["read_only_upstream"]:
        bind(bindings, identities_path.parent / entry["path"], entry["sha256"])
    identities = {r["compound_id"]: r for r in read(identities_path)}
    fingerprints = np.load(sidecar["fingerprints_path"], allow_pickle=False)
    require(fingerprints.shape == (2, 2048) and np.isin(fingerprints, [0, 1]).all(),
            "Invalid nominal chemical fingerprint array")
    rows = sidecar["rows"]
    require(len(rows) == 20 and len({r["condition_id"] for r in rows}) == 20,
            "Chemical sidecar requires exactly 20 unique conditions")
    by_id = {r["condition_id"]: r for r in rows}
    candidates, observations, mappings, references = [], [], [], []
    for source in challenge["active_conditions"]:
        require(source["condition_id"] in by_id, "Chemical condition identity mismatch")
        joined = by_id[source["condition_id"]]
        for key in ("state_id", "protocol_arm", "dose_uM", "exposure_duration_h",
                    "washout_at_h", "drug_free_recovery_h", "terminal_time_h"):
            require(joined[key] == source[key], "Chemical candidate dose/clock/state mismatch: " + key)
        require(joined["source_drug_id"] == source["drug_id"], "Chemical source drug mismatch")
        identity = identities[joined["compound_id"]]
        require(identity["nominal_published_structure_eligible"] is True and
                identity["fingerprint_row"] == joined["fingerprint_row"] and
                identity["inchikey"] == joined["full_structure_inchikey"] and
                identity["pubchem_cid"] == joined["pubchem_cid"], "Nominal chemical identity/row mismatch")
        index = joined["fingerprint_row"]
        require(isinstance(index, int) and index in (0, 1) and
                int(fingerprints[index].sum()) == identity["fingerprint_on_bits"], "Fingerprint row mismatch")
        cid = source["condition_id"]
        candidate = {"candidate_id": cid, "condition_id": cid, "source_id": "Vielhauer2013",
            "state_id": source["state_id"], "cell_line": "A549",
            "state_kind": "source_defined_group_context", "state_reference_id": None,
            "state_evidence": "parental_standard_culture_no_matched_RNA",
            "source_drug_id": source["drug_id"], "protocol_id": source["protocol_id"],
            "protocol_arm": source["protocol_arm"], "dose_uM": source["dose_uM"],
            "source_dose": {"value": source["dose_nM"], "unit": "nM"},
            "dose_scenario": "source_nominal", "actual_dose_verified": False,
            "chemical": {"compound_id": identity["compound_id"], "inchikey": identity["inchikey"],
                "identity_status": "nominal_published_graph_historical_batch_not_verified",
                "fingerprint_path": sidecar["fingerprints_path"], "fingerprint_row": index},
            "protocol": protocol(source["protocol_arm"]),
            "state_covariates": {"nominal_seeded_cells": 300000, "vessel": "T25",
                                 "preculture_h": 24, "initial_RNA": "not_available"},
            "endpoint_definition": {"assay": "Annexin_V_PI", "label": "early_plus_late_Annexin_positive",
                "denominator": source["denominator"], "sampling_gate": source["sampling_gate"],
                "harvest": "terminal_floating_plus_adherent"},
            "shared_control_id": next(r["condition_id"] for r in challenge["reference_conditions"]
                                      if r["source_figure"] == source["source_figure"]),
            "split_role": "retrospective_challenge_unassigned_for_training",
            "labels_seen_in_design": True}
        candidates.append(candidate)
        oid = cid + ":annexin24h"
        # Original source fields are labels/provenance only; never input covariates.
        observations.append({"observation_id": oid, "source_id": "Vielhauer2013",
            "condition_id": cid, "dose_scenarios_share_Y": False,
            "source_record": source, "unit": "fraction",
            "value": source["measured_endpoint"]["mean_fraction"],
            "uncertainty_kind": "pixel_reading_bounds_not_SEM", "raw_replicates_recovered": False})
        mappings.append({"candidate_id": cid, "observation_id": oid})
    references.extend({"source_id": "Vielhauer2013", "source_record": r}
                      for r in challenge["reference_conditions"])
    nav_status = "not_requested"
    if nav is not None:
        nav = Path(nav).resolve()
        n = load_nav(nav / "challenge-v1", nav / "chemical_review", ROOT)
        for name, digest in n["source_hashes"].items():
            bind(bindings, name, digest)
        for condition in n["conditions"]:
            primary = n["primary"].loc[n["primary"].curve_id.eq(condition["condition_group_id"])]
            require(len(primary) == 1, "Primary observation join mismatch")
            record = json.loads(primary.to_json(orient="records"))[0]
            if condition["is_vehicle"]:
                references.append({"source_id": "GonzalezGualda2020", "source_record": record})
                continue
            chem = condition["chemical"]
            oid = record["observation_id"]
            observations.append({"observation_id": oid, "source_id": "GonzalezGualda2020",
                "condition_id": condition["condition_group_id"], "dose_scenarios_share_Y": True,
                "source_record": record, "unit": "percent", "value": record["digitized_mean_percent"],
                "uncertainty_kind": "pixel_reading_bounds_not_SD_or_CI", "raw_replicates_recovered": False})
            for scenario, dose in n["contract"]["dose_scenarios"].items():
                cid = condition["condition_group_id"] + ":" + scenario + ":continuous24_t24"
                candidates.append({"candidate_id": cid, "condition_id": condition["condition_group_id"],
                    "source_id": "GonzalezGualda2020", "cell_line": "A549",
                    "state_id": condition["state_id"], "state_kind": "author_reported_group_state",
                    "state_evidence": condition["state_evidence"], "state_reference_id": None,
                    "source_drug_id": condition["challenge_treatment"], "protocol_id": condition["protocol_id"],
                    "protocol_arm": "continuous_24h", "dose_uM": dose,
                    "source_dose": {"value": dose, "unit": "uM"}, "dose_scenario": scenario,
                    "actual_dose_verified": False,
                    "chemical": {"compound_id": chem["identity_id"], "inchikey": chem["rdkit_inchikey"],
                        "identity_status": chem["status"], "fingerprint_row": chem["fingerprint_row"],
                        "fingerprint_path": str(nav / "chemical_review/morgan2048_chiral.npy")},
                    "protocol": protocol("continuous_24h", condition["cisplatin_washout_before_challenge"]),
                    "state_covariates": {key: condition[key] for key in (
                        "induction_compound", "induction_nominal_uM", "induction_days",
                        "nominal_seeded_cells_per_well", "hours_after_replating_before_challenge")},
                    "endpoint_definition": {"assay": "image_Annexin_V", "label": "Annexin_V_positive",
                        "denominator": "total_image_counted_cells", "sampling_gate": "not_applicable_image_count",
                        "harvest": "in_well_imaging"},
                    "shared_control_id": next(r["condition_group_id"] for r in n["conditions"]
                                              if r["is_vehicle"] and r["state_id"] == condition["state_id"]),
                    "split_role": "retrospective_challenge_unassigned_for_training", "labels_seen_in_design": True})
                mappings.append({"candidate_id": cid, "observation_id": oid})
        nav_status = "input_ready_both_reported_dose_scenarios_same_Y"
    requests = {}
    for candidate in candidates:
        validate_protocol(candidate)
        request = request_for(candidate)
        candidate["p_request_id"] = request["request_id"]
        requests[request["request_id"]] = request
    require(len({r["candidate_id"] for r in candidates}) == len(candidates), "Duplicate complete candidate ID")
    return {"candidate_records": candidates, "observations": observations,
            "observation_mapping": mappings, "reference_observations": references,
            "p_descriptor_requests": list(requests.values()),
            "contract": {"input_ready": True, "response_features_missing": True,
                "formal_training_ready": False, "nav_status": nav_status,
                "candidate_count": len(candidates), "unique_active_observations": len(observations),
                "p_request_count": len(requests), "model_execution_performed": False,
                "purpose": "retrospective_input_consumption_not_model_or_policy_validation"}}, bindings


def build(output, viel=VIEL, chemical=CHEMICAL, nav=NAV):
    output = Path(output).resolve()
    require(not output.exists(), "Use a new bundle directory; refusing overwrite")
    prepared, bindings = prepare(viel, chemical, nav)
    output.mkdir(parents=True)
    for name, value in prepared.items():
        (output / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False,
            allow_nan=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, **prepared["contract"],
        "inputs_sha256": bindings,
        "source_paths": {"viel": str(Path(viel).resolve()), "chemical": str(Path(chemical).resolve()),
                         "nav": None if nav is None else str(Path(nav).resolve())},
        "files_sha256": {name + ".json": sha(output / (name + ".json")) for name in prepared}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
        allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return EndpointProtocolInputs.load(output)


class EndpointProtocolInputs:
    """Independent input/label API. Unknowns remain None or named categories."""

    def __init__(self, prepared, manifest):
        self._data, self.manifest = prepared, manifest

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        manifest = read(directory / "manifest.json")
        require(manifest["schema_version"] == 1 and manifest["formal_training_ready"] is False,
                "Bundle schema/readiness mismatch")
        check_bindings(manifest["inputs_sha256"])
        expected, bindings = prepare(**manifest["source_paths"])
        require(bindings == data_file_bindings(manifest["inputs_sha256"]), "Incomplete recursive source bindings")
        require(set(manifest["files_sha256"]) == {k + ".json" for k in expected}, "Missing output bindings")
        for key, value in expected.items():
            path = directory / (key + ".json")
            require(sha(path) == manifest["files_sha256"][path.name] and read(path) == value,
                    "Output differs from reviewed source: " + key)
        require(all(manifest[k] == v for k, v in expected["contract"].items()), "Readiness/count mismatch")
        return cls(expected, manifest)

    @property
    def candidate_ids(self):
        return tuple(r["candidate_id"] for r in self._data["candidate_records"])

    def candidate(self, candidate_id):
        return next(json.loads(json.dumps(r)) for r in self._data["candidate_records"]
                    if r["candidate_id"] == candidate_id)

    def observations(self):
        """One label record per original active observation, never per scenario."""
        return json.loads(json.dumps(self._data["observations"]))

    def observation_mapping(self):
        return json.loads(json.dumps(self._data["observation_mapping"]))

    def model_input(self, candidate_id, descriptor_file=None, descriptor_sha256=None):
        row = self.candidate(candidate_id)
        p = row["protocol"]
        chemical = row["chemical"]
        path = chemical["fingerprint_path"]
        require(sha(path) == self.manifest["inputs_sha256"][str(Path(path).resolve())],
                "Chemical array changed after bundle load")
        fingerprint = np.load(path, allow_pickle=False)[chemical["fingerprint_row"]].copy()
        result = {"candidate_id": candidate_id, "candidate_record": row,
                  "feature_names": FEATURE_NAMES,
                  "protocol_values": (row["dose_uM"], p["exposure_duration_h"], p["washout_at_h"],
                                      p["drug_free_recovery_h"], p["terminal_time_h"]),
                  "protocol_known_mask": (True, True, p["washout_at_h"] is not None, True, True),
                  "protocol_categories": {k: p[k] for k in (
                      "washout_status", "prior_drug_retention", "floating_at_washout_recovered")},
                  "chemical_fingerprint": fingerprint, "p_descriptor": None,
                  "response_features_status": "pending"}
        if descriptor_file is not None:
            require(descriptor_sha256 is not None and sha(descriptor_file) == descriptor_sha256,
                    "Actual descriptor file requires matching explicit SHA256")
            records = read(descriptor_file)
            require(isinstance(records, list), "Descriptor records must be a list")
            allowed = {"request_id", "compound_id", "dose_uM", "reference_protocol_arm",
                "reference_terminal_time_h", "descriptor_role", "reference_source_id", "reference_state_id",
                "reference_measurement_time_h", "reference_cell_line", "model_sha256", "preprocessing_sha256",
                "gene_axis_sha256", "observed_or_predicted", "feature_names", "values"}
            require(all(set(r) == allowed for r in records), "Descriptor schema or label leakage")
            require(len({r["request_id"] for r in records}) == len(records), "Duplicate P descriptor request")
            matches = [r for r in records if r["request_id"] == row["p_request_id"]]
            require(len(matches) == 1, "Real P descriptor absent for this request")
            record = matches[0]
            request = request_for(row)
            for key in ("compound_id", "dose_uM", "reference_protocol_arm", "reference_terminal_time_h",
                        "descriptor_role", "reference_cell_line"):
                require(record[key] == request[key], "P descriptor request identity/protocol mismatch")
            require(record["reference_measurement_time_h"] == 24 and record["observed_or_predicted"] == "predicted",
                    "Only declared standard continuous24h predicted reference descriptors supported")
            for key in ("reference_source_id", "reference_state_id", "model_sha256", "preprocessing_sha256",
                        "gene_axis_sha256"):
                require(isinstance(record[key], str) and record[key].strip(), "Descriptor provenance missing")
            for key in ("model_sha256", "preprocessing_sha256", "gene_axis_sha256"):
                require(len(record[key]) == 64 and all(c in "0123456789abcdef" for c in record[key]),
                        "Invalid descriptor provenance SHA256")
            values = np.asarray(record["values"], dtype=float)
            names = record["feature_names"]
            require(values.ndim == 1 and len(values) > 0 and len(names) == len(values)
                    and len(set(names)) == len(names) and all(isinstance(x, str) and x for x in names)
                    and np.isfinite(values).all(), "Invalid actual descriptor vector/names")
            result = {**result, "p_descriptor": {**record, "values": values},
                      "response_features_status": "supplied_standard_reference_transfer_unvalidated"}
        return result
