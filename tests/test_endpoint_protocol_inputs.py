"""Real-input contract regressions; synthetic descriptors exist only in tests."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.lung.data.endpoint_protocol_inputs import (
    BASE, CHEMICAL, EndpointProtocolInputs, build, prepare, sha, validate_protocol,
)


@pytest.fixture(scope="module")
def bundle():
    return EndpointProtocolInputs.load(BASE / "endpoint-protocol-inputs-v1")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_actual_common_endpoint_preserves_distinct_pulse_and_continuous(bundle):
    pair = [bundle.model_input(cid) for cid in bundle.candidate_ids
            if cid.startswith("vielhauer2013_a549_bocnho_50pm")]
    assert len(pair) == 2
    assert {r["protocol_values"] for r in pair} == {
        (0.00005, 24, None, 0, 24), (0.00005, 6, 6, 18, 24)}
    assert pair[0]["candidate_id"] != pair[1]["candidate_id"]
    assert pair[0]["candidate_record"]["p_request_id"] == pair[1]["candidate_record"]["p_request_id"]
    np.testing.assert_array_equal(pair[0]["chemical_fingerprint"], pair[1]["chemical_fingerprint"])
    assert all(r["p_descriptor"] is None for r in pair)
    assert bundle.manifest["candidate_count"] == 28
    assert bundle.manifest["p_request_count"] == 14


def test_no_Y_or_pixel_information_in_predictor_inputs(bundle):
    forbidden = {"value", "mean_fraction", "digitized_mean_percent", "measured_endpoint",
                 "pixel_lower_fraction", "pixel_upper_fraction", "reading_lower_percent",
                 "bar_center_x_px", "mean_edge_y_px", "published_n", "n_status"}

    def check(value, path=()):
        if isinstance(value, dict):
            # A reported concentration has a legitimate numeric ``value``.
            # Permit it only in this dose object, never in an observation.
            excluded = forbidden - {"value"} if path[-1:] == ("source_dose",) else forbidden
            assert not excluded.intersection(value)
            for key, child in value.items():
                check(child, path + (key,))
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child, path)

    for cid in bundle.candidate_ids:
        check(bundle.model_input(cid))
    assert len(bundle.observations()) == 24
    assert {r["unit"] for r in bundle.observations()} == {"fraction", "percent"}


def test_nav_scenarios_do_not_multiply_observations_or_default_unknowns(bundle):
    rows = [bundle.candidate(cid) for cid in bundle.candidate_ids if cid.startswith("Gonzalez")]
    assert len(rows) == 8
    labels = [r for r in bundle.observations() if r["source_id"] == "GonzalezGualda2020"]
    assert len(labels) == 4 and all(r["dose_scenarios_share_Y"] for r in labels)
    mapping = bundle.observation_mapping()
    for label in labels:
        members = [r for r in mapping if r["observation_id"] == label["observation_id"]]
        assert len(members) == 2
        candidates = [bundle.candidate(r["candidate_id"]) for r in members]
        assert {r["dose_uM"] for r in candidates} == {10, 11}
        assert len({r["condition_id"] for r in candidates}) == 1
    senescent = next(r for r in rows if r["state_id"] == "cisplatin_induced_senescent")
    result = bundle.model_input(senescent["candidate_id"])
    assert result["protocol_categories"]["prior_drug_retention"] == "not_explicitly_reported"
    assert result["protocol_values"][2] is None and result["protocol_known_mask"][2] is False
    assert senescent["state_covariates"]["nominal_seeded_cells_per_well"] == 4000


@pytest.mark.parametrize("field,value", [("exposure_duration_h", 24), ("terminal_time_h", 6),
                                         ("washout_at_h", None), ("time_origin", "generator_tau")])
def test_wrong_clock_semantics_are_rejected(bundle, field, value):
    row = next(bundle.candidate(cid) for cid in bundle.candidate_ids if "pulse" in cid)
    row["protocol"][field] = value
    with pytest.raises(ValueError):
        validate_protocol(row)


@pytest.mark.parametrize("field,value", [("dose_uM", 24), ("fingerprint_row", 0),
                                         ("terminal_time_h", 6), ("source_drug_id", "cbi_indole2")])
def test_chemical_join_cannot_hide_dose_identity_or_time_mismatch(tmp_path, field, value):
    source = json.loads(CHEMICAL.read_text(encoding="utf-8"))
    source["rows"][0][field] = value
    path = tmp_path / "sidecar.json"
    dump(path, source)
    with pytest.raises(ValueError, match="mismatch"):
        prepare(chemical=path, nav=None)


def test_changed_derived_content_rejected_even_with_updated_output_hash(tmp_path, bundle):
    directory = tmp_path / "bundle"
    shutil.copytree(BASE / "endpoint-protocol-inputs-v1", directory)
    rows = json.loads((directory / "candidate_records.json").read_text(encoding="utf-8"))
    rows[0]["injected_label"] = 99
    dump(directory / "candidate_records.json", rows)
    manifest = copy.deepcopy(bundle.manifest)
    manifest["files_sha256"]["candidate_records.json"] = sha(directory / "candidate_records.json")
    dump(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="differs from reviewed source"):
        EndpointProtocolInputs.load(directory)


def test_changed_source_hash_and_existing_output_refused(tmp_path):
    sidecar = tmp_path / "sidecar.json"
    shutil.copyfile(CHEMICAL, sidecar)
    directory = tmp_path / "bundle"
    build(directory, chemical=sidecar, nav=None)
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        EndpointProtocolInputs.load(directory)
    with pytest.raises(ValueError, match="refusing overwrite"):
        build(directory)


def synthetic_descriptor(bundle, cid):
    row = bundle.candidate(cid)
    return {"request_id": row["p_request_id"], "compound_id": row["chemical"]["compound_id"],
        "dose_uM": row["dose_uM"], "reference_protocol_arm": "continuous_24h",
        "reference_terminal_time_h": 24, "descriptor_role": "standard_reference",
        "reference_source_id": "synthetic_test_only", "reference_state_id": "synthetic_test_only",
        "reference_measurement_time_h": 24, "reference_cell_line": "A549",
        "model_sha256": "a" * 64, "preprocessing_sha256": "b" * 64, "gene_axis_sha256": "c" * 64,
        "observed_or_predicted": "predicted", "feature_names": ["synthetic_test_feature"], "values": [2.5]}


def test_supplied_descriptor_joins_both_protocols_but_never_erases_protocol(tmp_path, bundle):
    ids = [cid for cid in bundle.candidate_ids if cid.startswith("vielhauer2013_a549_bocnho_50pm")]
    path = tmp_path / "synthetic-test-descriptor.json"
    dump(path, [synthetic_descriptor(bundle, ids[0])])
    one, two = [bundle.model_input(cid, path, sha(path)) for cid in ids]
    np.testing.assert_array_equal(one["p_descriptor"]["values"], two["p_descriptor"]["values"])
    assert one["protocol_values"] != two["protocol_values"]
    assert one["p_descriptor"]["reference_protocol_arm"] == "continuous_24h"


@pytest.mark.parametrize("mutation", ["wrong_protocol", "wrong_dose", "label_leak", "hash_change", "empty"])
def test_descriptor_provenance_join_and_labels_are_checked(tmp_path, bundle, mutation):
    cid = bundle.candidate_ids[0]
    record = synthetic_descriptor(bundle, cid)
    path = tmp_path / "synthetic-test-descriptor.json"
    if mutation == "wrong_protocol":
        record["reference_protocol_arm"] = "pulse_6h_washout_recovery18h"
    elif mutation == "wrong_dose":
        record["dose_uM"] = 1000
    elif mutation == "label_leak":
        record["observed_Annexin"] = 50
    elif mutation == "empty":
        record["values"] = []
    dump(path, [record])
    digest = sha(path)
    if mutation == "hash_change":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        bundle.model_input(cid, path, digest)
