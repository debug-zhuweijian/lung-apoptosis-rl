"""Source-bound physical events. No labels, model kwargs or model execution.

Only pinned input metadata is decoded. Some recursively hashed upstream JSON
also contains Y: hashing bytes is integrity checking, not historical blinding.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..models.full_protocol import (DOSE_UNITS, EVENT_TYPES, FEATURE_SCHEMA,
                                    SCALAR_FIELDS, FullPhysicalProtocolBatch)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1"
SOURCE_HASH = "2284d771170ee67fc81793e35e1465e805abef5ca08ea221f35818533f307075"
BOTHE = ROOT / "experiments/20261011-data-alignment-round17/bothe_protocol_source"
BOTHE_HASHES = {
    "PMC8403771.xml": "836ec5b9703faeb6cdf48489b81bd8e422457ea29149154ecd545039845f473e",
    "E-MTAB-9923.sdrf.txt": "cf7d41065c4f7ab2626014ffccfd57b08850360d1d262b605e03fa52e7175df6",
}
OUTPUT = ROOT / "experiments/20261011-data-alignment-round18/full-protocol-inputs-v2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(bindings):
    for path, digest in bindings.items():
        require(sha(path) == digest, "Source hash mismatch: " + str(path))


def event(kind, *, vehicle=False, fingerprint=None, unit="unknown", **values):
    require(set(values) <= set(SCALAR_FIELDS), "Unknown physical field")
    return {"event_type": kind, "vehicle": vehicle, "dose_unit": unit,
            "fingerprint": fingerprint,
            "values": {key: values.get(key) for key in SCALAR_FIELDS}}


def prepare():
    # Authenticate manifest and every declared source/output before decoding inputs.
    check({str(SOURCE / "manifest.json"): SOURCE_HASH})
    manifest = read(SOURCE / "manifest.json")
    bindings = {**manifest["inputs_sha256"],
                **{str(SOURCE / name): digest for name, digest in manifest["files_sha256"].items()},
                str(SOURCE / "manifest.json"): SOURCE_HASH,
                str(ROOT / "src/lung/data/endpoint_protocol_inputs.py"): manifest["consumer_sha256"],
                **{str(BOTHE / name): digest for name, digest in BOTHE_HASHES.items()}}
    check(bindings)
    candidates = read(SOURCE / "candidate_records.json")
    selected = [row for row in candidates if row["source_id"] == "Vielhauer2013"]
    require(len(selected) == 20 and len({r["candidate_id"] for r in selected}) == 20,
            "Expected 20 unique Viel source candidates")
    condition_path = next(Path(p) for p in bindings if Path(p).name == "bocnho_conditions.csv")
    with condition_path.open(encoding="utf-8-sig", newline="") as stream:
        conditions = {r["condition_id"]: r for r in csv.DictReader(stream)}
    records = []
    for row in selected:
        source = conditions[row["condition_id"]]
        require(source["is_vehicle"] == "false", "Only active Viel candidates mapped")
        for key in ("protocol_id", "protocol_arm", "cell_line", "state_id"):
            require(row[key] == source[key], "Source metadata join mismatch: " + key)
        require(row["source_drug_id"] == source["drug_id"], "Source chemical mismatch")
        p, dose = row["protocol"], row["source_dose"]
        require(dose == {"value": float(source["dose_nM"]), "unit": "nM"}, "Source dose mismatch")
        require(row["dose_uM"] == dose["value"] / 1000, "Explicit source unit mismatch")
        for key in ("exposure_duration_h", "drug_free_recovery_h", "terminal_time_h"):
            require(p[key] == float(source[key]), "Source physical time mismatch: " + key)
        exposure, recovery = p["exposure_duration_h"], p["drug_free_recovery_h"]
        require(p["exposure_intervals"] == [[0, exposure]] and p["time_origin"] == "challenge_drug_start",
                "Source interval/time origin changed")
        chem = row["chemical"]
        fingerprint = {"path": chem["fingerprint_path"], "row": chem["fingerprint_row"],
                       "identity_status": chem["identity_status"]}
        require(str(Path(fingerprint["path"]).resolve()) in bindings, "Unbound fingerprint")
        events = [event("drug_interval", fingerprint=fingerprint, unit="nM", dose=dose["value"],
                        start_hours=0, end_hours=exposure, exposure_hours=exposure)]
        if source["protocol_arm"] == "pulse_6h_washout_recovery18h":
            require(exposure == 6 and recovery == 18 and p["washout_at_h"] == float(source["washout_at_h"]),
                    "Pulse timeline changed")
            require(p["media_events"] == [{"at_h": 6, "event": "media_decant_rinse_fresh_media"}],
                    "Wash/media provenance changed")
            # Source says rinsed, not a count; do not manufacture wash_count=1.
            events += [event("wash", start_hours=6, end_hours=6),
                       event("media_change", start_hours=6, end_hours=6, media_change=1),
                       event("recovery", start_hours=6, end_hours=24, recovery_hours=18)]
        else:
            require(source["protocol_arm"] == "continuous_24h" and exposure == 24 and recovery == 0
                    and p["washout_at_h"] is None and p["media_events"] == [], "Continuous timeline changed")
        records.append({"candidate_id": row["candidate_id"], "source_id": row["source_id"],
                        "source_condition_id": row["condition_id"], "state_id": row["state_id"],
                        "role": row["split_role"], "cell_line": row["cell_line"],
                        "panel_id": source["source_figure"], "endpoint_hours": p["terminal_time_h"],
                        "time_origin": p["time_origin"], "events": events,
                        "stock_form": None, "reference_RNA_status": "pending_not_available",
                        "condition_kwargs_status": "pending_no_verified_state_reference",
                        "evidence": {"file": str(condition_path), "figure": source["source_figure"],
                                     "text": "Vielhauer2013_PMC_fulltext.md Fig3C/Fig5C and Methods Washout experiments"}})
    records.extend(_bothe_records())
    return records, bindings


def _bothe_records():
    # Exact SDRF stimulus vocabulary plus Fig3A and Methods re-induction clocks.
    with (BOTHE / "E-MTAB-9923.sdrf.txt").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    histories = {}
    for row in rows:
        require(row["Characteristics[cell line]"] == "A549", "Bothe cell line mismatch")
        stimulus = row["Factor Value[stimulus]"]
        histories.setdefault(stimulus, set()).add(row["Source Name"])
    expected = {a + "+washout+" + b for a in ("dexamethasone", "ethanol")
                for b in ("dexamethasone", "ethanol")}
    require(set(histories) == expected and all(len(samples) == 3 for samples in histories.values()),
            "Bothe four-history/three-replicate identity mismatch")
    records = []
    for stimulus, samples in sorted(histories.items()):
        first, _, second = stimulus.split("+")
        def interval(compound, start):
            vehicle = compound == "ethanol"
            return event("vehicle_interval" if vehicle else "drug_interval", vehicle=vehicle,
                         # Source says percent, without v/v; registry has no plain
                         # percent. Preserve it in metadata, do not assume v/v.
                         unit="unknown" if vehicle else "nM", dose=None if vehicle else 100,
                         start_hours=start, end_hours=start + 4, exposure_hours=4)
        records.append({"candidate_id": "Bothe2021:E-MTAB-9923:" + stimulus,
                        "source_id": "Bothe2021:E-MTAB-9923", "source_condition_id": stimulus,
                        "source_samples": sorted(samples), "state_id": "A549_source_culture",
                        "cell_line": "A549", "role": "unassigned_history_RNA_auxiliary",
                        "panel_id": "Fig3A/E", "endpoint_hours": 32, "time_origin": "first_treatment_start",
                        "events": [interval(first, 0), event("wash", start_hours=4, end_hours=4, wash_count=2),
                                   event("media_change", start_hours=4, end_hours=4, media_change=1),
                                   event("recovery", start_hours=4, end_hours=28, recovery_hours=24),
                                   interval(second, 28)], "stock_form": None,
                        "medium_scope": "5percent_FBS_no_extra_hormone_not_absolute_hormone_free",
                        "source_vehicle_dose": {"value": .1, "unit": "percent_unspecified_basis",
                                                "tensor_status": "masked_registry_has_no_plain_percent"},
                        "reference_RNA_status": "pending_raw_counts_not_recovered",
                        "condition_kwargs_status": "pending_unverified_Dex_fingerprint_and_state_reference",
                        "evidence": {"file": str(BOTHE / "PMC8403771.xml"),
                                     "figure": "Fig3A/E", "section": "Hormone re-induction treatments",
                                     "sample_table": str(BOTHE / "E-MTAB-9923.sdrf.txt")}})
    return records


class FullProtocolInputs:
    """Tensor API accepts source candidate identities only, never caller vectors."""
    def __init__(self):
        self._records, self.source_bindings = prepare()

    def _authenticated_records(self):
        # Rebuild from fixed source hashes. Never accept mutable caches as the
        # authority, including a caller deleting a source binding before use.
        records, bindings = prepare()
        require(self.source_bindings == bindings, "Cached source bindings changed")
        require(self._records == records, "Cached mapped records changed")
        return records

    @classmethod
    def load(cls, directory=OUTPUT):
        directory = Path(directory).resolve()
        manifest = read(directory / "manifest.json")
        require(manifest["consumer_sha256"] == sha(__file__), "Consumer code changed")
        require(manifest["feature_schema"] == FEATURE_SCHEMA and manifest["formal_training_ready"] is False,
                "Input schema/readiness changed")
        check(manifest["source_bindings"])
        check({str(directory / name): digest for name, digest in manifest["files_sha256"].items()})
        instance = cls()
        require(instance.source_bindings == manifest["source_bindings"], "Incomplete source bindings")
        require(read(directory / "records.json") == instance._records, "Mapped records differ from actual sources")
        return instance

    @property
    def candidate_ids(self):
        return tuple(r["candidate_id"] for r in self._authenticated_records())

    def candidate(self, candidate_id):
        matches = [r for r in self._authenticated_records() if r["candidate_id"] == candidate_id]
        require(len(matches) == 1, "Unknown source-bound candidate ID")
        return json.loads(json.dumps(matches[0]))

    def tensor_batch(self, candidate_ids):
        actual = self._authenticated_records()
        require(isinstance(candidate_ids, (list, tuple)) and len(candidate_ids) > 0
                and all(isinstance(x, str) for x in candidate_ids), "Source candidate IDs required")
        by_id = {r["candidate_id"]: r for r in actual}
        require(all(cid in by_id for cid in candidate_ids), "Unknown source-bound candidate ID")
        records = [by_id[cid] for cid in candidate_ids]
        b, s = len(records), max(len(r["events"]) for r in records)
        fp = torch.zeros((b, s, 2048), dtype=torch.float32)
        fk, vehicle = torch.zeros((b, s), dtype=torch.bool), torch.zeros((b, s), dtype=torch.bool)
        kind, unit = torch.zeros((b, s), dtype=torch.long), torch.zeros((b, s), dtype=torch.long)
        values = torch.zeros((b, s, len(SCALAR_FIELDS)), dtype=torch.float32)
        known = torch.zeros_like(values, dtype=torch.bool)
        for i, row in enumerate(records):
            for j, e in enumerate(row["events"]):
                kind[i, j], unit[i, j] = EVENT_TYPES.index(e["event_type"]), DOSE_UNITS.index(e["dose_unit"])
                vehicle[i, j] = e["vehicle"]
                for k, field in enumerate(SCALAR_FIELDS):
                    if e["values"][field] is not None:
                        values[i, j, k], known[i, j, k] = e["values"][field], True
                if e["fingerprint"] is not None:
                    spec = e["fingerprint"]
                    array = np.load(spec["path"], allow_pickle=False)
                    require(array.ndim == 2 and array.shape[1] == 2048 and np.isin(array, [0, 1]).all(),
                            "Source fingerprint shape/bits invalid")
                    require(type(spec["row"]) is int and 0 <= spec["row"] < len(array), "Source fingerprint row invalid")
                    fp[i, j], fk[i, j] = torch.from_numpy(array[spec["row"]].astype(np.float32)), True
        batch = FullPhysicalProtocolBatch(fp, fk, vehicle, kind, unit, values, known,
                                          torch.tensor([len(r["events"]) for r in records], dtype=torch.long),
                                          torch.tensor([r["endpoint_hours"] for r in records], dtype=torch.float32),
                                          torch.ones(b, dtype=torch.bool))
        batch.validate()
        return batch


def build(output=OUTPUT):
    output = Path(output).resolve()
    require(not output.exists(), "New output directory required; refusing overwrite")
    reader = FullProtocolInputs()
    batch = reader.tensor_batch(reader.candidate_ids)
    output.mkdir(parents=True)
    records_path = output / "records.json"
    records_path.write_text(json.dumps(reader._records, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    receipt = {"schema_version": 1, "feature_schema": FEATURE_SCHEMA, "candidate_count": len(reader.candidate_ids),
               "source_counts": {"Vielhauer2013": 20, "Bothe2021:E-MTAB-9923": 4},
               "scalar_fields": list(SCALAR_FIELDS), "known_counts": batch.known.sum((0, 1)).tolist(),
               "fingerprint_known_events": int(batch.fingerprint_known.sum()),
               "event_count": int(batch.lengths.sum()), "endpoint_known_count": int(batch.endpoint_known.sum()),
               "tensor_shape": list(batch.fingerprint.shape), "cpu_tensor_validate": "passed",
               "source_bindings": reader.source_bindings, "consumer_sha256": sha(__file__),
               "files_sha256": {"records.json": sha(records_path)},
               "decoded_source_files": [str(SOURCE / "manifest.json"), str(SOURCE / "candidate_records.json"),
                                        "bocnho_conditions.csv", str(BOTHE / "E-MTAB-9923.sdrf.txt")],
               "label_JSON_decoded": False, "historical_label_blinding_claimed": False,
               "stock_form_status": "unknown_not_in_tensor_schema", "formal_training_ready": False,
               "point_event_time_convention": "wash/media timestamps mark authored transition_clock_not_measured_operation_duration",
               "vehicle_dose_scope": "Bothe_0.1percent_preserved_as_metadata_not_assumed_percent_v_v",
               "real_model_forward_count": 0, "PT_load_count": 0,
               "pending": ["Martin bridge_v2 mapping not audited in this version",
                           "Bothe Dex verified fingerprint and sample counts",
                           "Viel matched reference/endpoint RNA", "verified base_condition kwargs",
                           "fullprotocol producer and matching weights", "independent scientific chain evaluation"]}
    (output / "manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt
