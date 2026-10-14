"""Separate nominal Dex graph enrichment; frozen v2 source consumer stays intact.

The new graph is not certification of the experimental stock. No Y decoder,
weights, forward pass or producer qualification is provided by this module.
"""
import json
from pathlib import Path

import numpy as np

from . import full_protocol_inputs as base

ROOT = base.ROOT
ENRICHMENT = ROOT / "experiments/20261011-data-alignment-round19/bothe_dexamethasone_source"
ENRICHMENT_HASH = "dc22f8d7bd53c4b2b62d9edffeba9a60f2acc9a03b334075a0678f1d1d769f59"
BASE_CODE_HASH = "6d067664e80cd9f7035387fc904c0c6271d6792b4d934b58aec2fe4854de6a37"
MODEL_CODE_HASH = "c6bdce150ed8a0cc53e3e26d3f438597805f918c6a4a4e1fa4f6304e1aa64e27"
OUTPUT = ROOT / "experiments/20261011-data-alignment-round19/full-protocol-inputs-v3"
SCHEMA = "source_bound_full_physical_protocol_inputs_v3"
DEX_KEY = "UREBDLICKHMUKA-CXSFZGCWSA-N"


def prepare():
    base.check({str(Path(base.__file__).resolve()): BASE_CODE_HASH,
                str(ROOT / "src/lung/models/full_protocol.py"): MODEL_CODE_HASH,
                str(ENRICHMENT / "manifest.json"): ENRICHMENT_HASH})
    manifest = base.read(ENRICHMENT / "manifest.json")
    base.require(manifest["schema"] == "bothe-dexamethasone-source-enrichment-v1"
                 and manifest["source_graph_scope"] == "nominal Dexamethasone only",
                 "Enrichment source identity changed")
    enrichment_bindings = {str(ENRICHMENT / "manifest.json"): ENRICHMENT_HASH,
                           str(Path(base.__file__).resolve()): BASE_CODE_HASH,
                           str(ROOT / "src/lung/models/full_protocol.py"): MODEL_CODE_HASH}
    for name, digest in manifest["frozen_inputs_sha256"].items():
        path = (ROOT / name).resolve()
        base.require(path.is_relative_to(ROOT), "Frozen source escapes root")
        enrichment_bindings[str(path)] = digest
    for name, entry in manifest["outputs"].items():
        path = (ENRICHMENT / name).resolve()
        base.require(path.is_relative_to(ENRICHMENT), "Enrichment output escapes source")
        base.require(path.stat().st_size == entry["bytes"], "Enrichment source size changed")
        enrichment_bindings[str(path)] = entry["sha256"]
    base.check(enrichment_bindings)
    records, frozen_bindings = base.prepare()
    # Fresh source records, not a mutable v2 reader cache, are the base authority.
    bindings = {**frozen_bindings, **enrichment_bindings}
    base.require(all(bindings[k] == v for k, v in frozen_bindings.items()), "Conflicting frozen source digest")
    identity = base.read(ENRICHMENT / "chemical_identity.json")
    official = base.read(ENRICHMENT / "dexamethasone_properties.json")["PropertyTable"]["Properties"]
    base.require(len(official) == 1 and official[0]["CID"] == 5743
                 and official[0]["InChIKey"] == DEX_KEY and identity["nominal_graph"] == official[0],
                 "Official nominal graph identity mismatch")
    base.require(identity["chemical_id"] == "pubchem:5743"
                 and identity["nominal_identity_resolved"] is True
                 and identity["source_material"]["experimental_material_graph_certified"] is False
                 and identity["source_material"]["stock_form"] is None
                 and identity["protocol_supported"]["vehicle_percent_basis"] is None,
                 "Nominal enrichment cannot certify stock or percent basis")
    spec = identity["fingerprint"]
    base.require(spec == {"path": str(ENRICHMENT / "morgan2048_chiral.npy"), "row": 0,
                          "identity_status": "nominal_pubchem_reference_historical_batch_not_verified"},
                 "Nominal source fingerprint spec changed")
    metadata = base.read(ENRICHMENT / "fingerprint_metadata.json")
    array = np.load(spec["path"], allow_pickle=False)
    base.require(array.shape == (1, 2048) and array.dtype == np.uint8 and np.isin(array, [0, 1]).all()
                 and metadata["radius"] == 2 and metadata["bits"] == 2048
                 and metadata["includeChirality"] is True and metadata["nominal_only"] is True
                 and np.flatnonzero(array[0]).tolist() == metadata["on_bits"],
                 "Nominal fingerprint convention/values changed")
    event_bindings = base.read(ENRICHMENT / "candidate_event_bindings.json")
    source_records = Path(event_bindings["source_records"]).resolve()
    base.require(str(source_records) in bindings
                 and bindings[str(source_records)] == event_bindings["source_records_sha256"],
                 "Event binding ancestry mismatch")
    actual_by_id = {row["candidate_id"]: row for row in records}
    expected_events = set()
    for row in records:
        if row["source_id"] != "Bothe2021:E-MTAB-9923":
            continue
        first, wash, second = row["source_condition_id"].split("+")
        base.require(wash == "washout", "Bothe source history changed")
        for index, compound in ((0, first), (4, second)):
            if compound == "dexamethasone":
                expected_events.add((row["candidate_id"], index))
    supplied = event_bindings["bindings"]
    keys = [(entry["candidate_id"], entry["event_index"]) for entry in supplied]
    base.require(len(keys) == len(set(keys)) == 4 and set(keys) == expected_events,
                 "Enrichment must bind exactly the four actual source Dex events")
    for entry in supplied:
        row = actual_by_id[entry["candidate_id"]]
        index = entry["event_index"]
        base.require(type(index) is int, "Source event index must be integer")
        event = row["events"][index]
        base.require(entry["chemical_id"] == identity["chemical_id"] and entry["fingerprint"] == spec
                     and entry["event_type"] == event["event_type"] == "drug_interval"
                     and event["vehicle"] is False and event["fingerprint"] is None
                     and event["dose_unit"] == "nM" and event["values"]["dose"] == 100
                     and entry["stock_form"] is None and entry["stock_form_known"] is False
                     and entry["identity_scope"] == "nominal_name_graph_only",
                     "Source event/compound/nominal scope mismatch")
    # These are fresh source-derived copies. No old v2 asset is patched.
    enriched = []
    for row in records:
        if row["source_id"] == "Bothe2021:E-MTAB-9923":
            events = [{**event, "fingerprint": dict(spec)}
                      if (row["candidate_id"], i) in expected_events else event
                      for i, event in enumerate(row["events"])]
            row = {**row, "events": events,
                   "nominal_chemical_scope": "pubchem_5743_graph_only_stock_material_unverified",
                   "condition_kwargs_status": "pending_no_verified_state_reference_and_sample_counts",
                   "enrichment_evidence": str(ENRICHMENT / "manifest.json")}
        enriched.append(row)
    return enriched, bindings


class FullProtocolInputsV3(base.FullProtocolInputs):
    """Reuse v2 tensor mechanics, with an independently authenticated v3 source."""
    def __init__(self):
        self._records, self.source_bindings = prepare()

    def _authenticated_records(self):
        records, bindings = prepare()
        base.require(self.source_bindings == bindings, "Cached v3 source bindings changed")
        base.require(self._records == records, "Cached v3 mapped records changed")
        return records

    @classmethod
    def load(cls, directory=OUTPUT):
        directory = Path(directory).resolve()
        manifest = base.read(directory / "manifest.json")
        base.require(manifest["schema"] == SCHEMA and manifest["consumer_sha256"] == base.sha(__file__)
                     and manifest["base_consumer_sha256"] == BASE_CODE_HASH
                     and manifest["feature_schema"] == base.FEATURE_SCHEMA
                     and manifest["formal_training_ready"] is False, "V3 consumer/schema identity changed")
        reader = cls()
        base.require(manifest["source_bindings"] == reader.source_bindings, "Incomplete v3 source bindings")
        base.require(set(manifest["files_sha256"]) == {"records.json"}, "V3 record binding schema changed")
        base.check({str(directory / name): digest for name, digest in manifest["files_sha256"].items()})
        base.require(base.read(directory / "records.json") == reader._records,
                     "V3 mapped records differ from actual sources")
        return reader


def build(output=OUTPUT):
    output = Path(output).resolve()
    base.require(not output.exists(), "New output directory required; refusing overwrite")
    reader = FullProtocolInputsV3()
    actual = reader._authenticated_records()
    batch = reader.tensor_batch(reader.candidate_ids)
    output.mkdir(parents=True)
    with (output / "records.json").open("x", encoding="utf-8") as stream:
        json.dump(actual, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    receipt = {"schema": SCHEMA, "feature_schema": base.FEATURE_SCHEMA,
               "consumer_sha256": base.sha(__file__), "base_consumer_sha256": BASE_CODE_HASH,
               "source_bindings": reader.source_bindings,
               "files_sha256": {"records.json": base.sha(output / "records.json")},
               "candidate_count": len(actual), "source_counts": {"Vielhauer2013": 20, "Bothe2021:E-MTAB-9923": 4},
               "event_count": int(batch.lengths.sum()), "fingerprint_known_events": int(batch.fingerprint_known.sum()),
               "known_counts": batch.known.sum((0, 1)).tolist(), "scalar_fields": list(base.SCALAR_FIELDS),
               "tensor_shape": list(batch.fingerprint.shape), "cpu_tensor_validate": "passed",
               "nominal_Dex_CID": 5743, "nominal_Dex_full_inchikey": DEX_KEY,
               "chemical_scope": "nominal_graph_only_experimental_stock_material_not_certified",
               "stock_form_status": "unknown", "vehicle_percent_basis": "unknown_preserved_mask",
               "formal_training_ready": False, "producer_qualified": False,
               "label_JSON_decoded": False, "historical_label_blinding_claimed": False,
               "real_model_forward_count": 0, "PT_load_count": 0,
               "pending": ["Martin source event mapping", "Bothe sample counts and state reference",
                           "Viel matched reference/endpoint RNA", "verified base_condition kwargs",
                           "fullprotocol producer and matching weights", "independent scientific chain evaluation"]}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt
