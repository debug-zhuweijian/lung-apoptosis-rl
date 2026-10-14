"""Label-free reader for the reviewed protocol bundle; P production is pending.

The legacy descriptor API is deliberately not called. There is currently no
audited producer of request-keyed standard-reference descriptors: even a
self-consistent cache receipt cannot make arbitrary numbers production features.
"""

import json
from pathlib import Path

import numpy as np

from ..data.endpoint_protocol_inputs import BASE, FEATURE_NAMES, request_for, validate_protocol
from .config import StageUnavailable, file_hash


REVIEWED_BUNDLE_SHA256 = "2284d771170ee67fc81793e35e1465e805abef5ca08ea221f35818533f307075"
INPUT_FILES = ("candidate_records.json", "p_descriptor_requests.json")


def _checked(path, digest):
    path = Path(path).resolve()
    if not path.is_file() or file_hash(path) != digest:
        raise StageUnavailable("PROTOCOL_FEATURE_ARTIFACT_MISSING_OR_CHANGED: " + str(path))
    return path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class EndpointProtocolFeatureInputs:
    """Read only frozen candidate/request JSON and fingerprint arrays.

    No observations, source images, error bars, legacy prepare/load or label
    deserialization are used. The hard-coded reviewed manifest hash is a trust
    anchor for this one bundle, not a general caller-provided certification.
    """

    @classmethod
    def load(cls, directory=BASE / "endpoint-protocol-inputs-v1"):
        directory = Path(directory).resolve()
        manifest = _read(_checked(directory / "manifest.json", REVIEWED_BUNDLE_SHA256))
        rows, requests = [_read(_checked(directory / name, manifest["files_sha256"][name]))
                          for name in INPUT_FILES]
        if (len(rows) != 28 or len(requests) != 14
                or len({r["candidate_id"] for r in rows}) != 28
                or len({r["request_id"] for r in requests}) != 14):
            raise StageUnavailable("PROTOCOL_INPUT_IDENTITY_MISMATCH")
        for row in rows:
            validate_protocol(row)
            expected = request_for(row)
            if expected not in requests or row["p_request_id"] != expected["request_id"]:
                raise StageUnavailable("PROTOCOL_REQUEST_IDENTITY_MISMATCH")
            chemical = row["chemical"]
            _checked(chemical["fingerprint_path"],
                     manifest["inputs_sha256"][str(Path(chemical["fingerprint_path"]).resolve())])
        obj = cls()
        obj._rows, obj._manifest = rows, manifest
        return obj

    @property
    def candidate_ids(self):
        return tuple(row["candidate_id"] for row in self._rows)

    def model_input(self, candidate_id, *, cache_manifest=None):
        matches = [r for r in self._rows if r["candidate_id"] == candidate_id]
        if len(matches) != 1:
            raise StageUnavailable("PROTOCOL_CANDIDATE_UNAVAILABLE")
        row = json.loads(json.dumps(matches[0]))
        validate_protocol(row)
        chemical, protocol = row["chemical"], row["protocol"]
        path = _checked(chemical["fingerprint_path"], self._manifest["inputs_sha256"][
            str(Path(chemical["fingerprint_path"]).resolve())])
        fingerprint = np.load(path, allow_pickle=False)[chemical["fingerprint_row"]].copy()
        if fingerprint.shape != (2048,) or not np.isfinite(fingerprint).all():
            raise StageUnavailable("PROTOCOL_FINGERPRINT_INVALID")
        if cache_manifest is not None:
            self._reject_unreviewed_cache(cache_manifest, row)
        return {"candidate_id": candidate_id, "candidate_record": row,
                "feature_names": FEATURE_NAMES,
                "protocol_values": tuple(row["dose_uM"] if k == "dose_uM" else protocol[k]
                                         for k in FEATURE_NAMES),
                "protocol_known_mask": (True, True, protocol["washout_at_h"] is not None, True, True),
                "protocol_categories": {k: protocol[k] for k in (
                    "washout_status", "prior_drug_retention", "floating_at_washout_recovered")},
                "chemical_fingerprint": fingerprint, "p_descriptor": None,
                "response_features_status": "pending_unsupported_producer"}

    def require_response_features(self, candidate_id, *, cache_manifest=None):
        self.model_input(candidate_id, cache_manifest=cache_manifest)
        raise StageUnavailable("PROTOCOL_P_CACHE_REQUIRED: pending_unsupported_producer")

    def _reject_unreviewed_cache(self, path, row):
        """Check identity/artifact claims, but never accept an unaudited producer.

        This is a rejection-only receipt envelope, not a supported cache format.
        Feature values are never deserialized, including legacy list descriptors.
        """
        path = Path(path).resolve()
        if not path.is_file():
            raise StageUnavailable("PROTOCOL_FEATURE_ARTIFACT_MISSING_OR_CHANGED: " + str(path))
        receipt = _read(path)
        if not isinstance(receipt, dict):
            raise StageUnavailable("PROTOCOL_LEGACY_DESCRIPTOR_NOT_PRODUCTION_CACHE")
        if (receipt.get("candidate_record") != row
                or receipt.get("request") != request_for(row)
                or receipt.get("input_files_sha256") != {
                    k: self._manifest["files_sha256"][k] for k in INPUT_FILES}):
            raise StageUnavailable("PROTOCOL_CACHE_REQUEST_OR_PROTOCOL_MISMATCH")
        artifacts = receipt.get("artifacts")
        required = {"producer_source", "checkpoint", "gene_axis", "preprocessing",
                    "reference", "descriptor_definition", "feature_table"}
        if not isinstance(artifacts, dict) or set(artifacts) != required:
            raise StageUnavailable("PROTOCOL_CACHE_REAL_ARTIFACTS_REQUIRED")
        for item in artifacts.values():
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise StageUnavailable("PROTOCOL_CACHE_REAL_ARTIFACTS_REQUIRED")
            _checked(path.parent / item["path"], item["sha256"])
        # Existing external-condition-world-score-v1 emits population/bio/
        # context/history, not this standardP request descriptor. No producer
        # version, checkpoint counters/data identity or fixed vector definition
        # has been audited for this bridge. Do not infer them from a receipt.
        raise StageUnavailable("PROTOCOL_P_CACHE_UNSUPPORTED_PRODUCER: producer audit and fixed descriptor definition pending")
