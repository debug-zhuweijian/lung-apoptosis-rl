"""Versioned label-free bridge to the audited standard-reference cache loader."""
from pathlib import Path
from ..data.endpoint_protocol_inputs import BASE
from .endpoint_protocol_features import EndpointProtocolFeatureInputs
from .config import StageUnavailable


class EndpointProtocolFeatureInputsV2(EndpointProtocolFeatureInputs):
    @classmethod
    def load(cls, directory=BASE / "endpoint-protocol-inputs-v1"):
        obj = super().load(directory)
        obj.directory = Path(directory).resolve()
        return obj

    def model_input(self, candidate_id, *, cache_manifest=None):
        packet = super().model_input(candidate_id)
        if cache_manifest is not None:
            from .endpoint_protocol_producer import load_standard_cache
            descriptors = load_standard_cache(cache_manifest, self)
            descriptor = descriptors.get(packet['candidate_record']['p_request_id'])
            if descriptor is None:
                raise StageUnavailable('PROTOCOL_P_REQUEST_NOT_IN_CACHE')
            packet = {**packet, 'p_descriptor': descriptor,
                      'response_features_status': 'ready_standard_reference_cache'}
        return packet

    def require_response_features(self, candidate_id, *, cache_manifest=None):
        packet = self.model_input(candidate_id, cache_manifest=cache_manifest)
        if packet['p_descriptor'] is None:
            raise StageUnavailable('PROTOCOL_P_CACHE_REQUIRED: pending_standard_reference_cache')
        return packet
