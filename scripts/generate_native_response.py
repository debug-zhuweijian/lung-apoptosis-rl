"""Metadata preflight or generation from declared matching AE/response weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from lung.data.native_response_units import NativeResponseUnits
from lung.engine.native_response_producer import NativeMetadata, preflight, read, write_new, produce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'generate'))
    parser.add_argument('--units', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-metadata')
    parser.add_argument('--request-id')
    parser.add_argument('--role', default='fit')
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ablation', choices=('full_reference', 'same_reference_mean'), default='full_reference')
    args = parser.parse_args()
    units = NativeResponseUnits(ROOT, ROOT / args.units,
        allowed_roles=(args.role,), allow_test=args.allow_test)
    coordinates = NativeMetadata.from_joint(ROOT, units)
    metadata = read(ROOT / args.model_metadata) if args.model_metadata else None
    if args.mode == 'preflight':
        # Metadata-only full inventory, not expression reads or tensor weights.
        requests = read(units.asset / 'requests.json')
        result = preflight(requests, coordinates.bindings, metadata)
        write_new(ROOT / args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'bindings'}, ensure_ascii=False))
        return
    if not args.request_id or metadata is None:
        parser.error('generate requires --request-id and --model-metadata')
    request = units.request(args.request_id, role=args.role)
    # Unsupported bulk/pulse requests are blocked before opening checkpoints.
    from lung.engine.native_response_producer import protocol_status, require
    require(protocol_status(request) == 'native_single_exposure_endpoint_only', 'NATIVE_PROTOCOL_OR_DECODER_PENDING')
    descriptor = produce(units, coordinates, request, metadata, ROOT / args.output,
        role=args.role, seed=args.seed, steps=args.steps, ablation=args.ablation, device=args.device)
    print(json.dumps({'cache_identity': descriptor.cache_identity, 'execution_kind': descriptor.execution_kind}))


if __name__ == '__main__':
    main()
