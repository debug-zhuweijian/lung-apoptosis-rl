"""Read real Martin E/D input tensors without ML; explicit real forward is future work.

Default readback does not construct a model, load PT, optimize, or read final RNA.
real-untrained-forward is a separately reviewable engineering command, not science.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'src'):
    sys.path.insert(0, str(directory))

from lung.data.martin_native_rna import BASE, DOMAINS, read_json, sha, write_json
from lung.data.martin_source_inputs import MartinSourceInputs


def summary(batch):
    mask, raw = batch.observed_mask, batch.raw_values
    observed = raw[mask]
    inverse = (batch.model_values * batch.scale + batch.offset if batch.domain == 'log2TPM'
               else batch.model_values.expm1())
    error = (inverse[mask] - observed).abs()
    return {'request_id': batch.request_id, 'part': batch.part, 'domain': batch.domain, 'units': batch.units,
            'system': batch.system, 'fold_id': batch.fold_id, 'role': batch.role, 'source_id': batch.source_id,
            'background': batch.background, 'protocol': batch.protocol, 'sample_ids': list(batch.sample_ids),
            'shape': list(raw.shape), 'observed_genes_per_sample': mask.sum(-1).tolist(),
            'missing_genes_per_sample': (~mask).sum(-1).tolist(),
            'negative_observed': int((observed < 0).sum()),
            'noninteger_estimated_counts': int((observed != observed.round()).sum()) if batch.domain == 'estimated_counts' else None,
            'max_inverse_absolute_error': float(error.max()),
            'max_inverse_relative_error': float((error / observed.abs().clamp_min(1.)).max()),
            'measurement_clock_h': batch.measurement_clock.tolist(),
            'candidate_observation_clock_h': batch.physical.endpoint_hours.tolist(),
            'physical_event_counts': batch.physical.lengths.tolist(),
            'metadata_known_mask': batch.metadata_known[0].tolist(),
            'missing_remains_NaN': bool(torch.isnan(raw[~mask]).all()),
            'model_constructed': False, 'model_forward': False, 'PT_load': False,
            'functional_Y_read': False, 'scientific_qualified': False}


def readback(output):
    contracts, rows, groups, samples = {}, [], {}, set()
    source = ROOT / BASE / 'native-rna-inputs-v1'
    original = {d: np.load(source / ('joint_' + d + '.npy'), mmap_mode='r', allow_pickle=False) for d in DOMAINS}
    try:
        with MartinSourceInputs(ROOT) as reader:
            requests = read_json(source / 'cycle_requests.json')
            selected = {}
            for request in requests:
                if request['role'] == 'fit':
                    selected.setdefault((request['background_id'], request['protocol_id']), request)
            assert len(selected) == 16
            for request in selected.values():
                rid, fold = request['request_id'], request['fold_id']
                contract = reader.contract(system='cycle', fold_id=fold)
                contracts[fold] = contract
                group = reader.protocol.condition(rid, system='cycle', fold_id=fold, role='fit')['group']
                groups[group['group_id']] = {'source': group['source_id'], 'background': group['source_background'],
                                             'events': len(group['events']), 'candidate_endpoint_h': group['endpoint_hours']}
                for domain in DOMAINS:
                    parts = reader.read_request(rid, system='cycle', fold_id=fold, role='fit',
                                                 domain=domain, supervision=True)
                    for name, batch in parts.items():
                        columns = request['reference_column_indices' if name == 'reference' else 'target_column_indices']
                        assert np.array_equal(batch.raw_values.numpy(), original[domain][columns], equal_nan=True)
                        assert set(batch.sample_ids) <= set(contract['fit_sample_ids'])
                        samples.update(batch.sample_ids)
                        rows.append(summary(batch))
        result = {'schema': 'martin-source-E-D-real-input-readback-v1', 'groups': groups, 'rows': rows,
                  'unique_groups': len(groups), 'unique_reported_events': sum(g['events'] for g in groups.values()),
                  'unique_RNA_GSM': len(samples), 'tensor_views': len(rows),
                  'real_data_preparation_passed': True, 'models_constructed': 0, 'real_model_forwards': 0,
                  'PT_loads': 0, 'optimizer_steps': 0, 'functional_Y_reads': 0, 'P_outputs': 0, 'G_outputs': 0, 'Q_outputs': 0,
                  'scientific_qualified': False,
                  'code_sha256': {name: sha(ROOT / name) for name in
                                    ('src/lung/data/martin_source_inputs.py', 'src/lung/models/martin_source.py',
                                     'scripts/verify_martin_source_model.py')}}
        write_json(output / 'real_input_readback.json', result)
        write_json(output / 'input_contracts.json', contracts)
        return {k: v for k, v in result.items() if k not in ('rows', 'groups')}
    finally:
        for values in original.values():
            values._mmap.close()


def real_untrained_forward(args, output):
    # Intentionally not executed during Round22: requires the outstanding human
    # model-run authorization. There is no automatic fallback from readback.
    from lung.models.martin_source import MartinSourceModel
    if not all((args.request_id, args.system, args.fold_id)):
        raise ValueError('EXPLICIT_REGISTERED_REQUEST_SYSTEM_FOLD_REQUIRED')
    torch.manual_seed(args.seed)
    with MartinSourceInputs(ROOT) as reader:
        parts = reader.read_request(args.request_id, system=args.system, fold_id=args.fold_id,
                                     role='fit', domain=args.domain, supervision=True)
        batch = parts['target']
        model = MartinSourceModel(batch.gene_ids, batch.contract, width=args.width).eval()
        with torch.no_grad():
            generated = model(batch)
            loss = model.observation_loss(generated, batch)
        result = {'schema': 'martin-source-E-D-real-untrained-forward-v1', 'model_identity': model.identity,
                  'seed': args.seed, 'dtype': str(next(model.parameters()).dtype),
                  'model_parameter_count': sum(p.numel() for p in model.parameters()),
                  'request': {**summary(batch), 'model_constructed': True, 'model_forward': True},
                  'real_model_forwards': 1, 'PT_loads': 0, 'optimizer_steps': 0,
                  'observed_reconstruction_loss_for_finiteness_only': float(loss),
                  'untrained_random_parameters': True, 'scientific_qualified': False, 'P_outputs': 0, 'G_outputs': 0, 'Q_outputs': 0}
        write_json(output / 'real_untrained_forward.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('readback', 'real-untrained-forward'), default='readback')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-id')
    parser.add_argument('--system', choices=('lobo', 'cycle'))
    parser.add_argument('--fold-id')
    parser.add_argument('--domain', choices=DOMAINS, default='log2TPM')
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = (args.output if args.output.is_absolute() else ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'experiments') or output.exists():
        raise ValueError('NEW_PROJECT_EXPERIMENT_OUTPUT_REQUIRED')
    output.mkdir(parents=True)
    result = readback(output) if args.mode == 'readback' else real_untrained_forward(args, output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
