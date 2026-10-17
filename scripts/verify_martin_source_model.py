"""Read real Martin E/D input tensors without models or final RNA."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'src'):
    sys.path.insert(0, str(directory))

from lung.data.martin_native_rna import BASE, DOMAINS, read_json, write_json
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
                  'scientific_qualified': False}
        write_json(output / 'real_input_readback.json', result)
        write_json(output / 'input_contracts.json', contracts)
        return {k: v for k, v in result.items() if k not in ('rows', 'groups')}
    finally:
        for values in original.values():
            values._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = (args.output if args.output.is_absolute() else ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'experiments') or output.exists():
        raise ValueError('NEW_PROJECT_EXPERIMENT_OUTPUT_REQUIRED')
    output.mkdir(parents=True)
    result = readback(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
