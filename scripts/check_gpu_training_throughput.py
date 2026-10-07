"""Fixed-input GPU execution benchmark; no optimizer steps or formal training."""

from dataclasses import replace
from pathlib import Path
import argparse
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np
from scipy import sparse
import torch

from lung.engine.config import load_config
from lung.engine.checkpoint import seed_all, atomic_json
from lung.models import ModelConfig, ResponseModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-ode', action='store_true')
    parser.add_argument('--cases', help='Comma separated query_chunk:microbatch cases')
    parser.add_argument('--cells', type=int, default=2)
    parser.add_argument('--output', default='logs/environment_checks/gpu_training_throughput.json')
    args = parser.parse_args()
    seed_all(20260924)
    torch.set_num_threads(4)
    config = ModelConfig(**load_config()['response']['model'])
    assets = ROOT / '08-datasets/LungCancer_A549_Integration/training-v1'
    index = json.loads((assets / 'fit_cache/index.json').read_text(encoding='utf-8'))
    if not 1 <= args.cells <= 8:
        raise ValueError('Bounded execution check supports 1..8 cells')
    values = sparse.load_npz(assets / index['parts'][0]['matrix'])[:2*args.cells].toarray()
    with np.load(assets / 'normalization.npz') as norm:
        values = (values - norm['mean']) / norm['scale']
    x = torch.tensor(values[:args.cells], device='cuda:0', dtype=torch.float32)
    y = torch.tensor(values[args.cells:], device='cuda:0', dtype=torch.float32)
    noise = torch.randn_like(x)
    tau = torch.linspace(.2, .8, args.cells, device='cuda:0')
    graph = (np.arange(config.n_genes)[:, None] + np.arange(33)[None, :]) % config.n_genes
    fingerprint = torch.rand(1, 2048, device='cuda:0')
    def condition(model, n):
        scalar = lambda v, dtype=torch.float32: torch.full((n,), v, device='cuda:0', dtype=dtype)
        return model.encode_condition(fingerprint=fingerprint.expand(n, -1), dose=scalar(1),
            duration=scalar(24), background=scalar(0, torch.long), source=scalar(0, torch.long),
            protocol=scalar(0, torch.long), elapsed=scalar(0), horizon=scalar(24))
    seed_all(20260924)
    template = ResponseModel(config, torch.from_numpy(graph))
    with torch.no_grad():
        for block in template.velocity_model.blocks:
            for layer in block.modulations:
                layer[-1].bias.fill_(.03)
    state = {key: value.clone() for key, value in template.state_dict().items()}
    del template
    report = {'formal_training': False, 'optimizer_steps': 0, 'precision': 'float32_no_TF32',
              'genes': config.n_genes, 'layers': [config.encoder_layers, config.pad_layers],
              'graph': 'synthetic_ring_33', 'input': f'{2*args.cells}_real_T_cells_existing_T_normalization',
              'cells_per_timed_step': args.cells,
              'device': torch.cuda.get_device_name(0), 'cases': []}
    output = ROOT / args.output
    baseline = None
    cases = ([tuple(map(int, pair.split(':'))) for pair in args.cases.split(',')] if args.cases else
             [(256, 1), (512, 1), (1024, 1), (2048, 1), (1024, 2), (2048, 2), (4096, 2)])
    for chunk, micro in cases:
        model = None
        try:
            torch.cuda.empty_cache()
            shape = replace(config, query_chunk=chunk)
            model = ResponseModel(shape, torch.from_numpy(graph)).to('cuda:0')
            model.load_state_dict(state)
            def step():
                model.zero_grad(set_to_none=True)
                losses = []
                for start in range(0, len(x), micro):
                    end = min(start + micro, len(x))
                    loss = model.cfm_loss(x[start:end], y[start:end], condition(model, end-start),
                                          noise[start:end], tau[start:end])
                    (loss * (end-start) / len(x)).backward()
                    losses.append(loss.detach() * (end-start) / len(x))
                return torch.stack(losses).sum()
            step()  # Warm CUDA libraries before timing.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss = step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            grads = {name: p.grad.cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
            if baseline is None:
                baseline = grads
                baseline_loss = loss.detach().cpu()
            errors = [float((grads[k] - baseline[k]).abs().max()) for k in baseline]
            equal = all(torch.allclose(grads[k], baseline[k], rtol=5e-4, atol=5e-6) for k in baseline)
            entry = {'query_chunk': chunk, 'microbatch': micro, 'status': 'passed' if equal else 'gradient_mismatch',
                     'cfm_cells_seconds': elapsed, 'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                     'loss_abs_difference': float((loss.detach().cpu() - baseline_loss).abs()),
                     'max_gradient_abs_difference': max(errors), 'gradients_equivalent': equal}
            report['cases'].append(entry)
            print(json.dumps(entry), flush=True)
            atomic_json(output, report)
            del grads, loss
        except torch.OutOfMemoryError:
            report['cases'].append({'query_chunk': chunk, 'microbatch': micro, 'status': 'cuda_oom'})
            print(json.dumps(report['cases'][-1]), flush=True)
            atomic_json(output, report)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    feasible = [c for c in report['cases'] if c['status'] == 'passed' and c['peak_allocated_gb'] <= 5.5]
    best = min(feasible, key=lambda item: item['cfm_cells_seconds'])
    report['selected'] = best
    report['speedup_vs_first_case'] = report['cases'][0]['cfm_cells_seconds'] / best['cfm_cells_seconds']
    atomic_json(output, report)
    if args.full_ode:
        model = ResponseModel(replace(config, query_chunk=best['query_chunk']), torch.from_numpy(graph)).to('cuda:0')
        model.load_state_dict(state)
        micro = best['microbatch']
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        sample = model.sample(x[:micro], condition(model, micro), noise=noise[:micro], steps=32, checkpoint=True)
        sample.square().mean().backward()
        torch.cuda.synchronize()
        report['full_ode'] = {'steps': 32, 'microbatch': micro, 'seconds': time.perf_counter() - start,
            'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'finite_output': bool(torch.isfinite(sample).all()),
            'finite_gradients': all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)}
        atomic_json(output, report)
        print(json.dumps(report['full_ode']), flush=True)
    print(json.dumps({'selected': report['selected'], 'speedup': report['speedup_vs_first_case']}), flush=True)


if __name__ == '__main__':
    main()
