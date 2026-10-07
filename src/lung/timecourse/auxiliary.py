"""Candidate full-gene auxiliary regressors and drug-group calibration."""
import hashlib
import json
import math
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lung.engine.checkpoint import atomic_save
from lung.models.objectives import rbf_kernel


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(population):
    if isinstance(population, torch.Tensor):
        x = population.to(dtype=torch.float32)
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('INVALID_POPULATION')
        return torch.cat([x.mean(0), x.std(0, correction=0)])
    x = np.asarray(population, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('INVALID_POPULATION')
    return np.concatenate([x.mean(0), x.std(0)])


def population_error(predicted, observed, bandwidth=1., *, device_cache_bytes=1024**3):
    """FP64 full-population V-MMD with bounded device caching, never subsampling."""
    device = predicted.device if isinstance(predicted, torch.Tensor) else 'cpu'
    x, y = (torch.as_tensor(value) for value in (predicted, observed))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('POPULATION_GENE_AXIS_MISMATCH')
    if not len(x) or not len(y):
        return {'status': 'empty', 'mean_mse': None, 'mmd_v': None, 'error': None}
    required = 8 * (x.numel() + y.numel())
    if torch.device(device).type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(device_cache_bytes, free // 4)
    else:
        budget = device_cache_bytes
    if required <= budget:
        x, y = (value.to(device=device, dtype=torch.float64) for value in (x, y))
    def block(value, start):
        return value[start:start+64].to(device=device, dtype=torch.float64)
    def mean(value):
        total = torch.zeros(value.shape[1], dtype=torch.float64, device=device)
        for start in range(0, len(value), 64):
            total.add_(block(value, start).sum(0))
        return total / len(value)
    mse = float((mean(x) - mean(y)).square().mean())
    if min(len(x), len(y)) < 2:
        return {'status': 'single_cell', 'mean_mse': mse, 'mmd_v': None, 'error': None}
    def kernel_mean(a, b):
        total = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, len(a), 64):
            left = block(a, start)
            for offset in range(0, len(b), 64):
                right = block(b, offset)
                total = total + rbf_kernel(left, right, bandwidth).sum()
        return total / (len(a) * len(b))
    mmd = max(0., float(kernel_mean(x, x) + kernel_mean(y, y) - 2 * kernel_mean(x, y)))
    return {'status': 'valid', 'mean_mse': mse, 'mmd_v': mmd, 'error': mse + mmd}


def calibrate(rows, alpha=.1, eps=1e-6):
    if not 0 < alpha < 1:
        raise ValueError('INVALID_ALPHA')
    groups = {}
    for row in rows:
        if row['error'] is None:
            continue
        score = float(row['error']) / (float(row['u']) + eps)
        if not np.isfinite(score) or score < 0:
            raise ValueError('INVALID_CALIBRATION_SCORE')
        groups[row['drug']] = max(groups.get(row['drug'], 0.), score)
    rank = math.ceil((len(groups) + 1) * (1 - alpha))
    bounded = rank <= len(groups)
    return {'status': 'bounded' if bounded else 'unbounded', 'alpha': alpha,
            'rank': rank, 'n_groups': len(groups), 'group_scores': groups,
            'quantile': sorted(groups.values())[rank - 1] if bounded else None,
            'eps': eps, 'scope': 'one exchangeable new drug group; not selected-best coverage'}


def select_gate(rows, seed=1729, bootstrap=2000):
    groups = {}
    for row in rows:
        if row['off']['error'] is not None and row['on']['error'] is not None:
            groups.setdefault(row['drug'], []).append([row['off']['error'], row['on']['error']])
    if len(groups) < 2:
        return {'enabled': False, 'reason': 'insufficient_D_groups', 'n_groups': len(groups)}
    values = np.array([np.mean(x, axis=0) for x in groups.values()])
    delta = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(seed)
    upper = float(np.quantile(delta[rng.integers(len(delta), size=(bootstrap, len(delta)))].mean(1), .95))
    gain = float(-delta.mean() / max(values[:, 0].mean(), 1e-12))
    return {'enabled': bool(gain >= .01 and upper < 0), 'relative_gain': gain,
            'bootstrap_upper': upper, 'n_groups': len(groups), 'role': 'D_outer_OOF'}


def architecture(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, outputs))


def fit_regressor(x, y, path, config, identity, resume=False, progress=None):
    """Fixed epoch endpoint, resumable optimizer; never accepts a validation target."""
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('INVALID_AUXILIARY_TRAINING_DATA')
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {**identity, 'x_hash': array_hash(x), 'y_hash': array_hash(y), 'config': config}
    signature = digest(identity)
    device = torch.device(config.get('device', 'cpu'))
    torch.manual_seed(int(config.get('seed', 1729)))
    width = int(config.get('hidden', 64))
    model = architecture(x.shape[1], y.shape[1], width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get('lr', .001)))
    epoch = 0
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('AUXILIARY_RESUME_IDENTITY_MISMATCH')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch = saved['epoch']
    epochs = int(config.get('epochs', 1000))
    if epochs < 1:
        raise ValueError('INVALID_EPOCH_TARGET')
    if epoch == epochs:
        return str(path)
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    batch = int(config.get('batch_size', 32))
    metrics_path = path.with_suffix('.metrics.csv')
    from torch.utils.tensorboard import SummaryWriter
    event_dir = (Path(config['log_root']) / path.parent.relative_to(Path(config['log_anchor'])) / path.stem / 'events'
                 if config.get('log_root') and config.get('log_anchor') else path.parent / (path.stem + '.events'))
    writer = SummaryWriter(str(event_dir), purge_step=epoch + 1)
    for current in range(epoch, epochs):
        model.train()
        losses = []
        for start in range(0, len(x), batch):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(xt[start:start+batch]) - yt[start:start+batch]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('NONFINITE_AUXILIARY_LOSS')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        atomic_save(path, {'signature': signature, 'identity': identity, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'epoch': current + 1,
                          'shape': [x.shape[1], y.shape[1], width]})
        fresh = not metrics_path.exists()
        with metrics_path.open('a', newline='', encoding='utf-8') as stream:
            csv_writer = csv.writer(stream)
            if fresh:
                csv_writer.writerow(['epoch', 'train_mse'])
            csv_writer.writerow([current + 1, float(np.mean(losses))])
        writer.add_scalar('train/mse', float(np.mean(losses)), current + 1)
        writer.flush()
        if progress:
            try:
                progress(identity.get('role', 'auxiliary'), current + 1, epochs)
            except BaseException:
                writer.close()
                raise
    writer.close()
    return str(path)


def load_frozen_regressor(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    with torch.random.fork_rng(devices=[]):
        model = architecture(*saved['shape']).to(device)
    model.load_state_dict(saved['model'])
    return model.eval().requires_grad_(False)


def infer(path, features, device='cpu'):
    model = path if isinstance(path, nn.Module) else load_frozen_regressor(path, device)
    with torch.no_grad():
        value = model(torch.as_tensor(features, dtype=torch.float32, device=device).reshape(1, -1))[0]
        return value if isinstance(features, torch.Tensor) else value.cpu().numpy()
