"""Fixed shared gene-coordinate features; targets are separate supervision fields."""
from pathlib import Path
import numpy as np
import torch

from lung.engine.response import projected
from .auxiliary import stats, infer, population_error, fit_regressor, file_hash, load_frozen_regressor
from .response import predict_response, load_frozen_response


def project(x, data):
    with torch.no_grad():
        value = projected(torch.as_tensor(x, dtype=torch.float32), data)
        return value if isinstance(x, torch.Tensor) else value.cpu().numpy()


def concatenate(values):
    if isinstance(values[0], torch.Tensor):
        return torch.cat([torch.as_tensor(v, dtype=torch.float32, device=values[0].device) for v in values])
    return np.concatenate(values).astype(np.float32)


def condition_vector(data, cid):
    row = data.conditions.loc[cid]
    fp = np.asarray(row['fingerprint'], dtype=np.float32)
    return np.concatenate([fp, [np.log1p(float(row['dose'])), float(row['duration']) / 72.]])


def reference(data, cid, n, seed):
    return data.sample_controls(cid, n, np.random.default_rng(seed))['control']


def base_features(data, cid, generated, control):
    return concatenate([stats(generated), stats(control), condition_vector(data, cid)])


def backward_features(data, cid, target):
    return concatenate([stats(target), condition_vector(data, cid)])


def chain_features(data, cid, generated, control, backward, device):
    reverse = infer(backward, backward_features(data, cid, generated), device)
    return concatenate([base_features(data, cid, generated, control), reverse - control.mean(0)])


def usable_ids(data, drugs):
    return [cid for cid in data.condition_ids(drugs=drugs, time=72)
            if int(data.conditions.loc[cid, 'n_cells']) > 0
            and int(data.conditions.loc[cid, 'reference_n']) > 0]


def unavailable(data, drugs):
    return [{'condition_id': str(cid), 'drug': str(data.conditions.loc[cid, 'group_id']),
             'n_cells': int(data.conditions.loc[cid, 'n_cells']),
             'status': str(data.conditions.loc[cid, 'status'])}
            for cid in data.condition_ids(drugs=drugs, time=72)
            if int(data.conditions.loc[cid, 'n_cells']) < 2 or int(data.conditions.loc[cid, 'reference_n']) == 0]


def train_backward(data, drugs, path, cfg, identity, resume, progress):
    x, y = [], []
    for cid in usable_ids(data, drugs):
        target = data.sample_target(cid, None, np.random.default_rng(cfg['seed']))
        control = reference(data, cid, cfg['population'], cfg['seed'])
        x.append(backward_features(data, cid, target))
        y.append(control.mean(0))
    return fit_regressor(x, y, path, cfg, identity, resume, progress)


def prediction_records(data, drugs, response, backward, cfg, correction=None):
    records = []
    models = [load_frozen_response(path, data, cfg['device']) for path in response]
    backward = load_frozen_regressor(backward, cfg['device'])
    correction = load_frozen_regressor(correction, cfg['device']) if correction else None
    for cid in usable_ids(data, drugs):
        seed = cfg['seed']
        raw = predict_response(response, data, cid, cfg['population'], seed, cfg['device'], models=models, as_tensor=True)
        generated = project(raw, data)
        control = torch.as_tensor(reference(data, cid, cfg['population'], seed), device=cfg['device'])
        features = chain_features(data, cid, generated, control, backward, cfg['device'])
        assert isinstance(features, torch.Tensor) and isinstance(generated, torch.Tensor)
        # This is the first target read in this function: features above are target-free.
        target = data.sample_target(cid, None, np.random.default_rng(seed))
        corrected = project(generated + infer(correction, features, cfg['device']), data) if correction else generated
        records.append({'condition_id': str(cid), 'drug': str(data.conditions.loc[cid, 'group_id']),
                        'features': features.cpu().numpy(), 'residual': target.mean(0) - generated.mean(0).cpu().numpy(),
                        'off': population_error(generated, target, data.bandwidth),
                        'on': population_error(corrected, target, data.bandwidth)})
    return records


def train_correction(records, path, cfg, identity, resume, progress):
    return fit_regressor([r['features'] for r in records], [r['residual'] for r in records],
                         path, cfg, identity, resume, progress)


def record_metrics(records):
    return [{k: v for k, v in row.items() if k not in ('features', 'residual')} for row in records]


def upstream(paths):
    return {str(Path(p).resolve()): file_hash(p) for p in paths}
