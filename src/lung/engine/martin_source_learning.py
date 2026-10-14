"""Independent Martin source E/D fit/selection, never an old training entry.

Fit uses unique GSM observations, with two domain views of each same sample.
The equal-domain reconstruction objective is a new project candidate, not a
measured gain. Final RNA, P, state C, functional G and Q are never enabled here.
"""
from dataclasses import dataclass, fields, replace
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch

from ..data.martin_source_inputs import MartinSourceInputs, canonical, digest, require, sha
from ..data.martin_native_rna import DOMAINS
from ..models.full_protocol import FullPhysicalProtocolBatch
from ..models.martin_source import MartinSourceModel
from .checkpoint import atomic_json, atomic_save, load_checkpoint, restore_rng, rng_state, seed_all
from .latent_pipeline import _EarlyStop

SCHEMA = 'martin-independent-source-E-D-learning-v1'
OBJECTIVE = 'equal_unique_GSM_equal_domain_masked_reconstruction_MSE_v1'


def one_sample(batch, index):
    physical = FullPhysicalProtocolBatch(**{f.name: getattr(batch.physical, f.name)[index:index+1].clone()
                                           for f in fields(batch.physical)})
    return replace(batch, raw_values=batch.raw_values[index:index+1].clone(),
                   model_values=batch.model_values[index:index+1].clone(),
                   observed_mask=batch.observed_mask[index:index+1].clone(),
                   measurement_clock=batch.measurement_clock[index:index+1].clone(),
                   metadata_values=batch.metadata_values[index:index+1].clone(),
                   metadata_known=batch.metadata_known[index:index+1].clone(), physical=physical,
                   sample_ids=(batch.sample_ids[index],))


def observation_identity(batch):
    """Reference candidate protocols are unused by reference reconstruction."""
    def tensor_identity(tensor):
        value = tensor.detach().cpu().numpy()
        import hashlib
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    common = {'sample_id': batch.sample_ids[0], 'source': batch.source_id, 'background': batch.background,
              'part': batch.part, 'domain': batch.domain, 'units': batch.units,
              'role': batch.role, 'contract': batch.contract}
    for name in ('raw_values', 'model_values', 'observed_mask', 'offset', 'scale', 'measurement_clock'):
        common[name] = tensor_identity(getattr(batch, name))
    if batch.part == 'target':
        common.update(protocol=batch.protocol,
                      physical={f.name: tensor_identity(getattr(batch.physical, f.name)) for f in fields(batch.physical)},
                      metadata=tensor_identity(batch.metadata_values), metadata_known=tensor_identity(batch.metadata_known))
    return common


@dataclass(frozen=True)
class SourceObservations:
    role: str
    contract: dict
    by_domain: dict
    counts: dict

    @property
    def sample_ids(self):
        return tuple(sorted(self.by_domain[DOMAINS[0]]))

    @property
    def identity(self):
        return digest({d: {sid: observation_identity(self.by_domain[d][sid]) for sid in self.sample_ids}
                       for d in DOMAINS})


def collect_observations(packets, *, role, contract):
    require(role in ('fit', 'selection'), 'FINAL_NOT_ENABLED')
    result, views = {d: {} for d in DOMAINS}, {'reference': 0, 'target': 0}
    for batch in packets:
        batch.validate()
        require(batch.role == role and batch.contract == contract, 'COLLECTION_ROLE_OR_FOLD')
        views[batch.part] += len(batch.sample_ids)
        for index, sid in enumerate(batch.sample_ids):
            part = one_sample(batch, index)
            previous = result[batch.domain].get(sid)
            require(previous is None or observation_identity(previous) == observation_identity(part),
                    'DUPLICATE_GSM_OBSERVATION_CONFLICT')
            result[batch.domain].setdefault(sid, part)
    ids = set(result[DOMAINS[0]])
    require(bool(ids) and all(set(result[d]) == ids for d in DOMAINS), 'SAME_GSM_TWO_DOMAIN_VIEWS_REQUIRED')
    fit_ids = set(contract['fit_sample_ids'])
    require(ids == fit_ids if role == 'fit' else not ids & fit_ids, 'FIT_SELECTION_SAMPLE_MEMBERSHIP')
    counts = {name: sum(result[DOMAINS[0]][sid].part == name for sid in ids) for name in views}
    return SourceObservations(role, contract, result,
                              {'unique_GSM': len(ids), 'domain_views_per_GSM': 2,
                               'unique_reference_GSM': counts['reference'], 'unique_target_GSM': counts['target'],
                               'reference_domain_request_rows_before_dedup': views['reference'],
                               'target_domain_request_rows_before_dedup': views['target'],
                               'not_twice_as_many_biological_replicates': True})


def prepare_fold(root, *, system, fold_id):
    require(system == 'cycle', 'FIT_SELECTION_REQUIRES_EXISTING_CYCLE_PLAN')
    with MartinSourceInputs(root, allowed_roles=('fit', 'selection')) as reader:
        contract = {k: v for k, v in reader.contract(system=system, fold_id=fold_id).items() if k != 'gene_ids'}
        data = {}
        for role in ('fit', 'selection'):
            packets = []
            requests = reader.protocol.requests(system=system, fold_id=fold_id, role=role)
            for request in sorted(requests, key=lambda r: r['request_id']):
                for domain in DOMAINS:
                    packets.extend(reader.read_request(request['request_id'], system=system, fold_id=fold_id,
                                                       role=role, domain=domain, supervision=True).values())
            data[role] = collect_observations(packets, role=role, contract=contract)
    require(not set(data['fit'].sample_ids) & set(data['selection'].sample_ids), 'DISJOINT_FIT_SELECTION_REQUIRED')
    return data['fit'], data['selection']


def settings_from_config(config, *, width=32):
    ae = config.get('research', {}).get('latent', {}).get('ae', {})
    response = config['response']
    value = {'width': width, 'epochs': config.get('modular', {}).get('stages', {}).get('ae', {}).get('epochs', config['epochs']),
             'batch': ae.get('batch', response['ot_samples']), 'lr': ae.get('lr', response['lr']),
             'weight_decay': response['weight_decay'], 'grad_clip': response['grad_clip'],
             'seed': config['seed'], 'device': config['device'], 'dtype': 'float32',
             'early_stopping': config.get('early_stopping', False)}
    require(all(type(value[k]) is int and value[k] > 0 for k in ('width', 'epochs', 'batch'))
            and type(value['seed']) is int and value['seed'] >= 0, 'LEARNING_INTEGER_SETTINGS')
    require(all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in
                ('lr', 'weight_decay', 'grad_clip'))
            and value['lr'] > 0 and value['weight_decay'] >= 0 and value['grad_clip'] > 0, 'LEARNING_OPTIMIZER_SETTINGS')
    return value


def execution_dependencies():
    """Check execution versions before opening a locally generated resume PT."""
    models = Path(__file__).resolve().parents[1] / 'models'
    return {**{name: sha(models / name) for name in ('martin_source.py', 'full_protocol.py', 'encoders.py')},
            'checkpoint.py': sha(Path(__file__).with_name('checkpoint.py')),
            'latent_pipeline._EarlyStop': hashlib.sha256(inspect.getsource(_EarlyStop).encode()).hexdigest(),
            'torch_version': str(torch.__version__)}


def preflight(fit, selection, settings):
    require(fit.role == 'fit' and selection.role == 'selection' and fit.contract == selection.contract
            and not set(fit.sample_ids) & set(selection.sample_ids), 'FIT_SELECTION_DATA_ROLES')
    return {'schema': SCHEMA, 'objective': OBJECTIVE, 'contract': fit.contract, 'settings': settings,
            'fit_counts': fit.counts, 'selection_counts': selection.counts,
            'fit_observations_sha256': fit.identity, 'selection_observations_sha256': selection.identity,
            'learning_source_sha256': sha(__file__), 'optimizer': 'AdamW',
            'execution_dependencies': execution_dependencies(),
            'selection_used_for_gradients_or_coordinates': False, 'final_RNA_enabled': False,
            'new_shared_latent_or_P_qualification': False, 'scientific_qualified': False,
            'real_model_forwards': 0, 'PT_loads': 0, 'optimizer_steps': 0}


def _loss(model, data, sample_ids):
    require(data.role in ('fit', 'selection'), 'LEARNING_ROLE')
    losses = {d: [] for d in DOMAINS}
    for sid in sample_ids:
        for domain in DOMAINS:
            batch = data.by_domain[domain][sid]
            require(batch.role == data.role, 'LEARNING_SAMPLE_ROLE_CHANGED')
            losses[domain].append(model.observation_loss(model(batch), batch))
    values = {d: torch.stack(items).mean() for d, items in losses.items()}
    return torch.stack(list(values.values())).mean(), values


def fit_source(fit, selection, settings, output, *, resume=False, synthetic=False):
    """Explicit fitting only; CLI default preflight never calls this function."""
    report = preflight(fit, selection, settings)
    output = Path(output).resolve()
    require(type(resume) is bool and type(synthetic) is bool, 'LEARNING_STRICT_BOOL')
    identity = digest({**report, 'synthetic': synthetic})
    manifest = {**report, 'identity': identity, 'synthetic': synthetic}
    if resume:
        require(output.is_dir() and (output / 'run_manifest.json').is_file(), 'RESUME_RUN_MISSING')
        require(canonical(json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))) == canonical(manifest),
                'RESUME_INPUT_MODEL_OR_CONFIG_CHANGED')
    else:
        require(not output.exists(), 'NEW_SOURCE_LEARNING_EXPERIMENT_REQUIRED')
    seed_all(settings['seed'])
    example = fit.by_domain[DOMAINS[0]][fit.sample_ids[0]]
    model = MartinSourceModel(example.gene_ids, fit.contract, width=settings['width']).to(settings['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    stop = _EarlyStop({'early_stopping': settings['early_stopping']})
    epoch, cursor, order, history = 0, 0, None, []
    if resume:
        saved = load_checkpoint(output / 'last.pt', identity)
        model.load_registered_state(saved['model'], saved['metadata'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch, cursor, order = saved['epoch'], saved['extra']['cursor'], saved['extra']['order']
        history = saved['extra']['history']
        stop.load_state_dict(saved['extra']['early_stop'])
        require(type(epoch) is int and 0 <= epoch <= settings['epochs'] and type(cursor) is int
                and cursor >= 0 and type(saved['counters']['optimizer_step']) is int
                and saved['counters']['optimizer_step'] == stop.steps, 'RESUME_COUNTERS')
        restore_rng(saved['rng'])
    else:
        output.mkdir(parents=True)
        atomic_json(output / 'run_manifest.json', manifest)
    def save(name='last.pt'):
        metadata = {'model_identity': model.identity, 'training_role': 'fit', 'final_RNA_used': False,
                    'stage': SCHEMA, 'synthetic': synthetic, 'objective': OBJECTIVE, 'scientific_qualified': False}
        atomic_save(output / name, {'schema_version': 1, 'identity': identity, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'metadata': metadata,
                    'counters': {'optimizer_step': stop.steps}, 'rng': rng_state(),
                    'extra': {'cursor': cursor, 'order': order, 'history': history, 'early_stop': stop.state_dict()}})
    while epoch < settings['epochs'] and not stop.stopped:
        if order is None:
            order = torch.randperm(len(fit.sample_ids)).tolist()
        require(sorted(order) == list(range(len(fit.sample_ids))) and cursor <= len(order), 'RESUME_FIT_PERMUTATION')
        model.train()
        while cursor < len(order):
            end = min(cursor + settings['batch'], len(order))
            ids = [fit.sample_ids[i] for i in order[cursor:end]]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss(model, fit, ids)
            require(bool(torch.isfinite(loss)), 'NONFINITE_SOURCE_FIT_LOSS')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            stop.step_done()
            cursor = end
            save()
        model.eval()
        with torch.no_grad():
            metric, domains = _loss(model, selection, selection.sample_ids)
        require(bool(torch.isfinite(metric)), 'NONFINITE_SOURCE_SELECTION_LOSS')
        improved = stop.observe(float(metric))
        history = [*history, {'epoch': epoch + 1, 'optimizer_step': stop.steps,
                             'selection_loss': float(metric),
                             'selection_domain_loss': {d: float(v) for d, v in domains.items()}}]
        epoch, cursor, order = epoch + 1, 0, None
        if improved:
            save('best.pt')
        save()
        atomic_json(output / 'history.json', history)
    result = {'schema': SCHEMA, 'identity': identity, 'epochs_completed': epoch, 'optimizer_steps': stop.steps,
              'reason': 'selection_plateau' if stop.stopped else 'max_epochs', 'history': history,
              'fit_counts': fit.counts, 'selection_counts': selection.counts, 'synthetic': synthetic,
              'final_RNA_used': False, 'scientific_qualified': False, 'P_outputs': 0, 'G_outputs': 0, 'Q_outputs': 0}
    result = {**result, 'completion': {'status': 'complete', 'reason': result['reason']},
              'run_manifest_sha256': sha(output / 'run_manifest.json'),
              'selected_checkpoint': {'file': 'best.pt', 'sha256': sha(output / 'best.pt')},
              'last_checkpoint': {'file': 'last.pt', 'sha256': sha(output / 'last.pt')}}
    atomic_json(output / 'result.json', result)
    return result
