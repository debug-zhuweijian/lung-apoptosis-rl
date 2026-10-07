"""Strict nested D-only P/B/K fitting and complete-chain OOF uncertainty targets.

This is a candidate model protocol, not evidence of biological or prediction benefit.
"""
import json
from pathlib import Path
import numpy as np
import torch

from lung.engine.checkpoint import atomic_json, atomic_save
from lung.engine.config import implementation_identity
from .data import TimeData
from .response import train_response, predict_response, load_frozen_response
from .auxiliary import (digest, file_hash, fit_regressor, infer, calibrate, select_gate, load_frozen_regressor)
from .nested_features import (project, reference as matched_reference, chain_features, train_backward,
                              prediction_records, train_correction, record_metrics, upstream, unavailable)


def options(config):
    return {'epochs': config.get('epochs', 1000), 'seed': config.get('seed', 1729),
            'device': config.get('device', 'cuda:0'), 'hidden': 64, 'lr': .001,
            'batch_size': 32, 'population': 64,
            'log_root': config.get('log_root'), 'log_anchor': config.get('log_anchor'),
            **config.get('time_auxiliary', {})}


def folds(drugs, count):
    if count < 2 or len(drugs) % count:
        raise ValueError('DRUG_FOLDS_MUST_BE_EQUAL_AND_NONTRIVIAL')
    return [list(x) for x in np.array_split(np.asarray(drugs, dtype=str), count)]


def run_nested(root, config, output, resume=False, progress=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = options(config)
    data = TimeData(root, config, fit_drugs=[], allowed_target_drugs=[], role='nested_protocol')
    groups = data.drug_groups
    development = groups['D']
    if set(development) & (set(groups['C']) | set(groups['E'])) or set(groups['C']) & set(groups['E']):
        raise ValueError('OVERLAPPING_DRUG_ROLES')
    outer = folds(development, int(cfg.get('outer_folds', 4)))
    inner_count = int(cfg.get('inner_folds', 3))
    if not data.synthetic and (len(development), len(groups['C']), len(groups['E']), len(outer), inner_count, data.n_genes) != (28, 9, 10, 4, 3, 32461):
        raise ValueError('REAL_TIME_PROTOCOL_REQUIRES_D28_C9_E10_OUTER4_INNER3_FULL_GENES')
    identity = {'config': config, 'data': data.identity, 'coordinates': data.coordinate_identity,
                'outer_folds': outer, 'code': implementation_identity()}
    signature = digest(identity)
    identity_path = output / 'identity.json'
    if identity_path.exists():
        if not resume or json.loads(identity_path.read_text())['signature'] != signature:
            raise ValueError('NESTED_RESUME_IDENTITY_MISMATCH')
    else:
        atomic_json(identity_path, {'signature': signature, 'identity': identity})
    manifest_path = output / 'manifest.json'
    if resume and manifest_path.exists():
        completed = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(not Path(p).is_file() or file_hash(p) != h for p, h in completed['hashes'].items()):
            raise ValueError('NESTED_COMPLETE_ARTIFACT_MISMATCH')
        return {**completed, 'artifacts': completed['artifacts'] + [str(manifest_path)]}
    audit, artifacts = [], [str(identity_path)]

    pretrain = config.get('time', {}).get('shared24_pretrain', config.get('timecourse', {}).get('shared24_pretrain', {}))
    if isinstance(pretrain, bool):
        pretrain = {'enabled': pretrain}
    initial_checkpoints = None
    if pretrain.get('enabled', False):
        from .protocol import settings
        pretrain_config = {**config, 'timecourse': {**settings(config), 'epochs': int(pretrain.get('epochs', cfg['epochs']))}}
        start_view = data.view([], [], 'shared_known24_pretrain').training_view(known24_only=True)
        initial = train_response(start_view, pretrain_config, output / 'shared24_pretrain', resume=resume, progress=progress)
        initial_checkpoints = initial['checkpoints']
        artifacts.extend(initial['artifacts'] + initial_checkpoints)

    def model_identity(role, fit, excluded, parents=None):
        result = {'role': role, 'fit_drugs': list(fit), 'excluded_drugs': list(excluded),
                  'fit_scope': '72h drug targets', 'excluded_scope': '72h drug targets',
                  'known24_drugs': sum(groups.values(), []),
                  'initial_checkpoints': upstream(initial_checkpoints) if initial_checkpoints else {},
                  'generalization_scope': '72h held-out drug targets; known 24h of D/C/E permitted',
                  'coordinates': data.coordinate_identity, 'data': data.identity,
                  'parents': parents or {}, 'epochs': cfg['epochs'], 'signature': signature}
        audit.append(result)
        return result

    def pb(fit, excluded, folder):
        view = data.view(fit, fit, role=str(folder))
        p_identity = model_identity(str(folder) + '/P', fit, excluded)
        result = train_response(view, config, output / folder / 'P', resume=resume, progress=progress, initial_checkpoints=initial_checkpoints)
        paths = result.get('checkpoints', result.get('response'))
        if not paths:
            raise ValueError('RESPONSE_CHECKPOINTS_MISSING')
        artifacts.extend(result['artifacts'] + paths)
        p_identity['outputs'] = upstream(paths)
        b = train_backward(view, fit, output / folder / 'B.pt', cfg,
                           model_identity(str(folder) + '/B', fit, excluded, upstream(paths)), resume, progress)
        artifacts.append(b)
        return paths, b

    outer_records, final_k_records = [], []
    for oi, held in enumerate(outer):
        fold_path = output / f'outer{oi}' / 'completed.pt'
        if resume and fold_path.exists():
            completed = torch.load(fold_path, map_location='cpu', weights_only=False)
            if completed['signature'] != signature or any(file_hash(p) != h for p, h in completed['hashes'].items()):
                raise ValueError('COMPLETED_FOLD_IDENTITY_MISMATCH')
            audit.extend(completed['audit'])
            artifacts.extend(completed['artifacts'] + [str(fold_path)])
            outer_records.extend(completed['records'])
            final_k_records.extend(completed['records'])
            continue
        audit_start, artifact_start = len(audit), len(artifacts)
        fit = [d for d in development if d not in held]
        inner_records, inner_parents = [], []
        for ii, inner_held in enumerate(folds(fit, inner_count)):
            inner_fit = [d for d in fit if d not in inner_held]
            p, b = pb(inner_fit, held + inner_held + groups['C'] + groups['E'], f'outer{oi}/inner{ii}')
            held_view = data.view([], inner_held, role=f'outer{oi}/inner{ii}/supervision')
            inner_records.extend(prediction_records(held_view, inner_held, p, b, cfg))
            inner_parents.extend(p + [b])
        records_path = output / f'outer{oi}' / 'inner_features.pt'
        if not records_path.exists():
            atomic_save(records_path, inner_records)
        artifacts.append(str(records_path))
        k = train_correction(inner_records, output / f'outer{oi}' / 'K.pt', cfg,
                             model_identity(f'outer{oi}/K', fit, held + groups['C'] + groups['E'],
                                            upstream(inner_parents)), resume, progress)
        artifacts.append(k)
        p, b = pb(fit, held + groups['C'] + groups['E'], f'outer{oi}/full')
        held_view = data.view([], held, role=f'outer{oi}/OOF')
        records = prediction_records(held_view, held, p, b, cfg, k)
        for row in records:
            row['outer_fold'] = oi
        outer_records.extend(records)
        final_k_records.extend(records)
        atomic_save(fold_path, {'signature': signature, 'records': records, 'audit': audit[audit_start:],
                               'artifacts': artifacts[artifact_start:], 'hashes': upstream(artifacts[artifact_start:])})
        artifacts.append(str(fold_path))
        if progress:
            progress('time_nested_outer_folds', oi + 1, len(outer))

    gate = select_gate(outer_records, cfg['seed'])
    gate_path = output / 'gate.json'
    atomic_json(gate_path, gate)
    oof_path = output / 'D_oof.json'
    atomic_json(oof_path, {'role': 'D_outer_OOF', 'records': record_metrics(outer_records),
                           'unavailable_distribution': unavailable(data, development)})
    p, b = pb(development, groups['C'] + groups['E'], 'final')
    k = train_correction(final_k_records, output / 'final/K.pt', cfg,
                         model_identity('final/K', development, groups['C'] + groups['E'],
                                        upstream(artifacts)), resume, progress)
    usable = [r for r in outer_records if r['off']['error'] is not None and r['on']['error'] is not None]
    u = fit_regressor([r['features'] for r in usable],
                      [np.log1p([r['off']['error'], r['on']['error']]) for r in usable],
                      output / 'final/U.pt', cfg,
                      model_identity('final/U', development, groups['C'] + groups['E'], upstream([oof_path, k])),
                      resume, progress)
    result = {'response': p, 'B': b, 'K': k, 'U': u, 'gate': str(gate_path), 'D_oof': str(oof_path)}
    # All models and the gate are frozen before reading any C labels.
    calibration_rows = []
    frozen_u = load_frozen_regressor(u, cfg['device'])
    c_records = prediction_records(data.view([], groups['C'], 'calibration'), groups['C'], p, b, cfg, k)
    for row in c_records:
        estimates = np.maximum(np.expm1(np.clip(infer(frozen_u, row['features'], cfg['device']), 0, 50)), 0)
        for i, name in enumerate(('off', 'on')):
            calibration_rows.append({'drug': row['drug'], 'gate': name,
                                     'error': row[name]['error'], 'u': float(estimates[i])})
    calibration: dict = {name: calibrate([r for r in calibration_rows if r['gate'] == name], cfg.get('alpha', .1))
                   for name in ('off', 'on')}
    calibration['records'] = calibration_rows
    calibration['unavailable_distribution'] = unavailable(data, groups['C'])
    calibration_path = output / 'calibration.json'
    atomic_json(calibration_path, calibration)
    result['calibration'] = str(calibration_path)
    # E is accessed only after the complete model, gate and calibration freeze.
    e_records = prediction_records(data.view([], groups['E'], 'final_report'), groups['E'], p, b, cfg, k)
    for row in e_records:
        estimates = np.maximum(np.expm1(np.clip(infer(frozen_u, row['features'], cfg['device']), 0, 50)), 0)
        row['uncertainty'] = {}
        for index, name in enumerate(('off', 'on')):
            cal = calibration[name]
            bound = None if cal['quantile'] is None else cal['quantile'] * (float(estimates[index]) + cal['eps'])
            error = row[name]['error']
            row['uncertainty'][name] = {'u': float(estimates[index]), 'upper_bound': bound,
                                        'status': cal['status'], 'covered': None if bound is None or error is None else error <= bound}
    report_path = output / 'E_report.json'
    atomic_json(report_path, {'role': 'E_frozen_report', 'gate': gate, 'records': record_metrics(e_records),
                              'unavailable_distribution': unavailable(data, groups['E']),
                              'scope': 'expression empirical-population error; not biological endpoint'})
    audit_path = output / 'audit.json'
    atomic_json(audit_path, {'models': audit, 'calibration_drugs': groups['C'], 'report_drugs': groups['E']})
    result.update(E_report=str(report_path), audit=str(audit_path))
    artifacts.extend([k, u, str(gate_path), str(oof_path), str(calibration_path), str(report_path), str(audit_path)])
    artifacts.extend(str(p) for p in output.rglob('*.metrics.csv'))
    artifacts.extend(str(p) for p in output.rglob('events.out.tfevents.*'))
    if cfg.get('log_root'):
        artifacts.extend(str(p) for p in Path(cfg['log_root']).rglob('events.out.tfevents.*'))
    result['artifacts'] = sorted(set(str(Path(path).resolve()) for path in artifacts))
    result['hashes'] = upstream(result['artifacts'])
    result['manifest'] = str(output / 'manifest.json')
    result.update(checkpoints=p, predictor_manifest=result['manifest'], report=str(report_path), synthetic=bool(data.synthetic))
    atomic_json(result['manifest'], result)
    result['artifacts'].append(result['manifest'])
    return result


class FrozenTimePredictor:
    def __init__(self, root, config, artifacts):
        self.config, self.artifacts, self.cfg = config, artifacts, options(config)
        self.data = TimeData(root, config, fit_drugs=[], allowed_target_drugs=[], role='frozen_inference')
        self.gate = json.loads(Path(artifacts['gate']).read_text())
        self.calibration = json.loads(Path(artifacts['calibration']).read_text())
        for path, expected in artifacts.get('hashes', {}).items():
            if file_hash(path) != expected:
                raise ValueError('FROZEN_ARTIFACT_HASH_MISMATCH')
        self.models = [load_frozen_response(path, self.data, self.cfg['device']) for path in artifacts['response']]
        self.regressors = {key: load_frozen_regressor(artifacts[key], self.cfg['device']) for key in ('B', 'K', 'U')}

    def predict(self, condition_id, sample_count=64, seed=0, reference=None):
        raw = predict_response(self.artifacts['response'], self.data, condition_id, sample_count, seed,
            self.cfg['device'], reference=reference, models=self.models, as_tensor=True)
        off = project(raw, self.data)
        controls = torch.as_tensor(reference if reference is not None else
            matched_reference(self.data, condition_id, sample_count, seed), dtype=torch.float32, device=self.cfg['device'])
        features = chain_features(self.data, condition_id, off, controls, self.regressors['B'], self.cfg['device'])
        corrected = project(off + infer(self.regressors['K'], features, self.cfg['device']), self.data)
        estimates = infer(self.regressors['U'], features, self.cfg['device']).clamp(0, 50).expm1().clamp_min(0)
        index = int(self.gate['enabled'])
        calibration = self.calibration['on' if index else 'off']
        uncertainty = {'off': float(estimates[0]), 'on': float(estimates[1]), 'selected': float(estimates[index]),
                       'status': calibration['status'], 'upper_bound': None if calibration['quantile'] is None else
                       calibration['quantile'] * (float(estimates[index]) + calibration['eps'])}
        return {'raw': raw, 'corrected': corrected, 'projected': corrected if index else off,
                'uncertainty': uncertainty, 'gate': bool(index)}
