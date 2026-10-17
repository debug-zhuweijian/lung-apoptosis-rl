"""Native final RNA evaluation for the new full-P namespace, never fitting.

Default entry reads only authenticated metadata and frozen fit coordinates.
Explicit evaluation verifies completed matching AE/P before opening E RNA.
Native observation hours remain distinct from unknown active exposure.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lung.data.native_response_units import JOINT, clean, sha
from lung.models import rbf_mmd
from lung.models.objectives import physical_projection
from . import full_protocol_response as producer
from . import full_protocol_response_learning as learning
from .checkpoint import load_checkpoint
from .config import ROOT, config_hash
from .data import EngineData
from .response import tensor

SCHEMA = 'full-protocol-native-final-evaluation-v2'
COMPARISON_SCHEMA = 'full-protocol-native-final-three-variant-comparison-v1'
METRICS = ('standardized_mean_mse', 'standardized_population_mmd_v')
NATIVE = 'experiments/20261011-data-alignment-round17/native-response-units-v1'
NATIVE_MANIFEST_SHA256 = '7c04dcd7486224f6870e31d58f7e47a9ca7859213ef120d08c529f4b562ebb71'
METADATA_FILES = ('conditions.parquet', 'chemicals.parquet', 'role_condition_counts.parquet', 'control_links.parquet',
                  'split_groups.parquet', 'genes.parquet', 'normalization.npz', 'bandwidths.npz',
                  'bandwidth_pairs.parquet', 'neighbors.npz', 'fingerprints.npy', 'test.parquet')
NO_EFFECT_CLAIMS = {'state_increment_validated': False, 'Annexin_validated': False,
                    'full_exposure_protocol_generalization_validated': False,
                    'multistep_or_true_sequence_validated': False, 'whole_goal_complete': False}


def require(ok, reason):
    if not ok:
        raise ValueError('FULL_P_FINAL_' + reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    producer.write_new(path, value)


def partial_request(record):
    protocol = record['actual_protocol']
    require(record['measurement_level'] == 'single_cell' and record['gene_axis_id'] == 'joint_ensembl_32461'
            and record['unique_functional_Y'] is None and record['fold_id'] is None,
            'SINGLE_CELL_RNA_ONLY_DOMAIN')
    require(record['units'] == 'source_raw_counts_no_new_normalization'
            and record['decoder_domain'] == 'singlecell_raw_counts_common_ensembl'
            and record['reference_semantics'] == 'contemporaneous_vehicle_not_same_cell_t0', 'SOURCE_UNITS_REFERENCE')
    require(protocol['exposure_duration_unknown'] is True and protocol['exposure_duration_h'] is None
            and protocol['prior_drug_unknown'] is True and protocol['prior_drug'] is None
            and protocol['washout_unknown'] is True and protocol['washout_h'] is None
            and protocol['withdrawal_unknown'] is True, 'UNKNOWN_EXPOSURE_HISTORY_RETAINED')
    require(type(record['dose_uM']) in (int, float) and math.isfinite(record['dose_uM']) and record['dose_uM'] > 0
            and type(protocol['duration_h']) in (int, float) and math.isfinite(protocol['duration_h'])
            and protocol['duration_h'] > 0, 'NOMINAL_DOSE_OBSERVATION_TIME_REQUIRED')


def context_key(record):
    return (record['source_id'], record['background_id'], record['protocol_id'], record['actual_protocol']['duration_h'])


def compile_records(requests, conditions, role_counts, links, splits):
    """Metadata-only join; control permissions belong to role-cell counts."""
    table = conditions.set_index('condition_id', drop=False)
    require(table.index.is_unique and not role_counts[['role', 'condition_id']].duplicated().any(), 'UNIQUE_ROLE_METADATA')
    counts = role_counts.set_index(['role', 'condition_id']).n_cells
    final = [deepcopy(r) for r in requests if r['role'] == 'final_test' and r['measurement_level'] == 'single_cell']
    development = [r for r in requests if r['role'] in ('fit', 'selection') and r['measurement_level'] == 'single_cell']
    expected = set(role_counts.loc[role_counts.role.eq('final_test'), 'condition_id'])
    expected = {cid for cid in expected if bool(table.loc[cid, 'eligible']) and table.loc[cid, 'condition_type'] == 'treated'}
    require(final and len({r['condition_id'] for r in final}) == len(final)
            and {r['condition_id'] for r in final} == expected, 'COMPLETE_NATIVE_FINAL_ROLE_REQUIRED')
    fit = [r for r in development if r['role'] == 'fit']
    development_groups = {r['split_group_id'] for r in development}
    fit_chemicals = {r['chemical_id'] for r in fit}
    development_chemicals = {r['chemical_id'] for r in development}
    fit_doses = {(r['source_id'], r['dose_uM']) for r in fit}
    fit_times = {(r['source_id'], r['actual_protocol']['duration_h']) for r in fit}
    fit_contexts = {context_key(r) for r in fit}
    split_table = splits.set_index('drug_id')
    gaps = []
    for r in final:
        partial_request(r)
        row = table.loc[r['condition_id']]
        require(row['role'] == 'final_test' and row['outer_split'] == r['outer_split'] == 'test'
                and r['cell_index_table'] == JOINT + '/test.parquet'
                and row['chemical_or_experiment_group_id'] == r['chemical_id']
                and row['split_group_id'] == r['split_group_id']
                and split_table.loc[row['drug_id'], 'role'] == 'final_test'
                and split_table.loc[row['drug_id'], 'split_group_id'] == r['split_group_id'], 'FROZEN_CHEMICAL_ROLE_BINDING')
        require(r['split_group_id'] not in development_groups and r['chemical_id'] not in development_chemicals,
                'FINAL_CHEMICAL_LEAKS_DEVELOPMENT')
        require(all(row[k] == r[k] for k in ('source_id', 'background_id', 'protocol_id'))
                and float(row['dose_um']) == r['dose_uM'] and float(row['duration_h']) == r['actual_protocol']['duration_h']
                and int(counts.loc[('final_test', r['condition_id'])]) == r['target_cells'] > 0,
                'ACTION_TARGET_SOURCE_BINDING')
        refs = r['physical_reference_ids']
        if not refs:
            gaps.append({'condition_id': r['condition_id'], 'gap': 'no_original_final_reference_links'})
        linked = set(links.loc[links.condition_id.eq(r['condition_id']), 'control_condition_id'])
        available = []
        for cid in refs:
            require(cid in linked and cid in table.index and table.loc[cid, 'condition_type'] == 'control', 'REGISTERED_REFERENCE_LINK')
            reference = table.loc[cid]
            require(all(reference[k] == row[k] for k in ('source_id', 'background_id', 'protocol_id', 'duration_h', 'plate')),
                    'SAME_SOURCE_ROLE_CONTROL_CONTEXT')
            if r['source_id'].startswith('sciPlex'):
                require(reference['replicate_id'] == row['replicate_id'], 'SOURCE_CONTROL_REPLICATE')
            key = ('final_test', cid)
            if key not in counts or int(counts.loc[key]) < 1:
                gaps.append({'condition_id': r['condition_id'], 'control_id': cid, 'gap': 'no_original_final_role_control_cells'})
                continue
            weight = r['reference_sampling_weights'][cid]
            require(weight['cells'] == int(counts.loc[key]) and weight['well_probability'] == 1 / len(refs)
                    and weight['cell_probability_given_well'] == 1 / int(counts.loc[key]), 'ORIGINAL_REFERENCE_SAMPLING_WEIGHTS')
            available.append(cid)
        r['evaluation_ready'] = bool(refs) and len(available) == len(refs)
        r['holdout_scope'] = {'chemical_unseen_from_fit': r['chemical_id'] not in fit_chemicals,
            'chemical_unseen_from_fit_and_selection': True,
            'dose_value_unseen_within_source': (r['source_id'], r['dose_uM']) not in fit_doses,
            'observation_hours_unseen_within_source': (r['source_id'], r['actual_protocol']['duration_h']) not in fit_times,
            'dose_holdout_at_seen_chemical': False, 'time_holdout_at_seen_chemical': False,
            'source_background_protocol_observation_context_seen_in_fit': context_key(r) in fit_contexts,
            'full_exposure_history_holdout_identifiable': False}
        r['fit_mean_baseline_status'] = 'ready_fit_context_metadata_only' if context_key(r) in fit_contexts else 'pending_no_fit_context_targets'
    return sorted(final, key=lambda r: r['condition_id']), gaps


class NativeFinalMetadata:
    """Source-bound final-only view; no TrainingData or native RNA reader here."""
    def __init__(self, config, root=ROOT):
        self.root = Path(root).resolve()
        self.config = learning.joint_config(config)
        base = self.root / JOINT
        require(sha(base / 'manifest.json') == producer.inputs.PINS[JOINT + '/manifest.json'], 'FROZEN_JOINT_MANIFEST')
        self.manifest = read(base / 'manifest.json')
        require(self.manifest['status'] == 'complete' and self.manifest['preprocessing']['fitted_role'] == 'fit'
                and self.manifest['bandwidth']['role'] == 'fit', 'FIT_ONLY_COORDINATES_REQUIRED')
        self.bindings = {str(base / 'manifest.json'): sha(base / 'manifest.json')}
        for name in METADATA_FILES:
            path = base / name
            expected = self.manifest['files'][name]['sha256']
            require(sha(path) == expected, 'FROZEN_METADATA_CHANGED: ' + name)
            self.bindings[str(path)] = expected
        native = self.root / NATIVE
        require(sha(native / 'manifest.json') == NATIVE_MANIFEST_SHA256, 'FROZEN_NATIVE_MANIFEST')
        self.native_manifest = read(native / 'manifest.json')
        self.bindings[str(native / 'manifest.json')] = sha(native / 'manifest.json')
        for name in ('requests.json', 'gene_axes.json', 'summary.json'):
            require(sha(native / name) == self.native_manifest['files'][name], 'NATIVE_METADATA_CHANGED: ' + name)
            self.bindings[str(native / name)] = self.native_manifest['files'][name]
        require(self.native_manifest['schema'] == 'native-response-units-v1'
                and all(self.native_manifest['source_bindings'].get(JOINT + '/' + name) == self.manifest['files'][name]['sha256']
                        for name in ('conditions.parquet', 'control_links.parquet', 'role_condition_counts.parquet', 'split_groups.parquet', 'genes.parquet')),
                'ORIGINAL_NATIVE_SOURCE_BINDING')
        conditions = pd.read_parquet(base / 'conditions.parquet')
        counts = pd.read_parquet(base / 'role_condition_counts.parquet')
        requests = read(native / 'requests.json')
        self.records, self.gaps = compile_records(requests, conditions, counts,
            pd.read_parquet(base / 'control_links.parquet'), pd.read_parquet(base / 'split_groups.parquet'))
        self.cell_index = pd.read_parquet(base / 'test.parquet', filters=[('role', '==', 'final_test')],
            columns=['cell_id', 'condition_id', 'role', 'matrix_path', 'matrix_row', 'source_id', 'background_id', 'protocol_id'])
        require(self.cell_index.role.eq('final_test').all() and self.cell_index.cell_id.is_unique
                and not self.cell_index[['matrix_path', 'matrix_row']].duplicated().any(), 'FINAL_ROLE_CELL_INDEX_IDENTITY')
        self.cell_ids_by_condition = {cid: frozenset(frame.cell_id.astype(str))
                                     for cid, frame in self.cell_index.groupby('condition_id', sort=False)}
        for record in self.records:
            target_ids = self.cell_ids_by_condition.get(record['condition_id'], frozenset())
            control_ids = set().union(*(self.cell_ids_by_condition.get(cid, frozenset()) for cid in record['physical_reference_ids']))
            require(len(target_ids) == record['target_cells'] and not target_ids & control_ids, 'FINAL_CONTROL_TARGET_METADATA_IDS')
            require(all(len(self.cell_ids_by_condition.get(cid, ())) == record['reference_sampling_weights'][cid]['cells']
                        for cid in record['physical_reference_ids']), 'FINAL_CONTROL_ROLE_CELL_COUNTS')
            record['allowed_final_target_ids_sha256'] = config_hash(sorted(target_ids))
            record['allowed_final_control_ids_sha256'] = config_hash(sorted(control_ids))
            record['allowed_final_control_cells'] = len(control_ids)
            record['cell_metadata_role'] = 'final_test'
        table = conditions.set_index('condition_id')
        for cid, frame in self.cell_index.groupby('condition_id', sort=False):
            require(all(frame[k].eq(table.loc[cid, k]).all() for k in ('source_id', 'background_id', 'protocol_id')),
                    'FINAL_CELL_SOURCE_CONTEXT_BINDING')
        chemicals = pd.read_parquet(base / 'chemicals.parquet').set_index('drug_id')
        table = conditions.set_index('condition_id')
        for record in self.records:
            row = table.loc[record['condition_id']]
            chemical = chemicals.loc[row['drug_id']]
            require(chemical['chemical_or_experiment_group_id'] == record['chemical_id'], 'REGISTERED_CHEMICAL_IDENTITY')
            record['registered_chemical_identity'] = clean({k: chemical[k] for k in
                ('drug_name', 'identity_status', 'parent_id', 'dose_basis', 'experimental_material_verified')})
            record['source_nominal_concentration'] = clean({'value': row['source_dose_value'], 'unit': row['source_dose_unit'],
                'nominal_uM': row['dose_um'], 'basis': chemical['dose_basis'], 'fingerprint_row': row['fingerprint_row']})
        self.fit_records = sorted((r for r in requests if r['role'] == 'fit' and r['measurement_level'] == 'single_cell'),
                                  key=lambda r: r['condition_id'])
        training = {r['condition_id']: r for r in requests if r['role'] in ('fit', 'selection') and r['measurement_level'] == 'single_cell'}
        self.protocol_identity = config_hash({'native_manifest': self.native_manifest, 'requests': training, 'coverage': learning.COVERAGE})
        self.identity = config_hash({'signature': self.manifest['signature'], 'files': self.manifest['files']})
        genes = pd.read_parquet(base / 'genes.parquet').ensembl_id.astype(str).tolist()
        require(len(genes) == len(set(genes)) == 32461 and genes == read(native / 'gene_axes.json')['joint_ensembl_32461'], 'ORDERED_GENE_AXIS')
        with np.load(base / 'normalization.npz', allow_pickle=False) as archive:
            self.mu, self.scale = archive['mean'].copy(), archive['scale'].copy()
            require(archive['gene_ids'].astype(str).tolist() == genes, 'NORMALIZATION_GENE_AXIS')
        self.n_genes = len(genes)
        require(self.mu.shape == self.scale.shape == (self.n_genes,) and np.isfinite(self.mu).all()
                and np.isfinite(self.scale).all() and (self.scale > 0).all(), 'FIT_NORMALIZATION_VALUES')
        with np.load(base / 'bandwidths.npz', allow_pickle=False) as archive:
            widths = archive['bandwidths'].copy()
        require(widths.shape == (5,) and np.isfinite(widths).all() and (widths > 0).all()
                and np.allclose(widths / widths[2], [.25, .5, 1., 2., 4.])
                and float(widths[2]) == self.manifest['bandwidth']['median'], 'FROZEN_FIT_BANDWIDTH')
        pair_conditions = set(pd.read_parquet(base / 'bandwidth_pairs.parquet', columns=['condition_id']).condition_id)
        require(pair_conditions.issubset(set(counts.loc[counts.role.eq('fit'), 'condition_id'])), 'BANDWIDTH_FIT_ROLE_ONLY')
        self.bandwidth = float(widths[2])
        vocabulary = {k: sorted(conditions[k].dropna().unique().tolist()) for k in ('source_id', 'background_id', 'protocol_id')}
        self.context = {'gene_ids': genes, 'mean': self.mu, 'scale': self.scale, 'context_vocabulary': vocabulary}
        for name in ('src/lung/engine/full_protocol_native_evaluation.py', 'src/lung/engine/full_protocol_response_learning.py',
                     'src/lung/engine/full_protocol_response.py', 'src/lung/data/reader.py', 'src/lung/data/preprocess.py',
                     'src/lung/data/bandwidth.py', 'src/lung/models/objectives.py', 'scripts/evaluate_full_protocol_native.py',
                     'configs/data_joint_sciplex_v1.json'):
            self.bindings[str(self.root / name)] = sha(self.root / name)

    def report(self):
        return {'schema': SCHEMA, 'status': 'label_free_final_metadata_verified_model_evaluation_pending',
            'conditions': len(self.records), 'by_source_observation_hours': dict(Counter(
                r['source_id'] + '|' + str(r['actual_protocol']['duration_h']) for r in self.records)),
            'independent_split_groups': len({r['split_group_id'] for r in self.records}),
            'final_role_cell_index_rows': len(self.cell_index),
            'final_cell_metadata_read': True, 'cell_metadata_is_RNA_values': False,
            'gene_count': self.n_genes, 'data_identity': self.identity, 'protocol_identity': self.protocol_identity,
            'context_vocabulary': self.context['context_vocabulary'], 'bindings': self.bindings,
            'source_gaps': self.gaps, 'evaluation_metadata_ready': all(r['evaluation_ready'] for r in self.records),
            'fit_bandwidth': self.bandwidth, 'preprocessing_role': 'fit', 'bandwidth_role': 'fit',
            'raw_units': 'source_raw_counts_no_new_normalization', 'evaluation_domain': producer.DOMAIN,
            'holdout_counts': {k: sum(r['holdout_scope'][k] for r in self.records) for k in self.records[0]['holdout_scope']},
            'exposure_history_known': False, 'fit_mean_baseline_scope': 'same fit source/background/protocol/observed-hour target conditions',
            'final_RNA_read': False, 'PT_loads': 0, 'model_forwards': 0, 'optimizer_steps': 0,
            'T_V_representation_cache_constructed': False, 'functional_Y_read': False, **NO_EFFECT_CLAIMS}


def evaluation_settings(config, *, reference_mean_sensitivity=False):
    require(type(config['seed']) is int and config['seed'] >= 0, 'ORIGINAL_EVALUATION_SEED')
    value = {'population': config['population'], 'steps': config['response']['inference_ode_steps'],
             'microbatch': config['response'].get('inference_microbatch', 1), 'seed': config['seed'] + 901,
             'fit_mean_samples_per_condition': 32, 'device': config['device'],
             'reference_mean_sensitivity': reference_mean_sensitivity, 'MMD_estimator': 'empirical_population_V_unbiased_false',
             'settings_version': 'fullP_native_final_fixed_population_v2',
             'sampling': 'original same-role uniform control-well then uniform cell; target uniform cells with replacement'}
    require(all(type(value[k]) is int and value[k] > 0 for k in ('population', 'steps', 'microbatch', 'fit_mean_samples_per_condition'))
            and type(value['seed']) is int and value['seed'] >= 0 and type(reference_mean_sensitivity) is bool, 'FIXED_EVALUATION_SETTINGS')
    return value


def stream_seed(seed, purpose, identity):
    return int.from_bytes(hashlib.sha256(f'{seed}|{purpose}|{identity}'.encode('utf8')).digest()[:8], 'little') % (2**63 - 1)


def completed_namespace(view, response_run):
    """All static checks precede PT. Only this explicit future branch loads PT."""
    run = Path(response_run).resolve(); assets = run.with_name(run.name + '-assets')
    require(run.is_relative_to(ROOT / 'experiments') and assets.is_relative_to(ROOT / 'experiments'), 'NEW_NAMESPACE_PROJECT_BOUNDARY')
    declaration = read(run / 'model_declaration.json')
    manifest, plan = read(run / 'run_manifest.json'), read(assets / 'run_manifest.json')
    _, _, _, bindings = producer.bound_inputs()
    producer.validate_metadata(declaration, bindings)
    meta = manifest['metadata']; ae = plan['AE_declaration']
    origin = Path(ae['weights']).resolve().parents[2]
    require(origin.is_relative_to(ROOT / 'experiments'), 'AE_SOURCE_PROJECT_BOUNDARY')
    ae_manifest = read(origin / 'run_manifest.json')
    require(meta['synthetic'] is False and meta['stage'] == producer.STAGE and meta['data_identity'] == view.identity
            and meta['full_response_bindings'] == bindings and meta['training_protocol_coverage'] == learning.COVERAGE
            and manifest['schema'] == learning.SCHEMA and manifest['protocol_identity'] == view.protocol_identity
            and plan['config'] == view.config and plan['protocol_identity'] == view.protocol_identity
            and plan['bindings'] == bindings
            and manifest['settings'] == plan['settings'] == learning.settings_from_config(view.config)
            and meta['reference_contract'] == producer.reference_contract(manifest['settings']['reference_variant']),
            'COMPLETED_MATCHING_NAMESPACE_METADATA')
    require(origin.is_relative_to(ROOT / 'experiments') and read(origin / 'AE_result.json') == ae
            and ae_manifest['schema'] == learning.SCHEMA + ':joint-AE'
            and ae_manifest['config'] == learning.joint_ae_config(view.config)
            and ae_manifest['data_identity'] == view.identity,
            'COMPLETED_SOURCE_AE_DECLARATION')
    learning.verify_completed_ae_result(ae, origin, view.identity)
    files = {run / 'model_declaration.json', run / 'run_manifest.json', assets / 'run_manifest.json',
             origin / 'AE_result.json', origin / 'run_manifest.json', run / 'final.pt',
             Path(ae['weights']).resolve(), Path(ae['final_checkpoint']).resolve()}
    require(Path(declaration['representation']['checkpoint']).resolve() == assets / 'representation_weights.pt'
            and Path(declaration['response']['checkpoint']).resolve() == run / 'response_weights.pt'
            and read(assets / 'representation_declaration.json') == declaration['representation']
            and meta['representation_hash'] == declaration['representation']['checkpoint_sha256'], 'NAMESPACE_SERVING_PATHS')
    files.add(assets / 'representation_declaration.json')
    for name in ('representation', 'response'):
        path = Path(declaration[name]['checkpoint']).resolve()
        require(sha(path) == declaration[name]['checkpoint_sha256'], 'SERVING_FILE_CHANGED_BEFORE_PT')
        files.add(path)
    before = {str(p): sha(p) for p in files}
    last = load_checkpoint(run / 'final.pt', config_hash(manifest))
    extra = last['extra']
    require(last['metadata'] == meta and extra.get('training_completion', {}).get('status') == 'complete'
            and extra.get('selected_weights_sha256') == declaration['response']['checkpoint_sha256']
            and type(extra.get('best_optimizer_step')) is int and extra['best_optimizer_step'] > 0
            and extra['best_optimizer_step'] == declaration['response']['optimizer_step']
            and type(extra.get('best_epoch')) is int and extra['best_epoch'] > 0,
            'ACTUAL_TRAINING_COMPLETION_PROOF')
    selected = torch.load(run / 'response_weights.pt', weights_only=True, map_location='cpu')
    require(selected['metadata'] == meta and selected['counters']['optimizer_step'] == extra['best_optimizer_step']
            and selected['epoch'] == extra['best_epoch'], 'ACTUAL_COMPLETED_SELECTED_SERVING')
    exported = torch.load(assets / 'representation_weights.pt', weights_only=True, map_location='cpu')
    require(exported['metadata'].get('source_checkpoint_sha256') == ae['checkpoint_sha256']
            and exported['metadata'].get('context_cardinality_expansion_changes_no_AE_tensor') is True,
            'ACTUAL_EXPORTED_COMPLETED_AE_SOURCE')
    model = producer.load_model(declaration, bindings, view.context, device=view.config['device'])
    require(model.reference_contract == meta['reference_contract'] == declaration['response']['reference_contract'],
            'ACTUAL_TRAINED_REFERENCE_CONTRACT')
    require(all(sha(p) == digest for p, digest in before.items()), 'NAMESPACE_CHANGED_DURING_MODEL_READ')
    return model, before


class FinalRNAData:
    """Constructed only after matching completed model proof in evaluate()."""
    batch = learning.NativeProtocolData.batch

    def __init__(self, view):
        self.data = EngineData(view.config, roles=('E',), allow_test=True)
        require(self.data.identity == view.identity and self.data.reader.allowed_roles == frozenset({'final_test'})
                and self.data.reader.allow_test is True
                and set(self.data.condition_ids('E')) == {r['condition_id'] for r in view.records}, 'FINAL_ONLY_READER')
        self.requests = {r['condition_id']: r for r in view.records}

    def action_features(self, condition):
        return self.data.action_features(condition)


@torch.no_grad()
def generate(model, data, condition, reference, settings, noise):
    """Same native partial batch as training; never the Viel validate_batch."""
    variant = producer.declared_reference_variant({'reference_contract': model.reference_contract})
    reference_h = producer.prepare_reference_h(model.shared, reference, variant,
        device=settings['device'], microbatch=settings['microbatch'])
    parts = []
    for start in range(0, len(reference), settings['microbatch']):
        current = reference[start:start + settings['microbatch']]
        numeric, physical = data.batch(condition, len(current), settings['device'])
        require(physical.known[:, :, 0].all() and not physical.known[:, :, 1:].any()
                and torch.isnan(physical.values[:, :, 1:]).all(), 'NATIVE_UNKNOWN_MASK_CHANGED')
        h = reference_h[start:start + len(current)]
        z = model.full.sample_latent(h, physical, noise=noise[start:start + len(current)], steps=settings['steps'], **numeric)
        raw = model.full.member.decode(z)
        result = physical_projection(raw, model.mean, model.scale)
        require(result.valid.all(), 'INVALID_GENERATED_MASS_NO_ROW_DROP_OR_FALLBACK')
        parts.append(result.x)
    return torch.cat(parts)


def allowed_fit_target_ids(view):
    """Lazy T cell metadata authentication; default final preflight stays E-only."""
    path = view.root / JOINT / 'train.parquet'
    binding = {'path': str(path), 'sha256': view.manifest['files']['train.parquet']['sha256']}
    require(sha(path) == binding['sha256'], 'ORIGINAL_FIT_CELL_METADATA_CHANGED')
    rows = pd.read_parquet(path, filters=[('role', '==', 'fit')], columns=['cell_id', 'condition_id', 'role'])
    require(rows.role.eq('fit').all() and rows.cell_id.is_unique and sha(path) == binding['sha256'], 'ORIGINAL_FIT_CELL_METADATA_IDENTITY')
    allowed = {cid: frozenset(frame.cell_id.astype(str)) for cid, frame in rows.groupby('condition_id', sort=False)}
    require({r['condition_id'] for r in view.fit_records}.issubset(allowed), 'ORIGINAL_FIT_TARGET_CONDITIONS')
    return allowed, binding


def fit_context_means(view, settings, output):
    """Future data-only baseline from T targets, no V/E cells or refitting."""
    from lung.data.preprocess import standardized
    data = EngineData(view.config, roles=('T',), allow_test=False)
    require(data.identity == view.identity and data.reader.allowed_roles == frozenset({'fit'})
            and data.reader.allow_test is False, 'BASELINE_FIT_ONLY_READER')
    allowed_fit, fit_index = allowed_fit_target_ids(view)
    groups = defaultdict(list)
    for r in view.fit_records:
        require(r['role'] == 'fit', 'BASELINE_SOURCE_ROLE')
        groups[context_key(r)].append(r)
    needed = {context_key(r) for r in view.records}
    means, receipts = {}, []
    for context in sorted(needed):
        records = groups.get(context, [])
        if not records:
            receipts.append({'context': list(context), 'status': 'pending_no_fit_context_targets'})
            continue
        total = np.zeros(view.n_genes, dtype=np.float64); sampled = []
        for r in records:
            pool = data.reader.role_cells(r['condition_id'], role='fit')
            require(len(pool) > 0 and pool.role.eq('fit').all(), 'BASELINE_FIT_TARGET_ROWS')
            rng = np.random.default_rng(stream_seed(settings['seed'], 'fit-mean', r['condition_id']))
            rows = pool.iloc[rng.integers(len(pool), size=settings['fit_mean_samples_per_condition'])]
            require(set(rows.cell_id.astype(str)).issubset(allowed_fit[r['condition_id']]), 'BASELINE_ORIGINAL_FIT_TARGET_IDS')
            values = standardized(data.reader.rows.read(rows), view.mu, view.scale)
            require(np.isfinite(values).all(), 'BASELINE_NONFINITE')
            total += values.astype(np.float64).mean(0)
            sampled.append({'condition_id': r['condition_id'], 'cell_ids': rows.cell_id.astype(str).tolist(), 'role': 'fit'})
        key = config_hash(list(context))
        means[context] = (total / len(records)).astype(np.float32)
        np.save(output / ('fit-mean-' + key + '.npy'), means[context], allow_pickle=False)
        receipts.append({'context': list(context), 'status': 'actual_fit_context_target_mean', 'conditions': len(records),
            'samples_per_condition': settings['fit_mean_samples_per_condition'], 'sampled_fit_rows': sampled,
            'sampling': 'equal fit target-condition means; uniform cells with replacement', 'fit_mean_key': key,
            'original_fit_cell_index': fit_index,
            'array_sha256': sha(output / ('fit-mean-' + key + '.npy')), 'final_or_selection_rows_used': False})
    write(output / 'fit_baseline_receipts.json', receipts)
    require(sha(fit_index['path']) == fit_index['sha256'], 'BASELINE_FIT_METADATA_CHANGED_DURING_READ')
    return means


def metrics(predicted, observed, bandwidth):
    require(predicted.ndim == observed.ndim == 2 and predicted.shape[1] == observed.shape[1]
            and torch.isfinite(predicted).all() and torch.isfinite(observed).all(), 'FINITE_ORDERED_METRIC_ARRAYS')
    return {'standardized_mean_mse': float((predicted.mean(0) - observed.mean(0)).square().mean()),
            'standardized_population_mmd_v': float(rbf_mmd(predicted, observed, bandwidth, unbiased=False))}


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['split_group_id']].append(row)
    methods = sorted(set.intersection(*(set(row['methods']) for row in rows)))
    return {'conditions': len(rows), 'split_components': len(grouped),
        'group_equal_metrics': {method: {metric: float(np.mean([np.mean([r['methods'][method][metric] for r in group])
            for group in grouped.values()])) for metric in METRICS} for method in methods}}


def evaluate(config, response_run, output, *, reference_mean_sensitivity=False):
    """Explicit future real model/final RNA execution. No optimizer or fitting."""
    view = NativeFinalMetadata(config)
    settings = evaluation_settings(view.config, reference_mean_sensitivity=reference_mean_sensitivity)
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / 'experiments') and not output.exists(), 'NEW_EVALUATION_OUTPUT_REQUIRED')
    require(all(r['evaluation_ready'] for r in view.records), 'SOURCE_FINAL_CONTROL_GAPS')
    output.mkdir(parents=True)
    plan = {'schema': SCHEMA, 'metadata_view': view.report(), 'settings': settings, 'response_run': str(Path(response_run).resolve()),
            'test_used_for_selection': False, 'normalization_or_bandwidth_refitted': False,
            'reference_contract_status': 'pending_actual_completed_model_proof',
            'three_variant_comparison_complete': False, **NO_EFFECT_CLAIMS}
    write(output / 'request.json', plan)
    try:
        model, model_hashes = completed_namespace(view, response_run)
        plan.update(reference_contract_status='actual_completed_matching_model_verified',
            trained_reference_contract=model.reference_contract,
            independently_trained_zero_h_model=model.reference_contract['variant'] == 'zero_h')
        write(output / 'plan.json', plan)
        fit_means = fit_context_means(view, settings, output)
        data = FinalRNAData(view)
        rows = []
        with torch.no_grad():
            for r in view.records:
                cid = r['condition_id']
                rng = np.random.default_rng(stream_seed(settings['seed'], 'E-response', cid))
                drawn = data.data.sample_response(cid, settings['population'], rng, role='E')
                x, y = tensor(drawn['control'], settings['device']), tensor(drawn['target'], settings['device'])
                require(drawn['role'] == 'final_test' and not set(drawn['control_cell_ids']) & set(drawn['target_cell_ids']), 'FINAL_SAMPLE_ROLE_IDS')
                allowed_control_ids = set().union(*(view.cell_ids_by_condition[cid] for cid in r['physical_reference_ids']))
                require(set(drawn['control_cell_ids']).issubset(allowed_control_ids)
                        and set(drawn['target_cell_ids']).issubset(view.cell_ids_by_condition[cid]), 'ACTUAL_SAMPLED_IDS_FINAL_ROLE_BOUND')
                noise = torch.randn((len(x), model.full.member.config.latent_dim), device=settings['device'],
                    generator=torch.Generator(device=settings['device']).manual_seed(stream_seed(settings['seed'], 'P-noise', cid)))
                predicted = {'full_P': generate(model, data, cid, x, settings, noise), 'reference': x}
                mean = fit_means.get(context_key(r))
                if mean is not None:
                    predicted['fit_context_mean'] = tensor(mean, settings['device']).expand(len(y), -1)
                if settings['reference_mean_sensitivity']:
                    predicted['full_P_expression_mean_sensitivity'] = generate(model, data, cid, x.mean(0).expand_as(x), settings, noise)
                current = {'condition_id': cid, 'chemical_id': r['chemical_id'], 'split_group_id': r['split_group_id'],
                    'source_id': r['source_id'], 'observation_hours': r['actual_protocol']['duration_h'], 'dose_uM': r['dose_uM'],
                    'holdout_scope': r['holdout_scope'], 'raw_source_units': r['units'], 'metric_domain': producer.DOMAIN,
                    'methods': {name: metrics(value, y, view.bandwidth) for name, value in predicted.items()},
                    'reference_cell_ids': drawn['control_cell_ids'].tolist(), 'target_cell_ids': drawn['target_cell_ids'].tolist(),
                    'sampling_seed': stream_seed(settings['seed'], 'E-response', cid), 'noise_seed': stream_seed(settings['seed'], 'P-noise', cid),
                    'fit_mean_status': 'actual_fit_only_mean' if mean is not None else 'pending_no_fit_context_targets',
                    'same_cell_t0': False, 'individual_pairing': False, 'state_increment_conclusion': False,
                    'sensitivity_is_independent_trained_stateless_baseline': False}
                current['trained_reference_contract'] = model.reference_contract
                name = 'gene-moments-' + config_hash(cid) + '.npz'
                arrays = {}
                for method, values in {**predicted, 'observed': y}.items():
                    arrays[method + '_mean_standardized'] = values.mean(0).cpu().numpy()
                    arrays[method + '_mean_log1p_library10000'] = (values * model.scale + model.mean).mean(0).cpu().numpy()
                    arrays[method + '_variance_standardized'] = values.var(0, correction=0).cpu().numpy()
                np.savez_compressed(output / name, **arrays)
                current.update(gene_moments_file=name, gene_moments_sha256=sha(output / name))
                write(output / ('condition-' + config_hash(cid) + '.json'), current)
                rows.append(current)
        require(all(sha(p) == digest for p, digest in {**view.bindings, **model_hashes}.items()), 'FROZEN_ASSET_CHANGED_DURING_EVALUATION')
        by_source_time = defaultdict(list)
        for row in rows:
            by_source_time[row['source_id'] + '|' + str(row['observation_hours'])].append(row)
        summary = aggregate(rows)
        write(output / 'gene_ids.json', view.context['gene_ids'])
        result = {'schema': SCHEMA, 'status': 'complete_actual_native_final_expression_evaluation', 'conditions': len(rows),
            'independent_split_groups': summary['split_components'], 'group_equal_metrics': summary['group_equal_metrics'], 'settings': settings,
            'by_source_observation_hours': {key: aggregate(subset) for key, subset in by_source_time.items()},
            'frozen_model_hashes': model_hashes, 'data_identity': view.identity, 'final_RNA_read': True,
            'trained_reference_contract': model.reference_contract,
            'independently_trained_zero_h_model': model.reference_contract['variant'] == 'zero_h',
            'three_variant_comparison_complete': False, 'synthetic': False,
            'actual_P_population_predictions': len(rows) * (1 + int(settings['reference_mean_sensitivity'])),
            'observed_targets_used_only_for_metrics': True, 'optimizer_steps': 0, 'test_used_for_selection': False,
            'normalization_or_bandwidth_refitted': False, 'T_V_representation_cache_constructed': False,
            'functional_Y_read': False, 'inference_sensitivity_is_state_increment_experiment': False, **NO_EFFECT_CLAIMS}
        result['files'] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        write(output / 'result.json', result)
        return result
    except BaseException as error:
        write(output / 'failure.json', {'schema': SCHEMA, 'error_type': type(error).__name__, 'error': str(error),
            'completed_evaluation': False, 'partial_outputs_preserved': True, 'optimizer_steps': 0, **NO_EFFECT_CLAIMS})
        raise


def without_reference_variant(config):
    """Only the registered P input ablation may differ between compared fits."""
    value = deepcopy(config)
    research = value.get('research', {})
    if 'full_protocol_response' in research:
        research['full_protocol_response'].pop('reference_variant', None)
        if not research['full_protocol_response']:
            research.pop('full_protocol_response')
    if 'research' in value and not research:
        value.pop('research')
    return value


def comparison_namespace(view, plan, result):
    """Read closed namespace JSON and recorded PT identities, never PT bytes."""
    run = Path(plan['response_run']).resolve(); assets = run.with_name(run.name + '-assets')
    require(run.is_relative_to(ROOT / 'experiments'), 'COMPARISON_NAMESPACE_BOUNDARY')
    manifest, declared, training = read(run / 'run_manifest.json'), read(run / 'model_declaration.json'), read(assets / 'run_manifest.json')
    meta, settings, ae = manifest['metadata'], manifest['settings'], training['AE_declaration']
    origin = Path(ae['weights']).resolve().parents[2]
    require(origin.is_relative_to(ROOT / 'experiments'), 'COMPARISON_AE_BOUNDARY')
    ae_manifest = read(origin / 'run_manifest.json')
    files = {run / 'model_declaration.json', run / 'run_manifest.json', assets / 'run_manifest.json',
             assets / 'representation_declaration.json', origin / 'AE_result.json', origin / 'run_manifest.json'}
    recorded = result['frozen_model_hashes']
    require(all(recorded.get(str(p)) == sha(p) for p in files), 'COMPARISON_CLOSED_NAMESPACE_JSON_CHANGED')
    producer.validate_metadata(declared, meta['full_response_bindings'])
    contract = result['trained_reference_contract']
    variant = producer.declared_reference_variant({'reference_contract': contract})
    require(meta['synthetic'] is False and meta['stage'] == producer.STAGE and meta['data_identity'] == view.identity
            and declared['bindings'] == training['bindings'] == meta['full_response_bindings']
            and declared['bindings']['data_identity'] == view.identity
            and manifest['schema'] == learning.SCHEMA and manifest['protocol_identity'] == training['protocol_identity'] == view.protocol_identity
            and meta['training_protocol_coverage'] == learning.COVERAGE
            and settings == training['settings'] == learning.settings_from_config(training['config'])
            and settings['reference_variant'] == variant
            and meta['reference_contract'] == declared['response']['reference_contract'] == plan['trained_reference_contract'] == contract
            and without_reference_variant(training['config']) == without_reference_variant(view.config),
            'COMPARISON_MATCHING_TRAINED_VARIANT_METADATA')
    require(ae == read(origin / 'AE_result.json') and ae['schema'] == learning.AE_RESULT_SCHEMA
            and ae['data_identity'] == view.identity and ae['training_completion']['status'] == 'complete'
            and ae_manifest['schema'] == learning.SCHEMA + ':joint-AE'
            and ae_manifest['config'] == learning.joint_ae_config(training['config'])
            and ae_manifest['data_identity'] == view.identity, 'COMPARISON_ORIGINAL_COMPLETED_AE')
    expected_pt = {run / 'final.pt', run / 'response_weights.pt', assets / 'representation_weights.pt',
                   Path(ae['weights']).resolve(), Path(ae['final_checkpoint']).resolve()}
    require(all(p.is_relative_to(ROOT / 'experiments') and isinstance(recorded.get(str(p)), str)
                and len(recorded[str(p)]) == 64 for p in expected_pt), 'COMPARISON_RECORDED_PT_IDENTITIES')
    require(Path(declared['representation']['checkpoint']).resolve() == assets / 'representation_weights.pt'
            and Path(declared['response']['checkpoint']).resolve() == run / 'response_weights.pt'
            and read(assets / 'representation_declaration.json') == declared['representation']
            and recorded[str(assets / 'representation_weights.pt')] == declared['representation']['checkpoint_sha256'] == meta['representation_hash']
            and recorded[str(run / 'response_weights.pt')] == declared['response']['checkpoint_sha256']
            and recorded[str(Path(ae['weights']).resolve())] == ae['checkpoint_sha256']
            and recorded[str(Path(ae['final_checkpoint']).resolve())] == ae['final_checkpoint_sha256'], 'COMPARISON_RECORDED_MODEL_SOURCE_BINDING')
    family = {'original_AE_run': str(origin), 'AE_declaration': ae, 'AE_manifest': ae_manifest,
        'training_config': without_reference_variant(training['config']), 'training_settings': {k: v for k, v in settings.items() if k != 'reference_variant'},
        'P_metadata': {k: v for k, v in meta.items() if k not in ('reference_contract', 'representation_hash')}, 'cache_encoder': manifest['cache_encoder'],
        'representation_declaration': {k: v for k, v in declared['representation'].items() if k not in ('checkpoint', 'checkpoint_sha256')}}
    return variant, family, {str(p): recorded[str(p)] for p in files}


def read_completed_evaluation(view, directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / 'experiments'), 'COMPARISON_EVALUATION_BOUNDARY')
    result, plan = read(directory / 'result.json'), read(directory / 'plan.json')
    count = len(view.records)
    settings = evaluation_settings(view.config, reference_mean_sensitivity=result['settings']['reference_mean_sensitivity'])
    require(result['schema'] == plan['schema'] == SCHEMA and result['status'] == 'complete_actual_native_final_expression_evaluation'
            and result['synthetic'] is False and result['final_RNA_read'] is True and result['data_identity'] == view.identity
            and type(result['optimizer_steps']) is int and result['optimizer_steps'] == 0
            and result['test_used_for_selection'] is False and result['normalization_or_bandwidth_refitted'] is False
            and result['T_V_representation_cache_constructed'] is False and result['functional_Y_read'] is False
            and result['observed_targets_used_only_for_metrics'] is True and result['conditions'] == count
            and result['actual_P_population_predictions'] == count * (1 + int(settings['reference_mean_sensitivity']))
            and result['settings'] == plan['settings'] == settings and plan['metadata_view'] == view.report()
            and all(result[k] is False and plan[k] is False for k in NO_EFFECT_CLAIMS)
            and result['inference_sensitivity_is_state_increment_experiment'] is False
            and plan['reference_contract_status'] == 'actual_completed_matching_model_verified', 'COMPARISON_COMPLETED_REAL_EVALUATION')
    files = result['files']
    require(all(Path(name).name == name and Path(name).suffix in ('.json', '.npy', '.npz')
                and name != 'result.json' for name in files), 'COMPARISON_REPORT_ARRAY_FILES_ONLY_NO_PT')
    require(all(sha(directory / name) == digest for name, digest in files.items()),
            'COMPARISON_RESULT_FILE_CHANGED')
    require(set(files) == {p.name for p in directory.iterdir() if p.is_file() and p.name != 'result.json'}
            and {'request.json', 'plan.json', 'gene_ids.json', 'fit_baseline_receipts.json'}.issubset(files), 'COMPARISON_COMPLETE_FILE_MANIFEST')
    require(read(directory / 'gene_ids.json') == view.context['gene_ids'], 'COMPARISON_ORDERED_GENE_IDS')
    baseline_receipts = read(directory / 'fit_baseline_receipts.json')
    allowed_fit, fit_index = allowed_fit_target_ids(view)
    fits = {r['condition_id']: r for r in view.fit_records}
    contexts = {context_key(r) for r in view.records}
    require(len(baseline_receipts) == len(contexts) and {tuple(r['context']) for r in baseline_receipts} == contexts,
            'COMPARISON_COMPLETE_FIT_BASELINE_CONTEXTS')
    baseline_means = {}
    for receipt in baseline_receipts:
        context = tuple(receipt['context'])
        expected = {cid for cid, r in fits.items() if context_key(r) == context}
        if not expected:
            require(receipt['status'] == 'pending_no_fit_context_targets', 'COMPARISON_MISSING_FIT_BASELINE_NOT_INVENTED')
            continue
        sampled = receipt['sampled_fit_rows']; key = config_hash(list(context)); name = 'fit-mean-' + key + '.npy'
        require(receipt['status'] == 'actual_fit_context_target_mean' and receipt['final_or_selection_rows_used'] is False
                and receipt['conditions'] == len(expected) and receipt['samples_per_condition'] == settings['fit_mean_samples_per_condition']
                and len(sampled) == len(expected) and {r['condition_id'] for r in sampled} == expected
                and all(r['role'] == fits[r['condition_id']]['role'] == 'fit'
                        and len(r['cell_ids']) == settings['fit_mean_samples_per_condition']
                        and all(isinstance(cid, str) and cid for cid in r['cell_ids'])
                        and set(r['cell_ids']).issubset(allowed_fit[r['condition_id']]) for r in sampled)
                and receipt['original_fit_cell_index'] == fit_index
                and receipt['fit_mean_key'] == key and receipt['array_sha256'] == files[name], 'COMPARISON_ACTUAL_FIT_ONLY_BASELINE')
        baseline_means[context] = np.load(directory / name, allow_pickle=False)
        require(baseline_means[context].shape == (view.n_genes,) and np.isfinite(baseline_means[context]).all(), 'COMPARISON_FIT_MEAN_ARRAY')
    expected_files = {'request.json', 'plan.json', 'gene_ids.json', 'fit_baseline_receipts.json'}
    expected_files.update('fit-mean-' + config_hash(list(c)) + '.npy' for c in baseline_means)
    expected_files.update(prefix + config_hash(r['condition_id']) + suffix for r in view.records
                          for prefix, suffix in (('condition-', '.json'), ('gene-moments-', '.npz')))
    require(set(files) == expected_files, 'COMPARISON_NO_MISSING_OR_EXTRA_CONDITION_FILES')
    variant, family, model_json = comparison_namespace(view, plan, result)
    require(result['independently_trained_zero_h_model'] is (variant == 'zero_h')
            and plan['independently_trained_zero_h_model'] is (variant == 'zero_h'), 'COMPARISON_ACTUAL_ZERO_H_TRAINING_FLAG')
    rows, moments = [], {}
    for r in view.records:
        cid = r['condition_id']; row = read(directory / ('condition-' + config_hash(cid) + '.json'))
        require(row['condition_id'] == cid and all(row[k] == r[k] for k in ('chemical_id', 'split_group_id', 'source_id', 'dose_uM', 'holdout_scope'))
                and row['observation_hours'] == r['actual_protocol']['duration_h'] and row['raw_source_units'] == r['units']
                and row['metric_domain'] == producer.DOMAIN and row['trained_reference_contract'] == result['trained_reference_contract']
                and row['sampling_seed'] == stream_seed(settings['seed'], 'E-response', cid)
                and row['noise_seed'] == stream_seed(settings['seed'], 'P-noise', cid), 'COMPARISON_CONDITION_SOURCE_SEED')
        require(all(row[k] is False for k in ('same_cell_t0', 'individual_pairing', 'state_increment_conclusion',
                'sensitivity_is_independent_trained_stateless_baseline')), 'COMPARISON_SOURCE_INFERENCE_SEMANTICS')
        refs, targets = row['reference_cell_ids'], row['target_cell_ids']
        allowed = set().union(*(view.cell_ids_by_condition[x] for x in r['physical_reference_ids']))
        require(len(refs) == len(targets) == settings['population'] and set(refs).issubset(allowed)
                and set(targets).issubset(view.cell_ids_by_condition[cid]) and not set(refs) & set(targets), 'COMPARISON_ACTUAL_FINAL_IDS')
        name = 'gene-moments-' + config_hash(cid) + '.npz'
        require(row['gene_moments_file'] == name and row['gene_moments_sha256'] == files[name], 'COMPARISON_MOMENT_SOURCE_BINDING')
        actual_mean = context_key(r) in baseline_means
        require(row['fit_mean_status'] == ('actual_fit_only_mean' if actual_mean else 'pending_no_fit_context_targets'), 'COMPARISON_BASELINE_AVAILABILITY')
        methods = {'full_P', 'reference'} | ({'fit_context_mean'} if actual_mean else set())
        if settings['reference_mean_sensitivity']:
            methods.add('full_P_expression_mean_sensitivity')
        require(set(row['methods']) == methods and all(set(values) == set(METRICS)
                and all(type(v) in (int, float) and math.isfinite(v) for v in values.values())
                and values['standardized_mean_mse'] >= 0 for values in row['methods'].values()), 'COMPARISON_FINITE_STANDARDIZED_METRICS')
        with np.load(directory / name, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        require(set(arrays) == {method + suffix for method in methods | {'observed'}
                for suffix in ('_mean_standardized', '_mean_log1p_library10000', '_variance_standardized')}
                and all(x.shape == (view.n_genes,) and np.isfinite(x).all() for x in arrays.values()), 'COMPARISON_FULL_GENE_MOMENTS')
        require(all((arrays[method + '_variance_standardized'] >= 0).all()
                    and np.allclose(arrays[method + '_mean_log1p_library10000'],
                                    arrays[method + '_mean_standardized'] * view.scale + view.mu, rtol=2e-5, atol=1e-5)
                    for method in methods | {'observed'}), 'COMPARISON_MOMENT_COORDINATES')
        if actual_mean:
            require(np.allclose(arrays['fit_context_mean_mean_standardized'], baseline_means[context_key(r)], rtol=2e-5, atol=1e-6),
                    'COMPARISON_FIT_MEAN_MOMENT_BINDING')
        for method in methods:
            mse = float(np.mean((arrays[method + '_mean_standardized'] - arrays['observed_mean_standardized']) ** 2))
            require(math.isclose(mse, row['methods'][method]['standardized_mean_mse'], rel_tol=2e-5, abs_tol=1e-8), 'COMPARISON_MSE_MOMENT_READBACK')
        rows.append(row); moments[cid] = arrays
    summary = aggregate(rows)
    require(result['independent_split_groups'] == summary['split_components']
            and result['group_equal_metrics'] == summary['group_equal_metrics'], 'COMPARISON_GROUP_AGGREGATE_READBACK')
    identity = {'files': {str(directory / name): digest for name, digest in files.items()}, 'model_JSON': model_json,
                'result_sha256': sha(directory / 'result.json'), 'original_fit_cell_index': fit_index,
                'model_weight_bytes_read_during_comparison': False}
    return {'directory': str(directory), 'variant': variant, 'family': family, 'settings': settings, 'rows': rows,
            'moments': moments, 'fit_baseline_receipts': baseline_receipts, 'identity': identity}


def paired_variant_comparison(readbacks):
    require(len(readbacks) == 3 and {r['variant'] for r in readbacks} == {'observed', 'zero_h', 'mean_h'}
            and len({r['directory'] for r in readbacks}) == 3, 'COMPARISON_THREE_DISTINCT_TRAINED_VARIANTS')
    runs = {r['variant']: r for r in readbacks}; base = runs['observed']
    require(all(r['family'] == base['family'] and r['settings'] == base['settings']
                and r['fit_baseline_receipts'] == base['fit_baseline_receipts'] for r in readbacks), 'COMPARISON_SAME_AE_FIT_SELECTION_INFORMATION')
    paired, metric_rows = [], []
    for i, reference in enumerate(base['rows']):
        cid = reference['condition_id']; current = {}
        invariant = {k: v for k, v in reference.items() if k not in ('methods', 'trained_reference_contract', 'gene_moments_sha256')}
        for variant, run in runs.items():
            row = run['rows'][i]
            require(invariant == {k: v for k, v in row.items() if k not in ('methods', 'trained_reference_contract', 'gene_moments_sha256')},
                    'COMPARISON_SAME_CONDITION_DRAW_NOISE')
            baseline = {k: v for k, v in row['methods'].items() if k not in ('full_P', 'full_P_expression_mean_sensitivity')}
            require(baseline == {k: v for k, v in reference['methods'].items() if k not in ('full_P', 'full_P_expression_mean_sensitivity')},
                    'COMPARISON_SAME_BASELINE_METRICS')
            for name, values in base['moments'][cid].items():
                if not name.startswith('full_P'):
                    require(np.array_equal(values, run['moments'][cid][name]), 'COMPARISON_SAME_REFERENCE_TARGET_BASELINE_MOMENTS')
            current[variant] = row['methods']['full_P']
        differences = {other: {metric: current['observed'][metric] - current[other][metric] for metric in METRICS}
                       for other in ('zero_h', 'mean_h')}
        paired.append({'condition_id': cid, 'split_group_id': reference['split_group_id'], 'source_id': reference['source_id'],
            'observation_hours': reference['observation_hours'], 'metrics_by_trained_variant': current,
            'observed_minus_other': differences})
        metric_rows.append({'split_group_id': reference['split_group_id'], 'methods': {**current,
            **{'observed_minus_' + k: v for k, v in differences.items()}, **baseline}})
    groups = {group: aggregate([row for row in metric_rows if row['split_group_id'] == group])['group_equal_metrics']
              for group in sorted({row['split_group_id'] for row in metric_rows})}
    return {'paired_conditions': paired, 'per_split_component_metrics': groups, **aggregate(metric_rows)}


def compare(config, evaluation_directories, output=None):
    """Readonly completed-result comparison: no PT, RNA, fitting or selection."""
    view = NativeFinalMetadata(config)
    directories = [Path(p).resolve() for p in evaluation_directories]
    require(len(directories) == len(set(directories)) == 3, 'COMPARISON_THREE_OUTPUT_DIRECTORIES')
    require(all(p.is_relative_to(ROOT / 'experiments') for p in directories), 'COMPARISON_INPUT_PROJECT_BOUNDARY')
    output = Path(output).resolve() if output is not None else None
    if output is not None:
        require(output.is_relative_to(ROOT / 'experiments') and not output.exists(), 'NEW_COMPARISON_OUTPUT_REQUIRED')
        output.mkdir(parents=True)
    boundary = {'schema': COMPARISON_SCHEMA, 'evaluation_directories': list(map(str, directories)),
        'PT_loads': 0, 'model_forwards': 0, 'RNA_values_read_during_comparison': False, 'optimizer_steps': 0,
        'model_selection_or_refit': False, 'statistical_CI_or_test': False, **NO_EFFECT_CLAIMS}
    try:
        missing = [str(p / name) for p in directories for name in ('result.json', 'plan.json') if not (p / name).is_file()]
        if missing:
            result = {**boundary, 'status': 'pending_three_actual_completed_native_final_results',
                      'three_variant_comparison_complete': False, 'missing_actual_outputs': missing}
        else:
            runs = [read_completed_evaluation(view, p) for p in directories]
            result = {**boundary, 'status': 'complete_actual_three_variant_result_readback', 'three_variant_comparison_complete': True,
                'metadata_view': view.report(), 'comparison': paired_variant_comparison(runs),
                'source_results': {r['variant']: r['identity'] for r in runs}, 'same_original_AE_fit_selection_and_final_draw_verified': True,
                'difference_direction': 'observed minus other; negative favors lower discrepancy',
                'MMD_readback': 'closed original empirical V values; moments cannot reconstruct population MMD'}
            closed = {p: digest for r in runs for p, digest in {**r['identity']['files'], **r['identity']['model_JSON']}.items()}
            closed.update({str(p / 'result.json'): r['identity']['result_sha256'] for p, r in zip(directories, runs)})
            closed.update({r['identity']['original_fit_cell_index']['path']: r['identity']['original_fit_cell_index']['sha256'] for r in runs})
            require(all(sha(p) == digest for p, digest in {**view.bindings, **closed}.items()), 'COMPARISON_ASSET_CHANGED_DURING_READ')
        if output is not None:
            write(output / 'result.json', result)
        return result
    except BaseException as error:
        if output is not None:
            write(output / 'failure.json', {**boundary, 'error_type': type(error).__name__, 'error': str(error),
                  'three_variant_comparison_complete': False, 'partial_outputs_preserved': True})
        raise
