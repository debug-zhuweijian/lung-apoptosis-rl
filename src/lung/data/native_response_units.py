"""Frozen label-free native expression requests; no model or functional Y access."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

JOINT = '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1'
MARTIN = '08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1'
ROUND12 = 'experiments/20261010-data-alignment-round12'
SCHEMA = 'native-response-units-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return None if pd.isna(value) else value.item() if hasattr(value, 'item') else value


def build(root, output):
    """Metadata-only: hash bytes, read metadata, never deserialize expression/Y."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('NEW_VERSION_REQUIRED')
    bindings = {}
    def bind(relative, expected=None):
        relative = relative.replace('\\', '/')
        digest = sha(root / relative)
        if expected and digest != expected:
            raise ValueError('SOURCE_HASH_MISMATCH: ' + relative)
        bindings[relative] = digest
        return root / relative
    state = json.loads(bind(ROUND12 + '/state_source_manifest.json').read_text(encoding='utf-8'))
    for name in ('state_condition_inputs.csv', 'state_contexts.csv', 'state_action_overlap.csv'):
        bind(ROUND12 + '/' + name, state['outputs'][name]['sha256'])
    joint = json.loads(bind(JOINT + '/manifest.json', state['joint_manifest_sha256']).read_text(encoding='utf-8'))
    for name in ('conditions.parquet', 'control_links.parquet', 'role_condition_counts.parquet', 'split_groups.parquet', 'genes.parquet', 'source_shards.parquet', 'train.parquet', 'val.parquet', 'test.parquet'):
        bind(JOINT + '/' + name, joint['files'][name]['sha256'])
    source = pd.read_csv(root / ROUND12 / 'state_condition_inputs.csv')
    conditions = pd.read_parquet(root / JOINT / 'conditions.parquet').set_index('condition_id')
    counts = pd.read_parquet(root / JOINT / 'role_condition_counts.parquet').set_index(['role', 'condition_id']).n_cells
    genes = pd.read_parquet(root / JOINT / 'genes.parquet')
    axis = genes.ensembl_id.astype(str).tolist()
    if len(axis) != len(set(axis)):
        raise ValueError('GENE_IDENTITY_NOT_UNIQUE')
    requests = []
    for row in source.to_dict('records'):
        refs = json.loads(row['available_control_condition_ids_json'])
        if conditions.loc[row['condition_id'], 'role'] != row['role']:
            raise ValueError('ROLE_IDENTITY_MISMATCH')
        if not refs or int(counts.loc[(row['role'], row['condition_id'])]) != int(row['available_target_cells']):
            raise ValueError('TARGET_OR_REFERENCE_COUNT_MISMATCH')
        weights = {}
        for cid in refs:
            ref = conditions.loc[cid]
            for field in ('source_id', 'background_id', 'protocol_id', 'duration_h', 'plate'):
                if ref[field] != row[field]:
                    raise ValueError('REFERENCE_SCOPE_MISMATCH')
            if row['source_id'].startswith('sciPlex') and ref['replicate_id'] != row['replicate_id']:
                raise ValueError('REFERENCE_REPLICATE_MISMATCH')
            n = int(counts.loc[(row['role'], cid)])
            if n < 1:
                raise ValueError('EMPTY_REFERENCE')
            weights[cid] = {'well_probability': 1 / len(refs), 'cells': n, 'cell_probability_given_well': 1 / n}
        protocol = {key: clean(row[key]) for key in ('duration_h', 'source_time_h', 'reported_treatment_hours', 'time_semantics', 'exposure_duration_h', 'withdrawal_protocol')}
        protocol.update({'prior_drug': None, 'prior_drug_unknown': True, 'washout_h': None, 'washout_unknown': True,
                         'withdrawal_unknown': protocol['withdrawal_protocol'] in (None, 'not_recorded', 'unknown'), 'exposure_duration_unknown': protocol['exposure_duration_h'] is None})
        requests.append({'request_id': 'native:' + row['condition_id'], 'role': row['role'], 'fold_id': None,
            'source_id': row['source_id'], 'background_id': row['background_id'], 'protocol_id': row['protocol_id'],
            'reference_context_id': row['context_id'], 'reference_semantics': 'contemporaneous_vehicle_not_same_cell_t0',
            'measurement_level': 'single_cell', 'decoder_domain': 'singlecell_raw_counts_common_ensembl',
            'gene_axis_id': 'joint_ensembl_32461', 'units': 'source_raw_counts_no_new_normalization',
            'condition_id': row['condition_id'], 'chemical_id': row['chemical_or_experiment_group_id'],
            'dose_uM': float(row['dose_um']), 'actual_protocol': protocol,
            'action_group_id': row['action_group_id'], 'split_group_id': row['split_group_id'],
            'outer_split': row['outer_split'], 'cell_index_table': row['cell_index_table'],
            'physical_reference_ids': refs, 'reference_sampling_weights': weights,
            'target_cells': int(row['available_target_cells']), 'condition_weight': 1.0,
            'source_metadata': clean(row), 'unique_functional_Y': None})
    rna = json.loads(bind(MARTIN + '/rna_manifest.json').read_text(encoding='utf-8'))
    for name, item in rna['files'].items():
        bind(MARTIN + '/' + name, item['sha256'])
    if rna['common_axis']['sha256'] != bindings[JOINT + '/genes.parquet']:
        raise ValueError('GENE_MAPPING_IDENTITY_MISMATCH')
    samples = pd.read_csv(root / MARTIN / 'rna/rna_samples.csv')
    if len(samples) != 60 or not samples.gsm.is_unique or not samples.column_index.is_unique or not samples.functional_y.eq('not_attached').all():
        raise ValueError('RNA_SAMPLE_IDENTITY_INVALID')
    bulk_axis = pd.read_csv(root / MARTIN / 'rna/rna_genes.csv').source_gene_symbol.astype(str).tolist()
    if len(bulk_axis) != len(set(bulk_axis)):
        raise ValueError('BULK_GENE_IDENTITY_INVALID')
    chemistry = json.loads(bind(MARTIN + '/chemistry/manifest.json').read_text(encoding='utf-8'))
    items = json.loads(bind(MARTIN + '/chemistry/items.json', chemistry['files']['items.json']['sha256']).read_text(encoding='utf-8'))
    # Only source nominal chemistry metadata, never the function table named by its provenance.
    chemical = next(x for x in items if x['source_drug_name'] == 'osimertinib')
    backgrounds = sorted(samples.cell_line.unique())
    for fold in backgrounds:
        for (background, state_name), group in samples.loc[~samples.state.eq('parental_DMSO')].groupby(['cell_line', 'state']):
            references = samples.loc[samples.cell_line.eq(background) & samples.state.eq('parental_DMSO')]
            if len(group) != 3 or len(references) != 3:
                raise ValueError('BULK_REPLICATES_CHANGED')
            first = group.iloc[0]
            acute = state_name == 'acute_post_treatment'
            exposure = 24.0 if acute else 504.0
            washout = float(first.washout_hours)
            requests.append({'request_id': 'native:GSE193258:' + background + ':' + state_name + ':LOBO:' + fold,
                'role': 'final_test' if background == fold else 'fit', 'fold_id': 'leave_background_out:' + fold,
                'source_id': 'GSE193258', 'background_id': background, 'protocol_id': 'Martin2022_RNA_' + state_name,
                'reference_context_id': 'GSE193258:' + background + ':parental_DMSO',
                'reference_semantics': 'parallel_bulk_unpaired_RNA_not_same_cell_t0', 'measurement_level': 'bulk',
                'decoder_domain': 'bulk_source_log2TPM', 'gene_axis_id': 'GSE193258_source_symbol_19712',
                'units': rna['units']['log2TPM'], 'chemical_id': chemical['chemical_id'], 'dose_uM': 0.5,
                'actual_protocol': {'exposure_duration_h': exposure, 'observation_time_h': exposure + washout,
                    'prior_drug': None if acute else 'osimertinib', 'prior_drug_unknown': False,
                    'prior_drug_dose_nM': 0 if acute else 500, 'prior_drug_exposure_h': 0 if acute else 504,
                    'parental_drug_free_culture_days': int(first.parental_culture_days),
                    'washout_h': washout, 'washout_unknown': False,
                    'withdrawal_protocol': 'two_PBS_washes' if washout else 'not_described',
                    'withdrawal_unknown': not bool(washout), 'long_short_history': state_name},
                'action_group_id': 'GSE193258:osimertinib:500nM:' + state_name + ':' + str(washout),
                'split_group_id': 'LOBO:' + fold, 'physical_reference_ids': references.gsm.tolist(),
                'reference_column_indices': references.column_index.astype(int).tolist(),
                'target_sample_ids': group.gsm.tolist(), 'target_column_indices': group.column_index.astype(int).tolist(),
                'condition_weight': 1.0, 'replicate_weights': [1/3] * 3,
                'source_metadata': clean(group.to_dict('records')), 'unique_functional_Y': None,
                'function_72h_four_drug_target': False})
    for name in ('src/lung/data/native_response_units.py', 'scripts/build_native_response_units.py', 'src/lung/data/reader.py', 'src/lung/data/sparse_cache.py', 'src/lung/data/common.py', 'src/lung/data/sciplex_source.py', 'src/lung_data.py'):
        bind(name)
    summary = {'requests': len(requests), 'singlecell_requests': len(source), 'bulk_LOBO_requests': len(requests)-len(source),
        'bulk_unique_state_groups': 16, 'bulk_unique_treated_samples': 48, 'bulk_unique_reference_samples': 12,
        'by_source_role_level': dict(Counter('|'.join([r['source_id'], r['role'], r['measurement_level']]) for r in requests)),
        'by_source_exposure_history': dict(Counter('|'.join([r['source_id'], str(r['actual_protocol']['exposure_duration_h']), r['actual_protocol'].get('long_short_history', 'single_exposure')]) for r in requests)),
        'by_singlecell_source_endpoint_h': dict(Counter(r['source_id'] + '|' + str(r['actual_protocol']['duration_h']) for r in requests if r['measurement_level'] == 'single_cell')), 'round12_overlap': state['denominators'], 'unique_functional_Y': 0,
        'expression_deserialized': False, 'model_executed': False, 'function_labels_read': False}
    output.mkdir(parents=True)
    save(output / 'requests.json', requests)
    save(output / 'gene_axes.json', {'joint_ensembl_32461': axis, 'GSE193258_source_symbol_19712': bulk_axis})
    save(output / 'summary.json', summary)
    save(output / 'manifest.json', {'schema': SCHEMA, 'source_bindings': bindings,
        'files': {name: sha(output / name) for name in ('requests.json', 'gene_axes.json', 'summary.json')},
        'counts': summary, 'normalization': 'none; decoder domains kept separate',
        'limitations': ['no wet trajectory', 'reference not verified independent biological state',
            'no pulse RNA supervision', 'no four-drug functional 72h RNA target', 'LOBO copies not independent samples']})
    return summary


class NativeResponseUnits:
    """Strictly role-gated native count/abundance reader, no normalization or labels."""
    def __init__(self, root, asset, *, allowed_roles=('fit',), allow_test=False):
        self.root, self.asset = Path(root).resolve(), Path(asset).resolve()
        if 'final_test' in allowed_roles and not allow_test:
            raise PermissionError('FINAL_TEST_REQUIRES_EXPLICIT_ACCESS')
        self.allowed_roles = frozenset(allowed_roles)
        manifest = json.loads((self.asset / 'manifest.json').read_text(encoding='utf-8'))
        self.manifest = json.loads(json.dumps(manifest))
        if manifest['schema'] != SCHEMA:
            raise ValueError('SCHEMA_MISMATCH')
        for name, digest in manifest['source_bindings'].items():
            if sha(self.root / name) != digest:
                raise ValueError('SOURCE_BINDING_CHANGED: ' + name)
        for name, digest in manifest['files'].items():
            if sha(self.asset / name) != digest:
                raise ValueError('ASSET_CHANGED: ' + name)
        self._requests = json.loads((self.asset / 'requests.json').read_text(encoding='utf-8'))
        self.axes = json.loads((self.asset / 'gene_axes.json').read_text(encoding='utf-8'))
        self._by_id = {r['request_id']: r for r in self._requests}
        if len(self._by_id) != len(self._requests):
            raise ValueError('DUPLICATE_REQUEST')

    def requests(self, role='fit', *, fold_id=None):
        if role not in self.allowed_roles:
            raise PermissionError('ROLE_NOT_ALLOWED')
        return json.loads(json.dumps([r for r in self._requests if r['role'] == role and r['fold_id'] == fold_id]))

    def read(self, request_id, *, role, expected_gene_ids, max_rows=2):
        if role not in self.allowed_roles:
            raise PermissionError('ROLE_NOT_ALLOWED')
        request = self._by_id[request_id]
        if request['role'] != role:
            raise PermissionError('REQUEST_ROLE_MISMATCH')
        if list(expected_gene_ids) != self.axes[request['gene_axis_id']]:
            raise ValueError('ORDERED_GENE_IDENTITY_MISMATCH')
        if max_rows is not None and (not isinstance(max_rows, int) or max_rows < 1):
            raise ValueError('POSITIVE_ROW_COUNT_REQUIRED')
        if request['measurement_level'] == 'bulk':
            matrix = np.load(self.root / MARTIN / 'rna/log2TPM.npy', mmap_mode='r', allow_pickle=False)
            if matrix.shape != (len(expected_gene_ids), 60):
                raise ValueError('BULK_AXIS_SHAPE_MISMATCH')
            refs, targets = request['reference_column_indices'][:max_rows], request['target_column_indices'][:max_rows]
            return {'reference': np.array(matrix[:, refs].T), 'target': np.array(matrix[:, targets].T),
                'reference_ids': request['physical_reference_ids'][:max_rows], 'target_ids': request['target_sample_ids'][:max_rows],
                'units': request['units'], 'gene_axis_id': request['gene_axis_id'], 'unique_functional_Y': None}
        from .reader import training_rows
        genes = pd.read_parquet(self.root / JOINT / 'genes.parquet')
        manifest = json.loads((self.root / JOINT / 'manifest.json').read_text(encoding='utf-8'))
        reader = training_rows(self.root, self.root / JOINT, manifest['config'], genes)
        refs = pd.read_parquet(self.root / request['cell_index_table'], filters=[('role', '==', role), ('condition_id', 'in', request['physical_reference_ids'])])
        if max_rows is not None:
            refs = refs.head(max_rows)
        targets = pd.read_parquet(self.root / request['cell_index_table'], filters=[('role', '==', role), ('condition_id', '==', request['condition_id'])])
        if max_rows is not None:
            targets = targets.head(max_rows)
        if refs.empty or targets.empty or set(refs.cell_id) & set(targets.cell_id):
            raise ValueError('EMPTY_OR_OVERLAPPING_CELL_SELECTION')
        try:
            return {'reference': reader.read(refs), 'target': reader.read(targets),
                'reference_ids': refs.cell_id.tolist(), 'target_ids': targets.cell_id.tolist(),
                'units': request['units'], 'gene_axis_id': request['gene_axis_id'], 'unique_functional_Y': None}
        finally:
            if hasattr(reader, 'close'):
                reader.close()


    def reference(self, request_id, *, role, expected_gene_ids, max_rows=2):
        """Serving input only: select reference values, never target expression rows."""
        if role not in self.allowed_roles:
            raise PermissionError('ROLE_NOT_ALLOWED')
        request = self._by_id[request_id]
        if request['role'] != role:
            raise PermissionError('REQUEST_ROLE_MISMATCH')
        if list(expected_gene_ids) != self.axes[request['gene_axis_id']]:
            raise ValueError('ORDERED_GENE_IDENTITY_MISMATCH')
        if max_rows is not None and (not isinstance(max_rows, int) or max_rows < 1):
            raise ValueError('POSITIVE_ROW_COUNT_REQUIRED')
        if request['measurement_level'] == 'bulk':
            matrix = np.load(self.root / MARTIN / 'rna/log2TPM.npy', mmap_mode='r', allow_pickle=False)
            if matrix.shape != (len(expected_gene_ids), 60):
                raise ValueError('BULK_AXIS_SHAPE_MISMATCH')
            values = np.array(matrix[:, request['reference_column_indices'][:max_rows]].T)
            ids = request['physical_reference_ids'][:max_rows]
            weights = [1 / len(request['physical_reference_ids'])] * len(ids)
            reference_rows = [{'gsm': gid, 'column_index': column, 'role': role} for gid, column in zip(ids, request['reference_column_indices'][:max_rows])]
        else:
            from .reader import training_rows
            genes = pd.read_parquet(self.root / JOINT / 'genes.parquet')
            manifest = json.loads((self.root / JOINT / 'manifest.json').read_text(encoding='utf-8'))
            reader = training_rows(self.root, self.root / JOINT, manifest['config'], genes)
            rows = pd.read_parquet(self.root / request['cell_index_table'], filters=[('role', '==', role), ('condition_id', 'in', request['physical_reference_ids'])])
            if max_rows is not None:
                rows = rows.head(max_rows)
            if rows.empty:
                raise ValueError('EMPTY_REFERENCE')
            try:
                values, ids = reader.read(rows), rows.cell_id.tolist()
                weights = [request['reference_sampling_weights'][cid]['well_probability'] * request['reference_sampling_weights'][cid]['cell_probability_given_well'] for cid in rows.condition_id]
                reference_rows = clean(rows.to_dict('records'))
            finally:
                if hasattr(reader, 'close'):
                    reader.close()
        return {'reference': values, 'reference_ids': ids, 'reference_rows': reference_rows, 'reference_weights': weights, 'units': request['units'],
                'gene_axis_id': request['gene_axis_id'], 'unique_functional_Y': None,
                'target_expression_used': False, 'selection': 'full_reference' if max_rows is None else 'deterministic_source_order_prefix_not_random_sample', 'weights_sum': float(sum(weights)), 'prefix_weights_renormalized': False}

    def bulk_common_axis_mask(self):
        """Comparison mapping only: missing joint genes remain masked, no zero fill."""
        mapping = pd.read_csv(self.root / MARTIN / 'rna/common_axis_coverage.csv')
        return mapping.copy(deep=True)



    def request(self, request_id, *, role):
        if role not in self.allowed_roles or self._by_id[request_id]['role'] != role:
            raise PermissionError('REQUEST_ROLE_MISMATCH')
        return json.loads(json.dumps(self._by_id[request_id]))

