"""Registered observed Martin RNA, with separate LOBO/cycle role views.

No model, normalization, latent descriptor or functional Y is created here.
Reference and target are unpaired bulk biological replicates, not cell t0/fate.
"""
import hashlib
import json
from pathlib import Path
import shutil
from types import MappingProxyType

import numpy as np
import pandas as pd

BASE = '08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1'
NATIVE = 'experiments/20261011-data-alignment-round17/native-response-units-v1'
CYCLE = 'experiments/20261011-data-alignment-round17/multistate-role-plan-v1'
SCHEMA = 'martin-observed-native-rna-inputs-v1'
DOMAINS = ('log2TPM', 'estimated_counts')
ROLES = ('fit', 'selection', 'final_test')

# Verified Round21 source snapshot. Copies must have these same bytes; current
# manifests do not get to redefine roles, clocks, arrays, axes or generator code.
REGISTERED_SOURCE_HASHES = MappingProxyType({
    NATIVE + '/manifest.json': '7c04dcd7486224f6870e31d58f7e47a9ca7859213ef120d08c529f4b562ebb71',
    CYCLE + '/manifest.json': '99cd8202e4f67f59e68292b7b1179ed2d69fd4d8a6e991f6a92dbf18779eb0ab',
    CYCLE + '/plan.json': 'bd075a595a3751d56dd5740cfbbc180c5843898f93e7580e0dff0e60e7bfc40f',
    BASE + '/rna_manifest.json': 'dc135fdc4d02560bac26d0c1bbc27686c291fdff145145f7ca360299f5cc4031',
    BASE + '/rna/common_axis_coverage.csv': '43d59db5184268484276014c38c01761ffd76785026e98595552e1c69633b6fa',
    BASE + '/rna/estimated_counts.npy': '6efced97a4f4a93f1cccbe3a357a3e75b7059af73563508028eb7a7706226803',
    BASE + '/rna/geo_samples_source.json': '2f5f8c243f344e1a53777169c8f373073640ea7c75f8eb27dbfdfb2ae3550bba',
    BASE + '/rna/log2TPM.npy': 'd2c09844feb36e1ffa4059ddaf2aa476a191511583bab2e704c694f4415d3366',
    BASE + '/rna/parallel_baseline_samples.csv': '015db5c4dc223bcc10f69163fbcf0b7e453670f7d70aacb65aa3752002599a36',
    BASE + '/rna/quarantined_axis_symbols.csv': 'dbe921f93b5e1b967945f27efc2554c6adbacba45fb2434ca53828fdca3459b1',
    BASE + '/rna/rna_genes.csv': '0dacb35cc3f779cc844f5c84a5d62066ae59f0b3b34e9d462a0d9f7ca81483dd',
    BASE + '/rna/rna_samples.csv': 'd16f9f0456625174801991fee04ff2014e210362d6cfa505d57dd685842abab9',
    BASE + '/raw/GSE193258_RNAseq_estimated_counts.tsv.gz': '51c92720b7bf4a0d7d29f6b7f33e304bd9f858ba8182572260ddf7f1c4de6d33',
    BASE + '/raw/GSE193258_RNAseq_log2TPM_abundance.tsv.gz': 'c4bd1b3d7e158783347a7b678df70bec2c590728a59486d67fcb3a8e21750949',
    '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/genes.parquet': '4d38a758179d9123fff1ffda1f1eae85fe5f57faa31a3a6df01fcdefbb5314fc',
    NATIVE + '/requests.json': '5c7091bef25d205246862f0d1a7e4a5d3de6fe4631aaf932e0c27f7192c97d77',
    'src/lung/data/multistate_role_plan.py': '96cf85e22508e92a0ef27640fbc6ef8a424d397393e1814afbf168311152d56d',
    BASE + '/bridge_v2/metadata.json': '0de2edf3a743e2e02d57f8b1014f27185dcd10b0a27fc07e6f7815927aef4363',
    BASE + '/bridge_v2/manifest.json': 'c60525bcd32824e664dce807ee4fb0aea3d9477e0b858610c6d8d02f90b062ed',
    'scripts/prepare_martin_native_rna.py': '9d5c92214523920d767711eb46408926600bc9c583ffc98b6cbd9e18a1d96922',
    'src/lung/data/native_response_units.py': '5d0c89ca3ff7798355db9072eb288ff1592520a52f867ade262f48453977dbf1',
    'scripts/build_native_response_units.py': 'fadd5f7420cac954a56b994238af62236517bc040ea2607d3a5ddb492cdb935b',
    'scripts/materialize_martin2022_rna.py': 'cd1f1e34b0ca45195830265761d8ca50f422333173954a9093e4b71b5a107a93',
    'src/lung/data/martin_state_function.py': '12a05add419d096d9d74361c46be099d550659d003e48a238467084a2b756b0b',
    'scripts/build_martin_state_function.py': '65989c6c64f3d6e1e54c8f75a3a40b65d76e29bdb1429fb266ca6bfba9697616',
})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def validate_values(values, domain, shape):
    if (domain not in DOMAINS or not isinstance(values, np.ndarray)
            or values.dtype != np.dtype('float64') or values.shape != shape
            or not np.isfinite(values).all()
            or (domain == 'estimated_counts' and (values < 0).any())):
        raise ValueError('SOURCE_FLOAT64_OBSERVATIONS_REQUIRED')
    return values


def align_observed(values, source_rows, observed):
    """Exact sample-major joint view; absent genes stay NaN, never zero."""
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float64')
            or values.ndim != 2 or not np.isfinite(values).all()
            or not isinstance(observed, np.ndarray) or observed.dtype != np.dtype('bool')
            or observed.ndim != 1 or not isinstance(source_rows, np.ndarray)
            or source_rows.dtype.kind not in 'iu' or source_rows.shape != (int(observed.sum()),)
            or (source_rows < 0).any() or (source_rows >= len(values)).any()):
        raise ValueError('EXACT_ORDERED_MAPPING_REQUIRED')
    result = np.full((values.shape[1], len(observed)), np.nan, dtype=np.float64)
    result[:, observed] = values[source_rows, :].T
    return result


def _check_anchor(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
        raise ValueError('FIXED_REGISTERED_SOURCE_CHANGED: ' + relative)


def _registered_bindings(root):
    for relative, expected in REGISTERED_SOURCE_HASHES.items():
        _check_anchor(root, relative, expected)
    # Self-referential fixed hashes are impossible. The copy of the active loader
    # must instead match the executing trusted module, never an arbitrary file.
    relative = 'src/lung/data/martin_native_rna.py'
    _check_anchor(root, relative, sha(__file__))
    return {**REGISTERED_SOURCE_HASHES, relative: sha(__file__)}


def _source(root):
    """Reconstruct from fixed registered paths, not caller arrays or hash claims."""
    root = Path(root).resolve()
    base, native, cycle = root / BASE, root / NATIVE, root / CYCLE
    bindings = _registered_bindings(root)
    def bind(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError('SOURCE_PATH_OUTSIDE_PROJECT')
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('REGISTERED_SOURCE_BYTES_CHANGED')
        bindings[str(path.relative_to(root)).replace('\\', '/')] = actual
        return path
    nm = read_json(bind(native / 'manifest.json'))
    rna = read_json(bind(base / 'rna_manifest.json', nm['source_bindings'][BASE + '/rna_manifest.json']))
    for name, item in rna['files'].items():
        bind(base / name, item['sha256'])
    for item in rna['source_files'].values():
        bind(base / 'raw' / item['url'].rsplit('/', 1)[1], item['sha256'])
    if (rna['rna_gene_count'], rna['rna_sample_count'], rna['baseline_samples']) != (19712, 60, 12):
        raise ValueError('REGISTERED_RNA_COUNTS_CHANGED')
    samples = pd.read_csv(base / 'rna/rna_samples.csv')
    genes = pd.read_csv(base / 'rna/rna_genes.csv')
    if (len(samples) != 60 or samples.column_index.tolist() != list(range(60))
            or not samples.gsm.is_unique or not samples.matrix_column.is_unique
            or not samples.functional_y.eq('not_attached').all()
            or len(genes) != 19712 or genes.source_row_index.tolist() != list(range(19712))
            or not genes.source_gene_symbol.is_unique):
        raise ValueError('SOURCE_SAMPLE_OR_GENE_IDENTITY_CHANGED')
    matrices = {d: validate_values(np.load(base / ('rna/' + d + '.npy'), mmap_mode='r', allow_pickle=False),
                                  d, (19712, 60)) for d in DOMAINS}
    joint_path = root / rna['common_axis']['path']
    bind(joint_path, rna['common_axis']['sha256'])
    joint = pd.read_parquet(joint_path)
    coverage = pd.read_csv(base / 'rna/common_axis_coverage.csv')
    if (len(joint) != 32461 or not joint.ensembl_id.is_unique
            or coverage.common_axis_index.tolist() != list(range(32461))
            or coverage.ensembl_id.tolist() != joint.ensembl_id.tolist()
            or coverage.source_observed.dtype != np.dtype('bool')):
        raise ValueError('ORDERED_JOINT_COORDINATES_CHANGED')
    observed = coverage.source_observed.to_numpy(copy=True)
    rows = coverage.loc[observed, 'source_row_index'].astype(np.int64).to_numpy()
    if (int(observed.sum()) != 16167 or genes.iloc[rows].source_gene_symbol.tolist()
            != coverage.loc[observed, 'joint_gene_symbol'].tolist()
            or coverage.loc[~observed, 'source_row_index'].notna().any()):
        raise ValueError('SOURCE_JOINT_MASK_MAPPING_CHANGED')
    nr = read_json(bind(native / 'requests.json', nm['files']['requests.json']))
    lobo = [r for r in nr if r['measurement_level'] == 'bulk']
    if len(lobo) != 64 or len({r['request_id'] for r in lobo}) != 64:
        raise ValueError('OLD_LOBO_REQUEST_IDENTITY_CHANGED')
    for r in lobo:
        _validate_request(r, samples)
        heldout = r['fold_id'].removeprefix('leave_background_out:')
        if r['role'] != ('final_test' if r['background_id'] == heldout else 'fit'):
            raise ValueError('OLD_LOBO_ROLES_CHANGED')
    cm = read_json(bind(cycle / 'manifest.json'))
    plan = read_json(bind(cycle / 'plan.json', cm['plan_sha256']))
    if plan['schema'] != 'martin-multistate-twofit-selection-final-v1' or len(plan['folds']) != 4:
        raise ValueError('CYCLE_PLAN_CHANGED')
    # Verify metadata/code anchors; never parse observations.json or core Y.
    bind(root / 'src/lung/data/multistate_role_plan.py', cm['builder_source_sha256'])
    bind(base / 'bridge_v2/metadata.json', plan['source_metadata_sha256'])
    bind(base / 'bridge_v2/manifest.json', plan['source_manifest_sha256'])
    cycle_requests = _cycle_requests(lobo, plan)
    for name in ('src/lung/data/martin_native_rna.py', 'scripts/prepare_martin_native_rna.py'):
        bind(root / name)
    axes = {'source_symbols': genes.source_gene_symbol.tolist(), 'joint_ensembl_ids': joint.ensembl_id.tolist(),
            'common_to_source_rows': [int(v) if known else None for known, v in
                                      zip(observed, coverage.source_row_index.fillna(-1))],
            'units': rna['units'], 'normalization': 'none_source_values_preserved',
            'source_measurement_level': 'bulk_RNA_biological_replicates'}
    return {'root': root, 'base': base, 'bindings': bindings, 'matrices': matrices,
            'samples': json.loads(samples.to_json(orient='records')), 'axes': axes,
            'observed': observed, 'source_rows': rows, 'lobo': lobo, 'cycle': cycle_requests}


def _validate_request(r, samples):
    refs = samples.iloc[r['reference_column_indices']]
    targets = samples.iloc[r['target_column_indices']]
    p = r['actual_protocol']
    state = p['long_short_history']
    acute = state == 'acute_post_treatment'
    expected_exposure = 24. if acute else 504.
    if (len(refs) != 3 or len(targets) != 3 or not refs.state.eq('parental_DMSO').all()
            or not targets.state.eq(state).all() or not refs.cell_line.eq(r['background_id']).all()
            or not targets.cell_line.eq(r['background_id']).all()
            or refs.gsm.tolist() != r['physical_reference_ids'] or targets.gsm.tolist() != r['target_sample_ids']
            or set(refs.column_index) & set(targets.column_index) or r['dose_uM'] != .5
            or p['exposure_duration_h'] != expected_exposure
            or not targets.washout_hours.eq(p['washout_h']).all()
            or p['observation_time_h'] != expected_exposure + p['washout_h']
            or r['unique_functional_Y'] is not None or r['function_72h_four_drug_target'] is not False
            or r['decoder_domain'] != 'bulk_source_log2TPM'
            or r['reference_semantics'] != 'parallel_bulk_unpaired_RNA_not_same_cell_t0'):
        raise ValueError('SOURCE_BULK_REQUEST_MISMATCH')


def _cycle_requests(lobo, plan):
    by_group = {}
    for r in lobo:
        by_group[(r['background_id'], r['actual_protocol']['long_short_history'])] = r
    result = []
    for fold in plan['folds']:
        background_roles = fold['background_roles']
        if (set(background_roles) != set(ROLES) or tuple(len(background_roles[r]) for r in ROLES) != (2, 1, 1)
                or len({b for values in background_roles.values() for b in values}) != 4):
            raise ValueError('DISJOINT_CYCLE_BACKGROUNDS_REQUIRED')
        for role in ROLES:
            for name in background_roles[role]:
                background = 'H1975' if name == 'NCI-H1975' else name
                for (bg, state), r in by_group.items():
                    if bg == background:
                        result.append({**r, 'request_id': 'observedRNA:' + bg + ':' + state + ':' + fold['fold_id'],
                            'fold_id': fold['fold_id'], 'role': role,
                            'split_group_id': 'cycle:' + fold['fold_id'],
                            'role_plan_identity': {'schema': plan['schema'], 'fold_id': fold['fold_id'],
                                                   'background_roles': background_roles}})
    if len(result) != 64 or len({r['request_id'] for r in result}) != 64:
        raise ValueError('CYCLE_REQUEST_COUNT_CHANGED')
    return result


def _manifest(source, files):
    return {'schema': SCHEMA, 'loader_revision': 'fixed_registered_snapshot_v2',
            'source_bindings': source['bindings'], 'files': files,
            'counts': {'source_genes': len(source['axes']['source_symbols']), 'RNA_samples': 60,
                       'unique_reference_samples': 12, 'unique_target_samples': 48,
                       'unique_treated_groups': 16, 'lobo_role_requests': 64, 'cycle_role_requests': 64},
            'normalization': 'none_source_values_preserved', 'missing_values': 'NaN_plus_boolean_mask',
            'same_cell_t0': False, 'functional_Y_used': False, 'function_72h_four_drug_target': False,
            'model_execution_performed': False, 'model_decoder_weights_status': 'pending_not_granted_by_data',
            'model_or_function_qualified': False}


def prepare(root, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('NEW_RNA_INPUT_VERSION_REQUIRED')
    source = _source(root)
    output.mkdir(parents=True)
    for d in DOMAINS:
        shutil.copyfile(source['base'] / ('rna/' + d + '.npy'), output / ('source_' + d + '.npy'))
        np.save(output / ('joint_' + d + '.npy'), align_observed(source['matrices'][d], source['source_rows'], source['observed']), allow_pickle=False)
    np.save(output / 'joint_observed_mask.npy', source['observed'], allow_pickle=False)
    write_json(output / 'samples.json', source['samples'])
    write_json(output / 'coordinates.json', source['axes'])
    for system in ('lobo', 'cycle'):
        write_json(output / (system + '_requests.json'), source[system])
    files = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'manifest.json', _manifest(source, files))
    return MartinNativeRNA(root, output, allowed_roles=ROLES, allow_final=True)


class MartinNativeRNA:
    """Actual observed source arrays; permissions and ordered coordinates checked."""
    def __init__(self, root, directory, *, allowed_roles=('fit',), allow_final=False):
        if set(allowed_roles) - set(ROLES) or ('final_test' in allowed_roles and not allow_final):
            raise PermissionError('FINAL_OR_UNKNOWN_ROLE_NOT_AUTHORIZED')
        self.allowed_roles = frozenset(allowed_roles)
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        source = _source(self.root)
        m = read_json(self.directory / 'manifest.json')
        names = {'source_' + d + '.npy' for d in DOMAINS} | {'joint_' + d + '.npy' for d in DOMAINS}
        names |= {'samples.json', 'coordinates.json', 'lobo_requests.json', 'cycle_requests.json', 'joint_observed_mask.npy'}
        if set(m.get('files', {})) != names or m != _manifest(source, m['files']):
            raise ValueError('PACKAGE_SOURCE_OR_SCHEMA_CHANGED')
        for name, expected in m['files'].items():
            if sha(self.directory / name) != expected:
                raise ValueError('RNA_PACKAGE_BYTES_CHANGED')
        for name, expected in (('samples.json', source['samples']), ('coordinates.json', source['axes']),
                               ('lobo_requests.json', source['lobo']), ('cycle_requests.json', source['cycle'])):
            if read_json(self.directory / name) != expected:
                raise ValueError('SOURCE_RECONSTRUCTION_MISMATCH')
        self._axes = source['axes']
        self._requests = {system: {r['request_id']: r for r in source[system]} for system in ('lobo', 'cycle')}
        self._source_values, self._joint_values = {}, {}
        mask = np.load(self.directory / 'joint_observed_mask.npy', mmap_mode='r', allow_pickle=False)
        if mask.dtype != np.dtype('bool') or not np.array_equal(mask, source['observed']):
            raise ValueError('REGISTERED_GENE_MASK_CHANGED')
        self._mask = mask
        for d in DOMAINS:
            raw = np.load(self.directory / ('source_' + d + '.npy'), mmap_mode='r', allow_pickle=False)
            validate_values(raw, d, source['matrices'][d].shape)
            joint = np.load(self.directory / ('joint_' + d + '.npy'), mmap_mode='r', allow_pickle=False)
            if (not np.array_equal(raw, source['matrices'][d]) or joint.dtype != np.dtype('float64')
                    or not np.array_equal(joint, align_observed(source['matrices'][d], source['source_rows'], source['observed']), equal_nan=True)):
                raise ValueError('EXACT_SOURCE_OBSERVATIONS_CHANGED')
            self._source_values[d], self._joint_values[d] = raw, joint
        self._manifest = m

    @property
    def axes(self):
        return json.loads(json.dumps(self._axes))

    @property
    def manifest(self):
        return json.loads(json.dumps(self._manifest))

    def close(self):
        for values in (*self._source_values.values(), *self._joint_values.values(), self._mask):
            values._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def requests(self, *, system, fold_id, role):
        if role not in self.allowed_roles or system not in self._requests:
            raise PermissionError('ROLE_OR_ROLE_SYSTEM_NOT_AUTHORIZED')
        return json.loads(json.dumps([r for r in self._requests[system].values()
                                    if r['fold_id'] == fold_id and r['role'] == role]))

    def _read(self, request_id, *, system, role, domain, coordinate, target):
        if role not in self.allowed_roles or system not in self._requests:
            raise PermissionError('ROLE_OR_ROLE_SYSTEM_NOT_AUTHORIZED')
        r = self._requests[system].get(request_id)
        if r is None or r['role'] != role:
            raise PermissionError('REQUEST_ROLE_OR_SYSTEM_MISMATCH')
        if domain not in DOMAINS or coordinate not in ('source', 'joint'):
            raise ValueError('EXPLICIT_SOURCE_DOMAIN_COORDINATE_REQUIRED')
        columns = r['target_column_indices'] if target else r['reference_column_indices']
        values = (np.array(self._source_values[domain][:, columns].T, copy=True) if coordinate == 'source'
                  else np.array(self._joint_values[domain][columns, :], copy=True))
        mask = (np.ones(values.shape, dtype=bool) if coordinate == 'source'
                else np.broadcast_to(self._mask, values.shape).copy())
        values.setflags(write=False)
        mask.setflags(write=False)
        return {'values': values, 'observed_mask': mask,
                'sample_ids': list(r['target_sample_ids'] if target else r['physical_reference_ids']),
                'source_column_indices': list(columns), 'domain': domain, 'units': self._axes['units'][domain],
                'gene_ids': tuple(self._axes['source_symbols' if coordinate == 'source' else 'joint_ensembl_ids']),
                'coordinate': coordinate, 'request': json.loads(json.dumps(r)), 'role_system': system,
                'target_expression_used': target, 'unpaired_reference_target': True,
                'same_cell_t0': False, 'functional_Y': None, 'function_72h_four_drug_target': False,
                'normalization': 'none_source_values_preserved', 'model_or_function_qualified': False}

    def reference(self, request_id, *, system, role, domain='log2TPM', coordinate='source'):
        return self._read(request_id, system=system, role=role, domain=domain, coordinate=coordinate, target=False)

    def supervision(self, request_id, *, system, role, domain='log2TPM', coordinate='source'):
        return {'reference': self.reference(request_id, system=system, role=role, domain=domain, coordinate=coordinate),
                'target': self._read(request_id, system=system, role=role, domain=domain, coordinate=coordinate, target=True)}
