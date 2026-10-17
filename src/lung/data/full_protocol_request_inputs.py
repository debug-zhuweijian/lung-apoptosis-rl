"""Pinned real candidate inputs and explicit proxy plan. No model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .common import data_file_bindings

ROOT = Path(__file__).resolve().parents[3]
VERSION = "full-protocol-request-inputs-v1"
OUTPUT = ROOT / "experiments/20261011-data-alignment-round21/full-protocol-request-inputs-v1"
PINS = {
    '08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1/candidate_records.json': '426de2f6dd4ce5b5eb202a4c8b06c53bd26caeefa77b4d1942e16a8e2dfa87b4',
    'experiments/20261011-data-alignment-round20/predicted_expression_function_bridge/v2/bridge_metadata.json': '000e0b467b458295b5c037599e7eb200e5ff6eb2a799adf250762683ac5de704',
    'experiments/20261011-data-alignment-round20/predicted_expression_function_bridge/v2/metadata_receipt.json': 'b83bbeea4d3e1338948dcde024ac3d9705013b076753def5d0dd263ea1f41ccc',
    'experiments/20261011-data-alignment-round19/full-protocol-inputs-v3/manifest.json': '60ff10eecdb63e67a9df506e3ff022022474d70856d83933447f177cacb2abd5',
    'experiments/20261011-data-alignment-round19/full-protocol-inputs-v3/records.json': '044625fdbe549dc79a0a61a8d0cb5348c2391056284b86003b48d779cbe9f55a',
    'experiments/20261010-endpoint-feature-preflight-joint/final-v3/reference_inputs/reference.json': '0f5a9633aa8b347f1bd2d730b08f2bf38acb1f8a59f65dda8fa810ca5ba7a6d7',
    'experiments/20261010-endpoint-feature-preflight-joint/final-v3/reference_inputs/reference_fit_control_cells.parquet': '83f45a6dbbf7dc78cdb4c4716954c6eb8f1290f6cefbf270a3227014dc3dc3ee',
    '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/manifest.json': '513c9650acec6613d61c6f32be26d68ec477d7fb9c3c38f43742829443cbba05',
    '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/conditions.parquet': '28e4ad85978711b41e4bf27793df65b3e1a9618e5d83460d7e366fce7c995eac',
    '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/genes.parquet': '4d38a758179d9123fff1ffda1f1eae85fe5f57faa31a3a6df01fcdefbb5314fc',
    '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/normalization.npz': '386eb92f178327f0ba4f67215b073d18e9a044ba0de83ab81b77406cd8db8e1b',
}

BRIDGE = 'experiments/20261011-data-alignment-round20/predicted_expression_function_bridge/v2/bridge_metadata.json'
FULL = 'experiments/20261011-data-alignment-round19/full-protocol-inputs-v3'
REF = 'experiments/20261010-endpoint-feature-preflight-joint/final-v3/reference_inputs/reference.json'
JOINT = '08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


DECLARATIONS = {
    'union_candidates':32, 'endpoint_candidates':28, 'bothe_auxiliary_histories':4,
    'Viel_full_inputs_ready':20, 'Nav_identity_preserved_physical_pending':8,
    'unique_endpoint_Y_identities':24, 'not_32_independent_Y':True,
    'proxy_reference_rows':2431, 'P_outputs':0, 'G_outputs':0, 'Q_outputs':0,
    'PT_loads':0, 'model_forwards':0,
    'preparation_grants_model_or_scientific_qualification':False,
    'duration_contract':'new_full_input_active_exposure6_or24_horizon24_elapsed0_unvalidated_choice',
    'reference_transfer':'cross_source_transfer_unvalidated',
}


def validate_declarations(manifest, context):
    expected = {**DECLARATIONS, 'model_context_indexes':context['model_context_indexes']}
    for key, value in expected.items():
        actual = manifest.get(key)
        require(type(actual) is type(value) and actual == value,
                'REGISTERED_DECLARATION_CHANGED: ' + key)
    require(all(type(v) is int for v in manifest['model_context_indexes'].values()),
            'REGISTERED_CONTEXT_INDEX_TYPE_CHANGED')


def reconstruct():
    for name, digest in PINS.items():
        require(sha(ROOT / name) == digest, 'PINNED_SOURCE_CHANGED: ' + name)
    from .full_protocol_inputs_v3 import FullProtocolInputsV3
    physical_reader = FullProtocolInputsV3.load(ROOT / FULL)
    full = {r['candidate_id']: r for r in physical_reader._authenticated_records()}
    bridge = read(ROOT / BRIDGE)['rows']
    endpoint = {r['candidate_id']: r for r in read(ROOT / '08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1/candidate_records.json')}
    reference = read(ROOT / REF)
    cells = pd.read_parquet(reference['cell_index']['path'])
    require(len(cells) == 2431 and cells.cell_id.is_unique and cells.role.eq('fit').all(), 'FIT_PROXY_SCOPE_CHANGED')
    require(set(cells.condition_id) == set(reference['control_condition_ids']), 'PROXY_CONTROL_SCOPE_CHANGED')
    for field in ('source_id', 'background_id', 'protocol_id'):
        require(cells[field].eq(reference[field]).all(), 'PROXY_CONTEXT_CHANGED')
    conditions = pd.read_parquet(ROOT / JOINT / 'conditions.parquet')
    vocabulary = {k: sorted(conditions[k].dropna().unique().tolist())
                  for k in ('source_id', 'background_id', 'protocol_id')}
    indexes = {k: vocabulary[k].index(reference[k]) for k in vocabulary}
    genes = pd.read_parquet(ROOT / JOINT / 'genes.parquet').ensembl_id.astype(str).tolist()
    require(len(genes) == 32461 and len(genes) == len(set(genes)), 'GENE_AXIS_CHANGED')
    with np.load(ROOT / JOINT / 'normalization.npz', allow_pickle=False) as norm:
        mean, scale = norm['mean'].copy(), norm['scale'].copy()
    require(mean.shape == scale.shape == (len(genes),) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and (scale > 0).all(), 'NORMALIZATION_INVALID')
    rows = []
    for item in bridge:
        cid = item['candidate_id']
        p = full.get(cid)
        record = {'candidate_id': cid, 'source_id': item['source_id'],
                  'source_protocol_id': endpoint[cid]['protocol_id'],
                  'source_state_id': item['source_state_id'], 'actual_protocol': item['actual_protocol'],
                  'role_binding': item['role_binding'], 'observation_id': item['observation_id'],
                  'physical_status': 'available' if p else 'pending_source_physical_events',
                  'request_status': 'pending_physical_input', 'physical': p, 'base_kwargs': None}
        if p:
            require(item['full_physical_input'] == p and p['source_id'] == 'Vielhauer2013', 'FULL_BRIDGE_MISMATCH')
            proxy = item['legal_proxy_reference']
            require(all(proxy[k] == reference[k] for k in ('reference_id','data_identity', *vocabulary)),
                    'REGISTERED_PROXY_CHANGED')
            drug = [e for e in p['events'] if e['event_type'] == 'drug_interval']
            require(len(drug) == 1 and drug[0]['dose_unit'] == 'nM', 'ACTIVE_INTERVAL_REQUIRED')
            e = drug[0]
            require(e['values']['dose'] / 1000 == item['dose_uM'] and p['endpoint_hours'] == 24,
                    'DOSE_OR_ENDPOINT_CHANGED')
            record.update(request_status='numeric_inputs_ready_proxy_transfer_unvalidated',
                          base_kwargs={'fingerprint_source': e['fingerprint'], 'dose': item['dose_uM'],
                                       'duration': e['values']['exposure_hours'], 'elapsed': 0., 'horizon': 24.,
                                       'vehicle': e['vehicle'], 'background': indexes['background_id'],
                                       'source': indexes['source_id'], 'protocol': indexes['protocol_id']},
                          proxy_context={k: reference[k] for k in vocabulary},
                          reference_id=reference['reference_id'],
                          transfer_status='cross_source_transfer_unvalidated')
        rows.append(record)
    endpoint_ids = {r['candidate_id'] for r in rows}
    for cid, p in full.items():
        if cid not in endpoint_ids:
            rows.append({'candidate_id': cid, 'source_id': p['source_id'], 'source_state_id': p['state_id'],
                         'actual_protocol': None, 'physical': p, 'physical_status': 'available',
                         'request_status': 'pending_bulk_reference_context_decoder', 'base_kwargs': None,
                         'observation_id': None, 'role_binding': None})
    require(len(rows) == len({r['candidate_id'] for r in rows}) == 32, 'UNION_SCOPE_CHANGED')
    context = {'reference': reference, 'reference_rows': cells.to_dict('records'),
               'gene_ids': genes, 'mean': mean, 'scale': scale, 'context_vocabulary': vocabulary,
               'model_context_indexes': indexes, 'physical_reader': physical_reader}
    return rows, context


class FullProtocolRequestInputs:
    def __init__(self, version=VERSION):
        require(version == VERSION, 'UNREGISTERED_VERSION')
        self.version = version

    @classmethod
    def load(cls, version=VERSION):
        require(version == VERSION, 'UNREGISTERED_VERSION')
        manifest = read(OUTPUT / 'manifest.json')
        require(set(manifest) - {'consumer_sha256'} == set(DECLARATIONS) | {'schema','source_pins',
                    'files_sha256','model_context_indexes'}, 'REGISTERED_MANIFEST_SCHEMA_CHANGED')
        require(manifest['schema'] == VERSION and data_file_bindings(manifest['source_pins']) == PINS, 'REGISTERED_CONSUMER_CHANGED')
        require(manifest['files_sha256'] == {'requests.json': sha(OUTPUT / 'requests.json')},
                'REGISTERED_OUTPUT_CHANGED')
        rows, context = reconstruct()
        require(read(OUTPUT / 'requests.json') == rows, 'SOURCE_RECONSTRUCTION_MISMATCH')
        validate_declarations(manifest, context)
        return cls(version)

    @property
    def candidate_ids(self):
        return tuple(r['candidate_id'] for r in reconstruct()[0])

    def request(self, candidate_id):
        require(type(candidate_id) is str, 'REGISTERED_CANDIDATE_ID_REQUIRED')
        rows, _ = reconstruct()
        selected = [r for r in rows if r['candidate_id'] == candidate_id]
        require(len(selected) == 1, 'UNKNOWN_CANDIDATE')
        return selected[0]

    def numeric_kwargs(self, candidate_id, *, batch_size=1):
        require(type(batch_size) is int and batch_size > 0, 'INVALID_BATCH_SIZE')
        rows, context = reconstruct()
        selected = [r for r in rows if r['candidate_id'] == candidate_id]
        require(len(selected) == 1 and selected[0]['base_kwargs'] is not None, 'PENDING_BASE_KWARGS')
        action = selected[0]['base_kwargs']
        spec = action['fingerprint_source']
        fp = np.load(spec['path'], allow_pickle=False)[spec['row']].astype(np.float32)
        require(fp.shape == (2048,) and np.isin(fp, [0,1]).all(), 'INVALID_SOURCE_FINGERPRINT')
        kwargs = {'fingerprint': np.repeat(fp[None], batch_size, axis=0)}
        for key in ('dose','duration','elapsed','horizon','vehicle','background','source','protocol'):
            dtype = np.bool_ if key == 'vehicle' else np.int64 if key in ('background','source','protocol') else np.float32
            kwargs[key] = np.full(batch_size, action[key], dtype=dtype)
        physical = context['physical_reader'].tensor_batch([candidate_id] * batch_size)
        return kwargs, physical

    def reference_metadata(self):
        _, context = reconstruct()
        return {k:v for k,v in context.items() if k != 'physical_reader'}


def build(output=OUTPUT):
    output = Path(output)
    require(not output.exists(), 'REFUSING_EXISTING_OUTPUT')
    rows, context = reconstruct()
    output.mkdir(parents=True)
    path = output / 'requests.json'
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    receipt = {'schema': VERSION, 'source_pins': PINS,
               'files_sha256': {'requests.json': sha(path)}, 'union_candidates':32,
               **DECLARATIONS,
               'model_context_indexes':context['model_context_indexes'],
               'duration_contract':'new_full_input_active_exposure6_or24_horizon24_elapsed0_unvalidated_choice',
               'reference_transfer':'cross_source_transfer_unvalidated',
               'P_outputs':0, 'G_outputs':0, 'Q_outputs':0, 'PT_loads':0, 'model_forwards':0,
               'preparation_grants_model_or_scientific_qualification':False}
    (output / 'manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return receipt
