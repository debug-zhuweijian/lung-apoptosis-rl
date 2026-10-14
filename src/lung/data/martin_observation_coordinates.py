"""Registered fit-only reversible Martin log2TPM coordinates; no model."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .martin_native_rna import BASE, MartinNativeRNA

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / BASE / 'native-rna-inputs-v1'
OUTPUT = ROOT / BASE / 'observation-coordinates-v1'
SCHEMA = 'martin-observation-coordinates-v1'
PINS = {'src/lung/data/martin_native_rna.py': '4f85643234cd99f5b5e5b1f6a6127e49ca4ed2268b9050ec0e2fdc4a82a21352',
        BASE + '/native-rna-inputs-v1/manifest.json': 'dd13e4870baf785fff59081578fb89df209f70066346ec72c367429aabb2d745'}
DECLARATIONS = {'schema':SCHEMA,'domain':'log2TPM','coordinate':'joint',
                'author_transform':'log2(TPM+0.01)', 'affine':'fit_unique_GSM_float64_ddof0',
                'constant_scale':1, 'fold_count':8, 'gene_count':32461,
                'observed_genes':16167,'missing_genes':16294,
                'missing_values':'NaN_plus_mask', 'model_qualified':False,
                'scientific_qualified':False,'P_outputs':0,'G_outputs':0,'Q_outputs':0,
                'PT_loads':0,'model_forwards':0}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fit_unique_observed(entries):
    """Deduplicate source sample identity before moments; never multiply refs."""
    unique = {}
    for sample_id, values, mask in entries:
        require(values.dtype == np.float64 and mask.dtype == np.bool_ and values.shape == mask.shape,
                'EXPLICIT_FLOAT64_MASK_REQUIRED')
        if sample_id in unique:
            v, m = unique[sample_id]
            require(np.array_equal(m,mask) and np.array_equal(v,values,equal_nan=True), 'DUPLICATE_SAMPLE_CONFLICT')
        else:
            unique[sample_id] = (values, mask)
    require(bool(unique),'EMPTY_FIT')
    ids = sorted(unique)
    values = np.stack([unique[k][0] for k in ids])
    masks = np.stack([unique[k][1] for k in ids])
    require(np.all(masks == masks[0]) and np.isfinite(values[masks]).all()
            and np.isnan(values[~masks]).all(), 'OBSERVED_MASK_OR_VALUES_CHANGED')
    observed = masks[0]
    mean, scale = np.full(values.shape[1],np.nan), np.full(values.shape[1],np.nan)
    mean[observed] = values[:,observed].mean(0,dtype=np.float64)
    std = values[:,observed].std(0,ddof=0,dtype=np.float64)
    constant = np.zeros(values.shape[1],dtype=bool)
    constant[observed] = std == 0
    scale[observed] = np.where(std == 0,1.,std)
    return ids,mean,scale,observed,constant


def affine(values, mask, mean, scale, *, inverse=False):
    require(values.dtype == np.float64 and mask.dtype == np.bool_ and values.shape == mask.shape,
            'EXPLICIT_FLOAT64_MASK_REQUIRED')
    require(values.shape[-1] == len(mean) and np.isfinite(values[mask]).all()
            and np.isnan(values[~mask]).all(),'REGISTERED_OBSERVATION_VALUES_REQUIRED')
    result = np.full_like(values,np.nan)
    full_mean, full_scale = np.broadcast_to(mean,values.shape),np.broadcast_to(scale,values.shape)
    result[mask] = (values[mask]*full_scale[mask]+full_mean[mask] if inverse
                    else (values[mask]-full_mean[mask])/full_scale[mask])
    return result


def reconstruct():
    for path,digest in PINS.items():
        require(sha(ROOT/path) == digest,'FIXED_SOURCE_CHANGED')
    # Original reader authenticates arrays/roles/code against its own fixed sources.
    with MartinNativeRNA(ROOT,SOURCE,allowed_roles=('fit',)) as reader:
        output = {}
        for system in ('lobo','cycle'):
            requests = read(SOURCE / (system+'_requests.json'))
            for fold in sorted({r['fold_id'] for r in requests}):
                fit = reader.requests(system=system,fold_id=fold,role='fit')
                entries = []
                for request in fit:
                    parts = reader.supervision(request['request_id'],system=system,role='fit',
                                               domain='log2TPM',coordinate='joint')
                    for part in parts.values():
                        entries += [(sid,v,m) for sid,v,m in zip(part['sample_ids'],part['values'],part['observed_mask'])]
                ids,mean,scale,mask,constant = fit_unique_observed(entries)
                require(len(ids) == (45 if system == 'lobo' else 30),'UNIQUE_FIT_SCOPE_CHANGED')
                output[(system,fold)] = {'fit_sample_ids':ids,'mean':mean,'scale':scale,
                                        'observed_mask':mask,'constant_mask':constant}
        axes = reader.axes
        bindings = reader.manifest
    return output,axes,bindings


def validate_manifest(manifest, expected):
    require(set(manifest) == set(expected),'MANIFEST_SCHEMA_CHANGED')
    for key,value in expected.items():
        require(type(manifest[key]) is type(value) and json.dumps(manifest[key],sort_keys=True,
                allow_nan=False) == json.dumps(value,sort_keys=True,allow_nan=False),
                'MANIFEST_DECLARATION_CHANGED: '+key)


def manifest_for(folds, axes, bindings, files):
    return {**DECLARATIONS,'source_pins':PINS,'source_manifest':bindings,
            'consumer_sha256':sha(__file__), 'ordered_gene_ids':axes['joint_ensembl_ids'],
            'folds':[{'system':s,'fold_id':f,'file':f'fold_{i}.npz',
                      'fit_sample_ids':v['fit_sample_ids'],'unique_fit_count':len(v['fit_sample_ids'])}
                     for i,((s,f),v) in enumerate(folds.items())], 'files_sha256':files}


def prepare():
    require(not OUTPUT.exists(),'EXCLUSIVE_OUTPUT_ALREADY_EXISTS')
    folds,axes,bindings = reconstruct()
    OUTPUT.mkdir(parents=True)
    files = {}
    for i,values in enumerate(folds.values()):
        path = OUTPUT / f'fold_{i}.npz'
        np.savez(path,**{k:v for k,v in values.items() if k != 'fit_sample_ids'})
        files[path.name] = sha(path)
    manifest = manifest_for(folds,axes,bindings,files)
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


class MartinObservationCoordinates:
    def __init__(self, *, allowed_roles=('fit',), allow_final=False):
        require(type(allow_final) is bool,'STRICT_FINAL_BOOL_REQUIRED')
        self.reader = MartinNativeRNA(ROOT,SOURCE,allowed_roles=allowed_roles,allow_final=allow_final)
        try:
            self._folds,axes,bindings = reconstruct()
            files = {f'fold_{i}.npz':sha(OUTPUT/f'fold_{i}.npz') for i in range(8)}
            validate_manifest(read(OUTPUT/'manifest.json'),manifest_for(self._folds,axes,bindings,files))
            for i,v in enumerate(self._folds.values()):
                with np.load(OUTPUT/f'fold_{i}.npz',allow_pickle=False) as arrays:
                    require(set(arrays.files) == {'mean','scale','observed_mask','constant_mask'},'COORDINATE_SCHEMA_CHANGED')
                    for key in arrays.files:
                        require(arrays[key].dtype == v[key].dtype and
                                np.array_equal(arrays[key],v[key],equal_nan=True),'FIT_RECONSTRUCTION_MISMATCH')
        except Exception:
            self.reader.close()
            raise

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def read_request(self, request_id, *, system, fold_id, role, domain='log2TPM',
                     coordinate='joint', supervision=False, finite_placeholder=False):
        require(domain == 'log2TPM' and coordinate == 'joint','REGISTERED_DOMAIN_AXIS_REQUIRED')
        require(type(supervision) is bool and type(finite_placeholder) is bool,'STRICT_BOOL_REQUIRED')
        require((system,fold_id) in self._folds,'UNKNOWN_FOLD_OR_SYSTEM')
        fresh,axes,bindings = reconstruct()
        files = {f'fold_{i}.npz':sha(OUTPUT/f'fold_{i}.npz') for i in range(8)}
        validate_manifest(read(OUTPUT/'manifest.json'),manifest_for(fresh,axes,bindings,files))
        for i,v in enumerate(fresh.values()):
            with np.load(OUTPUT/f'fold_{i}.npz',allow_pickle=False) as arrays:
                require(set(arrays.files) == {'mean','scale','observed_mask','constant_mask'},'COORDINATE_SCHEMA_CHANGED')
                require(all(arrays[k].dtype == v[k].dtype and
                            np.array_equal(arrays[k],v[k],equal_nan=True) for k in arrays.files),
                        'FIT_RECONSTRUCTION_MISMATCH')
        requests = self.reader.requests(system=system,fold_id=fold_id,role=role)
        require(request_id in {r['request_id'] for r in requests},'REQUEST_FOLD_ROLE_MISMATCH')
        parts = (self.reader.supervision(request_id,system=system,role=role,domain=domain,coordinate=coordinate)
                 if supervision else {'reference':self.reader.reference(request_id,system=system,role=role,domain=domain,coordinate=coordinate)})
        coords = fresh[(system,fold_id)]
        result = {}
        for name,part in parts.items():
            z = affine(part['values'],part['observed_mask'],coords['mean'],coords['scale'])
            inverse = affine(z,part['observed_mask'],coords['mean'],coords['scale'],inverse=True)
            result[name] = {**part,'model_values':np.where(part['observed_mask'],z,0.) if finite_placeholder else z,
                            'inverse_values':inverse,'constant_mask':coords['constant_mask'].copy(),
                            'affine_design':DECLARATIONS['affine'],'fit_sample_ids':list(coords['fit_sample_ids']),
                            'finite_placeholder':finite_placeholder,'model_qualified':False,'scientific_qualified':False,
                            'system':system,'fold_id':fold_id,'role':role}
        return result
