import numpy as np
import pytest
from lung.data import martin_observation_coordinates as m


def entries():
    mask = np.array([True,True,False])
    return [('a',np.array([0.,2.,np.nan]),mask),('b',np.array([2.,2.,np.nan]),mask)]


def test_duplicate_reference_does_not_weight_moments():
    a = m.fit_unique_observed(entries())
    b = m.fit_unique_observed(entries()+entries()*3)
    assert a[0] == b[0] == ['a','b']
    assert np.array_equal(a[1],b[1],equal_nan=True)
    assert a[1][0] == 1 and a[2][1] == 1 and a[4].tolist() == [False,True,False]


def test_mask_zero_constant_and_inverse():
    ids,mu,scale,mask,constant = m.fit_unique_observed(entries())
    values = np.stack([e[1] for e in entries()])
    masks = np.broadcast_to(mask,values.shape)
    z = m.affine(values,masks,mu,scale)
    inverse = m.affine(z,masks,mu,scale,inverse=True)
    assert inverse[0,0] == 0 and np.isnan(inverse[:,2]).all()
    assert np.max(np.abs(values[masks]-inverse[masks])) == 0


def test_fit_role_collection_ignores_holdout_and_deduplicates_actual():
    folds,_,_ = m.reconstruct()
    assert [len(v['fit_sample_ids']) for v in folds.values()] == [45]*4+[30]*4
    requests = m.read(m.SOURCE/'lobo_requests.json')
    for (system,fold),v in folds.items():
        if system == 'lobo':
            excluded = {sid for r in requests if r['fold_id']==fold and r['role']=='final_test'
                        for sid in r['target_sample_ids']+r['physical_reference_ids']}
            assert not excluded.intersection(v['fit_sample_ids'])


@pytest.mark.parametrize('key,value',[('model_qualified',0),('P_outputs',False),('scientific_qualified',True),('domain','estimated_counts'),('gene_count',19712)])
def test_declarations_strict_types_and_scope(key,value):
    expected = dict(m.DECLARATIONS)
    actual = dict(expected)
    actual[key] = value
    with pytest.raises(ValueError):
        m.validate_manifest(actual,expected)


def test_actual_readback_inverse_and_finite_placeholder():
    with m.MartinObservationCoordinates() as reader:
        fold = next(iter(reader._folds))
        request = reader.reader.requests(system=fold[0],fold_id=fold[1],role='fit')[0]
        parts = reader.read_request(request['request_id'],system=fold[0],fold_id=fold[1],role='fit',
                                    supervision=True,finite_placeholder=True)
        for part in parts.values():
            assert part['values'].shape == (3,32461) and part['observed_mask'].sum()==3*16167
            assert np.isfinite(part['model_values']).all()
            assert np.isnan(part['inverse_values'][~part['observed_mask']]).all()
            assert np.max(np.abs(part['values'][part['observed_mask']]-part['inverse_values'][part['observed_mask']])) < 1e-12


def test_final_access_only_explicitly_authorized():
    with pytest.raises(PermissionError):
        m.MartinObservationCoordinates(allowed_roles=('final_test',))
    with pytest.raises(ValueError):
        m.MartinObservationCoordinates(allow_final=1)
    with m.MartinObservationCoordinates(allowed_roles=('final_test',),allow_final=True) as reader:
        fold = next(iter(reader._folds))
        r = reader.reader.requests(system=fold[0],fold_id=fold[1],role='final_test')[0]
        part = reader.read_request(r['request_id'],system=fold[0],fold_id=fold[1],role='final_test')['reference']
        assert part['model_qualified'] is False


def test_domain_axis_system_fold_role_and_bool_rejected():
    with m.MartinObservationCoordinates() as reader:
        fold = next(iter(reader._folds))
        r = reader.reader.requests(system=fold[0],fold_id=fold[1],role='fit')[0]
        args = dict(system=fold[0],fold_id=fold[1],role='fit')
        for change in ({'domain':'estimated_counts'},{'coordinate':'source'},{'system':'unknown'},
                       {'fold_id':'unknown'},{'role':'final_test'},{'finite_placeholder':1}):
            with pytest.raises((ValueError,PermissionError)):
                reader.read_request(r['request_id'],**{**args,**change})


def test_source_hash_tamper_rejected(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda path:'0'*64)
    with pytest.raises(ValueError,match='FIXED_SOURCE_CHANGED'):
        m.reconstruct()


def test_self_rehashed_coordinate_array_tamper_rejected(monkeypatch):
    original = m.np.load
    class Tampered:
        def __init__(self, arrays):
            self.files = arrays.files
            self.values = {k:arrays[k].copy() for k in self.files}
            self.values['mean'][0] += 1
        def __enter__(self):
            return self
        def __exit__(self,*args):
            return None
        def __getitem__(self,key):
            return self.values[key]
    def changed(path,*args,**kwargs):
        arrays = original(path,*args,**kwargs)
        if str(path).endswith('fold_0.npz'):
            with arrays:
                return Tampered(arrays)
        return arrays
    monkeypatch.setattr(m.np,'load',changed)
    with pytest.raises(ValueError,match='FIT_RECONSTRUCTION_MISMATCH'):
        m.MartinObservationCoordinates()
