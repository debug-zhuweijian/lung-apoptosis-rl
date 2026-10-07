"""CPU synthetic actual CFM/MMD path; no real data or GPU training."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.engine import response
from lung.engine.config import load_config
from lung.engine.synthetic import smoke_config, SyntheticData
from lung.engine.checkpoint import load_checkpoint, atomic_save

class Data(SyntheticData):
    def __init__(self, config):
        super().__init__(config)
        self.visits = []
    def condition_ids(self, role):
        return ['T-0', 'T-1', 'T-2'] if role == 'T' else super().condition_ids(role)
    def sample_response(self, key, n, rng, role):
        self.visits.append((role, key))
        batch = super().sample_response(key, n, rng, role)
        batch['control'] *= .1
        batch['target'] = .2 + (batch['target'] - .2) * .1
        return batch

class Monitor:
    def __init__(self, root, stop=None):
        self.weights = root / 'weights'
        self.logdir = root / 'logs'
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.config = {'progress': False}
        self.records, self.stop = [], stop
    def progress(self, *args, **kwargs):
        return tqdm(*args, **{**kwargs, 'disable': True})
    def record(self, stage, step, metrics, axis='epoch'):
        self.records.append((axis, step, metrics))
        if (axis, step) == self.stop:
            raise KeyboardInterrupt('forced committed boundary')

def equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a) == set(b)
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b

@pytest.fixture
def cfg():
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    config=smoke_config(load_config())
    config['epochs']=2
    config['response'].update(members=1, microbatch=2, mmd_microbatch=2,
        ot_samples=2, cfm_samples=2, mmd_samples=2, train_ode_steps=1, inference_ode_steps=1)
    config['response']['model']['backbone']='gene_token'
    with patch('torch.cuda.is_available', return_value=False):
        yield config
    torch.set_num_threads(threads)

def checkpoint(mon):
    return mon.weights/'response/member_1/last.pt'

@pytest.mark.parametrize('boundary', ['first_update','last_group','before_validation','legacy_epoch','partial_group','before_log'])
def test_actual_training_resume_exact(tmp_path,cfg,boundary):
    base, interrupted = Monitor(tmp_path/'base'), Monitor(tmp_path/'resume')
    data0,data1=Data(cfg),Data(cfg)
    response.train_response(data0,cfg,base)
    if boundary=='before_log':
        original = response.save_training
        def stop_after_save(path, *args, **kwargs):
            original(path, *args, **kwargs)
            if kwargs.get('counters', {}).get('optimizer_step') == 1:
                raise KeyboardInterrupt('saved before log')
        with patch.object(response, 'save_training', side_effect=stop_after_save):
            with pytest.raises(KeyboardInterrupt): response.train_response(data1,cfg,interrupted)
    elif boundary=='partial_group':
        original = response.train_condition
        calls = []
        def stop_partial(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt('uncommitted gradients')
            return result
        with patch.object(response, 'train_condition', side_effect=stop_partial):
            with pytest.raises(KeyboardInterrupt): response.train_response(data1,cfg,interrupted)
        # Two visits in the uncommitted condition must be replayed; committed ones must not.
        assert len(data1.visits) == 6
        data1.visits = data1.visits[:4]
    elif boundary=='before_validation':
        with patch.object(response,'evaluate_response',side_effect=KeyboardInterrupt('before validation')):
            with pytest.raises(KeyboardInterrupt): response.train_response(data1,cfg,interrupted)
    else:
        interrupted.stop={'first_update':('optimizer_step',1), 'last_group':('optimizer_step',2),
                          'legacy_epoch':('epoch',1)}[boundary]
        with pytest.raises(KeyboardInterrupt): response.train_response(data1,cfg,interrupted)
    saved=load_checkpoint(checkpoint(interrupted))
    if boundary=='legacy_epoch':
        saved['extra']={'best':saved['extra']['best']}
        atomic_save(checkpoint(interrupted),saved)
    else:
        assert saved['epoch']==0
        assert saved['extra']['update_resume']['next_condition_index']==(2 if boundary in ('first_update','before_log','partial_group') else 3)
        assert saved['counters']['optimizer_step']==(1 if boundary in ('first_update','before_log','partial_group') else 2)
    interrupted.stop=None
    response.train_response(data1,cfg,interrupted,resume=True)
    expected,actual=load_checkpoint(checkpoint(base)),load_checkpoint(checkpoint(interrupted))
    for key in ('model','optimizer','rng','counters','extra','epoch'): equal(expected[key],actual[key])
    equal(data0.visits,data1.visits)
    expected_records = base.records[1:] if boundary == 'before_log' else base.records
    equal(expected_records,interrupted.records)

@pytest.mark.parametrize('corruption',['position','counter','optimizer','schedule'])
def test_corrupt_resume_rejected(tmp_path,cfg,corruption):
    mon=Monitor(tmp_path,('optimizer_step',1))
    with pytest.raises(KeyboardInterrupt): response.train_response(Data(cfg),cfg,mon)
    saved=load_checkpoint(checkpoint(mon))
    if corruption=='position': saved['extra']['update_resume']['next_condition_index']=1
    elif corruption=='counter': saved['counters']['optimizer_step']=99
    elif corruption=='optimizer': next(iter(saved['optimizer']['state'].values()))['step']+=1
    else: saved['extra']['update_resume']['total_updates']+=1
    atomic_save(checkpoint(mon),saved)
    with pytest.raises(ValueError,match='RESPONSE_RESUME_'):
        response.train_response(Data(cfg),cfg,mon,resume=True)
