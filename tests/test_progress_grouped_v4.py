"""Grouped progress is display-only: truthful counters, no CUDA or repaint thread."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.monitoring.response_progress import ResponseTrainingProgress
from lung.monitoring.run import RunMonitor
from test_response_progress_v2 import Clock, Bar, StageBar


def grouped(progress):
    progress.state.update(condition_batching='grouped_compact_observed_full_direct',
                          member=1, members=5, epoch=9)
    return progress


def test_grouped_resume_counter_throttle_and_close_once(tmp_path):
    stages, clock = [], Clock()
    class Tracked(StageBar):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.n = kwargs.get('initial', 0)
            self.closes = 0
        def close(self):
            self.closes += 1
    def factory(**kwargs):
        bar = Tracked(**kwargs)
        stages.append(bar)
        return bar
    path = tmp_path/'progress.json'
    progress = grouped(ResponseTrainingProgress(Bar(), bar_factory=factory, clock=clock, state_path=path))
    progress.condition_start(321,644,'first,second')
    assert progress.state['conditions_done'] == stages[0].n == 320
    initial_refreshes = stages[0].refresh_count
    for _ in range(8):
        progress('CFM grouped backward',1,1)
        progress('MMD grouped backward',2,2)
    assert stages[0].refresh_count == initial_refreshes
    progress.condition_done()
    assert progress.state['conditions_done'] == 321
    progress.condition_done()
    assert progress.state['conditions_done'] == 322
    clock.now=2
    progress.condition_start(323,644,'third,fourth')
    assert len(stages)==1 and not stages[0].closes and stages[0].n==322
    assert progress.state['conditions_per_second']==1.
    progress('CFM grouped backward',1,1)
    progress.close('interrupted')
    progress.close('interrupted')
    assert stages[0].closes==1
    state=json.loads(path.read_text(encoding='utf-8'))
    assert state['conditions_done']==322 and state['status']=='interrupted'
    assert state['cfm_pairs_total']==0 and state['ode_steps_reported']==0
    assert state['cfm_chunks_total']==1
    assert stages[0].kwargs['ncols']==80 and stages[0].kwargs['position']==0


def test_grouped_non_tty_persistent_epoch_bar(tmp_path):
    from tqdm.utils import disp_len
    output,clock=io.StringIO(),Clock()
    monitor=SimpleNamespace(config={'progress':True})
    main=RunMonitor.progress(monitor,total=644,disable=True,file=output,position=0)
    created=[]
    def factory(**kwargs):
        bar=RunMonitor.progress(monitor,file=output,**kwargs)
        created.append(bar)
        return bar
    progress=grouped(ResponseTrainingProgress(main,bar_factory=factory,clock=clock))
    progress.condition_start(641,644,'a,b')
    clock.now=1
    progress('CFM grouped backward',1,1)
    clock.now=2
    progress('MMD grouped backward',2,2)
    progress.condition_done()
    progress.condition_done()
    clock.now=3
    progress.condition_start(643,644,'c,d')
    progress('CFM grouped backward',1,1)
    progress('MMD grouped backward',2,2)
    progress.condition_done()
    progress.condition_done()
    clock.now=4
    progress.close()
    main.close()
    text=output.getvalue()
    (tmp_path/'non-tty.txt').write_text(text,encoding='utf-8')
    assert len(created)==1
    assert '640/644' in text and '642/644' in text and '644/644' in text
    assert 'P1/5 E9' in text and '条件/s' in text
    assert 'pair' not in text and 'ODE' not in text and '\x1b[A' not in text
    assert text.count('\n')==1
    assert all(disp_len(frame.rstrip('\n'))<=80 for frame in text.split('\r'))
    print('non_tty_capture',repr(text))


def test_real_grouped_callback_smoke(tmp_path):
    import torch
    from lung.engine.config import load_config
    from lung.engine.synthetic import smoke_config,SyntheticData
    from lung.engine.response import response_rngs
    from lung.engine.response_training import train_conditions
    from lung.models import ModelConfig,build_response_model
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config=smoke_config(load_config())
        cfg={**config['response'],'condition_microbatch':2,'coupling':'random_product_control',
             'cfm_samples':4,'microbatch':4,'mmd_samples':3,'mmd_microbatch':3,'train_ode_steps':2}
        data=SyntheticData(config)
        model=build_response_model(ModelConfig(**cfg['model']),torch.as_tensor(data.neighbors))
        progress=grouped(ResponseTrainingProgress(Bar(),clock=Clock(),state_path=tmp_path/'state.json'))
        progress.condition_start(1,644,'T-0,T-1')
        rng,extra_rng,generator=response_rngs(23,'cpu')
        rows=train_conditions(model,data,['T-0','T-1'],rng,cfg,'cpu',2,progress=progress,
                              mmd_rng=extra_rng,mmd_generator=generator)
        assert len(rows)==2 and progress.state['phase']=='MMD grouped backward'
        for _ in rows:
            progress.condition_done()
        progress.close()
        assert progress.state['conditions_done']==2
        assert progress.state['cfm_chunks_total']==1
        assert progress.state['cfm_pairs_total']==progress.state['ode_steps_reported']==0
    finally:
        torch.set_num_threads(old)
