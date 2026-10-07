import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.engine.training_control import TrainingControl, checkpoint_completion


def test_convergence_resume_finite_and_min_updates():
    cfg = {'early_stopping': {'patience': 2, 'min_delta': .1, 'min_updates': 4}}
    ctl = TrainingControl(cfg, 1000)
    ctl.advance(1, 1)
    assert ctl.observe(1.)
    with pytest.raises(FloatingPointError):
        ctl.observe(float('nan'))
    for epoch in (2, 3):
        ctl.advance(epoch, epoch)
        ctl.observe(.95)
    assert not ctl.stopped
    restored = TrainingControl(cfg, 1000, ctl.state_dict())
    restored.advance(4, 4)
    restored.observe(.94)
    assert restored.completion() == {'status': 'complete', 'target_epochs': 1000, 'actual_epochs': 4, 'actual_updates': 4, 'reason': 'validation_plateau'}
    saved = {'epoch': 4, 'counters': {'optimizer_step': 4}, 'extra': {'training_control': restored.state_dict()}}
    assert checkpoint_completion(saved, 1000)['actual_epochs'] == 4
    saved['epoch'] = 1000
    with pytest.raises(ValueError):
        checkpoint_completion(saved, 1000)


def test_schedule_and_save_clock(monkeypatch):
    cfg = {'training_schedule': {'mode': 'cfm_then_full', 'transition': 'validation_plateau', 'warmup_epochs': 1, 'patience': 1},
           'training_control': {'checkpoint_every_updates': 3, 'checkpoint_every_seconds': 10}}
    now = [0.]
    monkeypatch.setattr('lung.engine.training_control.time.monotonic', lambda: now[0])
    ctl = TrainingControl(cfg, 10)
    ctl.advance(1, 1)
    ctl.observe(1.)
    assert not ctl.checkpoint_due()
    ctl.advance(2, 2)
    ctl.observe(1.)
    assert ctl.stage == 'full'
    assert ctl.state_dict()['transition_reason'] == 'validation_plateau'
    now[0] = 11.
    assert ctl.checkpoint_due()
    ctl.checkpoint_saved()
    assert not ctl.checkpoint_due()
    ctl.advance(3, 5)
    assert ctl.checkpoint_due()
    assert not ctl.stopped


def test_default_never_stops_early_and_interruption_rejected():
    ctl = TrainingControl({}, 3)
    for epoch in (1, 2):
        ctl.advance(epoch, epoch)
        ctl.observe(1.)
    assert not ctl.stopped
    with pytest.raises(ValueError, match='TRAINING_NOT_COMPLETE'):
        checkpoint_completion({'epoch': 2, 'extra': {'training_control': ctl.state_dict()}}, 3)
    ctl.advance(3, 3)
    assert ctl.completion()['reason'] == 'max_epochs'


def test_old_max_epoch_cannot_complete_new_target():
    ctl = TrainingControl({}, 1)
    ctl.advance(1, 1)
    saved = {'epoch': 1, 'counters': {'optimizer_step': 1}, 'extra': {'training_control': ctl.state_dict()}}
    with pytest.raises(ValueError, match='TRAINING_NOT_COMPLETE'):
        checkpoint_completion(saved, 1000)
    restored = TrainingControl({}, 2, ctl.state_dict())
    assert not restored.stopped
    assert restored.state_dict()['target_epochs'] == 2
    with pytest.raises(ValueError, match='POLICY_CHANGED'):
        TrainingControl({'early_stopping': True}, 1, ctl.state_dict())
