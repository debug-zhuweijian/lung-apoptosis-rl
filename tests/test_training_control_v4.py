import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.engine.training_control import TrainingControl, checkpoint_completion


def test_convergence_finite_and_min_updates():
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
    ctl.advance(4, 4)
    ctl.observe(.94)
    assert ctl.completion() == {'status': 'complete', 'target_epochs': 1000, 'actual_epochs': 4, 'actual_updates': 4, 'reason': 'validation_plateau'}
    saved = {'epoch': 1, 'counters': {'optimizer_step': 1}, 'extra': {'training_completion': ctl.completion()}}
    assert checkpoint_completion(saved, 1000)['actual_epochs'] == 4
    for epoch, updates in ((5, 1), (1, 5)):
        changed = {**saved, 'epoch': epoch, 'counters': {'optimizer_step': updates}}
        with pytest.raises(ValueError, match='COMPLETION_COUNTER_MISMATCH'):
            checkpoint_completion(changed, 1000)
    saved['extra']['training_completion']['actual_epochs'] = -1
    with pytest.raises(ValueError):
        checkpoint_completion(saved, 1000)


def test_schedule_transition():
    cfg = {'training_schedule': {'mode': 'cfm_then_full', 'transition': 'validation_plateau', 'warmup_epochs': 1, 'patience': 1}}
    ctl = TrainingControl(cfg, 10)
    ctl.advance(1, 1)
    ctl.observe(1.)
    ctl.advance(2, 2)
    ctl.observe(1.)
    assert ctl.stage == 'full'
    assert ctl.state_dict()['transition_reason'] == 'validation_plateau'
    ctl.advance(3, 5)
    assert not ctl.stopped


def test_default_never_stops_early_and_interruption_rejected():
    ctl = TrainingControl({}, 3)
    for epoch in (1, 2):
        ctl.advance(epoch, epoch)
        ctl.observe(1.)
    assert not ctl.stopped
    with pytest.raises(ValueError, match='TRAINING_NOT_COMPLETE'):
        checkpoint_completion({'epoch': 2, 'extra': {'training_completion': ctl.completion()}}, 3)
    ctl.advance(3, 3)
    assert ctl.completion()['reason'] == 'max_epochs'


def test_completion_is_bound_to_its_own_target():
    ctl = TrainingControl({}, 1)
    ctl.advance(1, 1)
    saved = {'epoch': 1, 'counters': {'optimizer_step': 1}, 'extra': {'training_completion': ctl.completion()}}
    with pytest.raises(ValueError, match='TRAINING_COMPLETION_TARGET_CHANGED'):
        checkpoint_completion(saved, 1000)
    fresh = TrainingControl({}, 2)
    assert not fresh.stopped
    assert fresh.state_dict()['epoch'] == 0
