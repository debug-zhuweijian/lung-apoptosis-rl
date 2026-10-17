"""Convergence and completion metadata for a fresh training run."""
import copy
import math


class TrainingControl:
    def __init__(self, config, target_epochs):
        self.target_epochs = int(target_epochs)
        if self.target_epochs < 1:
            raise ValueError("POSITIVE_TARGET_EPOCHS_REQUIRED")
        early = config.get('early_stopping', False)
        self.enabled = bool(early.get('enabled', True)) if isinstance(early, dict) else bool(early)
        self.policy = {**config.get('training_control', {}), **(early if isinstance(early, dict) else {})}
        self.schedule = config.get('training_schedule', config.get('response', {}).get('training_schedule', {}))
        mode = self.schedule.get('mode', 'full')
        if mode not in ('full', 'cfm_then_full', 'one_step_then_full'):
            raise ValueError('UNKNOWN_TRAINING_SCHEDULE')
        if self.schedule.get('transition', 'epochs') not in ('epochs', 'validation_plateau'):
            raise ValueError('UNKNOWN_STAGE_TRANSITION')
        for key in ('patience', 'max_warmup_epochs'):
            if key in self.schedule and (isinstance(self.schedule[key], bool) or not isinstance(self.schedule[key], int) or self.schedule[key] < 1):
                raise ValueError('INVALID_SCHEDULE_' + key)
        if 'min_delta' in self.schedule and (not math.isfinite(float(self.schedule['min_delta'])) or float(self.schedule['min_delta']) < 0):
            raise ValueError('INVALID_SCHEDULE_min_delta')
        for key, default, minimum in [('patience', 10, 1), ('min_updates', 0, 0),
                ('validation_every_epochs', 1, 1), ('validation_every_updates', 0, 0)]:
            value = self.policy.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError('INVALID_TRAINING_CONTROL_' + key)
            self.policy[key] = value
        for key, default in [('min_delta', 0.)]:
            value = float(self.policy.get(key, default))
            if not math.isfinite(value) or value < 0:
                raise ValueError('INVALID_TRAINING_CONTROL_' + key)
            self.policy[key] = value
        if mode != 'full' and (not isinstance(self.schedule.get('warmup_epochs', 1), int)
                or not 0 < self.schedule.get('warmup_epochs', 1) < self.target_epochs):
            raise ValueError('SCHEDULE_REQUIRES_POSITIVE_WARMUP_AND_FULL_STAGE')
        self.state = dict(schema_version=1, target_epochs=self.target_epochs, epoch=0, updates=0, stage='full' if mode == 'full' else mode.split('_then')[0],
                          best=None, bad_validations=0, full_updates=0, full_validations=0, requires_validation=self.enabled or mode != 'full', last_validation_epoch=0, last_validation_update=0,
                          reason=None, status='running')

    @property
    def stage(self):
        return self.state['stage']

    @property
    def stopped(self):
        return self.state['status'] == 'complete'

    def advance(self, epoch, updates):
        if epoch < self.state['epoch'] or updates < self.state['updates']:
            raise ValueError('TRAINING_COUNTER_REGRESSION')
        if self.stage == 'full':
            self.state['full_updates'] += updates - self.state['updates']
        self.state.update(epoch=int(epoch), updates=int(updates))
        if self.stage != 'full' and (epoch >= min(self.schedule.get('max_warmup_epochs', self.target_epochs - 1), self.target_epochs - 1) or (self.schedule.get('transition', 'epochs') == 'epochs' and epoch >= self.schedule.get('warmup_epochs', 1))):
            self.state.update(stage='full', best=None, bad_validations=0, transition_reason='epoch_limit', transition_epoch=epoch)
        if epoch >= self.target_epochs:
            self.state.update(status='complete', reason='max_epochs')

    def validation_due(self):
        s, p = self.state, self.policy
        return (s['epoch'] - s['last_validation_epoch'] >= p['validation_every_epochs'] or
                (p['validation_every_updates'] > 0 and s['updates'] - s['last_validation_update'] >= p['validation_every_updates']))

    def observe(self, loss):
        value = float(loss)
        if not math.isfinite(value):
            raise FloatingPointError('NONFINITE_VALIDATION_LOSS')
        s = self.state
        if self.stage == 'full' and s['full_updates'] > 0:
            s['full_validations'] += 1
        s.update(last_validation_epoch=s['epoch'], last_validation_update=s['updates'])
        improved = s['best'] is None or value < s['best'] - (self.schedule.get('min_delta', self.policy['min_delta']) if self.stage != 'full' else self.policy['min_delta'])
        if improved:
            s.update(best=value, bad_validations=0)
        else:
            s['bad_validations'] += 1
        if self.stage != 'full' and self.schedule.get('transition') == 'validation_plateau' and s['epoch'] >= self.schedule.get('warmup_epochs', 1) and s['bad_validations'] >= self.schedule.get('patience', self.policy['patience']):
            s.update(stage='full', best=None, bad_validations=0, transition_reason='validation_plateau', transition_epoch=s['epoch'])
        if self.enabled and self.stage == 'full' and s['full_validations'] > 0 and s['updates'] >= self.policy['min_updates'] and s['bad_validations'] >= self.policy['patience']:
            s.update(status='complete', reason='validation_plateau')
        return improved

    def state_dict(self):
        return copy.deepcopy(self.state)

    def completion(self):
        return dict(status=self.state['status'], target_epochs=self.target_epochs,
                    actual_epochs=self.state['epoch'], actual_updates=self.state['updates'], reason=self.state['reason'])


def checkpoint_completion(saved, target_epochs):
    """Validate completion of this run, independently of the selected best epoch."""
    completion = saved.get('extra', {}).get('training_completion', {})
    if completion.get('status') != 'complete':
        raise ValueError('TRAINING_NOT_COMPLETE')
    if completion.get('target_epochs') != target_epochs:
        raise ValueError('TRAINING_COMPLETION_TARGET_CHANGED')
    reason = completion.get('reason')
    if reason not in ('max_epochs', 'validation_plateau'):
        raise ValueError('TRAINING_NOT_COMPLETE')
    epochs, updates = completion.get('actual_epochs'), completion.get('actual_updates')
    if type(epochs) is not int or not 0 <= epochs <= target_epochs or type(updates) is not int or updates < 1:
        raise ValueError('COMPLETION_COUNTER_MISMATCH')
    selected_epoch = saved.get('epoch')
    selected_updates = saved.get('counters', {}).get('optimizer_step')
    if (type(selected_epoch) is not int or not 0 <= selected_epoch <= epochs
            or type(selected_updates) is not int or not 0 < selected_updates <= updates):
        raise ValueError('COMPLETION_COUNTER_MISMATCH')
    if reason == 'max_epochs' and epochs != target_epochs:
        raise ValueError('TRAINING_NOT_COMPLETE')
    return dict(completion)
